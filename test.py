import json
import numpy as np
from stable_baselines3 import PPO
import pickle
from recommendation import CurriculumMultiStudentEnv
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

def visualize_curriculum_graph(graph, save_path="curriculum_graph.png"):
    """Visualize a portion of the curriculum graph"""
    plt.figure(figsize=(12, 8))
    
    # Create a hierarchical layout based on prerequisites
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Color nodes by category
    color_map = {"AI": "lightblue", "Security": "lightcoral", 
                 "Data Science": "lightgreen", "Elective": "lightyellow"}
    node_colors = [color_map.get(graph.nodes[node].get('category', 'Elective'), 'lightgray') 
                   for node in graph.nodes()]
    
    # Draw the graph
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, 
            node_size=1000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=cat)
                       for cat, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Curriculum Graph Structure", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {save_path}")

def analyze_student_results(results, students):
    """Analyze and display student recommendation results"""
    print("\n" + "="*60)
    print("STUDENT RECOMMENDATION ANALYSIS")
    print("="*60)
    
    total_rewards = []
    graduation_rates = []
    
    for student_id, result in results.items():
        # Find the student profile
        student = next((s for s in students if s['ID'] == student_id), None)
        if not student:
            continue
            
        print(f"\nStudent ID: {student_id}")
        print(f"Interests: {student['Interests']}")
        print(f"Starting GPA: {student['GPA']}")
        print(f"Total Reward: {result['total_reward']:.2f}")
        print(f"Graduated: {result.get('graduated', 'Unknown')}")
        
        # Show course recommendations by term
        print("Course Recommendations by Term:")
        for term, courses in enumerate(result['recommended_courses'], 1):
            if isinstance(courses, list) and any(courses):
                course_names = [f"Course_{i}" for i, selected in enumerate(courses) if selected]
                print(f"  Term {term}: {course_names}")
        
        total_rewards.append(result['total_reward'])
        graduation_rates.append(1 if result.get('graduated', False) else 0)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Average Total Reward: {np.mean(total_rewards):.2f}")
    print(f"Reward Standard Deviation: {np.std(total_rewards):.2f}")
    print(f"Graduation Rate: {np.mean(graduation_rates)*100:.1f}%")
    print(f"Total Students Analyzed: {len(results)}")

def main():
    # Load curriculum graph
    print("Loading curriculum graph...")
    with open("AI-Powered-Academic-Advisor/curriculum.pkl", "rb") as f:
        curi = pickle.load(f)
    
    # Visualize curriculum graph
    print("Creating curriculum graph visualization...")
    visualize_curriculum_graph(curi)
    
    # Load test students
    print("Loading test students...")
    with open("AI-Powered-Academic-Advisor/test_student_profiles.json") as f:
        test_students = json.load(f)
    
    print(f"Loaded {len(test_students)} test students")
    
    # Create test environment
    env = CurriculumMultiStudentEnv(test_students, curi)
    
    # Load trained model
    print("Loading trained model...")
    model = PPO.load("AI-Powered-Academic-Advisor/ppo_curriculum")
    
    results = {}
    detailed_results = {}
    
    print("\nRunning recommendations for test students...")
    
    for student_idx, student in enumerate(test_students):
        print(f"Processing student {student_idx + 1}/{len(test_students)}: {student['ID']}")
        
        obs, _ = env.reset()
        done = False
        recommended_courses = []
        total_reward = 0
        term_rewards = []
        term_info = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Save recommended courses for this term
            course_selection = action.tolist() if hasattr(action, 'tolist') else action
            recommended_courses.append(course_selection)
            
            # Get course names for this term
            selected_courses = [course for i, course in enumerate(curi.nodes) 
                              if course_selection[i] == 1]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            term_rewards.append(reward)
            
            # Store detailed info (convert numpy types to Python types)
            term_info.append({
                "selected_courses": selected_courses,
                "reward": float(reward),
                "term_gpa": float(info.get("term_gpa", 0)),
                "passed": int(info.get("passed", 0)),
                "failed": int(info.get("failed", 0)),
                "graduation_progress": float(info.get("graduation_progress", 0))
            })
        
        # Store results (convert numpy types to Python types for JSON serialization)
        results[student['ID']] = {
            "recommended_courses": recommended_courses,
            "total_reward": float(total_reward),
            "graduated": bool(info.get("graduated", False)),
            "final_gpa": float(obs.get("GPA", [0])[0]) if "GPA" in obs else 0.0,
            "graduation_progress": float(info.get("graduation_progress", 0))
        }
        
        detailed_results[student['ID']] = {
            "student_profile": student,
            "term_details": term_info,
            "summary": results[student['ID']]
        }
    
    # Save results with JSON-safe data types
    print("\nSaving results...")
    
    # Convert any remaining numpy types to Python types
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Convert results to JSON-serializable format
    json_safe_results = convert_to_json_serializable(results)
    json_safe_detailed_results = convert_to_json_serializable(detailed_results)
    
    with open("AI-Powered-Academic-Advisor/recommended_courses_results.json", "w") as f:
        json.dump(json_safe_results, f, indent=4)
    
    with open("AI-Powered-Academic-Advisor/detailed_results.json", "w") as f:
        json.dump(json_safe_detailed_results, f, indent=4)
    
    # Analyze results
    analyze_student_results(results, test_students)
    
    # Create performance visualization
    create_performance_plots(results, test_students)
    
    print(f"\nCompleted analysis for {len(test_students)} students.")
    print("Files saved:")
    print("- recommended_courses_results.json")
    print("- detailed_results.json")
    print("- curriculum_graph.png")
    print("- performance_analysis.png")

def create_performance_plots(results, students):
    """Create performance visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    rewards = [result['total_reward'] for result in results.values()]
    graduation_status = [result['graduated'] for result in results.values()]
    final_gpas = [result['final_gpa'] for result in results.values()]
    
    # Plot 1: Reward distribution
    axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Total Rewards')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Number of Students')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Graduation rate
    grad_counts = [sum(graduation_status), len(graduation_status) - sum(graduation_status)]
    axes[0, 1].pie(grad_counts, labels=['Graduated', 'Not Graduated'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title('Graduation Rate')
    
    # Plot 3: Final GPA distribution
    axes[1, 0].hist(final_gpas, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Final GPAs')
    axes[1, 0].set_xlabel('Final GPA')
    axes[1, 0].set_ylabel('Number of Students')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward vs GPA scatter
    axes[1, 1].scatter(final_gpas, rewards, alpha=0.6, color='purple')
    axes[1, 1].set_title('Total Reward vs Final GPA')
    axes[1, 1].set_xlabel('Final GPA')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("AI-Powered-Academic-Advisor/performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()