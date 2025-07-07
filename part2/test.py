import json
import numpy as np
from stable_baselines3 import PPO
import pickle
import os
from recommendation import CurriculumMultiStudentEnv, create_heuristic_recommender
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd

def check_file_dependencies():
    """Check if all required files exist"""
    required_files = [
        "AI-Powered-Academic-Advisor/data/curriculum.pkl",
        "AI-Powered-Academic-Advisor/data/test_student_profiles.json",
        "AI-Powered-Academic-Advisor/results/ppo_curriculum.zip"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def load_test_data():
    """Load and validate test data"""
    print("Loading test data...")
    
    # Load curriculum graph
    try:
        with open("AI-Powered-Academic-Advisor/data/curriculum.pkl", "rb") as f:
            curriculum = pickle.load(f)
        print(f"✓ Curriculum loaded: {len(curriculum.nodes)} courses, {len(curriculum.edges)} prerequisites")
    except Exception as e:
        print(f"✗ Error loading curriculum: {e}")
        return None, None
    
    # Load student profiles
    try:
        with open("AI-Powered-Academic-Advisor/data/test_student_profiles.json") as f:
            students = json.load(f)
        print(f"✓ Student profiles loaded: {len(students)} students")
        
        # Create test subset if too many students
        if len(students) > 20:
            test_students = students[:20]
            print(f"Using first 20 students for testing")
        else:
            test_students = students
            
    except Exception as e:
        print(f"✗ Error loading student profiles: {e}")
        return None, None
    
    return curriculum, test_students

def validate_environment(curriculum, students):
    """Validate environment setup"""
    print("\nValidating environment setup...")
    
    try:
        # Test environment creation
        env = CurriculumMultiStudentEnv(students, curriculum)
        obs, _ = env.reset()
        
        print("✓ Environment created successfully")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Number of courses: {len(curriculum.nodes)}")
        
        # Test a few steps
        for i in range(3):
            # Create a valid action (select 3-4 courses)
            eligible_courses = obs['eligible_courses']
            action = np.zeros(len(eligible_courses), dtype=int)
            eligible_indices = np.where(eligible_courses == 1)[0]
            
            if len(eligible_indices) >= 3:
                selected_indices = np.random.choice(eligible_indices, 
                                                  min(4, len(eligible_indices)), 
                                                  replace=False)
                action[selected_indices] = 1
            
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                obs, _ = env.reset()  # Reset for next student
        
        print("✓ Environment step testing successful")
        env.close() if hasattr(env, 'close') else None
        return True
        
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
        return False

def load_trained_model():
    """Load the trained PPO model"""
    print("\nLoading trained model...")
    
    try:
        model = PPO.load("AI-Powered-Academic-Advisor/results/ppo_curriculum")
        print("✓ Trained model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please ensure the model was trained and saved properly")
        return None

def run_rl_recommendations(model, curriculum, students):
    """Run RL-based recommendations"""
    print("\nRunning RL-based recommendations...")
    
    env = CurriculumMultiStudentEnv(students, curriculum)
    results = {}
    
    for student_idx, student in enumerate(students):
        print(f"Processing student {student_idx + 1}/{len(students)}: {student['ID']}")
        
        try:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            term_count = 0
            recommended_courses = []
            term_details = []
            
            while not done and term_count < 12:  # Limit to 12 terms
                # Get model prediction
                action, _ = model.predict(obs, deterministic=True)
                
                # Convert action to course selection
                selected_courses = []
                for i, select in enumerate(action):
                    if select == 1:
                        course_name = list(curriculum.nodes())[i]
                        selected_courses.append(course_name)
                
                recommended_courses.append(selected_courses)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                term_count += 1
                
                # Store term details
                term_details.append({
                    'term': term_count,
                    'selected_courses': selected_courses,
                    'reward': float(reward),
                    'term_gpa': float(info.get('current_gpa', 0)),
                    'graduation_progress': float(info.get('graduation_progress', 0)),
                    'passed': info.get('passed', 0),
                    'failed': info.get('failed', 0)
                })
            
            # Store results
            results[student['ID']] = {
                'total_reward': float(total_reward),
                'graduated': bool(info.get('graduated', False)),
                'final_gpa': float(info.get('current_gpa', 0)),
                'terms_completed': term_count,
                'recommended_courses': recommended_courses,
                'term_details': term_details
            }
            
        except Exception as e:
            print(f"Error processing student {student['ID']}: {e}")
            results[student['ID']] = {
                'error': str(e),
                'total_reward': 0,
                'graduated': False
            }
    
    return results

def run_heuristic_recommendations(curriculum, students):
    """Run heuristic-based recommendations for comparison"""
    print("\nRunning heuristic-based recommendations...")
    
    heuristic_recommender = create_heuristic_recommender(curriculum)
    results = {}
    
    for student_idx, student in enumerate(students):
        print(f"Processing student {student_idx + 1}/{len(students)}: {student['ID']}")
        
        try:
            # Simulate course progression using heuristic
            current_profile = student.copy()
            total_reward = 0
            term_count = 0
            recommended_courses = []
            
            while term_count < 12:  # Limit to 12 terms
                # Get heuristic recommendations
                recommendations = heuristic_recommender(current_profile, max_courses=4)
                
                if not recommendations:
                    break  # No more courses to recommend
                
                recommended_courses.append(recommendations)
                
                # Simulate course outcomes (simplified)
                passed_courses = []
                for course in recommendations:
                    # Simple success probability based on GPA
                    success_prob = 0.8 + (current_profile.get('GPA', 2.0) - 2.0) * 0.1
                    if np.random.random() < success_prob:
                        grade = np.random.choice(['A', 'B', 'C'], p=[0.4, 0.4, 0.2])
                        current_profile['Completed_Courses'][course] = grade
                        passed_courses.append(course)
                        total_reward += 5  # Basic reward for passing
                
                # Update GPA
                all_grades = [g for g in current_profile['Completed_Courses'].values() if g != 'F']
                if all_grades:
                    grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
                    current_profile['GPA'] = np.mean([grade_points[g] for g in all_grades])
                
                term_count += 1
                
                # Check graduation
                completed_count = len([c for c, g in current_profile['Completed_Courses'].items() if g != 'F'])
                if completed_count >= len(curriculum.nodes):
                    total_reward += 50  # Graduation bonus
                    break
            
            # Store results
            completed_count = len([c for c, g in current_profile['Completed_Courses'].items() if g != 'F'])
            results[student['ID']] = {
                'total_reward': float(total_reward),
                'graduated': completed_count >= len(curriculum.nodes),
                'final_gpa': float(current_profile.get('GPA', 0)),
                'terms_completed': term_count,
                'recommended_courses': recommended_courses
            }
            
        except Exception as e:
            print(f"Error processing student {student['ID']}: {e}")
            results[student['ID']] = {
                'error': str(e),
                'total_reward': 0,
                'graduated': False
            }
    
    return results

def analyze_and_compare_results(rl_results, heuristic_results, students):
    """Analyze and compare RL vs heuristic results"""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: RL vs HEURISTIC")
    print("="*60)
    
    # Extract metrics
    rl_metrics = extract_metrics(rl_results)
    heuristic_metrics = extract_metrics(heuristic_results)
    
    # Print comparison
    print(f"{'Metric':<25} {'RL Agent':<15} {'Heuristic':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('avg_reward', 'Avg Reward'),
        ('graduation_rate', 'Graduation Rate'),
        ('avg_gpa', 'Avg Final GPA'),
        ('avg_terms', 'Avg Terms')
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        rl_val = rl_metrics[metric_key]
        heur_val = heuristic_metrics[metric_key]
        diff = rl_val - heur_val
        
        if metric_key == 'graduation_rate':
            print(f"{metric_name:<25} {rl_val:.1%}<15 {heur_val:.1%}<15 {diff:+.1%}<15")
        else:
            print(f"{metric_name:<25} {rl_val:.2f}<15 {heur_val:.2f}<15 {diff:+.2f}<15")

    # Statistical significance (simple t-test)
    from scipy import stats
    
    rl_rewards = [r.get('total_reward', 0) for r in rl_results.values()]
    heur_rewards = [r.get('total_reward', 0) for r in heuristic_results.values()]
    
    try:
        t_stat, p_value = stats.ttest_ind(rl_rewards, heur_rewards)
        print(f"\nStatistical significance (t-test): p = {p_value:.4f}")
        if p_value < 0.05:
            print("✓ Difference is statistically significant")
        else:
            print("✗ Difference is not statistically significant")
    except:
        print("Could not perform statistical test")
    
    return rl_metrics, heuristic_metrics

def extract_metrics(results):
    """Extract key metrics from results"""
    valid_results = [r for r in results.values() if 'error' not in r]
    
    if not valid_results:
        return {
            'avg_reward': 0,
            'graduation_rate': 0,
            'avg_gpa': 0,
            'avg_terms': 0
        }
    
    return {
        'avg_reward': np.mean([r['total_reward'] for r in valid_results]),
        'graduation_rate': np.mean([r['graduated'] for r in valid_results]),
        'avg_gpa': np.mean([r['final_gpa'] for r in valid_results]),
        'avg_terms': np.mean([r['terms_completed'] for r in valid_results])
    }

def create_visualization(rl_results, heuristic_results, save_dir="AI-Powered-Academic-Advisor/results"):
    """Create comprehensive visualization of results"""
    print("\nCreating performance visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for plotting
    rl_rewards = [r.get('total_reward', 0) for r in rl_results.values() if 'error' not in r]
    heur_rewards = [r.get('total_reward', 0) for r in heuristic_results.values() if 'error' not in r]
    
    rl_gpas = [r.get('final_gpa', 0) for r in rl_results.values() if 'error' not in r]
    heur_gpas = [r.get('final_gpa', 0) for r in heuristic_results.values() if 'error' not in r]
    
    rl_grad_rates = [r.get('graduated', False) for r in rl_results.values() if 'error' not in r]
    heur_grad_rates = [r.get('graduated', False) for r in heuristic_results.values() if 'error' not in r]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RL Agent vs Heuristic Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward comparison
    axes[0, 0].hist(rl_rewards, alpha=0.7, label='RL Agent', bins=15, color='blue')
    axes[0, 0].hist(heur_rewards, alpha=0.7, label='Heuristic', bins=15, color='red')
    axes[0, 0].set_title('Total Reward Distribution')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: GPA comparison
    axes[0, 1].hist(rl_gpas, alpha=0.7, label='RL Agent', bins=15, color='blue')
    axes[0, 1].hist(heur_gpas, alpha=0.7, label='Heuristic', bins=15, color='red')
    axes[0, 1].set_title('Final GPA Distribution')
    axes[0, 1].set_xlabel('Final GPA')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Graduation rate comparison
    rl_grad_count = [sum(rl_grad_rates), len(rl_grad_rates) - sum(rl_grad_rates)]
    heur_grad_count = [sum(heur_grad_rates), len(heur_grad_rates) - sum(heur_grad_rates)]
    
    x = np.arange(2)
    width = 0.35
    
    axes[1, 0].bar(x - width/2, [rl_grad_count[0]/len(rl_grad_rates), 
                                 rl_grad_count[1]/len(rl_grad_rates)], 
                   width, label='RL Agent', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, [heur_grad_count[0]/len(heur_grad_rates), 
                                 heur_grad_count[1]/len(heur_grad_rates)], 
                   width, label='Heuristic', color='red', alpha=0.7)
    axes[1, 0].set_title('Graduation Rate Comparison')
    axes[1, 0].set_xlabel('Outcome')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Graduated', 'Not Graduated'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Box plot comparison
    combined_data = [rl_rewards, heur_rewards]
    axes[1, 1].boxplot(combined_data, labels=['RL Agent', 'Heuristic'])
    axes[1, 1].set_title('Reward Distribution Comparison')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {os.path.join(save_dir, 'comparison_analysis.png')}")

def save_results(rl_results, heuristic_results, save_dir="AI-Powered-Academic-Advisor/results"):
    """Save results to files"""
    print("\nSaving results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save RL results
    with open(os.path.join(save_dir, 'rl_recommendations.json'), 'w') as f:
        json.dump(rl_results, f, indent=2)
    
    # Save heuristic results
    with open(os.path.join(save_dir, 'heuristic_recommendations.json'), 'w') as f:
        json.dump(heuristic_results, f, indent=2)
    
    # Save comparison summary
    rl_metrics = extract_metrics(rl_results)
    heuristic_metrics = extract_metrics(heuristic_results)
    
    summary = {
        'rl_metrics': rl_metrics,
        'heuristic_metrics': heuristic_metrics,
        'comparison': {
            'reward_improvement': rl_metrics['avg_reward'] - heuristic_metrics['avg_reward'],
            'graduation_rate_improvement': rl_metrics['graduation_rate'] - heuristic_metrics['graduation_rate'],
            'gpa_improvement': rl_metrics['avg_gpa'] - heuristic_metrics['avg_gpa']
        }
    }
    
    with open(os.path.join(save_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Results saved to {save_dir}")

def main():
    """Main testing function"""
    print("="*60)
    print("AI-POWERED ACADEMIC ADVISOR - COMPREHENSIVE TESTING")
    print("="*60)
    
    # Check file dependencies
    if not check_file_dependencies():
        print("\n❌ Missing required files. Please run curriculum.py and train.py first.")
        return
    
    # Load data
    curriculum, students = load_test_data()
    if curriculum is None or students is None:
        print("\n❌ Failed to load required data.")
        return
    
    # Validate environment
    if not validate_environment(curriculum, students):
        print("\n❌ Environment validation failed.")
        return
    
    # Load trained model
    model = load_trained_model()
    if model is None:
        print("\n❌ Failed to load trained model.")
        return
    
    # Run RL recommendations
    rl_results = run_rl_recommendations(model, curriculum, students)
    
    # Run heuristic recommendations
    heuristic_results = run_heuristic_recommendations(curriculum, students)
    
    # Analyze and compare results
    rl_metrics, heuristic_metrics = analyze_and_compare_results(rl_results, heuristic_results, students)
    
    # Create visualizations
    create_visualization(rl_results, heuristic_results)
    
    # Save results
    save_results(rl_results, heuristic_results)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files generated:")
    print("- rl_recommendations.json")
    print("- heuristic_recommendations.json")
    print("- comparison_summary.json")
    print("- comparison_analysis.png")

if __name__ == "__main__":
    main()