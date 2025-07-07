import json
import pickle
from stable_baselines3 import PPO
from recommendation import CurriculumEnv
import matplotlib.pyplot as plt
import os  # For path handling

def test_advisor():
    # Load data and model
    with open("AI-Powered-Academic-Advisor/data/curriculum.pkl", "rb") as f:
        curriculum = pickle.load(f)
    with open("AI-Powered-Academic-Advisor/data/test_student_profiles.json") as f:
        test_students = json.load(f)
    
    model = PPO.load("AI-Powered-Academic-Advisor/results/ppo_advisor_simple")

    results = {}
    for student in test_students:
        env = CurriculumEnv(student, curriculum)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        
        # Store detailed results including selected courses
        results[student['ID']] = {
            'reward': float(total_reward),
            'gpa': float(info.get('gpa', 0)),
            'graduated': bool(info.get('graduation_progress', 0) >= 100),
            'terms_completed': int(info.get('current_term', 0)),
            'selected_courses': info.get('selected_courses', []),
            'passed_courses': info.get('passed', 0),
            'failed_courses': info.get('failed', 0)
        }
        env.close()

    # Print results to console
    print("\n=== Test Results ===")
    for student_id, data in results.items():
        print(f"{student_id}:")
        print(f"  Reward: {data['reward']:.1f}")
        print(f"  Graduated: {'Yes' if data['graduated'] else 'No'}")
        print(f"  Final GPA: {data['gpa']:.2f}")
        print(f"  Terms: {data['terms_completed']}")
        print(f"  Courses taken: {len(data['selected_courses'])}")
        print(f"  Passed: {data['passed_courses']}, Failed: {data['failed_courses']}")

    # Save results to JSON file
    results_dir = "AI-Powered-Academic-Advisor/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "test_results.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved detailed results to: {results_file}")

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [r['reward'] for r in results.values()])
    plt.title("Test Results - Total Reward per Student")
    plt.ylabel("Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_rewards.png"))
    print(f"Saved test results plot to: {os.path.join(results_dir, 'test_rewards.png')}")

if __name__ == "__main__":
    test_advisor()