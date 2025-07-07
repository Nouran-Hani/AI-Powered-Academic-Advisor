from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from recommendation import CurriculumMultiStudentEnv

class TrainingStatsCallback(BaseCallback):
    """Custom callback to track training statistics"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.graduation_rates = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Get info from the environment
            infos = self.locals.get('infos', [{}])
            if infos and len(infos) > 0:
                info = infos[0]
                
                # Track graduation status
                if 'graduated' in info:
                    self.graduation_rates.append(1 if info['graduated'] else 0)
                
                # Track episode reward and length
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                else:
                    # Fallback to manual tracking
                    episode_reward = self.locals.get('episode_rewards', [0])
                    if hasattr(episode_reward, '__len__') and len(episode_reward) > 0:
                        self.episode_rewards.append(episode_reward[0])
                    
                    episode_length = self.locals.get('episode_lengths', [0])
                    if hasattr(episode_length, '__len__') and len(episode_length) > 0:
                        self.episode_lengths.append(episode_length[0])
        
        return True

def create_training_plots(stats_callback, save_path="AI-Powered-Academic-Advisor/results/training_stats.png"):
    """Create training performance plots"""
    if not stats_callback.episode_rewards:
        print("No training data available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Performance Statistics', fontsize=16, fontweight='bold')
    
    # Moving average function
    def moving_average(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Episode rewards
    rewards = stats_callback.episode_rewards
    episodes = range(len(rewards))
    axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
    if len(rewards) > 50:
        avg_rewards = moving_average(rewards, 50)
        axes[0, 0].plot(range(49, len(rewards)), avg_rewards, color='red', 
                       linewidth=2, label='Moving Average (50 episodes)')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    if stats_callback.episode_lengths:
        lengths = stats_callback.episode_lengths
        episodes = range(len(lengths))
        axes[0, 1].plot(episodes, lengths, alpha=0.3, color='green', label='Episode Length')
        if len(lengths) > 50:
            avg_lengths = moving_average(lengths, 50)
            axes[0, 1].plot(range(49, len(lengths)), avg_lengths, color='orange', 
                           linewidth=2, label='Moving Average (50 episodes)')
        axes[0, 1].set_title('Episode Lengths Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Graduation rates
    if stats_callback.graduation_rates:
        grad_rates = stats_callback.graduation_rates
        episodes = range(len(grad_rates))
        axes[1, 0].plot(episodes, grad_rates, alpha=0.3, color='purple', label='Graduation Status')
        if len(grad_rates) > 50:
            avg_grad = moving_average(grad_rates, 50)
            axes[1, 0].plot(range(49, len(grad_rates)), avg_grad, color='red', 
                           linewidth=2, label='Moving Average (50 episodes)')
        axes[1, 0].set_title('Graduation Success Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Graduation Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution
    axes[1, 1].hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Distribution of Episode Rewards')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {save_path}")

def main():
    """Main training function"""
    # Create necessary directories
    os.makedirs("AI-Powered-Academic-Advisor/data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Load curriculum graph
    print("Loading curriculum graph...")
    try:
        with open("AI-Powered-Academic-Advisor/data/curriculum.pkl", "rb") as f:
            curi = pickle.load(f)
        print(f"Curriculum loaded successfully with {len(curi.nodes)} courses")
    except FileNotFoundError:
        print("Error: curriculum.pkl not found. Please run curriculum.py first.")
        return
    
    # Load student training data
    print("Loading student profiles...")
    try:
        with open("AI-Powered-Academic-Advisor/data/student_profiles.json") as f:
            train_students = json.load(f)
        print(f"Loaded {len(train_students)} training students")
    except FileNotFoundError:
        print("Error: student_profiles.json not found. Please run students.py first.")
        return
    
    # Validate that we have enough students
    if len(train_students) < 10:
        print(f"Warning: Only {len(train_students)} students loaded. Need at least 10 for training.")
        return
    
    # Create environment factory function
    def make_env():
        """Factory function to create a new environment instance"""
        return CurriculumMultiStudentEnv(train_students, curi)
    
    # Test environment creation
    print("Testing environment creation...")
    try:
        test_env = make_env()
        obs, _ = test_env.reset()
        print("Environment created successfully")
        print(f"Observation space: {test_env.observation_space}")
        print(f"Action space: {test_env.action_space}")
        test_env.close()
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Use vectorized environment for faster training
    print("Creating vectorized training environment...")
    try:
        vec_env = make_vec_env(make_env, n_envs=2)  # Reduced to 2 environments for stability
        print("Vectorized environment created successfully")
    except Exception as e:
        print(f"Error creating vectorized environment: {e}")
        return
    
    # Create separate evaluation environment
    eval_env = make_env()
    
    # Set up custom callback for tracking training stats
    stats_callback = TrainingStatsCallback(verbose=1)
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=5000,  # Evaluate every 5K steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Evaluate on 5 episodes
        verbose=1
    )
    
    # Initialize PPO model with enhanced hyperparameters
    print("Initializing PPO model...")
    try:
        model = PPO(
            "MultiInputPolicy", 
            vec_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,  # Reduced for faster training
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./logs/tensorboard/",
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
            )
        )
        print("PPO model initialized successfully")
    except Exception as e:
        print(f"Error initializing PPO model: {e}")
        return
    
    # Train model with callbacks
    print("Starting training...")
    print("Training will run for 100,000 timesteps with evaluation every 5,000 steps")
    
    try:
        model.learn(
            total_timesteps=100_000,  # Reduced for faster training
            callback=[eval_callback, stats_callback],
            progress_bar=True
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Save the final model
    print("Saving final model...")
    try:
        model.save("AI-Powered-Academic-Advisor/results/ppo_curriculum")
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # Create training performance plots
    print("Creating training performance plots...")
    try:
        create_training_plots(stats_callback, "AI-Powered-Academic-Advisor/results/training_stats.png")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Clean up
    vec_env.close()
    eval_env.close()
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if stats_callback.episode_rewards:
        print(f"Total episodes completed: {len(stats_callback.episode_rewards)}")
        print(f"Average episode reward: {np.mean(stats_callback.episode_rewards):.2f}")
        print(f"Best episode reward: {max(stats_callback.episode_rewards):.2f}")
        print(f"Worst episode reward: {min(stats_callback.episode_rewards):.2f}")
        
        if len(stats_callback.episode_rewards) > 50:
            print(f"Final 50 episodes average: {np.mean(stats_callback.episode_rewards[-50:]):.2f}")
    
    if stats_callback.graduation_rates:
        print(f"Overall graduation rate: {np.mean(stats_callback.graduation_rates)*100:.1f}%")
        if len(stats_callback.graduation_rates) > 50:
            print(f"Final 50 episodes graduation rate: {np.mean(stats_callback.graduation_rates[-50:])*100:.1f}%")
    
    print("\nTraining completed successfully!")
    print("Files saved:")
    print("- Model: AI-Powered-Academic-Advisor/results/ppo_curriculum")
    print("- Training plots: AI-Powered-Academic-Advisor/results/training_stats.png")
    print("- Logs: ./logs/")

if __name__ == "__main__":
    main()