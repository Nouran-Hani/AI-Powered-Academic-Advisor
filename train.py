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
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            # Get info from the environment
            info = self.locals.get('infos', [{}])[0]
            
            # Track graduation status
            if 'graduated' in info:
                self.graduation_rates.append(1 if info['graduated'] else 0)
            
            # Track episode reward and length
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        
        return True

def create_training_plots(stats_callback, save_path="training_stats.png"):
    """Create training performance plots"""
    if not stats_callback.episode_rewards:
        print("No training data available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Performance Statistics', fontsize=16, fontweight='bold')
    
    # Moving average function
    def moving_average(data, window=100):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Episode rewards
    rewards = stats_callback.episode_rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Episode Rewards')
    if len(rewards) > 100:
        avg_rewards = moving_average(rewards, 100)
        axes[0, 0].plot(range(99, len(rewards)), avg_rewards, color='red', 
                       linewidth=2, label='Moving Average (100 episodes)')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    lengths = stats_callback.episode_lengths
    axes[0, 1].plot(lengths, alpha=0.3, color='green', label='Episode Length')
    if len(lengths) > 100:
        avg_lengths = moving_average(lengths, 100)
        axes[0, 1].plot(range(99, len(lengths)), avg_lengths, color='orange', 
                       linewidth=2, label='Moving Average (100 episodes)')
    axes[0, 1].set_title('Episode Lengths Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Graduation rates
    if stats_callback.graduation_rates:
        grad_rates = stats_callback.graduation_rates
        axes[1, 0].plot(grad_rates, alpha=0.3, color='purple', label='Graduation Status')
        if len(grad_rates) > 100:
            avg_grad = moving_average(grad_rates, 100)
            axes[1, 0].plot(range(99, len(grad_rates)), avg_grad, color='red', 
                           linewidth=2, label='Moving Average (100 episodes)')
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
    # Create necessary directories
    os.makedirs("AI-Powered-Academic-Advisor", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Load curriculum graph
    print("Loading curriculum graph...")
    with open("AI-Powered-Academic-Advisor/curriculum.pkl", "rb") as f:
        curi = pickle.load(f)
    
    # Load student training data
    print("Loading student profiles...")
    with open("AI-Powered-Academic-Advisor/student_profiles.json") as f:
        train_students = json.load(f)
    
    print(f"Loaded {len(train_students)} training students")
    print(f"Curriculum has {len(curi.nodes)} courses")
    
    # Validate that we have at least 100 students as required
    if len(train_students) < 100:
        print(f"Warning: Only {len(train_students)} students loaded. Challenge requires 100 students.")
    
    # Create environment factory function
    def make_env():
        """Factory function to create a new environment instance"""
        return CurriculumMultiStudentEnv(train_students, curi)
    
    # Use vectorized environment for faster training
    print("Creating vectorized training environment...")
    vec_env = make_vec_env(make_env, n_envs=4)
    
    # Create separate evaluation environment
    eval_env = CurriculumMultiStudentEnv(train_students, curi)
    
    # Set up custom callback for tracking training stats
    stats_callback = TrainingStatsCallback(verbose=1)
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=10000,  # Evaluate every 10K steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,  # Evaluate on 10 episodes
        verbose=1
    )
    
    # Initialize PPO model with enhanced hyperparameters
    print("Initializing PPO model...")
    model = PPO(
        "MultiInputPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Train model with callbacks
    print("Starting training...")
    print("Training will run for 200,000 timesteps with evaluation every 10,000 steps")
    
    model.learn(
        total_timesteps=200_000,
        callback=[eval_callback, stats_callback],
        progress_bar=True
    )
    
    # Save the final model
    print("Saving final model...")
    model.save("AI-Powered-Academic-Advisor/ppo_curriculum")
    
    # Create training performance plots
    print("Creating training performance plots...")
    create_training_plots(stats_callback, "AI-Powered-Academic-Advisor/training_stats.png")
    
    # Print training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    if stats_callback.episode_rewards:
        print(f"Total episodes completed: {len(stats_callback.episode_rewards)}")
        print(f"Average episode reward: {np.mean(stats_callback.episode_rewards):.2f}")
        print(f"Best episode reward: {max(stats_callback.episode_rewards):.2f}")
        print(f"Final 100 episodes average: {np.mean(stats_callback.episode_rewards[-100:]):.2f}")
    
    if stats_callback.graduation_rates:
        print(f"Overall graduation rate: {np.mean(stats_callback.graduation_rates)*100:.1f}%")
        print(f"Final 100 episodes graduation rate: {np.mean(stats_callback.graduation_rates[-100:])*100:.1f}%")
    
    print("\nTraining completed successfully!")
    print("Files saved:")
    print("- Model: AI-Powered-Academic-Advisor/ppo_curriculum")

main()