from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import json
import pickle
import random
from recommendation import CurriculumEnv  # Your curriculum environment

def train_advisor():
    # Load data
    with open("AI-Powered-Academic-Advisor/data/curriculum.pkl", "rb") as f:
        curriculum = pickle.load(f)
    
    with open("AI-Powered-Academic-Advisor/data/student_profiles.json") as f:
        students = json.load(f)
    
    # Create environment without double-wrapping
    def make_env():
        student = random.choice(students)
        env = CurriculumEnv(student, curriculum)
        print("Observation space:", env.observation_space)  # Should show Box(140,)
        return env
    
    env = make_vec_env(make_env, n_envs=2)
    
    # Use MlpPolicy for Box observations
    model = PPO(
        "MlpPolicy",  # Changed from MultiInputPolicy
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        tensorboard_log="./logs/"
    )
    
    # Train
    checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/",
    name_prefix="rl_model"
    )

    model.learn(
        total_timesteps=200000,  # Train longer for better results
        callback=[checkpoint_callback],
        progress_bar=True
    )
    
    # Save
    model.save("AI-Powered-Academic-Advisor/results/ppo_advisor_simple")
    env.close()

if __name__ == "__main__":
    train_advisor()