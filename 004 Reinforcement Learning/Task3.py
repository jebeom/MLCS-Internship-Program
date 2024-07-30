############################################
# Title     : Using RL models from stable_baseline
# Author    : Jebeom Chae 
# Date      : 2024-07-22
# env       : Acrobot
# Model     : DQN
# Goal      : Swing above the given height
############################################

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

class BestRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(BestRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = log_dir
        self.best_mean_reward = -np.inf
        self.all_rewards = []
        self.all_timesteps = []
        self.best_rewards = []
        self.best_timesteps = []
        self.n_episodes = 0

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[-1]) > 0:
            self.n_episodes += 1
            ep_reward = self.model.ep_info_buffer[-1]['r']
            ep_timesteps = self.model.ep_info_buffer[-1]['l']
            self.all_rewards.append(ep_reward)
            self.all_timesteps.append(self.num_timesteps)
            
            if self.n_episodes % self.check_freq == 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    self.best_rewards.append(mean_reward)
                    self.best_timesteps.append(self.num_timesteps)
                    if self.verbose > 0:
                        print(f"Saving new best model at episode {self.n_episodes}, timestep {self.num_timesteps} with mean reward {mean_reward:.2f}")
        return True

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
env = DummyVecEnv([lambda: Monitor(gym.make("Acrobot-v1"), filename=os.path.join(log_dir, "monitor.csv"))])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=50000,
             learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99,
             train_freq=4, gradient_steps=1, target_update_interval=1000,
             device=device)

total_timesteps = 300000
check_freq = 2000
callback = BestRewardCallback(check_freq=check_freq, log_dir=log_dir)

# Train
model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)

plt.figure(figsize=(12, 6))
plt.plot(callback.all_timesteps, callback.all_rewards, color='blue', alpha=0.6, label='All Rewards')
plt.plot(callback.best_timesteps, callback.best_rewards, 'ro', markersize=5, label='Saved Model Points')
plt.title('Learning Curve')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

# Test ( visualization )
env_display = gym.make("Acrobot-v1", render_mode="human")
obs, _ = env_display.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_display.step(action)
    if terminated or truncated:
        obs, _ = env_display.reset()

env_display.close()
