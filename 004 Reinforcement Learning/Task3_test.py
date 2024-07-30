############################################
# Title     : Value Iteration
# Author    : Jebeom Chae 
# Date      : 2024-07-19
# env       : Acrobot
# Model     : DQN
# Gaol      : Swing above the given height
############################################

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Acrobot Env
env = gym.make("CartPole-v1")
env = Monitor(env)

# DQN Model 
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=50000,
             learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99,
             train_freq=4, gradient_steps=1, target_update_interval=1000)

total_timesteps = 50000
eval_interval = 2000
timesteps = []
mean_rewards = []

# Train
for step in range(0, total_timesteps, eval_interval):
    model.learn(total_timesteps, reset_num_timesteps=False, progress_bar=True)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    timesteps.append(step + eval_interval)
    mean_rewards.append(mean_reward)
    print(f"Timestep: {step + eval_interval}, Mean Reward: {mean_reward:.2f}")

# Train Result Plot
plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_rewards)
plt.title('Mean Reward vs Timesteps')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.grid(True)
plt.show()


# Test
env_display = gym.make("Acrobot-v1", render_mode="human")
obs, _ = env_display.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_display.step(action)
    if terminated or truncated:
        obs, _ = env_display.reset()

env_display.close()