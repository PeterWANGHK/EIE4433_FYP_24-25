# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:03:50 2025

@author: WANG ZIAN
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from slam_env_v2 import SLAMEnv

# Wrap SLAM environment
env = DummyVecEnv([lambda: SLAMEnv()])

# Define PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train PPO model
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_slam")
