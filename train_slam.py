# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:44:19 2025

@author: WANG ZIAN
"""

# train_slam.py
from stable_baselines3 import PPO
from slam_env import DynamicSLAMEnv
import numpy as np
import torch.nn as nn

def create_policy_network():
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)  # Output: linear and angular velocity
    )

def main():
    # Initialize environment
    initial_params = np.array([
        12.0, 0.2, 0.05, 0.5, 0.1, 0.5, 20, 
        12.0, 1.5, 3.0, 0.5, 1.0
    ])
    
    env = DynamicSLAMEnv(initial_params)
    
    # PPO hyperparameters
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": {
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
        }
    }
    
    # Initialize PPO agent
    model = PPO("MultiInputPolicy", env, **ppo_params, verbose=1)
    
    # Training
    total_timesteps = 1000000
    model.learn(
        total_timesteps=total_timesteps,
        callback=None  # Add callbacks if needed
    )
    
    # Save the trained model
    model.save("dynamic_slam_ppo_model")

if __name__ == "__main__":
    main()