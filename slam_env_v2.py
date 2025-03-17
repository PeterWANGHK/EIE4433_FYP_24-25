# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:48:51 2025

@author: WANG ZIAN
"""

import gym
import numpy as np
from gym import spaces

class SLAMEnv(gym.Env):
    def __init__(self):
        super(SLAMEnv, self).__init__()
        
        # Define action space (modifying SLAM parameters)
        self.action_space = spaces.Box(low=np.array([0.1, 0.01]), high=np.array([1.0, 0.5]), dtype=np.float32)
        
        # Define state space (SLAM parameters + robot pose)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        # Reset SLAM parameters and robot state
        self.state = np.random.uniform(low=-1.0, high=1.0, size=(5,))
        return self.state

    def step(self, action):
        # Simulate SLAM behavior and compute reward
        distance_to_obstacle = np.random.uniform(0.1, 5.0)
        collision = 1 if distance_to_obstacle < 0.2 else 0
        
        # Compute rewards
        R_dist = -1 / (distance_to_obstacle + 1e-3)  # Avoid division by zero
        R_col = -1 if collision else 0
        R_exp = 0.5  # Assume some exploration benefit
        
        reward = R_dist + R_col + R_exp
        done = False  # Continue training
        
        next_state = np.random.uniform(low=-1.0, high=1.0, size=(5,))
        return next_state, reward, done, {}
