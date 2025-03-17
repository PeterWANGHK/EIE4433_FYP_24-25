# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:22:36 2025

@author: peter
"""

# slam_env.py
import gym
from gym import spaces
import numpy as np

class DynamicSLAMEnv(gym.Env):
    def __init__(self, initial_params):
        super(DynamicSLAMEnv, self).__init__()
        self.initial_params = initial_params
        self.params = initial_params.copy()
        
        # State space definition
        self.observation_space = spaces.Dict({
            'robot_pose': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),  # x, y, theta
            'map_info': spaces.Box(low=0, high=1, shape=(100, 100)),  # Example map size
            'obstacle_positions': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 2))  # Max 10 obstacles
        })
        
        # Action space definition (linear and angular velocities)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # min linear and angular velocity
            high=np.array([1.0, 1.0]),   # max linear and angular velocity
            dtype=np.float32
        )
        
        # Reward function parameters
        self.alpha = 0.4  # distance reward weight
        self.beta = 0.3   # collision reward weight
        self.gamma = 0.2  # exploration reward weight
        self.delta = 0.1  # smoothness reward weight
        
    def calculate_reward(self, state, action):
        # Distance reward
        d_min = self.calculate_min_obstacle_distance(state)
        R_dist = -1.0 if d_min < 0.5 else np.exp(-d_min)
        
        # Collision reward
        R_col = -10.0 if d_min < 0.2 else 0.0
        
        # Exploration reward
        R_exp = self.calculate_exploration_reward(state)
        
        # Smoothness reward
        R_smooth = -np.sum(np.abs(action))
        
        # Total reward
        R_total = (self.alpha * R_dist + 
                  self.beta * R_col + 
                  self.gamma * R_exp + 
                  self.delta * R_smooth)
        
        return R_total
    
    def step(self, action):
        # Update robot state based on action
        next_state = self.update_state(action)
        
        # Calculate reward
        reward = self.calculate_reward(next_state, action)
        
        # Check if episode is done
        done = self.check_termination()
        
        # Additional info
        info = {
            'min_obstacle_distance': self.calculate_min_obstacle_distance(next_state),
            'exploration_ratio': self.calculate_exploration_ratio()
        }
        
        return next_state, reward, done, info
    def reset(self, seed=None, options=None):
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset environment state
        self.params = self.initial_params.copy()
        self.current_pose = np.zeros(3)  # Reset robot pose
        self.pose_history = []
        self.ground_truth_history = []
        
        # Create initial observation
        initial_observation = {
            'robot_pose': self.current_pose,
            'map_info': np.zeros((100, 100)),  # Initial empty map
            'obstacle_positions': np.zeros((10, 2))  # Initial obstacle positions
        }
        
        # Reset info dictionary
        reset_info = {
            'initial_params': self.initial_params.copy(),
            'environment_initialized': True
        }
        
        return initial_observation, reset_info

if __name__ == '__main__':
    print("SLAM Environment Module")
