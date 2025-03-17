# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:27:02 2025

@author: peter
"""

import pytest
from slam_env import DynamicSLAMEnv
import numpy as np

def test_environment_initialization():
    initial_params = np.array([12.0, 0.2, 0.05, 0.5, 0.1, 0.5, 
                             20, 12.0, 1.5, 3.0, 0.5, 1.0])
    env = DynamicSLAMEnv(initial_params)
    assert np.array_equal(env.params, initial_params)

def test_step_function():
    env = DynamicSLAMEnv(np.ones(12))
    action = np.zeros(12)
    obs, reward, done, info = env.step(action)
    assert obs.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

if __name__ == '__main__':
    pytest.main([__file__])