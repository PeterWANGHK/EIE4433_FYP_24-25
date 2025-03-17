# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:31:02 2025

@author: WANG ZIAN
"""

import numpy as np
#import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from stable_baselines3.common.utils import explained_variance

# Function to load KITTI trajectory files
def load_kitti_trajectory(file_path):
    data = np.loadtxt(file_path, usecols=(1, 2))  # Extract only x, y coordinates
    return data

# Load trajectories from KITTI format
ground_truth = load_kitti_trajectory("KITTI_01_gt.txt")
estimated_trajectory = load_kitti_trajectory("KITTI_01_ORB.txt")

# Load feature matches (modify as needed)
feature_matches = np.load("feature_matches.npy")  # Ensure this matches KITTI format

# Compute Absolute Trajectory Error (ATE)
def compute_ate(gt, est):
    return np.mean(np.linalg.norm(gt - est, axis=1))

# Compute Relative Pose Error (RPE)
def compute_rpe(gt, est):
    diffs = np.linalg.norm(np.diff(gt, axis=0) - np.diff(est, axis=0), axis=1)
    return np.mean(diffs)

# Compute Map Consistency Score (MCS)
def compute_mcs(gt, est):
    distances = [np.min(distance.cdist([p], gt)) for p in est]
    return 100 - (np.mean(distances) * 10)

# Compute Localization Drift (m/min)
def compute_localization_drift(est, time_interval=60):
    total_time = len(est) // time_interval
    drift_per_min = [np.linalg.norm(est[i] - est[i - time_interval]) for i in range(time_interval, len(est), time_interval)]
    return np.mean(drift_per_min)

# Compute Path Smoothness Index (PSI)
def compute_psi(est):
    angles = np.arctan2(np.diff(est[:, 1]), np.diff(est[:, 0]))
    return np.mean(np.abs(np.diff(angles)))

# Compute Feature Matching Accuracy
def compute_feature_matching_accuracy(matches, total_features):
    return (len(matches) / total_features) * 100

# Compute Loop Closure Frequency (per 100m)
def compute_loop_closure_frequency(gt, est, threshold=2.0):
    loop_closures = sum(1 for i in range(len(est)) if np.min(distance.cdist([est[i]], gt[:i])) < threshold)
    return loop_closures / (len(est) / 100)

# Simulated PPO policy logs (Replace with actual PPO policy outputs)
ppo_log_probs = np.random.uniform(0.1, 0.9, size=(1000,))
ppo_old_probs = np.random.uniform(0.1, 0.9, size=(1000,))
ppo_new_probs = np.random.uniform(0.1, 0.9, size=(1000,))

# Compute PPO Policy Entropy
def compute_policy_entropy(log_probs):
    return -np.mean(log_probs * np.log(log_probs + 1e-9))

# Compute PPO KL Divergence
def compute_kl_divergence(old_probs, new_probs):
    return np.mean(np.sum(old_probs * np.log(old_probs / (new_probs + 1e-9) + 1e-9), axis=1))

# Compute all metrics
metrics = {
    "ATE (m)": compute_ate(ground_truth, estimated_trajectory),
    "RPE (m/frame)": compute_rpe(ground_truth, estimated_trajectory),
    "Map Consistency Score (%)": compute_mcs(ground_truth, estimated_trajectory),
    "Localization Drift (m/min)": compute_localization_drift(estimated_trajectory),
    "Path Smoothness Index (PSI)": compute_psi(estimated_trajectory),
    "Feature Matching Accuracy (%)": compute_feature_matching_accuracy(feature_matches, total_features=5000),
    "Loop Closure Frequency (per 100m)": compute_loop_closure_frequency(ground_truth, estimated_trajectory),
    "PPO Policy Entropy": compute_policy_entropy(ppo_log_probs),
    "PPO KL Divergence": compute_kl_divergence(ppo_old_probs, ppo_new_probs),
}

# Convert metrics to DataFrame
df_metrics = pd.DataFrame(metrics, index=[0])

# Print results
print(df_metrics)

# Plot ATE and RPE over iterations
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(ground_truth)), np.linalg.norm(ground_truth - estimated_trajectory, axis=1), label="ATE")
plt.plot(np.arange(len(ground_truth)-1), np.linalg.norm(np.diff(ground_truth, axis=0) - np.diff(estimated_trajectory, axis=0), axis=1), label="RPE")
plt.xlabel("Time Step")
plt.ylabel("Error (m)")
plt.title("ATE & RPE Over Time")
plt.legend()
plt.show()
