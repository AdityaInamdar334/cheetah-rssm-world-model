#!/usr/bin/env python3
"""
Data Collection Script for Dreamer-style RSSM (Robotic World Model)
Task: Cheetah Run (DeepMind Control Suite)
Observations: 64x64 RGB pixels
Policy: Random policy
Steps: 20,000
Target OS: macOS (CPU execution)
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

# Configure OpenGL backend for MuJoCo rendering on macOS.
# GLFW is the standard/default rendering backend for macOS.
os.environ['MUJOCO_GL'] = 'glfw'

try:
    from dm_control import suite
except ImportError as e:
    print("Error: dm_control is not installed or not in path.")
    print("Please install the packages in requirements.txt first:")
    print("pip install -r requirements.txt")
    sys.exit(1)

import torch
# Disable PyTorch MPS/GPU usage for this data collection script (run on CPU as requested).
device = torch.device("cpu")
print(f"Data collection running on device: {device}")


def collect_dataset(num_steps=20000, height=64, width=64, camera_id=0, save_path="cheetah_run_dataset.pkl"):
    """
    Collects a dataset of random policy interactions from dm_control cheetah run.
    Each step contains: (obs, action, reward, next_obs, done)
    where obs and next_obs are 64x64 RGB images (uint8, shape HWC).
    """
    print(f"Initializing 'cheetah' domain with 'run' task...")
    try:
        env = suite.load(domain_name="cheetah", task_name="run")
    except Exception as e:
        print(f"Failed to load environment: {e}")
        print("Please check your MuJoCo and dm_control installations.")
        sys.exit(1)

    action_spec = env.action_spec()
    print(f"Action spec shape: {action_spec.shape}, limits: [{action_spec.minimum[0]}, {action_spec.maximum[0]}]")

    # Replay buffer list to store transitions
    replay_buffer = []

    # Reset environment to start
    timestep = env.reset()
    
    # Render initial observation
    # dm_control physics.render returns an RGB numpy array of shape (height, width, 3)
    obs = env.physics.render(height=height, width=width, camera_id=camera_id)

    print(f"Starting data collection of {num_steps} steps...")
    
    episode_reward = 0.0
    episode_steps = 0
    episodes_completed = 0
    all_episode_rewards = []

    # Using tqdm for a progress bar, print updates every 1,000 steps manually or let tqdm handle it.
    # To satisfy "Prints progress every 1,000 steps", we will also do a stdout print at those intervals.
    for step in range(1, num_steps + 1):
        # Sample action uniformly within bounds
        action = np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape
        ).astype(np.float32)

        # Apply action in environment
        timestep = env.step(action)

        # Get next observation (pixels), reward, and termination status
        next_obs = env.physics.render(height=height, width=width, camera_id=camera_id)
        
        # Reward is None on some timestep types (like reset, but not step, but handle it safely)
        reward = float(timestep.reward) if timestep.reward is not None else 0.0
        
        # done is True if we reached the end of an episode (e.g. time limit)
        done = timestep.last()

        # Save transition dictionary
        transition = {
            'obs': obs,            # shape: (64, 64, 3), dtype: uint8
            'action': action,      # shape: (action_dim,), dtype: float32
            'reward': reward,      # float
            'next_obs': next_obs,  # shape: (64, 64, 3), dtype: uint8
            'done': done           # bool
        }
        replay_buffer.append(transition)

        # Update statistics
        episode_reward += reward
        episode_steps += 1

        # Print progress every 1,000 steps
        if step % 1000 == 0:
            last_reward_str = f"{all_episode_rewards[-1]:.2f}" if all_episode_rewards else "N/A"
            print(f"Collected {step}/{num_steps} steps ({(step/num_steps)*100:.1f}%) | "
                  f"Current Episode Steps: {episode_steps} | "
                  f"Last Completed Episode Reward: {last_reward_str}")

        if done:
            # Episode finished, reset env
            all_episode_rewards.append(episode_reward)
            episodes_completed += 1
            
            # Reset the environment
            timestep = env.reset()
            obs = env.physics.render(height=height, width=width, camera_id=camera_id)
            episode_reward = 0.0
            episode_steps = 0
        else:
            # Transition to next step
            obs = next_obs

    # If the last episode was not completed, record its partial reward for stats
    if episode_steps > 0:
        all_episode_rewards.append(episode_reward)

    print(f"\nData collection finished! Collected {len(replay_buffer)} total steps.")
    print(f"Completed {episodes_completed} episodes.")
    print(f"Average reward per episode: {np.mean(all_episode_rewards):.2f} (Min: {np.min(all_episode_rewards):.2f}, Max: {np.max(all_episode_rewards):.2f})")

    # Save to pickle file
    print(f"Saving dataset to {save_path}...")
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved dataset to {save_path} ({os.path.getsize(save_path) / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Failed to save dataset: {e}")


if __name__ == "__main__":
    collect_dataset(
        num_steps=20000,
        height=64,
        width=64,
        camera_id=0,
        save_path="cheetah_run_dataset.pkl"
    )
