import os
import pickle
import numpy as np
import torch

class EpisodeReplayBuffer:
    """
    A Replay Buffer that segments a transition dataset into episodes and samples
    contiguous sequences of transitions. Essential for recurrent models like RSSM.
    """
    def __init__(self, data_path="data.pkl", seq_len=20):
        self.seq_len = seq_len
        self.episodes = []
        
        # Fallback to cheetah_run_dataset.pkl if data.pkl doesn't exist
        if not os.path.exists(data_path):
            fallback_path = "cheetah_run_dataset.pkl"
            if os.path.exists(fallback_path):
                print(f"Dataset '{data_path}' not found. Falling back to '{fallback_path}'.")
                data_path = fallback_path
            else:
                raise FileNotFoundError(f"Neither '{data_path}' nor '{fallback_path}' could be found.")

        print(f"Loading dataset from {data_path}...")
        with open(data_path, "rb") as f:
            transitions = pickle.load(f)
        
        # Segment transitions into episodes based on the 'done' flag
        current_episode = []
        for t in transitions:
            current_episode.append(t)
            if t['done']:
                self.episodes.append(current_episode)
                current_episode = []
        
        # Add any trailing transitions as an episode if it didn't end with done
        if len(current_episode) > 0:
            self.episodes.append(current_episode)
            
        print(f"Loaded {len(transitions)} transitions across {len(self.episodes)} episodes.")
        
        # Filter out episodes that are shorter than the sequence length
        self.episodes = [ep for ep in self.episodes if len(ep) >= self.seq_len]
        print(f"Valid episodes (length >= {self.seq_len}): {len(self.episodes)}")

    def sample_batch(self, batch_size=32, device="cpu"):
        """
        Samples a batch of sequence transitions.
        Returns tensors with shapes:
          - obs:      (batch_size, seq_len, 3, 64, 64)
          - action:   (batch_size, seq_len, action_dim)
          - reward:   (batch_size, seq_len, 1)
          - next_obs: (batch_size, seq_len, 3, 64, 64)
          - done:     (batch_size, seq_len, 1)
        """
        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_next_obs = []
        batch_done = []

        for _ in range(batch_size):
            # Select a random episode
            ep_idx = np.random.randint(0, len(self.episodes))
            episode = self.episodes[ep_idx]
            
            # Select a random starting timestep for the sequence
            start_t = np.random.randint(0, len(episode) - self.seq_len + 1)
            seq = episode[start_t : start_t + self.seq_len]
            
            # Extract lists of transition fields
            obs_seq = [t['obs'] for t in seq]
            action_seq = [t['action'] for t in seq]
            reward_seq = [t['reward'] for t in seq]
            next_obs_seq = [t['next_obs'] for t in seq]
            done_seq = [t['done'] for t in seq]
            
            batch_obs.append(obs_seq)
            batch_action.append(action_seq)
            batch_reward.append(reward_seq)
            batch_next_obs.append(next_obs_seq)
            batch_done.append(done_seq)

        # Convert to numpy arrays, scale images to [0, 1] range and transpose to (channels, height, width)
        # obs/next_obs shape in pickle: (batch_size, seq_len, 64, 64, 3)
        obs_arr = np.array(batch_obs, dtype=np.float32) / 255.0
        obs_arr = np.transpose(obs_arr, (0, 1, 4, 2, 3)) # (B, T, C, H, W)

        next_obs_arr = np.array(batch_next_obs, dtype=np.float32) / 255.0
        next_obs_arr = np.transpose(next_obs_arr, (0, 1, 4, 2, 3)) # (B, T, C, H, W)

        action_arr = np.array(batch_action, dtype=np.float32)
        reward_arr = np.array(batch_reward, dtype=np.float32)[..., np.newaxis] # add features/channel dim
        done_arr = np.array(batch_done, dtype=np.float32)[..., np.newaxis] # add features/channel dim

        # Move tensors to the designated device
        obs_tensor = torch.from_numpy(obs_arr).to(device)
        action_tensor = torch.from_numpy(action_arr).to(device)
        reward_tensor = torch.from_numpy(reward_arr).to(device)
        next_obs_tensor = torch.from_numpy(next_obs_arr).to(device)
        done_tensor = torch.from_numpy(done_arr).to(device)

        return obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor
