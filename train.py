#!/usr/bin/env python3
"""
Dreamer-style World Model (RSSM) Training Script
Optimized for macOS CPU execution.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import EpisodeReplayBuffer
from world_model import WorldModel

# Default configurations
LR = 3e-4
BATCH_SIZE = 32
SEQ_LEN = 20
TOTAL_STEPS = 100000
LOG_INTERVAL = 500
CHECKPOINT_INTERVAL = 10000
CHECKPOINT_DIR = "checkpoints"
FINAL_MODEL_PATH = "final_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dreamer-style RSSM World Model")
    parser.add_argument("--data_path", type=str, default="data.pkl", 
                        help="Path to the pickle dataset file")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, 
                        help="Number of training steps to run")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, 
                        help="Sequence length for temporal rollouts")
    parser.add_argument("--lr", type=float, default=LR, 
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--test_run", action="store_true", 
                        help="Helper flag to run a quick 100-step verification")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Adjust total steps if running a verification test
    total_steps = 100 if args.test_run else args.steps
    print(f"Starting training process. Total target steps: {total_steps}")

    # Detect and set device (using MPS GPU acceleration on macOS if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize Replay Buffer
    try:
        buffer = EpisodeReplayBuffer(data_path=args.data_path, seq_len=args.seq_len)
    except FileNotFoundError as e:
        print(f"Error initializing replay buffer: {e}")
        sys.exit(1)

    # Sample a dummy batch to inspect shapes and dynamically set action dimension
    print("Testing replay buffer sampling...")
    obs, action, reward, next_obs, done = buffer.sample_batch(batch_size=args.batch_size, device=device)
    print(f"Sampled shapes:")
    print(f"  Observations (obs):     {obs.shape} (B, T, C, H, W)")
    print(f"  Actions (action):       {action.shape} (B, T, action_dim)")
    print(f"  Rewards (reward):       {reward.shape} (B, T, 1)")
    print(f"  Next Obs (next_obs):    {next_obs.shape} (B, T, C, H, W)")
    print(f"  Dones (done):           {done.shape} (B, T, 1)")
    
    action_dim = action.size(-1)

    # Instantiate World Model
    print("Instantiating World Model...")
    model = WorldModel(action_dim=action_dim, h_dim=200, s_dim=30).to(device)
    
    # Calculate parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"World Model instantiated with {num_params:,} trainable parameters.")

    # Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Mixed Precision configuration
    use_amp = False
    amp_device = "cpu"
    amp_dtype = torch.bfloat16

    if device.type == "cpu":
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                _ = model(obs[:2], action[:2], done[:2])
            use_amp = True
            amp_device = "cpu"
            amp_dtype = torch.bfloat16
            print("Bfloat16 mixed precision is supported on CPU and will be used.")
        except Exception as e:
            print(f"Bfloat16 mixed precision CPU check failed: {e}. Falling back to standard FP32.")
    elif device.type == "cuda":
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(obs[:2], action[:2], done[:2])
            use_amp = True
            amp_device = "cuda"
            amp_dtype = torch.float16
            print("Float16 mixed precision is supported on CUDA and will be used.")
        except Exception as e:
            print(f"Float16 mixed precision CUDA check failed: {e}. Falling back to standard FP32.")
    else:
        print("Using standard FP32 precision (recommended for MPS/GPU stability).")

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Tracking variables
    step_losses = []
    step_recon_losses = []
    step_kl_losses = []
    step_reward_losses = []
    
    start_time = time.time()

    # Progress bar using tqdm
    pbar = tqdm(range(1, total_steps + 1), desc="Training Steps", unit="step")
    
    for step in pbar:
        # Sample sequence batch from replay buffer
        obs, action, reward, next_obs, done = buffer.sample_batch(batch_size=args.batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)

        # Helper training step inside autocast context if supported
        if use_amp:
            with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                loss, recon_loss, kl_loss, reward_loss = compute_loss(model, obs, action, reward, done)
            loss.backward()
        else:
            loss, recon_loss, kl_loss, reward_loss = compute_loss(model, obs, action, reward, done)
            loss.backward()

        # Gradient clipping to prevent exploding gradients (standard in RSSM training)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()

        # Record losses for logging
        step_losses.append(loss.item())
        step_recon_losses.append(recon_loss.item())
        step_kl_losses.append(kl_loss.item())
        step_reward_losses.append(reward_loss.item())

        # Update progress bar postfix with current metrics
        if step % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "reward": f"{reward_loss.item():.4f}"
            })

        # Periodic print logging
        if step % LOG_INTERVAL == 0:
            avg_loss = sum(step_losses[-LOG_INTERVAL:]) / LOG_INTERVAL
            avg_recon = sum(step_recon_losses[-LOG_INTERVAL:]) / LOG_INTERVAL
            avg_kl = sum(step_kl_losses[-LOG_INTERVAL:]) / LOG_INTERVAL
            avg_reward = sum(step_reward_losses[-LOG_INTERVAL:]) / LOG_INTERVAL
            
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_secs = (total_steps - step) / steps_per_sec
            eta_mins = eta_secs / 60
            
            print(f"\n[Step {step}/{total_steps}] | Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, Reward: {avg_reward:.4f}) | "
                  f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_mins:.1f}m")

        # Periodic checkpointing
        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"world_model_step_{step}.pt")
            print(f"\nSaving checkpoint to {checkpoint_path}...")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)

    # Save final model
    print(f"\nTraining complete! Saving final model to {FINAL_MODEL_PATH}...")
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print("Final model saved successfully.")


def compute_loss(model, obs, action, reward, done):
    """
    Computes the loss components for training the World Model:
      - Reconstruction loss (MSE)
      - Reward prediction loss (MSE)
      - KL Divergence loss with a free bits threshold of 1.0
    """
    # Forward pass: roll out RSSM
    recon_obs, pred_reward, post_means, post_stds, prior_means, prior_stds = model(obs, action, done)

    # 1. Reconstruction Loss (MSE)
    # Target image is obs (normalized [0, 1])
    recon_loss = F.mse_loss(recon_obs, obs, reduction='mean')

    # 2. Reward Prediction Loss (MSE)
    reward_loss = F.mse_loss(pred_reward, reward, reduction='mean')

    # 3. KL Loss with Free Bits
    # KL(q || p) for diagonal Gaussians:
    # log(sigma_p / sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2) - 0.5
    kl_div = torch.log(prior_stds / post_stds) + (post_stds**2 + (post_means - prior_means)**2) / (2.0 * prior_stds**2) - 0.5
    
    # Sum over the stochastic state dimensions (s_dim) to get KL per timestep
    kl_sum = kl_div.sum(dim=-1) # (B, T)
    
    # Apply free bits constraint (clip minimum KL loss per step to 1.0)
    free_bits = 1.0
    kl_clipped = torch.max(kl_sum, torch.tensor(free_bits, device=obs.device))
    kl_loss = kl_clipped.mean()

    # Total loss
    total_loss = recon_loss + kl_loss + reward_loss

    return total_loss, recon_loss, kl_loss, reward_loss


if __name__ == "__main__":
    main()
