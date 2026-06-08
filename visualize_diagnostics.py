#!/usr/bin/env python3
"""
World Model Diagnostics Plotter
Evaluates average reconstruction error and reward predictions over time,
saving them as a matplotlib diagnostic figure.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import EpisodeReplayBuffer
from world_model import WorldModel

def generate_diagnostics(model_path="final_model.pt", data_path="cheetah_run_dataset.pkl", 
                         seq_len=40, context_len=10, save_path="diagnostics.png"):
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device for diagnostics: {device}")

    # Load dataset
    buffer = EpisodeReplayBuffer(data_path=data_path, seq_len=seq_len)
    
    # Sample 16 sequences for averaging stats
    batch_size = 16
    obs, action, reward, next_obs, done = buffer.sample_batch(batch_size=batch_size, device=device)
    
    action_dim = action.size(-1)

    # Load Model
    model = WorldModel(action_dim=action_dim, h_dim=200, s_dim=30).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    model.eval()

    # Roll out model
    step_pixel_mses = []
    gt_rewards = []
    pred_rewards = []
    
    with torch.no_grad():
        h, s = model.rssm.get_initial_state(batch_size, device)

        for t in range(seq_len):
            # Compute prior dynamics step for t > 0
            if t > 0:
                mask = 1.0 - done[:, t-1]
                h = h * mask
                s = s * mask
                h, s_prior, prior_mean, prior_std = model.rssm.dynamics_step(s, h, action[:, t-1])
            else:
                prior_mean, prior_std = model.rssm.compute_prior(h)
                s_prior = model.rssm.reparameterize(prior_mean, prior_std)

            # Compute posterior representation step
            embed = model.encoder(obs[:, t])
            s_post, post_mean, post_std = model.rssm.representation_step(h, embed)

            # Decide state for rollout
            if t < context_len:
                s = s_post
                # Decode context reconstruction
                pred_img = model.decoder(h, s_post)
            else:
                s = s_prior
                # Decode open-loop prediction
                pred_img = model.decoder(h, s_prior)

            # Predict reward
            reward_pred = model.reward_predictor(h, s)

            # Calculate pixel MSE for this step
            step_mse = torch.mean((pred_img - obs[:, t]) ** 2, dim=[1, 2, 3]) # average over C, H, W for each batch item
            step_pixel_mses.append(step_mse.cpu().numpy())

            gt_rewards.append(reward[:, t].cpu().numpy())
            pred_rewards.append(reward_pred.cpu().numpy())

    # Shape: (seq_len, batch_size)
    step_pixel_mses = np.array(step_pixel_mses)
    gt_rewards = np.squeeze(np.array(gt_rewards))   # (seq_len, batch_size)
    pred_rewards = np.squeeze(np.array(pred_rewards)) # (seq_len, batch_size)

    # Average over the batch dimension
    avg_pixel_mse = np.mean(step_pixel_mses, axis=1)
    avg_gt_reward = np.mean(gt_rewards, axis=1)
    avg_pred_reward = np.mean(pred_rewards, axis=1)

    print("Generating diagnostic plots...")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = np.arange(seq_len)

    # Plot 1: Image MSE Error Curve
    ax1.plot(steps, avg_pixel_mse, color="blue", linewidth=2.5, label="Mean Squared Error")
    ax1.axvline(x=context_len - 1, color="red", linestyle="--", linewidth=1.5, 
                label=f"Open-Loop Dreaming Start (Step {context_len})")
    ax1.set_title("Image Prediction Error Over Time", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Sequence Timestep", fontsize=12)
    ax1.set_ylabel("Pixel MSE (Lower is Better)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Reward Prediction vs Ground Truth
    ax2.plot(steps, avg_gt_reward, color="black", linestyle="-", linewidth=2, label="Ground Truth Reward")
    ax2.plot(steps, avg_pred_reward, color="orange", linestyle="--", linewidth=2, label="Predicted Reward (RSSM)")
    ax2.axvline(x=context_len - 1, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title("Reward Prediction Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Sequence Timestep", fontsize=12)
    ax2.set_ylabel("Reward Signal", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Diagnostics plot successfully saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    generate_diagnostics(
        model_path="final_model.pt",
        data_path="cheetah_run_dataset.pkl",
        seq_len=40,
        context_len=10,
        save_path="diagnostics.png"
    )
