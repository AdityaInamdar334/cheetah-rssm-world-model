#!/usr/bin/env python3
"""
World Model Evaluation & Visualizer
Loads 'final_model.pt', runs closed-loop context initialization followed by
open-loop dreaming (prior predictions), and saves the results as an animated GIF.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import EpisodeReplayBuffer
from world_model import WorldModel

def visualize_predictions(model_path="final_model.pt", data_path="cheetah_run_dataset.pkl", 
                          seq_len=40, context_len=10, save_path="prediction_comparison.gif"):
    # Set device to CPU or MPS/CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device for visualization: {device}")

    # Load dataset
    buffer = EpisodeReplayBuffer(data_path=data_path, seq_len=seq_len)
    
    # Sample a sequence batch (batch size 1 for visualization)
    obs, action, reward, next_obs, done = buffer.sample_batch(batch_size=1, device=device)
    
    action_dim = action.size(-1)

    # Load Model
    print(f"Loading trained World Model from {model_path}...")
    model = WorldModel(action_dim=action_dim, h_dim=200, s_dim=30).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    model.eval()

    # Roll out model
    recon_obs = []
    predicted_obs = []
    
    with torch.no_grad():
        B, T, C, H, W = obs.size()
        h, s = model.rssm.get_initial_state(B, device)

        for t in range(T):
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

            # Sequence Rollout Decision
            if t < context_len:
                # Closed loop (Context phase): Feed the actual posterior state
                s = s_post
                # Decode from posterior (closed-loop reconstruction)
                recon = model.decoder(h, s_post)
                recon_obs.append(recon)
                predicted_obs.append(recon)
            else:
                # Open loop (Dreaming phase): Feed the predicted prior state (no real pixels)
                s = s_prior
                # Decode prediction from prior
                pred = model.decoder(h, s_prior)
                predicted_obs.append(pred)
                
                # Reconstruct posterior for baseline comparisons
                recon = model.decoder(h, s_post)
                recon_obs.append(recon)

        # Concatenate steps along sequence dim (T, C, H, W)
        recon_obs = torch.cat(recon_obs, dim=0)
        predicted_obs = torch.cat(predicted_obs, dim=0)

    # Move tensors to CPU and denormalize back to uint8 images
    obs_cpu = (obs[0].permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    recon_cpu = (recon_obs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_cpu = (predicted_obs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    print(f"Generating side-by-side comparison frames (GIF: {save_path})...")
    frames = []

    for t in range(T):
        # Slice frames for time step t
        gt_frame = obs_cpu[t]
        recon_frame = recon_cpu[t]
        pred_frame = pred_cpu[t]

        # Stack frames side by side: Ground Truth | Reconstruction | Prediction
        combined_img_arr = np.hstack([gt_frame, recon_frame, pred_frame])
        
        # Convert to PIL Image for rendering annotations
        # Output width will be 64 * 3 = 192, height is 64
        # Scale up the image by a factor of 4 for better viewing (768 x 256)
        scale = 4
        pil_img = Image.fromarray(combined_img_arr)
        pil_img = pil_img.resize((64 * 3 * scale, 64 * scale), resample=Image.Resampling.NEAREST)
        
        draw = ImageDraw.Draw(pil_img)
        
        # Draw column title labels
        label_y = 5
        draw.text((10, label_y), "Ground Truth", fill=(255, 255, 255))
        draw.text((64 * scale + 10, label_y), "Reconstruction", fill=(255, 255, 255))
        
        if t < context_len:
            draw.text((128 * scale + 10, label_y), f"Context ({t}/{context_len})", fill=(0, 255, 0))
        else:
            draw.text((128 * scale + 10, label_y), f"Open-Loop ({t-context_len})", fill=(255, 50, 50))
            # Draw vertical visual divider at open-loop start boundary
            draw.line([(128 * scale, 0), (128 * scale, 64 * scale)], fill=(255, 0, 0), width=2)

        # Draw frame number
        draw.text((10, 64 * scale - 20), f"Step: {t}", fill=(255, 255, 0))

        frames.append(pil_img)

    # Save frames as an animated GIF
    try:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=150, # 150ms per frame
            loop=0
        )
        print(f"Successfully generated and saved comparison GIF to: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

    # Compute and print MSE errors
    recon_mse = np.mean((obs_cpu / 255.0 - recon_cpu / 255.0) ** 2)
    pred_mse = np.mean((obs_cpu[context_len:] / 255.0 - pred_cpu[context_len:] / 255.0) ** 2)
    print(f"\nReconstruction MSE (Closed-loop): {recon_mse:.6f}")
    print(f"Prediction MSE (Open-loop dreaming): {pred_mse:.6f}")


if __name__ == "__main__":
    visualize_predictions(
        model_path="final_model.pt",
        data_path="cheetah_run_dataset.pkl",
        seq_len=40,
        context_len=10,
        save_path="prediction_comparison.gif"
    )
