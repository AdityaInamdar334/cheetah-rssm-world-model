#!/usr/bin/env python3
"""
Web Showcase Asset Exporter
Extracts individual frames from a sampled rollout sequence and saves them as PNG files
under web/assets/ for the interactive web slider.
"""

import os
import torch
import numpy as np
from PIL import Image

from utils import EpisodeReplayBuffer
from world_model import WorldModel

def export_web_assets(model_path="final_model.pt", data_path="cheetah_run_dataset.pkl", 
                      seq_len=40, context_len=10, output_dir="web/assets"):
    # Create directories
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load dataset & model
    buffer = EpisodeReplayBuffer(data_path=data_path, seq_len=seq_len)
    obs, action, reward, next_obs, done = buffer.sample_batch(batch_size=1, device=device)
    action_dim = action.size(-1)

    model = WorldModel(action_dim=action_dim, h_dim=200, s_dim=30).to(device)
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
            # Prior step
            if t > 0:
                mask = 1.0 - done[:, t-1]
                h = h * mask
                s = s * mask
                h, s_prior, prior_mean, prior_std = model.rssm.dynamics_step(s, h, action[:, t-1])
            else:
                prior_mean, prior_std = model.rssm.compute_prior(h)
                s_prior = model.rssm.reparameterize(prior_mean, prior_std)

            # Posterior step
            embed = model.encoder(obs[:, t])
            s_post, post_mean, post_std = model.rssm.representation_step(h, embed)

            # closed loop vs open loop
            if t < context_len:
                s = s_post
                recon = model.decoder(h, s_post)
                recon_obs.append(recon)
                predicted_obs.append(recon)
            else:
                s = s_prior
                pred = model.decoder(h, s_prior)
                predicted_obs.append(pred)
                
                recon = model.decoder(h, s_post)
                recon_obs.append(recon)

        # Stack sequence
        recon_obs = torch.cat(recon_obs, dim=0)
        predicted_obs = torch.cat(predicted_obs, dim=0)

    # Convert to CPU arrays
    obs_cpu = (obs[0].permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    recon_cpu = (recon_obs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_cpu = (predicted_obs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    print("Exporting individual PNG frames for web dashboard...")
    scale = 4 # Upscale images to 256x256 for quality viewing

    for t in range(T):
        gt_img = Image.fromarray(obs_cpu[t]).resize((64*scale, 64*scale), resample=Image.Resampling.NEAREST)
        recon_img = Image.fromarray(recon_cpu[t]).resize((64*scale, 64*scale), resample=Image.Resampling.NEAREST)
        pred_img = Image.fromarray(pred_cpu[t]).resize((64*scale, 64*scale), resample=Image.Resampling.NEAREST)

        gt_img.save(os.path.join(output_dir, f"gt_{t}.png"))
        recon_img.save(os.path.join(output_dir, f"recon_{t}.png"))
        pred_img.save(os.path.join(output_dir, f"pred_{t}.png"))

    print(f"Successfully exported {T * 3} frames to {output_dir}")

if __name__ == "__main__":
    export_web_assets()
