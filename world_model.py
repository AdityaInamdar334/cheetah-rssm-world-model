import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    CNN Encoder that embeds 64x64 RGB observations into a 1024-dimensional feature vector.
    Matches standard Dreamer CNN encoder specifications.
    """
    def __init__(self, in_channels=3, embed_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),   # 64x64 -> 31x31
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # 31x31 -> 14x14
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),           # 14x14 -> 6x6
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),          # 6x6 -> 2x2
            nn.ELU(),
        )
        # Flattened shape: 256 * 2 * 2 = 1024
        self.fc = nn.Linear(1024, embed_dim) if embed_dim != 1024 else nn.Identity()

    def forward(self, obs):
        # Input shape: (B, C, H, W)
        h = self.conv(obs)
        h = torch.flatten(h, start_dim=1) # (B, 1024)
        return self.fc(h)


class Decoder(nn.Module):
    """
    Transpose CNN Decoder that reconstructs 64x64 RGB observations from latent states (h_t, s_t).
    Matches standard Dreamer CNN decoder specifications.
    """
    def __init__(self, latent_dim, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2, padding=0), # 1x1 -> 5x5
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0),   # 5x5 -> 13x13
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0),    # 13x13 -> 30x30
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=6, stride=2, padding=0), # 30x30 -> 64x64
        )

    def forward(self, h, s):
        # Concatenate deterministic state h and stochastic state s
        x = torch.cat([h, s], dim=-1)
        x = self.fc(x)
        x = x.view(-1, 1024, 1, 1) # Reshape for deconvolution
        x = self.deconv(x)
        return x # Output shape: (B, C, H, W)


class RewardPredictor(nn.Module):
    """
    MLP that predicts a scalar reward from the current latent state (h_t, s_t).
    """
    def __init__(self, latent_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        return self.net(x)


class RSSMCell(nn.Module):
    """
    Recurrent State Space Model (RSSM) Cell.
    Consists of:
      - Deterministic State transition: GRU Cell
      - Stochastic State prior: p(s_t | h_t)
      - Stochastic State posterior: q(s_t | h_t, e_t)
    """
    def __init__(self, action_dim, h_dim=200, s_dim=30, min_std=0.1):
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.min_std = min_std

        # GRU updates deterministic state h_t from previous action and stochastic state
        self.gru = nn.GRUCell(input_size=action_dim + s_dim, hidden_size=h_dim)

        # Prior model: h_t -> mean, std
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * s_dim) # mean and raw_std
        )

        # Posterior model: h_t, embed_t -> mean, std
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + 1024, 200), # 1024 is the observation embed_dim
            nn.ELU(),
            nn.Linear(200, 2 * s_dim) # mean and raw_std
        )

    def get_initial_state(self, batch_size, device):
        """
        Returns zero-initialized deterministic and stochastic states.
        """
        h = torch.zeros(batch_size, self.h_dim, device=device)
        s = torch.zeros(batch_size, self.s_dim, device=device)
        return h, s

    def compute_prior(self, h):
        """
        Computes the prior distribution p(s_t | h_t).
        """
        prior_out = self.prior_net(h)
        mean, raw_std = torch.chunk(prior_out, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def compute_posterior(self, h, embed):
        """
        Computes the posterior distribution q(s_t | h_t, e_t).
        """
        inputs = torch.cat([h, embed], dim=-1)
        post_out = self.post_net(inputs)
        mean, raw_std = torch.chunk(post_out, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def reparameterize(self, mean, std):
        """
        Samples from Gaussian using the reparameterization trick.
        """
        epsilon = torch.randn_like(mean)
        return mean + std * epsilon

    def dynamics_step(self, prev_s, prev_h, action):
        """
        Deterministic dynamics step: computes next h_t and prior over s_t.
        """
        inputs = torch.cat([prev_s, action], dim=-1)
        h = self.gru(inputs, prev_h)
        prior_mean, prior_std = self.compute_prior(h)
        s_prior = self.reparameterize(prior_mean, prior_std)
        return h, s_prior, prior_mean, prior_std

    def representation_step(self, h, embed):
        """
        Posterior representation step: computes posterior over s_t and samples from it.
        """
        post_mean, post_std = self.compute_posterior(h, embed)
        s_post = self.reparameterize(post_mean, post_std)
        return s_post, post_mean, post_std


class WorldModel(nn.Module):
    """
    Unified World Model that wraps Encoder, Decoder, RewardPredictor, and RSSMCell.
    """
    def __init__(self, action_dim, h_dim=200, s_dim=30):
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim

        self.encoder = Encoder()
        self.rssm = RSSMCell(action_dim, h_dim, s_dim)
        self.decoder = Decoder(h_dim + s_dim)
        self.reward_predictor = RewardPredictor(h_dim + s_dim)

    def forward(self, obs, action, done):
        """
        Performs a sequential forward rollout of the RSSM over a batch of sequences.
        Inputs:
          - obs:      (B, T, C, H, W)
          - action:   (B, T, action_dim)
          - done:     (B, T, 1)
        Outputs:
          - recon_obs:     (B, T, C, H, W) reconstructed observations
          - pred_reward:   (B, T, 1) predicted rewards
          - post_means:    (B, T, s_dim) posterior means
          - post_stds:     (B, T, s_dim) posterior standard deviations
          - prior_means:   (B, T, s_dim) prior means
          - prior_stds:    (B, T, s_dim) prior standard deviations
        """
        B, T, C, H, W = obs.size()
        device = obs.device

        # Initialize hidden states to zero
        h, s = self.rssm.get_initial_state(B, device)

        # Pre-allocate containers for sequence data
        post_h_seq = []
        post_s_seq = []
        post_means = []
        post_stds = []
        prior_means = []
        prior_stds = []

        for t in range(T):
            # Roll out dynamics (prior) for t > 0
            if t > 0:
                # Reset states if episode ended at t-1 (using done[:, t-1])
                mask = 1.0 - done[:, t-1]
                h = h * mask
                s = s * mask

                h, _, prior_mean, prior_std = self.rssm.dynamics_step(s, h, action[:, t-1])
            else:
                # At t=0, compute the prior from initial zero hidden state h
                prior_mean, prior_std = self.rssm.compute_prior(h)

            # Update representation (posterior) with observation
            embed = self.encoder(obs[:, t])
            s, post_mean, post_std = self.rssm.representation_step(h, embed)

            # Store states and distributions
            post_h_seq.append(h)
            post_s_seq.append(s)
            post_means.append(post_mean)
            post_stds.append(post_std)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)

        # Stack sequences along time dimension (B, T, ...)
        post_h_seq = torch.stack(post_h_seq, dim=1) # (B, T, h_dim)
        post_s_seq = torch.stack(post_s_seq, dim=1) # (B, T, s_dim)
        post_means = torch.stack(post_means, dim=1) # (B, T, s_dim)
        post_stds = torch.stack(post_stds, dim=1)   # (B, T, s_dim)
        prior_means = torch.stack(prior_means, dim=1) # (B, T, s_dim)
        prior_stds = torch.stack(prior_stds, dim=1)   # (B, T, s_dim)

        # Flatten batch and time dimensions for parallel decoder/predictor forward passes
        flat_h = post_h_seq.view(B * T, -1)
        flat_s = post_s_seq.view(B * T, -1)

        # Reconstruct image and predict reward
        flat_recon_obs = self.decoder(flat_h, flat_s) # (B*T, C, H, W)
        flat_pred_reward = self.reward_predictor(flat_h, flat_s) # (B*T, 1)

        # Reshape outputs back to (B, T, ...) sequence shapes
        recon_obs = flat_recon_obs.view(B, T, C, H, W)
        pred_reward = flat_pred_reward.view(B, T, 1)

        return recon_obs, pred_reward, post_means, post_stds, prior_means, prior_stds
