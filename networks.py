"""
Neural network components:
  - GaussianPolicy  (SAC actor)
  - QNetwork        (SAC critic)
  - NoisePredictor  (diffusion denoiser)
  - Diffusion       (DDPM forward/reverse process)
  - StateEncoder    (ICM encoder)
  - ForwardModel    (ICM dynamics predictor)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HIDDEN_DIM, LATENT_DIM, DIFFUSION_STEPS, DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
# SAC components
# ═══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.net(state)
        return self.mean(h), torch.clamp(self.log_std(h), -20, 2)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = (
            normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob

    def get_action(self, state):
        with torch.no_grad():
            action, _ = self.sample(state)
            return action.cpu().numpy()[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion components
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class NoisePredictor(nn.Module):
    """Conditional noise predictor: takes (noised transition, timestep, relevance)."""

    def __init__(self, transition_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.rel_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(transition_dim + hidden * 2, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, transition_dim),
        )

    def forward(self, x, t, rel):
        t_emb = self.time_mlp(self.time_embed(t))
        r_emb = self.rel_mlp(rel)
        return self.net(torch.cat([x, t_emb, r_emb], dim=-1))


class Diffusion:
    """Simple DDPM with conditional generation."""

    def __init__(self, model: NoisePredictor, T: int = DIFFUSION_STEPS):
        self.model = model
        self.T = T

        betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_1m_alpha_bar = (1 - alpha_bar).sqrt()

    def loss(self, x0, rel, weights=None):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=DEVICE)
        noise = torch.randn_like(x0)
        xt = self.sqrt_alpha_bar[t, None] * x0 + self.sqrt_1m_alpha_bar[t, None] * noise
        pred = self.model(xt, t, rel)
        per_sample = ((pred - noise) ** 2).mean(dim=-1)
        if weights is not None:
            return (weights * per_sample).mean()
        return per_sample.mean()

    @torch.no_grad()
    def generate(self, shape, rel):
        x = torch.randn(shape, device=DEVICE)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)
            eps = self.model(x, t, rel)
            alpha = self.alphas[t, None]
            beta = self.betas[t, None]
            mean = (x - beta / self.sqrt_1m_alpha_bar[t, None] * eps) / alpha.sqrt()
            if i > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ICM (Intrinsic Curiosity Module) components
# ═══════════════════════════════════════════════════════════════════════════════

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = LATENT_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, s):
        return self.net(s)


class ForwardModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, h_s, a):
        return self.net(torch.cat([h_s, a], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Shift+scale scores to be non-negative with unit-ish variance."""
    std = scores.std() + 1e-8
    normed = (scores - scores.mean()) / std
    return normed - normed.min() + 1e-6
