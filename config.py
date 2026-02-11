"""
Hyperparameters and global config for PGR Safety experiments.
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ── Environment ──────────────────────────────────────────────────────────────
MAX_STEPS = 500
N_EPISODES = 5000

# ── Networks ─────────────────────────────────────────────────────────────────
HIDDEN_DIM = 256
LR = 3e-4
GAMMA = 0.99
TAU = 0.005

# ── Replay buffer ────────────────────────────────────────────────────────────
BATCH_SIZE = 256
BUFFER_SIZE = 50_000

# ── PGR / Diffusion ─────────────────────────────────────────────────────────
DIFFUSION_STEPS = 30
LATENT_DIM = 64
REPLAY_RATIO = 0.3          # fraction of each batch that is synthetic
UPDATES_PER_EPISODE = 10
PGR_START_BUFFER = 5_000    # min real transitions before PGR kicks in

# ── Rare-event memory ───────────────────────────────────────────────────────
RARE_BUFFER_SIZE = 500
RARE_BATCH_RATIO = 0.2      # fraction of diffusion batch from rare buffer
RARE_WEIGHT = 5.0            # loss weight multiplier for rare transitions
