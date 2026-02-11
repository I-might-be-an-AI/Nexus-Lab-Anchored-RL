"""
Agents:
  - SACAgent           – vanilla SAC baseline
  - SACPGRAgent        – SAC + curiosity-conditioned generative replay
  - SACPGRMemoryAgent  – SAC + PGR + rare-event memory bank
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import (
    DEVICE, LR, GAMMA, TAU, BATCH_SIZE,
    LATENT_DIM, REPLAY_RATIO, PGR_START_BUFFER,
    RARE_BATCH_RATIO, RARE_WEIGHT,
)
from buffers import ReplayBuffer, RareEventBuffer
from networks import (
    QNetwork, GaussianPolicy,
    StateEncoder, ForwardModel,
    NoisePredictor, Diffusion,
    normalize_scores,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Base SAC
# ═══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim).to(DEVICE)

        # Twin critics + targets
        self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Entropy temperature
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp()

        # Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

        # Replay
        self.buffer = ReplayBuffer(state_dim, action_dim)

    # ── Interaction ──────────────────────────────────────────────────────────

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.policy.get_action(s)

    def add_transition(self, s, a, r, c, ns, d):
        self.buffer.add(s, a, r, c, ns, d)

    # ── Update ───────────────────────────────────────────────────────────────

    def _soft_update(self):
        for tp, sp in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)
        for tp, sp in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

    def _sac_update(self, states, actions, rewards, next_states, dones):
        """Core SAC gradient step on a prepared batch."""

        # Critic targets
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q_target = torch.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            )
            target = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (
                q_target - self.alpha * next_log_probs
            )

        # Critic losses
        self.q1_opt.zero_grad()
        F.mse_loss(self.q1(states, actions), target).backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        F.mse_loss(self.q2(states, actions), target).backward()
        self.q2_opt.step()

        # Actor loss
        new_actions, log_probs = self.policy.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))

        self.policy_opt.zero_grad()
        (self.alpha.detach() * log_probs - q_new).mean().backward()
        self.policy_opt.step()

        # Alpha loss
        self.alpha_opt.zero_grad()
        (-(self.log_alpha * (log_probs + self.target_entropy).detach())).mean().backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        self._soft_update()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, _costs, next_states, dones = self.buffer.sample(BATCH_SIZE)
        self._sac_update(states, actions, rewards, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC + PGR (curiosity-conditioned diffusion replay)
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRAgent(SACAgent):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)

        self.transition_dim = state_dim + action_dim + 1 + 1 + state_dim  # s,a,r,c,s'

        # ICM
        self.encoder = StateEncoder(state_dim).to(DEVICE)
        self.fwd_model = ForwardModel(LATENT_DIM, action_dim).to(DEVICE)

        # Diffusion
        self.noise_pred = NoisePredictor(self.transition_dim).to(DEVICE)
        self.diffusion = Diffusion(self.noise_pred)

        # Optimizers for new components
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=LR)
        self.fwd_opt = optim.Adam(self.fwd_model.parameters(), lr=LR)
        self.diff_opt = optim.Adam(self.noise_pred.parameters(), lr=LR)

    # ── Curiosity scoring ────────────────────────────────────────────────────

    def _compute_curiosity(self, states, actions, next_states):
        """ICM forward-prediction error (detached)."""
        with torch.no_grad():
            h_s = self.encoder(states)
            h_ns = self.encoder(next_states)
            return 0.5 * ((self.fwd_model(h_s, actions) - h_ns) ** 2).mean(dim=-1)

    def _train_icm(self):
        """One gradient step on the ICM forward model."""
        s, a, _, _, ns, _ = self.buffer.sample(BATCH_SIZE)
        h_s, h_ns = self.encoder(s), self.encoder(ns)

        self.encoder_opt.zero_grad()
        self.fwd_opt.zero_grad()
        F.mse_loss(self.fwd_model(h_s, a), h_ns).backward()
        self.encoder_opt.step()
        self.fwd_opt.step()

    # ── Diffusion training ───────────────────────────────────────────────────

    def _train_diffusion(self, transitions, scores, weights=None):
        """One gradient step on the conditional diffusion model."""
        x0 = torch.FloatTensor(transitions).to(DEVICE)
        rel = torch.FloatTensor(scores[:, None]).to(DEVICE)
        w = torch.FloatTensor(weights).to(DEVICE) if weights is not None else None

        self.diff_opt.zero_grad()
        self.diffusion.loss(x0, rel, weights=w).backward()
        self.diff_opt.step()

    # ── Synthetic generation ─────────────────────────────────────────────────

    def _generate_synthetic(self, n_syn, scores_np):
        """Generate n_syn transitions conditioned on high-curiosity scores."""
        top_k = max(1, int(len(scores_np) * 0.1))
        top_scores = scores_np[np.argsort(scores_np)[-top_k:]]
        conds = np.maximum(
            0,
            np.random.choice(top_scores, n_syn) + np.random.normal(0, 0.1, n_syn),
        )

        syn = self.diffusion.generate(
            (n_syn, self.transition_dim),
            torch.FloatTensor(conds[:, None]).to(DEVICE),
        )

        syn_s  = syn[:, :self.state_dim]
        syn_a  = syn[:, self.state_dim : self.state_dim + self.action_dim]
        syn_r  = syn[:, self.state_dim + self.action_dim]
        syn_c  = syn[:, self.state_dim + self.action_dim + 1]
        syn_ns = syn[:, self.state_dim + self.action_dim + 2:]
        syn_d  = torch.zeros(n_syn, device=DEVICE)

        return syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d

    # ── Main training step ───────────────────────────────────────────────────

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            # Score a pool of transitions
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            # Update ICM
            self._train_icm()

            # Update diffusion on real data
            batch_idx = np.random.choice(len(idx), BATCH_SIZE, replace=True)
            self._train_diffusion(
                self.buffer.get_transitions(idx[batch_idx]),
                scores_np[batch_idx],
            )

            # Generate synthetic batch
            n_syn = int(BATCH_SIZE * REPLAY_RATIO)
            syn_s, syn_a, syn_r, _, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)

            # Mix real + synthetic
            real_s, real_a, real_r, _, real_ns, real_d = self.buffer.sample(BATCH_SIZE - n_syn)
            states      = torch.cat([real_s, syn_s])
            actions     = torch.cat([real_a, syn_a])
            rewards     = torch.cat([real_r, syn_r])
            next_states = torch.cat([real_ns, syn_ns])
            dones       = torch.cat([real_d, syn_d])
        else:
            states, actions, rewards, _, next_states, dones = self.buffer.sample(BATCH_SIZE)

        self._sac_update(states, actions, rewards, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC + PGR + Rare-event memory
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRMemoryAgent(SACPGRAgent):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        self.rare_buffer = RareEventBuffer(state_dim, action_dim)

    def add_transition(self, s, a, r, c, ns, d):
        self.buffer.add(s, a, r, c, ns, d)
        if c > 0:
            self.rare_buffer.add(s, a, r, c, ns, d)

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            # Score pool
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            # Update ICM
            self._train_icm()

            # ── Diffusion training with rare-event upweighting ───────────
            n_rare = int(BATCH_SIZE * RARE_BATCH_RATIO)
            n_normal = BATCH_SIZE - n_rare

            batch_idx = np.random.choice(len(idx), n_normal, replace=True)
            normal_trans = self.buffer.get_transitions(idx[batch_idx])
            normal_scores = scores_np[batch_idx]
            normal_weights = np.ones(n_normal)

            if len(self.rare_buffer) > 0:
                rare_trans = self.rare_buffer.get_transitions(min(n_rare, len(self.rare_buffer)))
                if rare_trans is not None:
                    rare_scores = np.ones(len(rare_trans)) * scores_np.max() * 2
                    rare_weights = np.ones(len(rare_trans)) * RARE_WEIGHT

                    all_trans = np.vstack([normal_trans, rare_trans])
                    all_scores = np.concatenate([normal_scores, rare_scores])
                    all_weights = np.concatenate([normal_weights, rare_weights])
                else:
                    all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights
            else:
                all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights

            all_weights = all_weights / all_weights.mean()  # normalize
            self._train_diffusion(all_trans, all_scores, all_weights)

            # Generate synthetic batch
            n_syn = int(BATCH_SIZE * REPLAY_RATIO)
            syn_s, syn_a, syn_r, _, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)

            # Mix real + synthetic
            real_s, real_a, real_r, _, real_ns, real_d = self.buffer.sample(BATCH_SIZE - n_syn)
            states      = torch.cat([real_s, syn_s])
            actions     = torch.cat([real_a, syn_a])
            rewards     = torch.cat([real_r, syn_r])
            next_states = torch.cat([real_ns, syn_ns])
            dones       = torch.cat([real_d, syn_d])
        else:
            states, actions, rewards, _, next_states, dones = self.buffer.sample(BATCH_SIZE)

        self._sac_update(states, actions, rewards, next_states, dones)
