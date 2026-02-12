"""
Agent implementations for the PGR Safety experiment.

Three agents, each building on the last:

  1. SACAgent           — vanilla Soft Actor-Critic. Learns from real data only.
                          This is our "no generative replay" baseline.

  2. SACPGRAgent        — SAC + Prioritized Generative Replay. Trains a diffusion
                          model on real transitions, conditions on curiosity, and
                          generates synthetic transitions to augment training.
                          Like the PGR paper (Wang et al., 2025).

  3. SACPGRMemoryAgent  — PGR + our rare-event memory bank. Same as above, but
                          also stores hazardous transitions in a separate buffer
                          and upweights them during diffusion training. This is
                          OUR contribution — preventing forgetting of catastrophic
                          events that the curiosity signal would otherwise
                          deprioritize once they become "familiar."
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import (
    DEVICE, LR, GAMMA, TAU, BATCH_SIZE,
    LATENT_DIM, REPLAY_RATIO, PGR_START_BUFFER,
    RARE_BATCH_RATIO, RARE_WEIGHT,
    CFG_P_UNCOND, CFG_GUIDANCE_SCALE,
    COST_LIMIT, LAMBDA_LR, LAMBDA_INIT,
)
from buffers import ReplayBuffer, RareEventBuffer
from networks import (
    QNetwork, GaussianPolicy,
    StateEncoder, ForwardModel,
    NoisePredictor, Diffusion,
    normalize_scores,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Base SAC Agent
#
# SAC (Soft Actor-Critic) is an off-policy RL algorithm that learns:
#   - A policy π(a|s)       — "what action to take" (the actor)
#   - Q-functions Q(s,a)    — "how good is this state-action pair" (critics)
#   - Temperature α         — balances reward vs exploration (entropy)
#
# The "soft" part: SAC adds an entropy bonus to the reward, encouraging
# the policy to be as random as possible while still achieving high reward.
# This helps with exploration and makes training more stable.
# ═══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    """Vanilla SAC baseline — learns purely from real environment experience."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ── Actor: the policy network ────────────────────────────────────
        self.policy = GaussianPolicy(state_dim, action_dim).to(DEVICE)

        # ── Critics: two Q-networks (double-Q trick to reduce overestimation)
        # We also keep "target" copies that update slowly for stable training.
        # The target networks provide the bootstrap target for Q-learning:
        #   Q_target(s', a') where a' ~ π(·|s')
        self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())  # start identical
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ── Entropy temperature (auto-tuned) ─────────────────────────────
        # α controls the reward-entropy tradeoff: J = E[r + α * entropy]
        # Higher α = more exploration, lower α = more exploitation.
        # SAC auto-tunes α so that the policy's entropy stays near a target.
        self.target_entropy = -action_dim  # heuristic: -dim(action_space)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp()

        # ── Optimizers ───────────────────────────────────────────────────
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

        # ── Lagrangian cost constraint ──────────────────────────────────
        # λ (lambda) penalizes the reward when the agent hits hazards:
        #   effective_reward = reward - λ * cost
        #
        # λ auto-tunes via dual gradient descent:
        #   If recent cost > COST_LIMIT → λ increases (penalize harder)
        #   If recent cost < COST_LIMIT → λ decreases (relax penalty)
        #
        # We store log(λ) and exponentiate so λ stays positive.
        # This is the same trick SAC uses for α (entropy temperature).
        self.log_lambda = torch.tensor(
            np.log(max(LAMBDA_INIT, 1e-8)),  # log(1.0) = 0.0
            requires_grad=True, device=DEVICE, dtype=torch.float32,
        )
        self.lam = self.log_lambda.exp().item()  # current λ value
        self.lambda_opt = optim.Adam([self.log_lambda], lr=LAMBDA_LR)

        # Track recent episode costs for λ update
        self.recent_costs = []  # costs from recent episodes

        # ── Replay buffer ────────────────────────────────────────────────
        self.buffer = ReplayBuffer(state_dim, action_dim)

    # ── Environment interaction ──────────────────────────────────────────

    def select_action(self, state):
        """Pick an action for the given state (used during env rollouts)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.policy.get_action(s)

    def add_transition(self, s, a, r, c, ns, d):
        """Store a transition in the replay buffer."""
        self.buffer.add(s, a, r, c, ns, d)

    # ── Core SAC update ──────────────────────────────────────────────────

    def _soft_update(self):
        """
        Slowly blend online Q-networks into target Q-networks.

        target = TAU * online + (1 - TAU) * target

        With TAU=0.005, the target network moves very slowly, providing
        a stable training signal. Without this, Q-learning is unstable
        because the target we're chasing keeps changing.
        """
        for tp, sp in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)
        for tp, sp in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

    def _sac_update(self, states, actions, rewards, costs, next_states, dones):
        """
        One full SAC gradient step with Lagrangian cost constraint.

        The key change from vanilla SAC:
          effective_reward = reward - λ * cost
        
        This makes the policy AVOID high-cost transitions. λ auto-tunes
        so the agent finds the best reward while staying under COST_LIMIT.

        Three updates happen:

        1. CRITICS (Q-networks):
           Target: y = (r - λ*c) + γ * (1-done) * (min Q_target(s', a') - α * log π(a'|s'))
           Loss:   MSE(Q(s,a), y)  for both Q1 and Q2

        2. ACTOR (policy):
           Loss:   E[α * log π(a|s) - min Q(s, a)]
           This pushes the policy toward actions with high Q and high entropy.

        3. TEMPERATURE (α):
           Loss:   -α * (log π(a|s) + target_entropy)
           If entropy is too low, α increases (more exploration).
           If entropy is too high, α decreases (more exploitation).

        Args:
            states, actions, rewards, costs, next_states, dones — batch of transitions
            (can be real, synthetic, or a mix)
        """
        # ── Compute cost-penalized reward ──────────────────────────────
        # This is the CORE of the safety mechanism:
        # effective_reward = reward - λ * cost
        # When λ is high and cost > 0, the effective reward becomes very negative,
        # teaching the Q-function that hazard transitions are BAD.
        effective_rewards = rewards - self.lam * costs

        # ── Step 1: Critic update ────────────────────────────────────────
        with torch.no_grad():
            # Sample next actions from current policy (for bootstrapping)
            next_actions, next_log_probs = self.policy.sample(next_states)

            # Double-Q: take the minimum of both target Q-networks
            # This prevents overestimation of Q-values
            q_target = torch.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            )

            # Bellman target using COST-PENALIZED reward:
            # y = (r - λ*c) + γ * (Q_target - α * log_prob)
            target = effective_rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (
                q_target - self.alpha * next_log_probs
            )

        # Update Q1
        self.q1_opt.zero_grad()
        F.mse_loss(self.q1(states, actions), target).backward()
        self.q1_opt.step()

        # Update Q2
        self.q2_opt.zero_grad()
        F.mse_loss(self.q2(states, actions), target).backward()
        self.q2_opt.step()

        # ── Step 2: Actor (policy) update ────────────────────────────────
        # Sample fresh actions from current policy
        new_actions, log_probs = self.policy.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))

        # Policy loss: maximize Q while maximizing entropy
        # = minimize (α * log_prob - Q)
        self.policy_opt.zero_grad()
        (self.alpha.detach() * log_probs - q_new).mean().backward()
        self.policy_opt.step()

        # ── Step 3: Temperature (α) update ───────────────────────────────
        # If entropy > target → α decreases (less exploration needed)
        # If entropy < target → α increases (need more exploration)
        self.alpha_opt.zero_grad()
        (-(self.log_alpha * (log_probs + self.target_entropy).detach())).mean().backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        # Slowly update target networks
        self._soft_update()

    def record_episode_cost(self, episode_cost):
        """
        Called at the end of each episode to record how much cost was incurred.
        After enough episodes, we update λ (the Lagrange multiplier).

        Dual gradient descent on λ:
          ∇λ = (avg_recent_cost - COST_LIMIT)
          If avg_cost > COST_LIMIT → gradient positive → λ increases → penalize more
          If avg_cost < COST_LIMIT → gradient negative → λ decreases → relax penalty

        We update every 10 episodes to reduce noise.
        """
        self.recent_costs.append(episode_cost)

        # Update λ every 10 episodes (smooths out noise)
        if len(self.recent_costs) >= 10:
            avg_cost = np.mean(self.recent_costs)
            self.recent_costs = []  # reset

            # Dual gradient step: push λ toward satisfying the constraint
            # Loss for λ: λ * (avg_cost - COST_LIMIT)
            # If avg_cost > limit → loss positive → λ should increase
            # If avg_cost < limit → loss negative → λ should decrease
            self.lambda_opt.zero_grad()
            lambda_loss = self.log_lambda.exp() * (avg_cost - COST_LIMIT)
            lambda_loss.backward()
            self.lambda_opt.step()

            # Update the cached value (used in _sac_update)
            self.lam = self.log_lambda.exp().item()

            # Clamp to prevent λ from going crazy
            # Min: 0.01 (always some penalty)  Max: 100.0 (don't over-penalize)
            self.lam = np.clip(self.lam, 0.01, 100.0)
            self.log_lambda.data.copy_(torch.tensor(np.log(self.lam), device=DEVICE))

    def train_step(self):
        """One training iteration: sample batch from buffer → SAC update."""
        if len(self.buffer) < BATCH_SIZE:
            return  # not enough data yet
        states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)
        self._sac_update(states, actions, rewards, costs, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC + PGR (Prioritized Generative Replay)
#
# This extends SAC with a conditional diffusion model that:
#   1. Learns to generate transitions from the replay buffer
#   2. Uses curiosity (ICM prediction error) as a relevance signal
#   3. Generates synthetic transitions biased toward high-curiosity regions
#   4. Mixes synthetic + real data for SAC training
#
# Key insight from the paper: conditioning on curiosity generates MORE DIVERSE
# transitions, which REDUCES overfitting of the Q-function to synthetic data.
# This is NOT about generating higher-quality data — it's about generating
# the RIGHT KIND of data.
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRAgent(SACAgent):
    """SAC + curiosity-conditioned diffusion replay (PGR paper approach)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)

        # Transition dimension: the diffusion model generates entire transitions
        # as flat vectors: [s, a, r, c, s'] all concatenated
        self.transition_dim = state_dim + action_dim + 1 + 1 + state_dim  # s,a,r,c,s'

        # ── ICM (Intrinsic Curiosity Module) ─────────────────────────────
        # Provides the relevance function F(τ) = prediction error
        self.encoder = StateEncoder(state_dim).to(DEVICE)
        self.fwd_model = ForwardModel(LATENT_DIM, action_dim).to(DEVICE)

        # ── Conditional diffusion model ──────────────────────────────────
        # Learns to generate transitions conditioned on curiosity scores
        self.noise_pred = NoisePredictor(self.transition_dim).to(DEVICE)
        self.diffusion = Diffusion(
            self.noise_pred,
            p_uncond=CFG_P_UNCOND,           # 25% of time, drop the condition
            guidance_scale=CFG_GUIDANCE_SCALE, # amplify conditioning at gen time
        )

        # ── Optimizers for the new components ────────────────────────────
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=LR)
        self.fwd_opt = optim.Adam(self.fwd_model.parameters(), lr=LR)
        self.diff_opt = optim.Adam(self.noise_pred.parameters(), lr=LR)

    # ── Curiosity scoring ────────────────────────────────────────────────

    def _compute_curiosity(self, states, actions, next_states):
        """
        Compute ICM curiosity scores for a batch of transitions.

        Curiosity = forward model prediction error in latent space:
            F(s, a, s') = 0.5 * ||g(h(s), a) - h(s')||²

        High score = the ICM can't predict this transition = it's novel.
        Low score = the ICM has seen enough similar transitions = familiar.

        Returns: (batch_size,) tensor of curiosity scores (detached — no grad)
        """
        with torch.no_grad():
            h_s = self.encoder(states)        # encode current state
            h_ns = self.encoder(next_states)  # encode next state (ground truth)
            # Prediction error: how far off is the forward model?
            pred_h_ns = self.fwd_model(h_s, actions)
            return 0.5 * ((pred_h_ns - h_ns) ** 2).mean(dim=-1)

    def _train_icm(self):
        """
        One gradient step to improve the ICM's forward prediction.

        We train the encoder and forward model jointly to minimize:
            ||ForwardModel(Encoder(s), a) - Encoder(s')||²

        As the ICM gets better at predicting, curiosity scores for familiar
        transitions drop — which is exactly what we want. Novel transitions
        remain high-curiosity because the ICM hasn't learned them yet.
        """
        s, a, _, _, ns, _ = self.buffer.sample(BATCH_SIZE)
        h_s, h_ns = self.encoder(s), self.encoder(ns)

        self.encoder_opt.zero_grad()
        self.fwd_opt.zero_grad()
        F.mse_loss(self.fwd_model(h_s, a), h_ns).backward()
        self.encoder_opt.step()
        self.fwd_opt.step()

    # ── Diffusion training ───────────────────────────────────────────────

    def _train_diffusion(self, transitions, scores, weights=None):
        """
        One gradient step on the conditional diffusion model.

        Teaches the diffusion model: "given this curiosity score as a condition,
        learn to generate transitions that look like the real ones with that
        score level."

        Args:
            transitions: (batch, transition_dim) — flat real transition vectors
            scores:      (batch,) — normalized curiosity scores for each
            weights:     (batch,) — optional per-sample loss weights
        """
        x0 = torch.FloatTensor(transitions).to(DEVICE)
        rel = torch.FloatTensor(scores[:, None]).to(DEVICE)  # (batch, 1)
        w = torch.FloatTensor(weights).to(DEVICE) if weights is not None else None

        self.diff_opt.zero_grad()
        self.diffusion.loss(x0, rel, weights=w).backward()
        self.diff_opt.step()

    # ── Synthetic generation ─────────────────────────────────────────────

    def _generate_synthetic(self, n_syn, scores_np):
        """
        Generate n_syn synthetic transitions using the diffusion model,
        conditioned on HIGH curiosity scores.

        Strategy (from PGR paper — "prompting"):
          1. Find the top 10% of curiosity scores in the current pool
          2. Randomly sample from these top scores + small noise
          3. Use these as the relevance condition for diffusion generation

        This biases generation toward transitions that are novel/surprising,
        which the paper shows increases diversity and reduces overfitting.

        Returns: tuple of (states, actions, rewards, costs, next_states, dones)
                 all as GPU tensors
        """
        # Pick conditioning values from the top 10% of curiosity scores
        top_k = max(1, int(len(scores_np) * 0.1))
        top_scores = scores_np[np.argsort(scores_np)[-top_k:]]

        # Sample conditions + small Gaussian noise for variety
        conds = np.maximum(
            0,
            np.random.choice(top_scores, n_syn) + np.random.normal(0, 0.1, n_syn),
        )

        # Generate! This runs the full reverse diffusion with CFG
        syn = self.diffusion.generate(
            (n_syn, self.transition_dim),
            torch.FloatTensor(conds[:, None]).to(DEVICE),
        )

        # Slice the flat vector back into individual components
        # Layout: [s (state_dim) | a (action_dim) | r (1) | c (1) | s' (state_dim)]
        syn_s  = syn[:, :self.state_dim]
        syn_a  = syn[:, self.state_dim : self.state_dim + self.action_dim]
        syn_r  = syn[:, self.state_dim + self.action_dim]
        syn_c  = syn[:, self.state_dim + self.action_dim + 1]
        syn_ns = syn[:, self.state_dim + self.action_dim + 2:]
        syn_d  = torch.zeros(n_syn, device=DEVICE)  # assume not terminal

        return syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d

    # ── Main training step ───────────────────────────────────────────────

    def train_step(self):
        """
        One PGR training iteration. This is the core algorithm (Algorithm 1 in paper):

        Phase 1 — Score & learn (when enough data):
          a) Sample a pool of transitions from the buffer
          b) Compute curiosity scores for the pool
          c) Train the ICM (so it gets better at predicting)
          d) Train the diffusion model on real transitions + their scores

        Phase 2 — Generate & mix:
          e) Generate synthetic transitions conditioned on high curiosity
          f) Mix real + synthetic data (REPLAY_RATIO controls the split)
          g) Run one SAC update on the mixed batch
        """
        if len(self.buffer) < BATCH_SIZE:
            return  # need enough data first

        # Only use PGR once we have enough data for the diffusion model
        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            # ── Phase 1: Score a pool and train components ───────────
            # Sample a large pool of transitions (up to 5000)
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            # Compute and normalize curiosity for the pool
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            # Train ICM to get better at prediction (makes curiosity more accurate)
            self._train_icm()

            # Train diffusion model: "learn to generate transitions like these,
            # given their curiosity scores as conditioning"
            batch_idx = np.random.choice(len(idx), BATCH_SIZE, replace=True)
            self._train_diffusion(
                self.buffer.get_transitions(idx[batch_idx]),
                scores_np[batch_idx],
            )

            # ── Phase 2: Generate synthetic data and mix ─────────────
            n_syn = int(BATCH_SIZE * REPLAY_RATIO)  # e.g. 30% of batch
            syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)

            # Fill the rest of the batch with real data
            real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(BATCH_SIZE - n_syn)

            # Concatenate real + synthetic
            states      = torch.cat([real_s, syn_s])
            actions     = torch.cat([real_a, syn_a])
            rewards     = torch.cat([real_r, syn_r])
            costs       = torch.cat([real_c, syn_c])
            next_states = torch.cat([real_ns, syn_ns])
            dones       = torch.cat([real_d, syn_d])
        else:
            # Before PGR kicks in, just train on real data (same as vanilla SAC)
            states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Run SAC update on the (possibly augmented) batch
        self._sac_update(states, actions, rewards, costs, next_states, dones)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC + PGR + Rare-event memory  (OUR CONTRIBUTION)
#
# Problem: PGR's curiosity relevance function prioritizes NOVEL transitions.
# Once the ICM learns to predict hazard transitions, they stop being novel
# and get deprioritized. The diffusion model then stops generating
# hazard-adjacent synthetic data, and the policy "forgets" to avoid hazards.
#
# Solution: maintain a separate small buffer of catastrophic transitions
# and inject them into diffusion training with upweighted loss. This forces
# the diffusion model to always know how to generate near-hazard transitions,
# even when the ICM no longer considers them novel.
# ═══════════════════════════════════════════════════════════════════════════════

class SACPGRMemoryAgent(SACPGRAgent):
    """PGR + rare-event memory bank for hazardous transitions."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        # Small dedicated buffer for catastrophic transitions
        self.rare_buffer = RareEventBuffer(state_dim, action_dim)

    def add_transition(self, s, a, r, c, ns, d):
        """
        Store transition in main buffer, AND if it's hazardous (cost > 0),
        also store it in the rare-event buffer.
        """
        self.buffer.add(s, a, r, c, ns, d)
        if c > 0:   # this transition hit a hazard!
            self.rare_buffer.add(s, a, r, c, ns, d)

    def train_step(self):
        """
        Same as PGR train_step, but with one key difference:
        when training the diffusion model, we MIX in rare-event transitions
        with higher loss weights.

        The diffusion training batch becomes:
          - 80% normal transitions from the main buffer (weight = 1.0)
          - 20% catastrophic transitions from rare buffer (weight = 5.0)

        This forces the diffusion model to learn the structure of hazardous
        transitions 5x more thoroughly than normal ones, preventing the
        "curiosity decay → forgetting" problem.
        """
        if len(self.buffer) < BATCH_SIZE:
            return

        use_pgr = len(self.buffer) > PGR_START_BUFFER

        if use_pgr:
            # Score a pool of transitions (same as base PGR)
            s, a, ns, r, c, d, idx = self.buffer.sample_with_idx(
                min(5000, len(self.buffer))
            )
            scores_np = normalize_scores(self._compute_curiosity(s, a, ns).cpu().numpy())

            # Train ICM
            self._train_icm()

            # ── KEY DIFFERENCE: diffusion training with rare-event injection ──
            n_rare = int(BATCH_SIZE * RARE_BATCH_RATIO)    # 20% rare
            n_normal = BATCH_SIZE - n_rare                  # 80% normal

            # Normal transitions from main buffer
            batch_idx = np.random.choice(len(idx), n_normal, replace=True)
            normal_trans = self.buffer.get_transitions(idx[batch_idx])
            normal_scores = scores_np[batch_idx]
            normal_weights = np.ones(n_normal)  # weight = 1.0

            # Inject rare transitions if available
            if len(self.rare_buffer) > 0:
                rare_trans = self.rare_buffer.get_transitions(min(n_rare, len(self.rare_buffer)))
                if rare_trans is not None:
                    # Give rare transitions artificially HIGH relevance scores
                    # so the diffusion model associates them with high-priority
                    rare_scores = np.ones(len(rare_trans)) * scores_np.max() * 2

                    # Upweight their loss contribution — forces the model to
                    # learn these transitions more thoroughly
                    rare_weights = np.ones(len(rare_trans)) * RARE_WEIGHT

                    # Combine normal + rare into one batch
                    all_trans = np.vstack([normal_trans, rare_trans])
                    all_scores = np.concatenate([normal_scores, rare_scores])
                    all_weights = np.concatenate([normal_weights, rare_weights])
                else:
                    all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights
            else:
                all_trans, all_scores, all_weights = normal_trans, normal_scores, normal_weights

            # Normalize weights so they average to 1.0
            # (otherwise total loss magnitude changes with rare buffer size)
            all_weights = all_weights / all_weights.mean()

            # Train diffusion on this mixed batch
            self._train_diffusion(all_trans, all_scores, all_weights)

            # ── Generate and mix (same as base PGR) ──────────────────────
            n_syn = int(BATCH_SIZE * REPLAY_RATIO)
            syn_s, syn_a, syn_r, syn_c, syn_ns, syn_d = self._generate_synthetic(n_syn, scores_np)

            real_s, real_a, real_r, real_c, real_ns, real_d = self.buffer.sample(BATCH_SIZE - n_syn)
            states      = torch.cat([real_s, syn_s])
            actions     = torch.cat([real_a, syn_a])
            rewards     = torch.cat([real_r, syn_r])
            costs       = torch.cat([real_c, syn_c])
            next_states = torch.cat([real_ns, syn_ns])
            dones       = torch.cat([real_d, syn_d])
        else:
            states, actions, rewards, costs, next_states, dones = self.buffer.sample(BATCH_SIZE)

        self._sac_update(states, actions, rewards, costs, next_states, dones)
