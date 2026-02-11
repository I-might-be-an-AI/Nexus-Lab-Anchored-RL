"""
Replay buffers: standard uniform replay and rare-event memory bank.
"""

import random
import numpy as np
import torch

from config import BUFFER_SIZE, RARE_BUFFER_SIZE, DEVICE


class ReplayBuffer:
    """Fixed-size ring buffer storing (s, a, r, c, s', done) transitions."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.costs = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def add(self, state, action, reward, cost, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.costs[idx]).to(DEVICE),
            torch.FloatTensor(self.next_states[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
        )

    def sample_with_idx(self, n: int):
        """Sample n transitions, also returning their buffer indices."""
        n = min(n, self.size)
        idx = np.random.choice(self.size, n, replace=False)
        return (
            torch.FloatTensor(self.states[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.next_states[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.costs[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
            idx,
        )

    def get_transitions(self, idx):
        """Return flat (s, a, r, c, s') arrays for given indices."""
        return np.concatenate([
            self.states[idx],
            self.actions[idx],
            self.rewards[idx].reshape(-1, 1),
            self.costs[idx].reshape(-1, 1),
            self.next_states[idx],
        ], axis=-1)

    def __len__(self):
        return self.size


class RareEventBuffer:
    """
    Small memory bank that retains rare / catastrophic transitions.
    Currently uses simple FIFO eviction.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = RARE_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: list[dict] = []

    def add(self, state, action, reward, cost, next_state, done):
        self.buffer.append({
            "state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": float(reward),
            "cost": float(cost),
            "next_state": np.array(next_state, dtype=np.float32),
            "done": float(done),
        })
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_transitions(self, batch_size: int):
        """Return flat transition arrays for a random subset."""
        batch_size = min(batch_size, len(self.buffer))
        if batch_size == 0:
            return None
        batch = random.sample(self.buffer, batch_size)
        return np.stack([
            np.concatenate([
                t["state"], t["action"], [t["reward"]], [t["cost"]], t["next_state"]
            ])
            for t in batch
        ])

    def __len__(self):
        return len(self.buffer)
