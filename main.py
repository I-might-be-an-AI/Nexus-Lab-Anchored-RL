#!/usr/bin/env python3
"""
PGR Safety Experiment — MuJoCo with Injected Hazards
=====================================================

Compares three agents on HalfCheetah with velocity/angle/gust hazards:
  1. SAC          – vanilla baseline
  2. PGR          – SAC + curiosity-conditioned generative replay
  3. PGR+Memory   – PGR + rare-event memory bank for catastrophic transitions

Usage:
  python main.py                     # defaults
  python main.py --episodes 1000     # quick test
  python main.py --env ant           # switch to HazardAnt
"""

import argparse
import random

import numpy as np
import torch

from config import DEVICE, SEED, N_EPISODES, MAX_STEPS, UPDATES_PER_EPISODE
from envs import HazardHalfCheetah, HazardAnt
from agents import SACAgent, SACPGRAgent, SACPGRMemoryAgent
from train import train_agent, plot_results, print_summary


ENV_REGISTRY = {
    "cheetah": HazardHalfCheetah,
    "ant": HazardAnt,
}

AGENT_REGISTRY = {
    "SAC": SACAgent,
    "PGR": SACPGRAgent,
    "PGR+Memory": SACPGRMemoryAgent,
}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def run_experiment(env_name: str, n_episodes: int, methods: list[str] | None = None):
    EnvClass = ENV_REGISTRY[env_name]
    if methods is None:
        methods = list(AGENT_REGISTRY.keys())

    print("=" * 80)
    print(f"HAZARD {env_name.upper()} EXPERIMENT")
    print("=" * 80)
    print(f"Device:   {DEVICE}")
    print(f"Episodes: {n_episodes}")
    print(f"Methods:  {', '.join(methods)}")
    print("=" * 80)

    results = {}

    for name in methods:
        AgentClass = AGENT_REGISTRY[name]

        print(f"\n{'─' * 80}")
        print(f"Training {name}")
        print(f"{'─' * 80}")

        seed_everything(SEED)
        env = EnvClass()
        agent = AgentClass(env.state_dim, env.action_dim)

        rewards, costs = train_agent(
            agent, env, n_episodes, name,
            updates_per_episode=UPDATES_PER_EPISODE,
            max_steps=MAX_STEPS,
        )

        results[name] = {
            "rewards": rewards,
            "costs": costs,
            "total_cost": sum(costs),
        }

        # Add env-specific stats if available
        if hasattr(env, "total_velocity_hazards"):
            results[name]["velocity_hazards"] = env.total_velocity_hazards
            results[name]["angle_hazards"] = env.total_angle_hazards

        print(
            f"→ {name}: Avg Reward={np.mean(rewards):.1f}, "
            f"Total Cost={sum(costs):.0f}"
        )
        env.close()

    print_summary(results, n_episodes)
    plot_results(results, save_path=f"hazard_{env_name}_experiment.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="PGR Safety Experiment")
    parser.add_argument("--env", choices=list(ENV_REGISTRY.keys()), default="cheetah")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument(
        "--methods", nargs="+", choices=list(AGENT_REGISTRY.keys()),
        default=None, help="Subset of methods to run (default: all)",
    )
    args = parser.parse_args()

    run_experiment(args.env, args.episodes, args.methods)


if __name__ == "__main__":
    main()
