#!/usr/bin/env python3
"""
PGR Safety Experiment — Main Entry Point
=========================================

Compares three agents on hazard-injected environments:
  1. SAC          — vanilla baseline (no generative replay)
  2. PGR          — SAC + curiosity-conditioned generative replay
  3. PGR+Memory   — PGR + rare-event memory (our contribution)

The key question: does the rare-event buffer help the agent avoid hazards
that PGR alone would "forget" once the curiosity signal decays?

Usage:
  python main.py                           # default: point env, 3000 episodes, all methods
  python main.py --episodes 500            # quick test
  python main.py --env ant                 # switch to HazardAnt (needs MuJoCo)
  python main.py --methods SAC PGR         # only run a subset of methods
"""

import argparse
import random

import numpy as np
import torch

from config import DEVICE, SEED, N_EPISODES, MAX_STEPS, UPDATES_PER_EPISODE
from envs import HazardHalfCheetah, HazardAnt, PointHazardEnv
from agents import SACAgent, SACPGRAgent, SACPGRMemoryAgent
from train import train_agent, plot_results, print_summary


# ── Registries: map CLI names to classes ─────────────────────────────────────
# Add new envs/agents here and they'll automatically be available via CLI.

ENV_REGISTRY = {
    "point": PointHazardEnv,       # fast, no dependencies beyond numpy
    "cheetah": HazardHalfCheetah,  # needs gymnasium[mujoco]
    "ant": HazardAnt,              # needs gymnasium[mujoco]
}

AGENT_REGISTRY = {
    "SAC": SACAgent,
    "PGR": SACPGRAgent,
    "PGR+Memory": SACPGRMemoryAgent,
}


def seed_everything(seed: int):
    """Set all random seeds for reproducibility across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def run_experiment(env_name: str, n_episodes: int, methods: list[str] | None = None):
    """
    Run the full experiment: train each method sequentially, collect results,
    print summary, and generate plots.

    Each method gets the SAME random seed so differences are due to the
    algorithm, not random initialization.

    Args:
        env_name:   key into ENV_REGISTRY (e.g. "point")
        n_episodes: total training episodes per method
        methods:    list of method names to run (default: all)

    Returns:
        dict mapping method_name → results dict
    """
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

        # Same seed for each method → fair comparison
        seed_everything(SEED)
        env = EnvClass()
        agent = AgentClass(env.state_dim, env.action_dim)

        # Run training
        rewards, costs = train_agent(
            agent, env, n_episodes, name,
            updates_per_episode=UPDATES_PER_EPISODE,
            max_steps=MAX_STEPS,
        )

        # Store results
        results[name] = {
            "rewards": rewards,
            "costs": costs,
            "total_cost": sum(costs),
        }

        # Add env-specific stats if available (e.g. velocity/angle hazard counts)
        if hasattr(env, "total_velocity_hazards"):
            results[name]["velocity_hazards"] = env.total_velocity_hazards
            results[name]["angle_hazards"] = env.total_angle_hazards

        print(
            f"→ {name}: Avg Reward={np.mean(rewards):.1f}, "
            f"Total Cost={sum(costs):.0f}"
        )

        # ── Trajectory visualization for point env ───────────────────────
        # Roll out the trained policy once and save the path it takes
        if env_name == "point":
            demo_env = EnvClass()
            seed_everything(SEED)
            s = demo_env.reset()
            traj = [s[:2].copy()]  # record (x, y) positions
            for _ in range(200):
                a = agent.select_action(s)
                s, _, _, done, _ = demo_env.step(a)
                traj.append(s[:2].copy())
                if done:
                    break
            # Save a visualization showing the path + hazard zones
            fig = demo_env.render_trajectory(traj)
            fig.savefig(f"trajectory_{name.replace('+', '_')}.png", dpi=120)
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"   Trajectory saved → trajectory_{name.replace('+', '_')}.png")
            demo_env.close()

        env.close()

    # ── Summary and plots ────────────────────────────────────────────────
    print_summary(results, n_episodes)
    plot_results(results, save_path=f"hazard_{env_name}_experiment.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="PGR Safety Experiment")
    parser.add_argument(
        "--env", choices=list(ENV_REGISTRY.keys()), default="point",
        help="Which environment to use (default: point — fast, no MuJoCo needed)",
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of training episodes (default: {N_EPISODES})",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(AGENT_REGISTRY.keys()),
        default=None,
        help="Subset of methods to run (default: all three)",
    )
    args = parser.parse_args()

    run_experiment(args.env, args.episodes, args.methods)


if __name__ == "__main__":
    main()
