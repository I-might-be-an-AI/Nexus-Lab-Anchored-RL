"""
Training loop and experiment runner.
"""

import time
from collections import deque

import numpy as np


def train_agent(agent, env, n_episodes: int, label: str, updates_per_episode: int, max_steps: int):
    """
    Run one agent on one environment for n_episodes.
    Returns (rewards_list, costs_list).
    """
    rewards = []
    costs = []
    cost_window = deque(maxlen=50)
    t0 = time.time()

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, cost, done, _ = env.step(action)
            agent.add_transition(state, action, reward, cost, next_state, done)
            ep_reward += reward
            ep_cost += cost
            state = next_state
            if done:
                break

        rewards.append(ep_reward)
        costs.append(ep_cost)
        cost_window.append(ep_cost)

        for _ in range(updates_per_episode):
            agent.train_step()

        if (ep + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"[{label:12s}] Ep {ep+1:4d}/{n_episodes}  "
                f"Reward: {np.mean(rewards[-25:]):7.1f}  "
                f"Cost(50): {sum(cost_window):6.1f}  "
                f"Total: {sum(costs):7.1f}  "
                f"Time: {elapsed:.0f}s"
            )

    return rewards, costs


def plot_results(results: dict, save_path: str = "hazard_cheetah_experiment.png"):
    """Generate a 2×2 summary figure and save to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    colors = {"SAC": "#ff7f0e", "PGR": "#1f77b4", "PGR+Memory": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Smoothed rewards
    ax = axes[0, 0]
    for name, data in results.items():
        sm = np.convolve(data["rewards"], np.ones(25) / 25, mode="valid")
        ax.plot(sm, label=name, color=colors.get(name, None), linewidth=2)
    ax.set_title("Episode Rewards (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Smoothed costs
    ax = axes[0, 1]
    for name, data in results.items():
        sm = np.convolve(data["costs"], np.ones(25) / 25, mode="valid")
        ax.plot(sm, label=name, color=colors.get(name, None), linewidth=2)
    ax.set_title("Episode Costs (smoothed) — LOWER IS SAFER")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost (hazard hits)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Cumulative cost
    ax = axes[1, 0]
    for name, data in results.items():
        ax.plot(np.cumsum(data["costs"]), label=name, color=colors.get(name, None), linewidth=2)
    ax.set_title("Cumulative Cost — LOWER IS SAFER")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Bar chart — safety vs performance
    ax = axes[1, 1]
    names = list(results.keys())
    x = np.arange(len(names))
    total_costs = [results[n]["total_cost"] for n in names]
    avg_rewards = [np.mean(results[n]["rewards"]) for n in names]

    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, total_costs, 0.4, label="Total Cost", color="red", alpha=0.7)
    bars2 = ax2.bar(x + 0.2, avg_rewards, 0.4, label="Avg Reward", color="green", alpha=0.7)
    ax.set_ylabel("Total Cost (↓ safer)", color="red")
    ax2.set_ylabel("Avg Reward (↑ better)", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Safety vs Performance")

    for bar, val in zip(bars1, total_costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.0f}", ha="center", va="bottom", fontsize=10, color="red",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")


def print_summary(results: dict, n_episodes: int):
    """Print a text summary table + relative cost comparisons."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Method':<14} {'Avg Reward':>12} {'Total Cost':>12} {'Early Cost':>12} {'Late Cost':>12}"
    print(header)
    print("-" * 80)

    for name, data in results.items():
        avg_reward = np.mean(data["rewards"])
        total_cost = data["total_cost"]
        early_cost = sum(data["costs"][: n_episodes // 4])
        late_cost = sum(data["costs"][-n_episodes // 4 :])
        print(f"{name:<14} {avg_reward:>12.1f} {total_cost:>12.0f} {early_cost:>12.0f} {late_cost:>12.0f}")

    print("\n" + "=" * 80)
    print("ANALYSIS — Lower cost = Safer")
    print("=" * 80)

    sac_cost = results["SAC"]["total_cost"]
    pgr_cost = results["PGR"]["total_cost"]
    mem_cost = results["PGR+Memory"]["total_cost"]

    def pct(a, b):
        return (a - b) / max(b, 1) * 100

    print(f"\nPGR vs SAC:        {pct(pgr_cost, sac_cost):+.1f}% cost change")
    print(f"PGR+Memory vs SAC: {pct(mem_cost, sac_cost):+.1f}% cost change")
    print(f"PGR+Memory vs PGR: {pct(mem_cost, pgr_cost):+.1f}% cost change")
