"""
UltraTrader Multi-Seed Evaluation Script - Robustness Analysis

This script measures robustness and stability across different random seeds,
similar to internal PM review processes at institutional trading desks.

For each seed, runs multiple episodes and computes:
- Mean reward and std
- Sharpe-like ratio (reward mean / reward std)
- Average maximum drawdown across episodes
- Average episode length

Outputs (saved to reports/):
1. seed_stats.csv: Per-seed performance statistics with risk metrics
2. equity_multiseed.png: Mean equity curve ± 1 std across all seeds/episodes

This analysis answers: "Is our strategy robust to different market conditions,
or did we just get lucky with one seed?"

Usage:
    python seedeval.py
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from ultra_trading_env import UltraTradingEnv


def compute_max_drawdown(equity_curve):
    """
    Compute maximum drawdown from equity curve.

    Args:
        equity_curve: List or array of equity values over time

    Returns:
        float: Maximum drawdown as fraction (0.0 to 1.0)
    """
    peak = -1e9
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def compute_sharpe_like(reward_list):
    """
    Compute Sharpe-like ratio from reward sequence.

    Args:
        reward_list: List of episode rewards

    Returns:
        float: Mean reward / std reward (higher is better)
    """
    if len(reward_list) < 2:
        return 0.0
    mean_r = float(np.mean(reward_list))
    std_r = float(np.std(reward_list)) + 1e-8
    return mean_r / std_r


def rollout_once(model, episodes=10):
    """
    Run multiple evaluation episodes and collect performance data.

    Args:
        model: Trained SAC model
        episodes: Number of episodes to run (default: 10)

    Returns:
        tuple: (rewards, lengths, curves, max_dds)
            - rewards: List of cumulative rewards per episode
            - lengths: List of episode lengths
            - curves: List of equity curves (each is a list of floats)
            - max_dds: List of max drawdown per episode
    """
    rewards, lengths, curves, max_dds = [], [], [], []

    for _ in range(episodes):
        env = UltraTradingEnv()
        obs, _ = env.reset()
        curve = [env.equity]
        total_reward = 0.0
        done = False
        trunc = False

        while not (done or trunc):
            action, _ = model.predict(np.array(obs, dtype=np.float32), deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            curve.append(info["equity"])

        rewards.append(total_reward)
        lengths.append(len(curve))
        curves.append(curve)
        max_dds.append(compute_max_drawdown(curve))

    return rewards, lengths, curves, max_dds


def main(
    model_path="checkpoints/ultra_sac_last.zip",
    seeds=(11, 22, 33, 44, 55, 66, 77, 88, 99, 111),
    episodes_per_seed=5
):
    """
    Multi-seed robustness evaluation with institutional risk metrics.

    Args:
        model_path: Path to trained model
        seeds: Tuple of random seeds to test
        episodes_per_seed: Episodes to run per seed

    Outputs:
        - reports/seed_stats.csv: Per-seed statistics
        - reports/equity_multiseed.png: Mean equity ± std visualization
    """
    print(f"\n{'='*70}")
    print(f"MULTI-SEED ROBUSTNESS EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Seeds: {len(seeds)}")
    print(f"Episodes per seed: {episodes_per_seed}")
    print(f"Total episodes: {len(seeds) * episodes_per_seed}")
    print(f"{'='*70}\n")

    model = SAC.load(model_path)

    all_blocks = []
    stats_rows = []

    for seed_idx, seed in enumerate(seeds):
        print(f"[Seed {seed_idx+1}/{len(seeds)}] seed={seed}")
        np.random.seed(seed)

        reward_list, length_list, curves, max_dd_list = rollout_once(
            model, episodes=episodes_per_seed
        )

        # Compute per-seed metrics
        mean_reward = float(np.mean(reward_list))
        std_reward = float(np.std(reward_list))
        sharpe = compute_sharpe_like(reward_list)
        avg_max_dd = float(np.mean(max_dd_list))
        avg_length = float(np.mean(length_list))

        print(f"  Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"  Sharpe-like: {sharpe:.3f}")
        print(f"  Avg Max DD: {avg_max_dd:.4f}")
        print(f"  Avg Length: {avg_length:.1f}\n")

        stats_rows.append((
            seed,
            mean_reward,
            std_reward,
            sharpe,
            avg_max_dd,
            avg_length
        ))

        # Align curves within this seed
        max_len_local = max(len(c) for c in curves)
        padded_local = []
        for curve in curves:
            if len(curve) < max_len_local:
                # Pad with last value
                curve = curve + [curve[-1]] * (max_len_local - len(curve))
            padded_local.append(curve)

        block = np.array(padded_local, dtype=np.float32)
        all_blocks.append(block)

    # Align all seed blocks to same global length
    global_T = max(b.shape[1] for b in all_blocks)
    all_normed = []
    for block in all_blocks:
        if block.shape[1] < global_T:
            # Pad with last column repeated
            tail = np.repeat(
                block[:, -1][:, None],
                repeats=(global_T - block.shape[1]),
                axis=1
            )
            block = np.concatenate([block, tail], axis=1)
        all_normed.append(block)

    # Combine all episodes across all seeds
    combined = np.concatenate(all_normed, axis=0)  # (total_episodes, T)
    mu = combined.mean(axis=0)
    sd = combined.std(axis=0)
    x = np.arange(len(mu))

    print(f"{'='*70}")
    print(f"AGGREGATE STATISTICS")
    print(f"{'='*70}")
    print(f"Total episodes: {combined.shape[0]}")
    print(f"Max episode length: {combined.shape[1]}")
    print(f"Mean final equity (across all): {mu[-1]:.4f} ± {sd[-1]:.4f}")
    print(f"{'='*70}\n")

    os.makedirs("reports", exist_ok=True)

    # Save per-seed stats CSV
    stats_path = os.path.join("reports", "seed_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "reward_mean", "reward_std", "sharpe_like", "avg_max_drawdown", "avg_ep_length"])
        for row in stats_rows:
            writer.writerow(row)
    print(f"✓ Saved: {stats_path}")

    # Plot multi-seed equity curve
    plt.figure(figsize=(12, 7))
    plt.plot(x, mu, linewidth=2, label="Equity (mean)", color='blue')
    plt.fill_between(x, mu - sd, mu + sd, alpha=0.3, label="±1 std", color='blue')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Initial")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.title(f"Equity Mean ± Std (multi-seed, risk-aware)\n{len(seeds)} seeds × {episodes_per_seed} episodes = {combined.shape[0]} total")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join("reports", "equity_multiseed.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✓ Saved: {out_png}")

    print(f"\n{'='*70}")
    print("MULTI-SEED EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
