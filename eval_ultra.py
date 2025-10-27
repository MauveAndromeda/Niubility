"""
UltraTrader Evaluation Script - Institutional Risk Analysis

Evaluates a trained SAC model with PM-style risk metrics and performance reporting.

Outputs (saved to reports/):
1. equity_curve_best.csv: Best episode equity trajectory
2. equity_curve_best.png: Best episode equity visualization
3. drawdown_curve_best.png: Best episode drawdown visualization
4. performance_summary_single.csv: Comprehensive risk metrics including:
   - Total reward (mean ± std across episodes)
   - Sharpe-like ratio (reward mean / reward std)
   - Maximum drawdown of best episode
   - Episode lengths

This mimics institutional PM review: not just "did we make money?" but
"how much risk did we take to get there?"

Usage:
    python eval_ultra.py
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


def eval_and_save(model_path="checkpoints/ultra_sac_last.zip", episodes=10):
    """
    Evaluate trained model and save performance reports with risk metrics.

    Args:
        model_path: Path to saved SAC model (default: checkpoints/ultra_sac_last.zip)
        episodes: Number of evaluation episodes (default: 10)

    Outputs:
        - reports/equity_curve_best.csv: Best episode equity data
        - reports/equity_curve_best.png: Best episode equity visualization
        - reports/drawdown_curve_best.png: Best episode drawdown visualization
        - reports/performance_summary_single.csv: Risk metrics summary
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"{'='*60}\n")

    model = SAC.load(model_path)

    best_curve = None
    best_final = -1e9
    all_rewards = []
    all_lengths = []

    for ep in range(episodes):
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

        all_rewards.append(total_reward)
        all_lengths.append(len(curve))
        final_equity = curve[-1]

        print(f"Episode {ep+1}/{episodes}: Reward={total_reward:.3f}, FinalEquity={final_equity:.4f}, Length={len(curve)}")

        if final_equity > best_final:
            best_final = final_equity
            best_curve = curve

    # Compute aggregate metrics
    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    mean_length = float(np.mean(all_lengths))
    std_length = float(np.std(all_lengths))
    sharpe_like = compute_sharpe_like(all_rewards)
    best_max_dd = compute_max_drawdown(best_curve)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Sharpe-like: {sharpe_like:.3f}")
    print(f"Episode Length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"Best Episode Final Equity: {best_final:.4f}")
    print(f"Best Episode Max Drawdown: {best_max_dd:.4f}")
    print(f"{'='*60}\n")

    os.makedirs("reports", exist_ok=True)

    # Save performance summary CSV
    summary_path = os.path.join("reports", "performance_summary_single.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["episodes", episodes])
        writer.writerow(["mean_reward", mean_reward])
        writer.writerow(["std_reward", std_reward])
        writer.writerow(["sharpe_like", sharpe_like])
        writer.writerow(["mean_episode_length", mean_length])
        writer.writerow(["std_episode_length", std_length])
        writer.writerow(["best_episode_final_equity", best_final])
        writer.writerow(["best_episode_max_drawdown", best_max_dd])
    print(f"✓ Saved: {summary_path}")

    # Save best episode equity curve CSV
    csv_path = os.path.join("reports", "equity_curve_best.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "equity"])
        for i, equity_val in enumerate(best_curve):
            writer.writerow([i, equity_val])
    print(f"✓ Saved: {csv_path}")

    # Plot equity curve
    x = np.arange(len(best_curve))
    plt.figure(figsize=(10, 6))
    plt.plot(x, best_curve, linewidth=2, label="Equity")
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Initial")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.title(f"Best Episode Equity Curve (Final={best_final:.4f}, MaxDD={best_max_dd:.4f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png_equity = os.path.join("reports", "equity_curve_best.png")
    plt.savefig(png_equity, dpi=150)
    plt.close()
    print(f"✓ Saved: {png_equity}")

    # Plot drawdown curve
    equity_arr = np.asarray(best_curve, dtype=np.float32)
    peak_arr = np.maximum.accumulate(equity_arr)
    dd_arr = (peak_arr - equity_arr) / np.maximum(peak_arr, 1e-9)

    plt.figure(figsize=(10, 6))
    plt.plot(x, dd_arr, linewidth=2, color='red', label="Drawdown")
    plt.fill_between(x, 0, dd_arr, alpha=0.3, color='red')
    plt.xlabel("Step")
    plt.ylabel("Drawdown (fraction)")
    plt.title(f"Best Episode Drawdown Curve (Max={best_max_dd:.4f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png_dd = os.path.join("reports", "drawdown_curve_best.png")
    plt.savefig(png_dd, dpi=150)
    plt.close()
    print(f"✓ Saved: {png_dd}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    eval_and_save()
