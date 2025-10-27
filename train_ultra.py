"""
UltraTrader Training Script - Institutional-Grade RL Trading Agent

Trains a Soft Actor-Critic (SAC) agent against a multi-agent market simulator
with 30+ expert ensemble and risk-aware reward shaping.

Key Upgrades:
1. Multi-Agent Market Simulation
   - Models retail (FOMO/panic), small funds, large institutions, market makers
   - Generates realistic stateful flow dynamics (AR(1) processes)
   - Liquidity stress events and market microstructure effects

2. 30+ Expert Ensemble
   - Technical/microstructure, macro/cross-asset, behavioral/flow experts
   - Risk/portfolio construction and arbitrage/structural signals
   - Consolidated into 60-dim feature vector per timestep

3. Institutional Reward Function
   - PROFIT_W (80×): Strongly reward profitability
   - HOLD_W (40×): Bonus for conviction trades (profit WITH size)
   - DD_W (50×): Penalize drawdown to maintain risk discipline
   - LEV_W (2×): Penalize over-leverage
   - Result: "Do nothing" is NOT optimal - must take GOOD risks

4. SAC Hyperparameters Tuned for Prop-Desk Style
   - Longer horizon (gamma=0.995) for multi-step planning
   - Larger replay buffer (200k) for better sample efficiency
   - Controlled entropy (ent_coef='auto_0.2') to avoid collapse
   - Deeper networks (256×256) to model complex market dynamics

Environment Variables:
    ULTRA_STEPS: Total training timesteps (default: 20000)
    ULTRA_SEED: Random seed for reproducibility (default: 123)

Usage:
    python train_ultra.py
"""
import os
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from ultra_trading_env import UltraTradingEnv


def make_env():
    """Create a monitored trading environment for training/evaluation."""
    return Monitor(UltraTradingEnv())


def main():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"CUDA available: {use_cuda}")
    print(f"Using {device} device")

    TIMESTEPS = int(os.getenv("ULTRA_STEPS", "20000"))  # Increased default for richer environment
    SEED      = int(os.getenv("ULTRA_SEED",  "123"))

    env = make_env()
    eval_env = make_env()

    # Institutional-grade SAC hyperparameters
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,        # Larger replay buffer for complex environment
        batch_size=256,
        tau=0.005,
        gamma=0.995,                # Longer horizon for multi-step planning
        train_freq=1,               # Train every step
        gradient_steps=1,
        ent_coef="auto_0.2",        # Controlled entropy (target=0.2) to avoid policy collapse
        policy_kwargs=dict(net_arch=[256, 256]),  # Deeper networks for complex signals
        verbose=1,
        seed=SEED,
        device=device,
    )

    os.makedirs("checkpoints/best", exist_ok=True)
    os.makedirs("checkpoints/periodic", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/best",
        log_path="logs",
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=5000,
        save_path="checkpoints/periodic",
        name_prefix="ultra_sac",
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        callback=[eval_cb, ckpt_cb],
    )

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ultra_sac_last.zip")
    print("Saved last model to checkpoints/ultra_sac_last.zip")

    # 小小推理演示
    test_env = UltraTradingEnv()
    obs, _ = test_env.reset()
    action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)

    print("Sample obs[0:5]:", np.asarray(obs, dtype=np.float32)[:5])
    print("Model action:", action, " -> [direction, exposure]")

if __name__ == "__main__":
    main()
