# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UltraTrader is a reinforcement learning-based trading system using Soft Actor-Critic (SAC) to train an agent on a custom trading environment. The system emphasizes profit maximization while maintaining risk controls through drawdown penalties and position limits.

## Core Architecture

### Trading Environment (`ultra_trading_env.py`)
- **UltraTradingEnv**: Custom Gymnasium environment simulating a trading environment
  - 200-dimensional observation space (5 core state variables + 195 market features)
  - 2-dimensional continuous action space: [direction ∈ [-1, 1], exposure ∈ [0, 0.6]]
  - Maximum episode length: 500 steps
  - Hard position limit: 60% (EXPOSURE_MAX = 0.6)

**Key State Variables**:
- `position_dir`: Direction of current position (-1 to 1)
- `position_size`: Size of current position (0 to 0.6)
- `equity`: Current account equity
- `equity_peak`: Historical peak equity
- Drawdown: Calculated as (peak - current) / peak

**Reward Structure** (ultra_trading_env.py:102-125):
The reward combines multiple components with specific weights:
- Profit driver: 60.0 × PnL (strongly encourages profitability)
- Anti-retail signal: 0.10 × contrarian positioning vs retail sentiment
- Smart money signal: 0.10 × alignment with institutional flow
- Drawdown penalty: 0.08 × drawdown
- Leverage penalty: 0.003 × exposure
- Final reward is tanh-bounded to prevent critic explosion

**Market Simulation**:
- Retail sentiment and institutional flow are generated via Gaussian noise
- Price returns incorporate both signals: `base_ret + 1.5e-3 * inst_flow - 1.0e-3 * retail_sent`
- Trading costs include commission (5e-4 per unit) and slippage (5e-4 per unit)

### Model Components (`ultra_advanced_trading_system.py`)
- **UltraExpertEnsemble**: Placeholder ensemble class (currently returns dummy outputs)
- **SACPolicyHead**: Placeholder policy head (currently returns dummy actions)

Note: These classes appear to be stubs for future development. The actual training uses SAC from stable-baselines3 with MlpPolicy.

### Training Pipeline (`train_ultra.py`)
Uses Stable-Baselines3 SAC with:
- Learning rate: 3e-4
- Buffer size: 100,000
- Batch size: 256
- Tau: 0.005, Gamma: 0.99
- Automatic entropy coefficient tuning
- CUDA support when available

**Callbacks**:
- EvalCallback: Evaluates every 2,000 steps, saves best model to `checkpoints/best/`
- CheckpointCallback: Periodic saves every 5,000 steps to `checkpoints/periodic/`

**Environment Variables**:
- `ULTRA_STEPS`: Total training timesteps (default: 10,000)
- `ULTRA_SEED`: Random seed (default: 123)

## Development Commands

### Install Dependencies
```bash
pip install -r requirements_ultra.txt
```

### Run Tests
```bash
pytest test_ultra.py
```

Run tests with coverage:
```bash
pytest --cov=. test_ultra.py
```

Run specific test:
```bash
pytest test_ultra.py::test_env_rollout_no_crash
```

### Training
```bash
python train_ultra.py
```

Set custom training parameters:
```bash
set ULTRA_STEPS=50000
set ULTRA_SEED=456
python train_ultra.py
```

### Evaluation

Evaluate trained model (single seed, 10 episodes):
```bash
python eval_ultra.py
```

Multi-seed evaluation with statistics:
```bash
python seedeval.py
```

Outputs are saved to `reports/`:
- `equity_curve_best.csv`: Best episode equity curve data
- `equity_curve_best.png`: Best episode equity visualization
- `drawdown_curve_best.png`: Best episode drawdown visualization
- `seed_stats.csv`: Per-seed performance statistics
- `equity_multiseed.png`: Mean equity curve across seeds with ±1 std bands

## Project Structure

```
├── ultra_trading_env.py          # Core trading environment
├── ultra_advanced_trading_system.py  # Model components (placeholder stubs)
├── train_ultra.py                # SAC training script
├── eval_ultra.py                 # Single evaluation script
├── seedeval.py                   # Multi-seed evaluation with statistics
├── test_ultra.py                 # Pytest unit tests
├── requirements_ultra.txt        # Python dependencies
├── checkpoints/                  # Saved models
│   ├── best/                    # Best model from evaluation
│   └── periodic/                # Periodic checkpoints
├── logs/                        # Training logs
└── reports/                     # Evaluation outputs (CSV, PNG)
```

## Key Implementation Details

### Reward Design Philosophy
The reward function (ultra_trading_env.py:116-122) is designed to:
1. Strongly incentivize profitability (60× multiplier on PnL)
2. Encourage contrarian behavior against retail sentiment
3. Align with institutional money flow
4. Penalize excessive drawdown and leverage
5. Use tanh to bound rewards and stabilize training

### Position Management
- Position is determined by: `direction × exposure`
- Turnover is calculated as: `|target_position - current_position|`
- Both commission and slippage scale linearly with turnover
- Position limits are enforced via action space bounds

### Testing Philosophy
Tests in `test_ultra.py` verify:
- Model output shapes are correct
- Action bounds are respected
- Environment spaces match expected dimensions
- Environment can complete rollouts without crashes
