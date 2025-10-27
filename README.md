# UltraTrader: Institutional-Grade RL Trading System

**Multi-agent market simulation with 30+ expert ensemble and risk-aware reward shaping**

[![Tests](https://img.shields.io/badge/tests-6%2F6%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![Framework](https://img.shields.io/badge/RL-Stable--Baselines3-orange)]()

## 🎯 Overview

UltraTrader is an institutional-grade reinforcement learning trading system that models realistic market microstructure through multi-agent simulation. Unlike toy environments with simple random noise, UltraTrader simulates **4 distinct market participant cohorts** (retail, small funds, large institutions, market makers) and aggregates signals from **30+ specialized trading experts**.

### Key Features

- **Multi-Agent Market Simulator**: Stateful AR(1) dynamics modeling retail sentiment, institutional flows, and liquidity stress
- **30+ Expert Ensemble**: Technical, macro, behavioral, risk, and arbitrage signals across 5 categories
- **Institutional Reward Shaping**: PM-style objective encouraging profitable conviction trades (not "do nothing")
- **Risk-Aware Evaluation**: Max drawdown, Sharpe-like ratios, multi-seed robustness analysis
- **Production Ready**: Fully tested, documented, backwards compatible with Stable-Baselines3 SAC

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Agent (SAC)                          │
│              Obs: 200-D  →  Action: [dir, exp]              │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
┌──────────────────┐          ┌──────────────────────┐
│ Market Simulator │          │  Expert Ensemble     │
│  4 Cohorts:      │          │  30+ Experts:        │
│  - Retail (250)  │          │  - Technical (5)     │
│  - Small Inst(50)│          │  - Macro (5)         │
│  - Large Inst(30)│          │  - Behavioral (4)    │
│  - Market Makers │          │  - Risk (4)          │
│  Outputs:        │          │  - Arbitrage (12)    │
│  - Sentiment     │          │  Output: 60 features │
│  - Flows         │          └──────────────────────┘
│  - Liquidity     │
└──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              UltraTradingEnv (Gymnasium)                    │
│  Reward: PROFIT_W×pnl + HOLD_W×conviction - DD_W×drawdown  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/MauveAndromeda/Niubility.git
cd Niubility

# Create virtual environment (recommended)
python -m venv .ultra_env
source .ultra_env/bin/activate  # On Windows: .ultra_env\Scripts\activate

# Install dependencies
pip install -r requirements_ultra.txt
```

## 🚀 Quick Start

### Train Agent

```bash
# Default: 20,000 steps
python train_ultra.py

# Custom configuration
set ULTRA_STEPS=50000
set ULTRA_SEED=456
python train_ultra.py
```

### Evaluate Performance

```bash
# Single evaluation with risk metrics
python eval_ultra.py

# Multi-seed robustness analysis
python seedeval.py
```

### Run Tests

```bash
pytest test_ultra.py -v
```

## 📊 Outputs

All evaluation outputs are saved to `reports/`:

- **`performance_summary_single.csv`**: Mean reward, Sharpe-like, max drawdown
- **`equity_curve_best.csv`**: Best episode equity trajectory
- **`equity_curve_best.png`**: Equity visualization
- **`drawdown_curve_best.png`**: Drawdown visualization
- **`seed_stats.csv`**: Per-seed robustness statistics
- **`equity_multiseed.png`**: Mean equity ± 1 std across seeds

## 🧠 Core Components

### 1. Multi-Agent Market Simulator

Models 4 market participant cohorts using AR(1) mean-reverting processes:

| Cohort | Count | Persistence | Volatility | Output |
|--------|-------|-------------|------------|--------|
| Retail | 250 | ρ=0.85 | σ=0.25 | sentiment ∈ [-1,1] |
| Small Institutions | 50 | ρ=0.90 | σ=0.15 | flow ∈ [-1,1] |
| Large Institutions | 30 | ρ=0.95 | σ=0.08 | flow ∈ [-1,1] |
| Market Makers | 20 | ρ=0.92 | σ=0.10 | liquidity ∈ [0,1] |

**Price Dynamics**:
```
return = base_noise + 2.0e-3×big_inst + 1.0e-3×small_inst - 0.8e-3×retail
slippage = commission × turnover × (1 / liquidity)
```

### 2. Expert Ensemble (30+ Experts)

| Category | Experts | Examples |
|----------|---------|----------|
| Technical/Microstructure | 5 | Trend Tracker, Mean Reversion, Vol Forecaster |
| Macro/Cross-Asset | 5 | Regime Detector, Rates Stress, Inflation Pressure |
| Behavioral/Flow | 4 | FOMO Heat, Panic Monitor, Smart Money Divergence |
| Risk/Portfolio | 4 | Tail Risk Alert, Leverage Recommender, DD Sentinel |
| Arbitrage/Structural | 12 | Stat Arb, Vol Arb, Regime Shift, Carry/Convexity |

Each expert analyzes market conditions from a specialized perspective, outputting 1-3 features. Total: **60-dimensional feature vector** per timestep.

### 3. Institutional Reward Function

```python
profit_component = 80.0 × pnl_pct              # Strong profit incentive
hold_bonus = 40.0 × pnl_pct × exposure_frac    # Conviction bonus (only when profitable)
dd_penalty = 50.0 × drawdown                   # Risk discipline
leverage_penalty = 2.0 × exposure_frac         # Moderate leverage

reward = tanh((profit + hold_bonus - dd_penalty - leverage_penalty) / 3.0)
```

**Key Insight**: "Do nothing" (flat position) is **NOT optimal** because you only earn the `hold_bonus` when profitable WITH size. Agent must take calculated risks.

## 🎓 Technical Details

### Observation Space (200-D)

| Dimensions | Component | Description |
|------------|-----------|-------------|
| 0-4 | Core State | position_dir, position_size, equity, equity_peak, drawdown |
| 5-8 | Market Sim | retail_sentiment, small_flow, big_flow, liquidity |
| 9 | Price | recent_return |
| 10-69 | Experts | 30 experts × 2 features = 60 dims |
| 70-199 | Padding | Random noise to reach 200-D |

### Action Space (2-D)

- `action[0]`: direction ∈ [-1, 1] (short to long)
- `action[1]`: exposure ∈ [0, 0.6] (max 60% position)

### SAC Hyperparameters

```python
learning_rate    = 3e-4
buffer_size      = 200,000      # Large replay for complex environment
gamma            = 0.995        # Longer horizon for multi-step planning
ent_coef         = "auto_0.2"   # Controlled entropy (target=0.2)
policy_net_arch  = [256, 256]   # Deep networks for 30+ expert signals
```

## 📈 Performance Metrics

Evaluation includes institutional PM-style risk metrics:

- **Sharpe-like Ratio**: reward_mean / reward_std (risk-adjusted performance)
- **Maximum Drawdown**: Peak-to-trough equity decline
- **Exposure Discipline**: Position sizing consistency
- **Multi-Seed Robustness**: Performance stability across random seeds

## 🧪 Testing

```bash
# Run all tests
pytest test_ultra.py -v

# Run with coverage
pytest --cov=. test_ultra.py

# Tests include:
# ✓ Expert ensemble output shape (60-D float32)
# ✓ Policy action bounds ([dir, exp])
# ✓ Market simulator state management
# ✓ Environment spaces (obs=200-D, action=2-D)
# ✓ Rollout completion without crashes
# ✓ Reward function validity
```

## 📁 Project Structure

```
UltraTrader/
├── ultra_trading_env.py              # Core environment + market simulator
├── ultra_advanced_trading_system.py  # Expert ensemble (30+ experts)
├── train_ultra.py                    # SAC training script
├── eval_ultra.py                     # Single evaluation with risk metrics
├── seedeval.py                       # Multi-seed robustness analysis
├── test_ultra.py                     # Pytest unit tests
├── requirements_ultra.txt            # Dependencies
├── CLAUDE.md                         # Development guide for Claude Code
├── NEXT_TASK.md                      # Implementation specification
├── .gitignore                        # Git ignore rules
├── checkpoints/                      # Saved models
│   ├── best/                        # Best model from evaluation
│   └── periodic/                    # Periodic checkpoints
├── logs/                            # Training logs
└── reports/                         # Evaluation outputs (CSV, PNG)
```

## 🔬 Research Background

This system transforms a toy SAC sandbox into an institutional-grade research platform by:

1. **Replacing i.i.d. noise** with stateful multi-agent market dynamics
2. **Expanding from 2 dummy experts** to 30+ specialized trading signals
3. **Upgrading reward** from simple PnL to risk-aware PM-style objectives
4. **Tuning SAC** for stability with complex 200-D observations
5. **Adding PM-style evaluation** (not just "did it train?" but "should we trade this?")

### Why This Matters

Traditional RL trading environments use unrealistic price dynamics (Gaussian noise, random walks). Real markets are driven by:
- Heterogeneous participants with different objectives
- Persistent flows (not i.i.d.)
- Liquidity stress events
- Multi-factor signals (not single indicators)

UltraTrader models these explicitly, creating a more realistic training ground for RL agents.

## 🤝 Contributing

This is a research sandbox. Contributions welcome:

- New expert implementations
- Alternative market simulators
- Reward function refinements
- Evaluation metrics
- Documentation improvements

## 📄 License

This project is provided as-is for research and educational purposes.

## 🙏 Acknowledgments

- Built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Environment follows [Gymnasium](https://gymnasium.farama.org/) API
- Inspired by institutional trading desk architecture

---

**Status**: ✅ All tests passing (6/6) | Ready for training and evaluation

**Version**: v2.0 Institutional Grade | Generated: 2025-10-27
