"""
Unit tests for UltraTrader institutional-grade components.

Tests verify correct shapes, bounds, and basic functionality of:
- UltraExpertEnsemble (30+ expert) model outputs
- SACPolicyHead action generation
- UltraTradingEnv environment spaces and rollouts
- Multi-agent market simulator integration
"""
import numpy as np
from ultra_advanced_trading_system import UltraExpertEnsemble, SACPolicyHead
from ultra_trading_env import UltraTradingEnv, train_loop_dummy, MultiAgentMarketSimulator


def test_expert_output_shape():
    """Test that expert ensemble returns correct dimensionality (60 features)."""
    ensemble = UltraExpertEnsemble()

    # Create fake environment context
    fake_context = {
        "recent_returns": 0.001,
        "retail_sentiment": 0.5,
        "small_inst_flow": -0.2,
        "big_inst_flow": 0.3,
        "drawdown": 0.05,
        "exposure": 0.3,
        "liquidity": 0.7
    }

    features = ensemble.forward(fake_context)

    # Should return 60-dimensional feature vector
    assert features.shape == (60,), f"Expected (60,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"


def test_policy_action_shape():
    """Test that policy head returns correct action dimensions."""
    policy = SACPolicyHead()

    # Test with different input sizes
    fused = np.zeros((4, 60), dtype=np.float32)
    actions = policy.act(fused)

    assert actions.shape == (4, 2), f"Expected (4, 2), got {actions.shape}"
    assert np.all(actions[:, 0] <= 1.0) and np.all(actions[:, 0] >= -1.0), "Direction out of bounds"
    assert np.all(actions[:, 1] <= 1.0) and np.all(actions[:, 1] >= 0.0), "Exposure out of bounds"


def test_market_simulator():
    """Test that market simulator maintains state and returns correct format."""
    sim = MultiAgentMarketSimulator()

    # Reset and run a few steps
    sim.reset()

    for _ in range(10):
        obs = sim.step()

        # Check all required keys present
        assert "retail_sentiment" in obs
        assert "small_inst_flow" in obs
        assert "big_inst_flow" in obs
        assert "liq_depth" in obs

        # Check bounds
        assert -1.0 <= obs["retail_sentiment"] <= 1.0
        assert -1.0 <= obs["small_inst_flow"] <= 1.0
        assert -1.0 <= obs["big_inst_flow"] <= 1.0
        assert 0.0 <= obs["liq_depth"] <= 1.0

def test_env_spaces():
    """Test that environment maintains correct observation and action space shapes."""
    env = UltraTradingEnv()

    # Observation must remain exactly 200 dimensions
    assert env.observation_space.shape == (200,), f"Obs space should be (200,), got {env.observation_space.shape}"

    # Action must remain exactly 2 dimensions [direction, exposure]
    assert env.action_space.shape == (2,), f"Action space should be (2,), got {env.action_space.shape}"

    # Test observation generation
    obs, _ = env.reset()
    assert obs.shape == (200,), f"Observation should be (200,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Observation should be float32, got {obs.dtype}"


def test_env_rollout_no_crash():
    """Test that environment can complete rollouts without crashing."""
    last_equity, total_reward = train_loop_dummy(steps=10)

    # Should return valid floats
    assert isinstance(last_equity, float), f"Expected float, got {type(last_equity)}"
    assert isinstance(total_reward, float), f"Expected float, got {type(total_reward)}"

    # Equity should be positive
    assert last_equity > 0, f"Equity should be positive, got {last_equity}"

    # Total reward over 10 steps should be reasonable (each step tanh-bounded to [-1, 1])
    assert -15.0 <= total_reward <= 15.0, f"Total reward suspiciously out of range: {total_reward}"


def test_env_reward_components():
    """Test that new institutional reward function produces valid outputs."""
    env = UltraTradingEnv()
    obs, _ = env.reset()

    # Take one step
    action = np.array([0.5, 0.3], dtype=np.float32)  # direction=0.5, exposure=0.3
    obs, reward, done, trunc, info = env.step(action)

    # Check reward is bounded
    assert -1.0 <= reward <= 1.0, f"Reward should be tanh-bounded, got {reward}"

    # Check info contains new keys
    assert "pnl_pct" in info, "Missing pnl_pct in info"
    assert "exposure_frac" in info, "Missing exposure_frac in info"
    assert "reward_raw" in info, "Missing reward_raw in info"
    assert "drawdown" in info, "Missing drawdown in info"

    # Check observation shape maintained
    assert obs.shape == (200,), f"Obs shape should remain (200,), got {obs.shape}"
