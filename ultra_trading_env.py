import math
import numpy as np
import gymnasium as gym


class MultiAgentMarketSimulator:
    """
    Multi-agent market microstructure simulator modeling heterogeneous market participants.

    This simulator models 4 cohorts of market participants:

    1. Retail Cohort (250 agents): FOMO chasers, panic sellers, gamblers, passive traders
       - React to recent price direction and emotional sentiment
       - Output: retail_sentiment ∈ [-1, 1] (bullish positive, bearish negative)

    2. Small Funds / Prop Desks / Family Offices (50 agents)
       - Opportunistic, tactical, chase short-term edge
       - De-risk when stressed
       - Output: small_inst_flow ∈ [-1, 1] (risk-on vs risk-off)

    3. Large Institutions (30 agents): Pensions, sovereigns, bank balance sheets
       - Slow-moving but high impact when rotating risk
       - Output: big_inst_flow ∈ [-1, 1], smoother/persistent

    4. Market Makers / Liquidity Providers (20 agents)
       - Provide liquidity, tighten spreads, widen when vol spikes
       - Output: liq_depth ∈ [0, 1] (1=liquid, 0=illiquid)

    Implementation uses AR(1)-like mean-reverting processes for realistic dynamics.
    """

    def __init__(self, seed=None):
        """Initialize simulator with stateful dynamics for each cohort."""
        self._rng = np.random.default_rng(seed)

        # AR(1) parameters for mean reversion
        self.retail_mean_revert = 0.85  # Strong mean reversion (emotional)
        self.small_inst_mean_revert = 0.90  # Moderate
        self.big_inst_mean_revert = 0.95  # Slow-moving (persistent)
        self.liq_mean_revert = 0.92  # Liquidity is somewhat sticky

        # Volatility/noise parameters
        self.retail_vol = 0.25
        self.small_inst_vol = 0.15
        self.big_inst_vol = 0.08
        self.liq_vol = 0.10

        # Initialize states
        self.reset()

    def reset(self):
        """Reset simulator to initial neutral state."""
        self.retail_sentiment = 0.0
        self.small_inst_flow = 0.0
        self.big_inst_flow = 0.0
        self.liq_depth = 0.7  # Start with reasonable liquidity

    def step(self):
        """
        Evolve market participant states one timestep using AR(1) dynamics.

        Returns:
            dict: {
                "retail_sentiment": float ∈ [-1, 1],
                "small_inst_flow": float ∈ [-1, 1],
                "big_inst_flow": float ∈ [-1, 1],
                "liq_depth": float ∈ [0, 1]
            }
        """
        # AR(1) dynamics: x_t = rho * x_{t-1} + noise
        # Retail: emotional, mean-reverting to neutral
        self.retail_sentiment = (
            self.retail_mean_revert * self.retail_sentiment
            + self._rng.normal(0.0, self.retail_vol)
        )
        self.retail_sentiment = float(np.clip(self.retail_sentiment, -1.0, 1.0))

        # Small institutions: tactical, moderate persistence
        self.small_inst_flow = (
            self.small_inst_mean_revert * self.small_inst_flow
            + self._rng.normal(0.0, self.small_inst_vol)
        )
        self.small_inst_flow = float(np.clip(self.small_inst_flow, -1.0, 1.0))

        # Large institutions: slow-moving, high persistence
        self.big_inst_flow = (
            self.big_inst_mean_revert * self.big_inst_flow
            + self._rng.normal(0.0, self.big_inst_vol)
        )
        self.big_inst_flow = float(np.clip(self.big_inst_flow, -1.0, 1.0))

        # Liquidity depth: mean-reverts to 0.7, spikes down on vol
        # Add occasional liquidity shocks (stress events)
        stress_shock = 0.0
        if self._rng.random() < 0.02:  # 2% chance of liquidity stress per step
            stress_shock = -self._rng.uniform(0.2, 0.5)

        self.liq_depth = (
            self.liq_mean_revert * self.liq_depth
            + (1.0 - self.liq_mean_revert) * 0.7  # Mean revert to 0.7
            + self._rng.normal(0.0, self.liq_vol)
            + stress_shock
        )
        self.liq_depth = float(np.clip(self.liq_depth, 0.0, 1.0))

        return {
            "retail_sentiment": self.retail_sentiment,
            "small_inst_flow": self.small_inst_flow,
            "big_inst_flow": self.big_inst_flow,
            "liq_depth": self.liq_depth
        }


class UltraTradingEnv(gym.Env):
    """
    UltraTradingEnv - Institutional-Grade RL Trading Environment

    Multi-agent market simulation with 30+ expert ensemble and risk-aware reward shaping.

    Key Features:
    - Observation: 200-d float32 vector (core state + market sim + expert signals + noise)
    - Action: [direction ∈ [-1, 1], exposure ∈ [0, 0.6]]
    - Reward: Institutional PM-style objective encouraging profitable risk-taking
    - Episode: Max 500 steps or equity drops below 0.2

    Reward Philosophy:
    The new reward function incentivizes GOOD risk-taking, not zero risk:
    - PROFIT_W: Large multiplier on positive PnL (80×) to drive profitability
    - HOLD_W: Bonus for being right WITH SIZE on (40×)
    - DD_W: Penalty for drawdown (50×) to maintain risk discipline
    - LEV_W: Penalty for high exposure (2×) to avoid over-leveraging
    - All bounded by tanh() for training stability

    This means "do nothing" (flat position) is NOT optimal - the agent must
    take calculated risks to earn HOLD_W bonuses when profitable.
    """

    metadata = {"render_modes": []}

    # Institutional reward weights (risk-aware PM objectives)
    PROFIT_W    = 80.0    # Reward for positive returns (strongly encourage profitability)
    HOLD_W      = 40.0    # Bonus for being right with size on (incentivize conviction)
    DD_W        = 50.0    # Penalty for drawdown (maintain risk discipline)
    LEV_W       = 2.0     # Penalty for carrying high exposure (avoid over-leverage)
    CLIP_SCALE  = 3.0     # Smooth tanh scaling to bound reward magnitude

    # Market/cost parameters
    COMMISSION_PER_UNIT = 5e-4
    SLIPPAGE_COEF       = 5e-4
    BASE_RET_SIGMA      = 2.5e-3

    # Position hard limit
    EXPOSURE_MAX        = 0.6

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        # obs: 200 维
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(200,), dtype=np.float32
        )

        # action: [方向(-1..1), 仓位(0..0.6)]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, self.EXPOSURE_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # episode 配置
        self.max_steps     = int(self.config.get("max_steps", 500))
        self.min_equity    = float(self.config.get("min_equity", 0.2))
        self.start_equity  = float(self.config.get("start_equity", 1.0))

        # Multi-agent market simulator
        self.market_sim = MultiAgentMarketSimulator()

        # Expert ensemble (30+ specialized trading signals)
        from ultra_advanced_trading_system import UltraExpertEnsemble
        self.experts = UltraExpertEnsemble()

        # 伪随机
        self._rng = np.random.default_rng()
        self._reset_state()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        self.market_sim.reset()  # Reset market simulator state
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # 动作裁剪
        dir_raw, exp_raw = np.asarray(action, dtype=np.float32)
        direction = float(np.clip(dir_raw, -1.0, 1.0))
        exposure  = float(np.clip(exp_raw, 0.0, self.EXPOSURE_MAX))

        # Get multi-agent market simulator state
        sim_obs = self.market_sim.step()
        retail_sent = sim_obs["retail_sentiment"]
        small_flow = sim_obs["small_inst_flow"]
        big_flow = sim_obs["big_inst_flow"]
        liq_depth = sim_obs["liq_depth"]

        # Base random return component
        base_ret = float(self._rng.normal(0.0, self.BASE_RET_SIGMA))

        # Market price dynamics influenced by institutional flows and retail sentiment
        # Big institutions have stronger impact (2.0e-3), small funds moderate (1.0e-3)
        # Retail sentiment creates contrarian opportunity (-0.8e-3)
        price_ret = (
            base_ret
            + 2.0e-3 * big_flow         # Large inst flow drives market
            + 1.0e-3 * small_flow        # Small inst adds tactical moves
            - 0.8e-3 * retail_sent       # Retail sentiment is often wrong (contrarian signal)
        )

        # 目标仓位 和 换手
        target_pos  = direction * exposure
        turnover    = abs(target_pos - self.position)

        # Transaction costs influenced by liquidity
        # Lower liquidity => higher slippage (liquidity stress penalty)
        liq_multiplier = 1.0 / max(liq_depth, 0.2)  # Avoid division by zero, min 0.2
        commission = self.COMMISSION_PER_UNIT * turnover
        slippage = self.SLIPPAGE_COEF * turnover * liq_multiplier

        # PnL: 仓位 * 市场收益 - 成本
        pnl_frac    = target_pos * price_ret - commission - slippage

        # 更新权益（乘法式资金曲线）
        self.equity = float(max(1e-3, self.equity * (1.0 + pnl_frac)))
        self.equity_peak = max(self.equity_peak, self.equity)

        drawdown    = (self.equity_peak - self.equity) / max(self.equity_peak, 1e-9)

        # Save position state
        self.position      = target_pos
        self.position_dir  = direction
        self.position_size = exposure

        # ============================================================
        # Institutional Risk-Aware Reward Shaping
        # ============================================================
        # Philosophy: Incentivize PROFITABLE risk-taking, not zero risk
        # "Do nothing" (flat) is sub-optimal because you need HOLD_W bonus

        # PnL metrics
        pnl_raw = self.equity - self.prev_equity
        pnl_pct = pnl_raw / max(self.prev_equity, 1e-6)

        # Risk metrics
        drawdown = (self.equity_peak - self.equity) / max(self.equity_peak, 1e-6)
        exposure_frac = self.position_size / self.EXPOSURE_MAX  # Normalize 0..1

        # Reward components:
        # 1. PROFIT: Reward all positive PnL (80× multiplier)
        profit_component = self.PROFIT_W * pnl_pct

        # 2. HOLD: Bonus for being right WITH SIZE on (40× multiplier)
        #    Only triggers when pnl_pct > 0 AND you have exposure
        #    Incentivizes conviction trades, not timid flat positions
        if pnl_pct > 0:
            hold_component = self.HOLD_W * pnl_pct * exposure_frac
        else:
            hold_component = 0.0

        # 3. DRAWDOWN: Penalty for being in drawdown (50× multiplier)
        dd_penalty = self.DD_W * drawdown

        # 4. LEVERAGE: Small penalty for high exposure (2× multiplier)
        #    Keeps position sizes reasonable
        leverage_penalty = self.LEV_W * exposure_frac

        # Combine all components
        reward_unclipped = (
            profit_component
            + hold_component
            - dd_penalty
            - leverage_penalty
        )

        # Bound reward with tanh for critic stability
        reward = float(np.tanh(reward_unclipped / self.CLIP_SCALE))

        # 终止条件
        terminated = bool(
            self.step_count >= self.max_steps or
            self.equity <= self.min_equity
        )
        truncated  = False

        # Get expert ensemble features
        env_context = {
            "recent_returns": price_ret,
            "retail_sentiment": retail_sent,
            "small_inst_flow": small_flow,
            "big_inst_flow": big_flow,
            "drawdown": drawdown,
            "exposure": self.position_size,
            "liquidity": liq_depth
        }
        expert_features = self.experts.forward(env_context)

        # Store for observation construction
        self.last_sim_obs = sim_obs
        self.last_price_ret = price_ret
        self.last_expert_features = expert_features

        info = {
            "step": self.step_count,
            "equity": self.equity,
            "drawdown": drawdown,
            "pnl_pct": pnl_pct,
            "pnl_frac": pnl_frac,
            "exposure_frac": exposure_frac,
            "reward_raw": reward_unclipped,
            "turnover": turnover,
            "commission": commission,
            "slippage": slippage,
            "retail_sentiment": retail_sent,
            "small_inst_flow": small_flow,
            "big_inst_flow": big_flow,
            "liq_depth": liq_depth,
            "price_ret": price_ret,
        }

        # Update prev_equity for next step
        self.prev_equity = self.equity

        return self._get_obs(), reward, terminated, truncated, info

    def _reset_state(self):
        self.step_count    = 0
        self.equity        = float(self.start_equity)
        self.equity_peak   = float(self.start_equity)
        self.prev_equity   = float(self.start_equity)  # Track for PnL calculation
        self.position      = 0.0
        self.position_dir  = 0.0
        self.position_size = 0.0
        self.last_sim_obs  = {
            "retail_sentiment": 0.0,
            "small_inst_flow": 0.0,
            "big_inst_flow": 0.0,
            "liq_depth": 0.7
        }
        self.last_price_ret = 0.0
        self.last_expert_features = np.zeros(60, dtype=np.float32)

    def _get_obs(self):
        """
        Build 200-dimensional observation vector combining:
        - Core position/equity state (5 dims)
        - Market simulator outputs (4 dims)
        - Recent price dynamics (1 dim)
        - Reserved for expert features (60 dims, to be populated in STEP 3)
        - Padding noise (130 dims)

        Total: 5 + 4 + 1 + 60 + 130 = 200
        """
        # Core trading state (5 dimensions)
        drawdown = (self.equity_peak - self.equity) / max(self.equity_peak, 1e-9)
        core = np.array([
            self.position_dir,
            self.position_size,
            self.equity,
            self.equity_peak,
            drawdown,
        ], dtype=np.float32)

        # Market simulator state (4 dimensions)
        sim_features = np.array([
            self.last_sim_obs["retail_sentiment"],
            self.last_sim_obs["small_inst_flow"],
            self.last_sim_obs["big_inst_flow"],
            self.last_sim_obs["liq_depth"],
        ], dtype=np.float32)

        # Recent price dynamics (1 dimension)
        price_features = np.array([self.last_price_ret], dtype=np.float32)

        # Expert ensemble features (60 dimensions) - from 30+ specialized experts
        expert_features = self.last_expert_features

        # Padding noise to reach exactly 200 dimensions (130 dimensions)
        noise = self._rng.uniform(-0.5, 0.5, size=(130,)).astype(np.float32)

        # Concatenate all components
        obs = np.concatenate([core, sim_features, price_features, expert_features, noise])
        assert obs.shape == (200,), f"Observation shape mismatch: {obs.shape}"
        return obs.astype(np.float32)


def train_loop_dummy(steps=10):
    """
    Smoke test: randomly play for 'steps' steps to ensure environment doesn't crash.

    Args:
        steps: Number of steps to run (default: 10)

    Returns:
        tuple: (last_equity: float, total_reward: float)
            - last_equity: Final equity value
            - total_reward: Cumulative reward over all steps
    """
    env = UltraTradingEnv()
    obs, _ = env.reset()
    total_r = 0.0
    last_equity = env.equity
    for _ in range(steps):
        a = env.action_space.sample()
        obs, r, done, trunc, info = env.step(a)
        total_r += r
        last_equity = info.get("equity", last_equity)
        if done or trunc:
            break
    return last_equity, total_r
