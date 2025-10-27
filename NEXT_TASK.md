# NEXT_TASK.md
Mission:
You're acting as the lead quantitative research engineer of a next-generation multi-agent trading platform.
You are allowed to directly modify files in this repo.
You must execute all steps in order, without waiting for approval.

Your environment:
- OS: Windows (PowerShell host)
- Path: C:\Users\Shadow\Desktop\UltraTrader
- Python: 3.13 in .ultra_env
- Hardware: CPU only (assume no CUDA, but still keep CUDA codepaths optional)
- Repo files we currently have:
  - ultra_trading_env.py
  - ultra_advanced_trading_system.py
  - train_ultra.py
  - eval_ultra.py
  - seedeval.py
  - test_ultra.py
  - requirements_ultra.txt
  - CLAUDE.md
  - checkpoints/, reports/ (runtime artifacts)

High-level goals of this session:
1. Move from "toy SAC agent in toy env" → "institutional-grade research sandbox".
2. Keep backwards compatibility (observation still 200-D; action still [direction, exposure]).
3. Add a multi-agent market simulator with heterogeneous actors (retail herding, hedge funds, market makers).
4. Expand the “experts” ensemble from ~15 roles to 30+ specialized experts (macro, vol, liquidity, etc.).
   Even if some experts are still partially stubbed, they must now have clean interfaces and structured outputs.
5. Fuse expert signals + market simulator outputs into the environment state.
6. Upgrade reward shaping to favor profitable, size-adjusted, risk-aware behavior.
7. Add professional risk metrics (drawdown, Sharpe-like, exposure discipline) to eval.
8. Clean up docs and prep a single commit.

You will produce code that can run (even with placeholder stochastic behaviors) so training/eval/tests still work.

--------------------------------------------------------------------------------
STEP 0. Global rules (follow these for ALL changes)

- Do NOT change:
  - observation shape seen by the agent: must remain (200,)
  - action shape: still 2 floats [direction ∈ [-1,1], exposure ∈ [0,1] or [0,EXPOSURE_MAX]]
  - agent interface into Stable Baselines SAC (SAC must still work)
- You MAY internally reorganize how that 200-D observation vector is constructed.
  For example: [core position stats | market microstructure | retail sentiment panel | institutional flow panel | expert ensemble signals | noise padding] as long as final shape is exactly 200.
- All new classes must be pure-Python / NumPy / standard lib only.
  No extra heavy dependencies beyond what’s already in requirements_ultra.txt.
- Evaluation scripts must continue writing PNG and CSV into reports/.
- Keep tests passing (you can update tests if needed, but the final repo must end with pytest green).
- You must add docstrings and inline comments like you’re submitting code to a serious quant fund.

At the end of this workflow you will:
- update environment
- update model stubs
- update training/eval scripts
- update tests
- run pytest
- create/prepare a git commit

--------------------------------------------------------------------------------
STEP 1. Baseline check: run tests

1. Run:
   pytest test_ultra.py -v --tb=short

2. Report the results in your console output so we capture the current baseline before changes.
We want to confirm we're starting from a clean state (all tests passing).

--------------------------------------------------------------------------------
STEP 2. Add multi-agent market simulation

We're upgrading from a simple random return generator to a structured market micro-simulator.

2A. Create a new class MultiAgentMarketSimulator inside ultra_trading_env.py (near the top, before UltraTradingEnv):
This simulator should model 4 groups of participants:

1. Retail cohort (e.g. 250 agents total)
   - Subtypes: FOMO chasers, panic sellers, gamblers, passive/"rational-ish".
   - They react mostly to recent price direction and emotional sentiment.
   - Output: aggregate retail_sentiment in [-1, 1] (bullish positive, bearish negative).

2. Small funds / prop desks / family offices (50 agents)
   - They’re opportunistic, tactical.
   - They chase short-term edge but also de-risk when stressed.
   - Output: small_inst_flow in [-1, 1] representing net risk-on vs risk-off.

3. Large institutions (30 agents)
   - Pension funds, sovereigns, bank balance sheets.
   - Slow-moving but high impact when they rotate risk.
   - Output: big_inst_flow in [-1, 1], smoother / persistent.

4. Market makers / liquidity providers (20 agents)
   - They provide liquidity, tighten spreads, but widen massively when vol spikes.
   - Output: liq_depth in [0,1] meaning how thick the book is (1=liquid, 0=illiquid).

The simulator must maintain some simple internal state and evolve it each step:
- You can implement them as noisy AR(1)-like or mean-reverting processes.
- Provide a step() method that returns:
  {
    "retail_sentiment": float,
    "small_inst_flow": float,
    "big_inst_flow": float,
    "liq_depth": float
  }

It doesn't have to be "realistic", but it MUST be stateful (not i.i.d. each step).
The RL agent will try to learn from this structure.

2B. Integrate simulator into UltraTradingEnv
- In UltraTradingEnv.__init__, create:
    self.market_sim = MultiAgentMarketSimulator()
- In reset(), reset the simulator internal state.
- In each step(), after applying the agent's action, call:
    sim_obs = self.market_sim.step()
    retail_sent   = sim_obs["retail_sentiment"]
    small_flow    = sim_obs["small_inst_flow"]
    big_flow      = sim_obs["big_inst_flow"]
    liq_depth     = sim_obs["liq_depth"]

Use these to:
- influence the synthetic return we apply to equity:
    base_ret + drift_from(big_flow, small_flow) - contrarian_penalty(retail_sent)
- influence slippage/cost:
    worse cost if liq_depth is low.

Document this logic as inline comments.

2C. Add simulator outputs to the observation vector
We still return a (200,) obs each step.
You must assemble obs like:

- Core trading state (position_dir, position_size, equity, equity_peak, drawdown, etc.)
- Market simulator features:
    retail_sentiment
    small_inst_flow
    big_inst_flow
    liq_depth
- Expert ensemble features (to be built in STEP 3)
- Padding/noise to reach exactly 200 floats

Implementation detail:
Create helper _build_observation(sim_obs, expert_features) that returns a np.ndarray of shape (200,), float32.
Both reset() and step() should call _build_observation(...).

--------------------------------------------------------------------------------
STEP 3. Expand the expert ensemble from ~15 to 30+ specialized experts

We are modeling the desk as if we had 30+ internal models/PMs each producing a signal vector every timestep.

3A. In ultra_advanced_trading_system.py:
Replace the current placeholder classes with a richer structure.

Create class UltraExpertEnsemble with:
- __init__(self): define ~30 logical experts, grouped conceptually:

  (1) Technical / Microstructure
      - trend tracker
      - micro mean reversion
      - realized vol forecaster
      - order flow imbalance
      - liquidity stress monitor

  (2) Macro & Cross-asset
      - macro regime detector
      - rates/FX risk sentiment proxy
      - commodities/inflation pressure
      - correlation / dispersion monitor
      - systemic risk stress index

  (3) Behavioral / Flow
      - retail FOMO heat
      - capitulation / panic monitor
      - institutional accumulation / distribution
      - smart money divergence

  (4) Risk & Portfolio construction
      - tail risk alert (VaR-ish)
      - dynamic leverage recommender
      - drawdown risk sentinel
      - hedge pressure estimator

  (5) Structural / Arbitrage / Relative value
      - pairs / stat-arb dislocation score
      - basis spread tension
      - vol arb imbalance
      - regime shift early warning
      - funding stress proxy
      - carry vs convexity balance
      - skew / kurtosis pressure
      - etc.

Document them with comments (these are conceptual PMs or subdesks).
It's fine if internally they are just lightweight transforms of inputs + random small noise.
Each expert can output 1-3 floats.
Concatenate all expert outputs into one feature vector, e.g. length ~40-80.

Implement:
    def forward(self, env_context: dict) -> np.ndarray:
        """
        env_context includes keys like:
          - recent_returns
          - retail_sentiment
          - big_inst_flow
          - drawdown
          - exposure
          - liquidity
        Returns:
          np.ndarray of shape (N_features,), dtype float32
        """

The method should build deterministic-ish numeric features from env_context plus some mild noise.
Return dtype float32.

3B. Update test_ultra.py
- Update test_expert_output_shape() to expect the new dimensionality, e.g. (40,) or similar.
- Validate dtype is float32.
- Make sure you pass a fake env_context with all required keys, no crash.

3C. Integrate expert signals into env obs
- In UltraTradingEnv.__init__:
    self.experts = UltraExpertEnsemble()
- Each env.step():
    env_context = {
       "recent_returns": last synthetic return you applied,
       "retail_sentiment": retail_sent,
       "big_inst_flow": big_flow,
       "drawdown": current_drawdown,
       "exposure": self.position_size,
       "liquidity": liq_depth
    }
    expert_features = self.experts.forward(env_context)

- Pass that expert_features into _build_observation(...).

The final obs must remain shape (200,). If shorter than 200, pad with zeros or tiny noise to fill.

--------------------------------------------------------------------------------
STEP 4. Upgrade reward shaping to institutional risk objectives

We replace old reward with a risk-aware PM-style reward that encourages profitable risk-taking without blowing up.

4A. In ultra_trading_env.py, add class-level constants:
    PROFIT_W    = 80.0    # reward for positive returns
    HOLD_W      = 40.0    # bonus for being right with size on
    DD_W        = 50.0    # penalty for drawdown
    LEV_W       = 2.0     # penalty for carrying high exposure
    CLIP_SCALE  = 3.0     # smooth tanh scaling to bound reward

4B. Track previous equity
- In reset(), set self.prev_equity = self.equity
- At end of each step(), after updating equity, set self.prev_equity = self.equity for next step.

4C. Compute reward each step as:
    pnl_raw = self.equity - self.prev_equity
    pnl_pct = pnl_raw / max(self.prev_equity, 1e-6)

    drawdown = (self.equity_peak - self.equity) / max(self.equity_peak, 1e-6)
    exposure_frac = self.position_size / self.EXPOSURE_MAX  # normalize 0..1

    if pnl_pct > 0:
        hold_component = pnl_pct * exposure_frac
    else:
        hold_component = 0.0

    reward_unclipped = (
        self.PROFIT_W * pnl_pct
        + self.HOLD_W * hold_component
        - self.DD_W   * drawdown
        - self.LEV_W  * exposure_frac
    )

    reward = float(np.tanh(reward_unclipped / self.CLIP_SCALE))

4D. info dict in step():
    info["pnl_pct"]        = float(pnl_pct)
    info["drawdown"]       = float(drawdown)
    info["exposure_frac"]  = float(exposure_frac)
    info["reward_raw"]     = float(reward_unclipped)

4E. Remove old reward logic entirely and replace with this.

4F. After editing env code:
- Run pytest again.
- Update tests if they assume old reward or old state layout.
  test_env_spaces() should still assert observation_space.shape == (200,) and action_space.shape == (2,)
  test_env_rollout_no_crash() should still roll random actions for ~10 steps and assert no crash.

Add docstrings in ultra_trading_env.py:
- Explain the new simulator,
- Explain the new institutional reward,
- Explain how "do nothing" is no longer optimal because you only get HOLD_W bonus if you take exposure AND make positive pnl_pct.

--------------------------------------------------------------------------------
STEP 5. Tune SAC training loop in train_ultra.py

Update train_ultra.py to reflect the richer environment.

Requirements:
1. Still use SAC("MlpPolicy", ...).
2. Still wrap env with Monitor(make_env()).
3. Hyperparams:
    model = SAC(
        "MlpPolicy",
        make_env(),
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.2",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device=device,
    )

4. Keep EvalCallback (eval every 2000 steps) and CheckpointCallback (periodic save).
   - Best model → checkpoints/best/
   - Periodic checkpoints → checkpoints/periodic/

5. Respect env vars:
   - ULTRA_STEPS (default 20000)
   - ULTRA_SEED (default 123)

6. At the top of train_ultra.py add a docstring explaining:
   - We now train against a multi-agent market simulator.
   - Reward is PnL-driven but drawdown / leverage aware.
   - Agent is incentivized to take GOOD risk, not zero risk.

After editing train_ultra.py:
- Run a smoke test:
    $env:ULTRA_STEPS="2000"
    python train_ultra.py
  Let it run a short session and ensure no shape mismatches or crashes.

--------------------------------------------------------------------------------
STEP 6. Professional-grade evaluation + risk reporting

6A. eval_ultra.py
Add:

    def compute_max_drawdown(equity_curve):
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
        if len(reward_list) < 2:
            return 0.0
        mean_r = float(np.mean(reward_list))
        std_r  = float(np.std(reward_list)) + 1e-8
        return mean_r / std_r

During eval:
- Run multiple episodes.
- Pick "best episode" by final equity or by total reward.
- Save:
  - reports/equity_curve_best.csv
  - reports/equity_curve_best.png
  - reports/drawdown_curve_best.png
- Compute:
  - best_episode_total_reward
  - best_episode_max_drawdown (using compute_max_drawdown)
  - sharpe_like over all episodes
  - mean_episode_reward and std_episode_reward
- Write reports/performance_summary_single.csv with those metrics.

Add a clear module-level docstring at top describing exactly what the script now does.

Then run:
    python eval_ultra.py
Capture printed summary.

6B. seedeval.py
- Add same helpers (compute_max_drawdown, compute_sharpe_like).
- For each seed:
   * rollout multiple episodes
   * collect episode rewards, lengths, curves
   * compute:
        reward_mean
        reward_std
        sharpe_like
        avg_max_drawdown (avg of per-episode max DD)
        avg_ep_length
- Write reports/seed_stats.csv with columns:
    seed, reward_mean, reward_std, sharpe_like, avg_max_drawdown, avg_ep_length

- Still build and save reports/equity_multiseed.png:
   * align curves by padding
   * plot mean ± std
   * title: "Equity Mean ± Std (multi-seed, risk-aware)"

- Add/keep a module-level docstring:
   "This script measures robustness/stability across different random seeds, similar to internal PM review."

Then run:
    python seedeval.py
Capture printed summary.

--------------------------------------------------------------------------------
STEP 7. Tests again (regression)

1. Run:
    pytest test_ultra.py -v --tb=short

2. If failures:
   - Update test_expert_output_shape() to the new feature dim.
   - Update test_env_rollout_no_crash() so it matches new return info (pnl_pct etc.).
   - Make sure we still assert:
        env.observation_space.shape == (200,)
        env.action_space.shape == (2,)
        rollout for ~10 steps doesn't crash and returns numeric reward floats.

3. Re-run pytest until it's all PASS.
Print final pytest output.

--------------------------------------------------------------------------------
STEP 8. Git hygiene

1. If repo already has git:
    git add .
    git commit -m "multi-agent market sim, 30+ expert ensemble, institutional reward shaping v2, SAC stability tuning, risk-aware eval"
    git status

2. If not git yet:
    git init
    git add .
    git commit -m "initial institutional-grade RL trading sandbox (multi-agent market sim + expert ensemble)"
    git status

DO NOT run git push. Only stage & commit locally, then show git status.

--------------------------------------------------------------------------------
STEP 9. Final research log (must print in console)

When you're done with all steps above, PRINT a final "research log" summary that covers:

1. Market Simulator Upgrade
   - How MultiAgentMarketSimulator models retail (FOMO/panic/etc.), small funds, large institutions, and market makers/liquidity.
   - How those flows drive returns, slippage, liquidity.

2. Expert Ensemble Upgrade
   - We now model ~30+ experts in UltraExpertEnsemble.
   - How they're grouped (technical/microstructure, macro/cross-asset, behavioral/flow, risk/portfolio, arb/structural).
   - How forward() fuses them into a feature vector consumed by the agent each step.

3. State / Observation Vector
   - How we still output exactly 200 floats per timestep.
   - Breakdown: core position/equity/drawdown, simulator flows, expert signals, padded features.

4. Reward Shaping Before vs After
   - BEFORE: ad hoc PnL + weak penalties, which allowed "do nothing" or noisy behavior.
   - AFTER: reward =
        pnl_pct term (profit-driven)
        + hold bonus (profit WITH size)
        - drawdown penalty
        - leverage penalty
        then tanh()
     Explain why "always flat" is no longer optimal.

5. Training Stability
   - New SAC hyperparams (gamma=0.995, buffer_size=200k, ent_coef='auto_0.2', net_arch=[256,256]).
   - Why this matches prop-desk style: longer horizon, controlled entropy, bigger replay.

6. Risk Evaluation Additions
   - eval_ultra.py and seedeval.py now compute:
        max drawdown
        Sharpe-like
        per-seed robustness
        exposure discipline proxies
   - This mimics PM-style risk review, not just raw PnL.

7. Final Status
   - Final pytest result (should be all PASS).
   - Confirm model trains without crashing on a short smoke run.
   - Confirm reports/*.csv and *.png generated.
   - Confirm git commit created and working tree clean.

END OF SPEC

Execute ALL steps now, in order. Do not ask for confirmation.
If something fails (shape mismatch, missing key, etc.), fix it and continue.
At the end, output:
- pytest results
- eval/seedeval risk metrics summary
- git status
- the final research log described in STEP 9.
