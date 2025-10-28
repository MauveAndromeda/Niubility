"""
risk_manager.py

Institutional-grade risk management module for UltraTrader.

Features:
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Kelly Criterion for optimal position sizing
- Dynamic risk adjustment based on market conditions
- Stop-loss and take-profit mechanisms
- Position limit enforcement
- Real-time risk monitoring and alerts
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
from scipy import stats


class RiskMetrics:
    """
    Calculate comprehensive risk metrics for portfolio management.
    """

    @staticmethod
    def value_at_risk(
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of historical returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: Calculation method - "historical", "parametric", or "monte_carlo"

        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        if method == "historical":
            # Historical VaR: use percentile of empirical distribution
            percentile = (1 - confidence) * 100
            var = -np.percentile(returns, percentile)
            return float(var)

        elif method == "parametric":
            # Parametric VaR: assume normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence)
            var = -(mu + z_score * sigma)
            return float(var)

        elif method == "monte_carlo":
            # Monte Carlo VaR: simulate returns
            mu = np.mean(returns)
            sigma = np.std(returns)
            n_simulations = 10000
            simulated = np.random.normal(mu, sigma, n_simulations)
            percentile = (1 - confidence) * 100
            var = -np.percentile(simulated, percentile)
            return float(var)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    @staticmethod
    def conditional_var(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that we're in the (1-confidence)% worst cases.

        Args:
            returns: Array of historical returns
            confidence: Confidence level (e.g., 0.95 for 95% CVaR)

        Returns:
            CVaR value (positive number representing expected tail loss)
        """
        if len(returns) == 0:
            return 0.0

        # Find the VaR threshold
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)

        # CVaR is the mean of all returns below the VaR threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return float(-var_threshold)

        cvar = -np.mean(tail_returns)
        return float(cvar)

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: Array of equity values over time

        Returns:
            Maximum drawdown as a fraction (0 to 1)
        """
        if len(equity_curve) == 0:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / np.maximum(peak, 1e-9)
        max_dd = float(np.max(drawdown))
        return max_dd

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate per period

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return == 0:
            return 0.0

        sharpe = mean_return / std_return
        return float(sharpe)

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
    ) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only penalizes downside volatility).

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate per period
            target_return: Target return threshold (default 0)

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        mean_return = np.mean(excess_returns)

        # Downside deviation: std of returns below target
        downside_returns = returns[returns < target_return] - target_return
        if len(downside_returns) == 0:
            return float(mean_return / 1e-9)  # Very high if no downside

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        sortino = mean_return / downside_std
        return float(sortino)

    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        annualization_factor: float = 252,
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Array of returns
            equity_curve: Array of equity values
            annualization_factor: Factor to annualize returns (252 for daily)

        Returns:
            Calmar ratio
        """
        if len(returns) == 0 or len(equity_curve) == 0:
            return 0.0

        # Annualized return
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        n_periods = len(returns)
        annualized_return = ((1 + total_return) ** (annualization_factor / n_periods)) - 1

        # Max drawdown
        max_dd = RiskMetrics.max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        calmar = annualized_return / max_dd
        return float(calmar)

    @staticmethod
    def information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """
        Calculate Information Ratio (excess return / tracking error).

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Ensure same length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]

        excess_returns = returns - benchmark_returns
        mean_excess = np.mean(excess_returns)
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        ir = mean_excess / tracking_error
        return float(ir)


class KellyPositionSizer:
    """
    Calculate optimal position sizes using Kelly Criterion.

    The Kelly Criterion maximizes long-term growth rate by sizing positions
    based on edge and odds. We use fractional Kelly to be more conservative.
    """

    @staticmethod
    def kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Calculate Kelly fraction for position sizing.

        Formula: f = (p * b - q) / b
        where:
            p = win probability
            q = loss probability (1 - p)
            b = avg_win / avg_loss (odds)
            fraction = fractional Kelly (e.g., 0.25 for quarter Kelly)

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning return
            avg_loss: Average losing return (positive number)
            fraction: Fractional Kelly (0 to 1), lower is more conservative

        Returns:
            Optimal position size as fraction of capital (0 to 1)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss

        # Full Kelly
        kelly = (p * b - q) / b

        # Apply fractional Kelly for safety
        kelly_frac = kelly * fraction

        # Clamp to reasonable bounds
        kelly_frac = float(np.clip(kelly_frac, 0.0, 1.0))

        return kelly_frac

    @staticmethod
    def kelly_from_returns(
        returns: np.ndarray,
        fraction: float = 0.25,
    ) -> float:
        """
        Calculate Kelly fraction from historical returns.

        Args:
            returns: Array of historical returns
            fraction: Fractional Kelly

        Returns:
            Optimal position size
        """
        if len(returns) == 0:
            return 0.0

        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        if len(winning_trades) == 0 or len(losing_trades) == 0:
            return 0.0

        win_rate = len(winning_trades) / len(returns)
        avg_win = np.mean(winning_trades)
        avg_loss = abs(np.mean(losing_trades))

        return KellyPositionSizer.kelly_fraction(win_rate, avg_win, avg_loss, fraction)


class RiskManager:
    """
    Comprehensive risk management system.

    Integrates multiple risk controls:
    - Position limits
    - VaR limits
    - Stop-loss / take-profit
    - Dynamic risk adjustment
    - Kelly-based position sizing
    """

    def __init__(self, config: Dict = None):
        """
        Initialize risk manager.

        Args:
            config: Risk management configuration dictionary
        """
        self.config = config or {}

        # Extract configuration
        self.max_position = self.config.get("max_position", 0.6)
        self.max_drawdown = self.config.get("max_drawdown", 0.20)
        self.var_limit_95 = self.config.get("var_limit_95", 0.10)
        self.var_limit_99 = self.config.get("var_limit_99", 0.15)

        # Position sizing config
        sizing_cfg = self.config.get("position_sizing", {})
        self.sizing_method = sizing_cfg.get("method", "kelly")
        self.kelly_fraction = sizing_cfg.get("kelly_fraction", 0.25)
        self.max_leverage = sizing_cfg.get("max_leverage", 1.0)

        # Stop loss / take profit
        sl_cfg = self.config.get("stop_loss", {})
        self.stop_loss_enabled = sl_cfg.get("enabled", True)
        self.stop_loss_threshold = sl_cfg.get("threshold", 0.05)

        tp_cfg = self.config.get("take_profit", {})
        self.take_profit_enabled = tp_cfg.get("enabled", True)
        self.take_profit_threshold = tp_cfg.get("threshold", 0.15)

        # State tracking
        self.recent_returns: List[float] = []
        self.recent_equity: List[float] = []
        self.alerts: List[str] = []

    def update_state(self, equity: float, pnl: float):
        """
        Update risk manager state with new equity and PnL.

        Args:
            equity: Current equity value
            pnl: Recent PnL (return)
        """
        self.recent_equity.append(equity)
        self.recent_returns.append(pnl)

        # Keep last 252 periods (roughly 1 year of daily data)
        if len(self.recent_returns) > 252:
            self.recent_returns.pop(0)
        if len(self.recent_equity) > 252:
            self.recent_equity.pop(0)

    def check_position_limits(self, position_size: float) -> Tuple[bool, str]:
        """
        Check if position size is within limits.

        Args:
            position_size: Proposed position size

        Returns:
            (is_valid, message)
        """
        if abs(position_size) > self.max_position:
            msg = f"Position size {position_size:.2f} exceeds limit {self.max_position:.2f}"
            self.alerts.append(msg)
            return False, msg

        return True, "OK"

    def check_drawdown_limit(self) -> Tuple[bool, str]:
        """
        Check if current drawdown exceeds limits.

        Returns:
            (is_valid, message)
        """
        if len(self.recent_equity) == 0:
            return True, "OK"

        equity_arr = np.array(self.recent_equity)
        current_dd = RiskMetrics.max_drawdown(equity_arr)

        if current_dd > self.max_drawdown:
            msg = f"Drawdown {current_dd:.2%} exceeds limit {self.max_drawdown:.2%}"
            self.alerts.append(msg)
            return False, msg

        return True, "OK"

    def check_var_limits(self) -> Tuple[bool, str]:
        """
        Check if VaR exceeds limits.

        Returns:
            (is_valid, message)
        """
        if len(self.recent_returns) < 20:
            return True, "Insufficient data for VaR"

        returns_arr = np.array(self.recent_returns)

        var_95 = RiskMetrics.value_at_risk(returns_arr, confidence=0.95)
        var_99 = RiskMetrics.value_at_risk(returns_arr, confidence=0.99)

        if var_95 > self.var_limit_95:
            msg = f"VaR(95%) {var_95:.2%} exceeds limit {self.var_limit_95:.2%}"
            self.alerts.append(msg)
            return False, msg

        if var_99 > self.var_limit_99:
            msg = f"VaR(99%) {var_99:.2%} exceeds limit {self.var_limit_99:.2%}"
            self.alerts.append(msg)
            return False, msg

        return True, "OK"

    def calculate_optimal_position_size(
        self,
        signal_direction: float,
        current_volatility: float = 1.0,
    ) -> float:
        """
        Calculate optimal position size based on configured method.

        Args:
            signal_direction: Trading signal direction (-1 to 1)
            current_volatility: Current market volatility scaling factor

        Returns:
            Optimal position size
        """
        if self.sizing_method == "fixed":
            # Fixed position size
            base_size = self.max_position * 0.5  # Use half of max as default

        elif self.sizing_method == "kelly":
            # Kelly criterion
            if len(self.recent_returns) < 10:
                base_size = 0.1  # Conservative default
            else:
                returns_arr = np.array(self.recent_returns)
                kelly_size = KellyPositionSizer.kelly_from_returns(
                    returns_arr, fraction=self.kelly_fraction
                )
                base_size = kelly_size * self.max_leverage

        else:
            base_size = self.max_position * 0.3  # Default conservative

        # Scale by signal strength
        position_size = base_size * abs(signal_direction)

        # Inverse scale by volatility (lower size in high vol)
        position_size = position_size / max(current_volatility, 0.5)

        # Clamp to limits
        position_size = float(np.clip(position_size, 0.0, self.max_position))

        return position_size

    def should_stop_loss(self, current_pnl: float, entry_price: float) -> bool:
        """
        Check if stop-loss should be triggered.

        Args:
            current_pnl: Current position PnL
            entry_price: Entry price for position

        Returns:
            True if stop-loss should trigger
        """
        if not self.stop_loss_enabled:
            return False

        if len(self.recent_equity) < 2:
            return False

        # Check if loss exceeds threshold
        loss_pct = current_pnl / max(self.recent_equity[-1], 1e-9)

        if loss_pct < -self.stop_loss_threshold:
            msg = f"Stop-loss triggered: {loss_pct:.2%}"
            self.alerts.append(msg)
            return True

        return False

    def should_take_profit(self, current_pnl: float) -> bool:
        """
        Check if take-profit should be triggered.

        Args:
            current_pnl: Current position PnL

        Returns:
            True if take-profit should trigger
        """
        if not self.take_profit_enabled:
            return False

        if len(self.recent_equity) < 2:
            return False

        # Check if profit exceeds threshold
        profit_pct = current_pnl / max(self.recent_equity[-1], 1e-9)

        if profit_pct > self.take_profit_threshold:
            msg = f"Take-profit triggered: {profit_pct:.2%}"
            self.alerts.append(msg)
            return True

        return False

    def get_risk_summary(self) -> Dict[str, float]:
        """
        Get comprehensive risk summary.

        Returns:
            Dictionary of risk metrics
        """
        if len(self.recent_returns) == 0:
            return {}

        returns_arr = np.array(self.recent_returns)
        equity_arr = np.array(self.recent_equity)

        summary = {
            "var_95": RiskMetrics.value_at_risk(returns_arr, 0.95),
            "cvar_95": RiskMetrics.conditional_var(returns_arr, 0.95),
            "var_99": RiskMetrics.value_at_risk(returns_arr, 0.99),
            "cvar_99": RiskMetrics.conditional_var(returns_arr, 0.99),
            "max_drawdown": RiskMetrics.max_drawdown(equity_arr),
            "sharpe_ratio": RiskMetrics.sharpe_ratio(returns_arr),
            "sortino_ratio": RiskMetrics.sortino_ratio(returns_arr),
            "calmar_ratio": RiskMetrics.calmar_ratio(returns_arr, equity_arr),
        }

        return summary

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []

    def get_alerts(self) -> List[str]:
        """Get all current alerts."""
        return self.alerts.copy()
