"""
performance_metrics.py

Comprehensive performance analytics for institutional trading systems.

Provides PM-grade metrics including:
- Sharpe, Sortino, Calmar ratios
- Information Ratio
- Win rate, profit factor
- Transaction cost analysis
- Slippage analysis
- Performance attribution
- Rolling window metrics
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    downside_volatility: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float

    # Trade metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float

    # Cost analysis
    total_commission: float
    total_slippage: float
    total_costs_pct: float

    # Other
    kelly_fraction: float
    best_trade: float
    worst_trade: float


class PerformanceAnalyzer:
    """
    Analyze trading system performance with institutional-grade metrics.
    """

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: float = 252):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize metrics (252 for daily, 12 for monthly)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float | None = None,
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rfr_per_period = rfr / self.annualization_factor

        excess_returns = returns - rfr_per_period
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0:
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(self.annualization_factor)
        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float | None = None,
        target_return: float = 0.0,
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) == 0:
            return 0.0

        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rfr_per_period = rfr / self.annualization_factor

        excess_returns = returns - rfr_per_period
        mean_excess = np.mean(excess_returns)

        # Downside deviation
        downside = returns[returns < target_return] - target_return
        if len(downside) == 0:
            return float(mean_excess) if mean_excess > 0 else 0.0

        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(self.annualization_factor)
        return float(sortino)

    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(returns) == 0 or len(equity_curve) < 2:
            return 0.0

        # Annualized return
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        n_periods = len(returns)
        annualized_return = ((1 + total_return) ** (self.annualization_factor / n_periods)) - 1

        # Max drawdown
        max_dd = self._calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        calmar = annualized_return / max_dd
        return float(calmar)

    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Calculate Information Ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]

        excess_returns = returns - benchmark_returns
        mean_excess = np.mean(excess_returns)
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        ir = (mean_excess / tracking_error) * np.sqrt(self.annualization_factor)
        return float(ir)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / np.maximum(peak, 1e-9)
        return float(np.max(drawdown))

    def _calculate_max_drawdown_duration(self, equity_curve: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods."""
        peak = np.maximum.accumulate(equity_curve)
        is_drawdown = equity_curve < peak

        if not np.any(is_drawdown):
            return 0

        # Find consecutive drawdown periods
        drawdown_periods = []
        current_dd_length = 0

        for in_dd in is_drawdown:
            if in_dd:
                current_dd_length += 1
            else:
                if current_dd_length > 0:
                    drawdown_periods.append(current_dd_length)
                current_dd_length = 0

        if current_dd_length > 0:
            drawdown_periods.append(current_dd_length)

        return int(max(drawdown_periods)) if drawdown_periods else 0

    def _calculate_var(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        percentile = (1 - confidence) * 100
        var = -np.percentile(returns, percentile)
        return float(var)

    def _calculate_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return float(-var_threshold)
        cvar = -np.mean(tail_returns)
        return float(cvar)

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (fraction of positive returns)."""
        if len(returns) == 0:
            return 0.0
        wins = np.sum(returns > 0)
        return float(wins / len(returns))

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (total wins / total losses)."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        total_wins = np.sum(wins) if len(wins) > 0 else 0.0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0

        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0

        return float(total_wins / total_losses)

    def _calculate_kelly_fraction(
        self,
        returns: np.ndarray,
        fraction: float = 0.25,
    ) -> float:
        """Calculate Kelly fraction for position sizing."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss

        kelly = (p * b - q) / b
        kelly_frac = kelly * fraction

        return float(np.clip(kelly_frac, 0.0, 1.0))

    def generate_report(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: List[Dict] | None = None,
        total_commission: float = 0.0,
        total_slippage: float = 0.0,
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            returns: Array of returns per period
            equity_curve: Array of equity values over time
            trades: Optional list of trade dictionaries
            total_commission: Total commission paid
            total_slippage: Total slippage cost

        Returns:
            PerformanceReport dataclass
        """
        if len(returns) == 0 or len(equity_curve) < 2:
            return self._empty_report()

        # Return metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        n_periods = len(returns)
        annualized_return = ((1 + total_return) ** (self.annualization_factor / n_periods)) - 1
        volatility = float(np.std(returns) * np.sqrt(self.annualization_factor))

        # Downside volatility
        downside_rets = returns[returns < 0]
        downside_vol = float(np.std(downside_rets) * np.sqrt(self.annualization_factor)) if len(downside_rets) > 0 else 0.0

        # Risk-adjusted ratios
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, equity_curve)

        # Information ratio (vs zero benchmark)
        benchmark_rets = np.zeros_like(returns)
        info_ratio = self.calculate_information_ratio(returns, benchmark_rets)

        # Risk metrics
        max_dd = self._calculate_max_drawdown(equity_curve)
        max_dd_duration = self._calculate_max_drawdown_duration(equity_curve)
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)
        cvar_99 = self._calculate_cvar(returns, 0.99)

        # Trade metrics
        num_trades = len(trades) if trades else len(returns)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # Trade duration (if trade data available)
        if trades and all('duration' in t for t in trades):
            durations = [t['duration'] for t in trades]
            avg_trade_duration = float(np.mean(durations))
        else:
            avg_trade_duration = 1.0

        # Cost analysis
        total_costs_pct = (total_commission + total_slippage) / max(equity_curve[0], 1e-9)

        # Kelly fraction
        kelly = self._calculate_kelly_fraction(returns)

        # Best/worst trades
        best_trade = float(np.max(returns)) if len(returns) > 0 else 0.0
        worst_trade = float(np.min(returns)) if len(returns) > 0 else 0.0

        return PerformanceReport(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=volatility,
            downside_volatility=downside_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_trade_duration,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs_pct=total_costs_pct,
            kelly_fraction=kelly,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

    def _empty_report(self) -> PerformanceReport:
        """Return empty report with zeros."""
        return PerformanceReport(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            var_95=0.0,
            cvar_95=0.0,
            var_99=0.0,
            cvar_99=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade_duration=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            total_costs_pct=0.0,
            kelly_fraction=0.0,
            best_trade=0.0,
            worst_trade=0.0,
        )

    def calculate_rolling_sharpe(
        self,
        returns: np.ndarray,
        window: int = 60,
    ) -> np.ndarray:
        """
        Calculate rolling Sharpe ratio.

        Args:
            returns: Return series
            window: Rolling window size

        Returns:
            Array of rolling Sharpe ratios
        """
        if len(returns) < window:
            return np.array([])

        rolling_sharpe = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            sharpe = self.calculate_sharpe_ratio(window_returns)
            rolling_sharpe.append(sharpe)

        return np.array(rolling_sharpe)

    def calculate_rolling_max_drawdown(
        self,
        equity_curve: np.ndarray,
        window: int = 60,
    ) -> np.ndarray:
        """
        Calculate rolling maximum drawdown.

        Args:
            equity_curve: Equity curve
            window: Rolling window size

        Returns:
            Array of rolling max drawdowns
        """
        if len(equity_curve) < window:
            return np.array([])

        rolling_dd = []
        for i in range(window, len(equity_curve) + 1):
            window_equity = equity_curve[i - window:i]
            dd = self._calculate_max_drawdown(window_equity)
            rolling_dd.append(dd)

        return np.array(rolling_dd)

    def print_report(self, report: PerformanceReport):
        """
        Print formatted performance report.

        Args:
            report: PerformanceReport to display
        """
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)

        print("\n--- RETURN METRICS ---")
        print(f"Total Return:        {report.total_return:>10.2%}")
        print(f"Annualized Return:   {report.annualized_return:>10.2%}")
        print(f"Volatility (Ann.):   {report.volatility:>10.2%}")
        print(f"Downside Vol (Ann.): {report.downside_volatility:>10.2%}")

        print("\n--- RISK-ADJUSTED RATIOS ---")
        print(f"Sharpe Ratio:        {report.sharpe_ratio:>10.3f}")
        print(f"Sortino Ratio:       {report.sortino_ratio:>10.3f}")
        print(f"Calmar Ratio:        {report.calmar_ratio:>10.3f}")
        print(f"Information Ratio:   {report.information_ratio:>10.3f}")

        print("\n--- RISK METRICS ---")
        print(f"Max Drawdown:        {report.max_drawdown:>10.2%}")
        print(f"Max DD Duration:     {report.max_drawdown_duration:>10d} periods")
        print(f"VaR (95%):           {report.var_95:>10.2%}")
        print(f"CVaR (95%):          {report.cvar_95:>10.2%}")
        print(f"VaR (99%):           {report.var_99:>10.2%}")
        print(f"CVaR (99%):          {report.cvar_99:>10.2%}")

        print("\n--- TRADE METRICS ---")
        print(f"Number of Trades:    {report.num_trades:>10d}")
        print(f"Win Rate:            {report.win_rate:>10.2%}")
        print(f"Profit Factor:       {report.profit_factor:>10.3f}")
        print(f"Avg Win:             {report.avg_win:>10.4f}")
        print(f"Avg Loss:            {report.avg_loss:>10.4f}")
        print(f"Avg Trade Duration:  {report.avg_trade_duration:>10.2f} periods")
        print(f"Best Trade:          {report.best_trade:>10.4f}")
        print(f"Worst Trade:         {report.worst_trade:>10.4f}")

        print("\n--- COST ANALYSIS ---")
        print(f"Total Commission:    {report.total_commission:>10.6f}")
        print(f"Total Slippage:      {report.total_slippage:>10.6f}")
        print(f"Total Costs (% NAV): {report.total_costs_pct:>10.2%}")

        print("\n--- POSITION SIZING ---")
        print(f"Kelly Fraction:      {report.kelly_fraction:>10.2%}")

        print("\n" + "=" * 70 + "\n")
