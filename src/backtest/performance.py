"""Performance metrics computed from backtest results."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult


@dataclass
class PerformanceMetrics:
    """Summary statistics of a backtest run."""
    total_pnl: float
    num_trades: int
    win_rate: float             # % of winning trades
    profit_factor: float        # gross gains / gross losses
    avg_pnl_per_trade: float
    sharpe_ratio: float         # annualized
    max_drawdown_pct: float     # % from peak
    max_drawdown_duration: int  # in bars
    calmar_ratio: float         # annualized return / max drawdown


def compute_performance(result: BacktestResult, bars_per_day: int = 120) -> PerformanceMetrics:
    """Compute performance metrics from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Output of BacktestEngine.run().
    bars_per_day : int
        Number of bars per trading day (5min bars × 20h session ≈ 240,
        but trading window 04:00-14:00 = 10h = 120 bars). Used for annualization.

    Returns
    -------
    PerformanceMetrics
    """
    trades = result.trades
    equity = result.equity_curve

    num_trades = len(trades)

    if num_trades == 0:
        return PerformanceMetrics(
            total_pnl=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_pnl_per_trade=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
        )

    # --- Trade-level metrics ---
    pnls = [t.pnl_net for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / num_trades

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / num_trades * 100

    gross_gains = sum(winners) if winners else 0.0
    gross_losses = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else float("inf")

    # --- Equity curve metrics ---
    eq = equity.values

    # Drawdown
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    max_drawdown_pct = abs(float(drawdown.min())) * 100

    # Drawdown duration (longest streak below peak)
    below_peak = eq < running_max
    max_dd_duration = 0
    current_duration = 0
    for bp in below_peak:
        if bp:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    # Sharpe ratio (annualized from bar returns)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(eq) / eq[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    bars_per_year = bars_per_day * 252
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Calmar ratio
    total_return = (eq[-1] - eq[0]) / eq[0] if eq[0] != 0 else 0.0
    n_bars = len(eq)
    with np.errstate(invalid="ignore"):
        annualized_return = (1 + total_return) ** (bars_per_year / max(n_bars, 1)) - 1
    if np.isnan(annualized_return) or np.isinf(annualized_return):
        annualized_return = 0.0
    calmar = (annualized_return * 100) / max_drawdown_pct if max_drawdown_pct > 0 else float("inf")

    return PerformanceMetrics(
        total_pnl=total_pnl,
        num_trades=num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_pnl_per_trade=avg_pnl,
        sharpe_ratio=float(sharpe),
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=float(calmar),
    )
