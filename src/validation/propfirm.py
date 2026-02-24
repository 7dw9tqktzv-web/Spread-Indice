"""Propfirm $150k account risk metrics.

Account: $150,000
Max daily loss: $4,500 (3%)
Trailing drawdown: $5,000
Target: $300/day

Computes daily P&L from trade-level data to check propfirm compliance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropfirmConfig:
    """Propfirm account parameters."""

    account_size: float = 150_000.0
    max_daily_loss: float = 4_500.0
    trailing_max_dd: float = 5_000.0
    daily_target: float = 300.0


@dataclass
class PropfirmResult:
    """Propfirm compliance metrics."""

    n_trading_days: int
    daily_pnls: np.ndarray
    max_daily_loss_observed: float  # worst single-day loss (negative)
    max_trailing_dd: float  # max trailing drawdown from equity peak (negative)
    n_days_exceed_daily_limit: int
    n_days_exceed_trailing_dd: int  # bars where trailing DD exceeds limit
    pct_days_profitable: float
    avg_daily_pnl: float
    is_compliant: bool  # True if NEVER exceeds either limit


def compute_propfirm_metrics(
    trade_entry_bars: np.ndarray,
    trade_exit_bars: np.ndarray,
    trade_pnls: np.ndarray,
    bar_dates: np.ndarray,
    equity_curve: np.ndarray,
    config: PropfirmConfig,
) -> PropfirmResult:
    """Compute propfirm compliance metrics.

    Parameters
    ----------
    trade_entry_bars, trade_exit_bars : np.ndarray
        Bar indices.
    trade_pnls : np.ndarray
        Net PnL per trade.
    bar_dates : np.ndarray
        Date (not datetime) for each bar index. Trades assigned to day by exit bar.
    equity_curve : np.ndarray
        Mark-to-market equity curve.
    config : PropfirmConfig
    """
    if len(trade_pnls) == 0:
        return PropfirmResult(
            n_trading_days=0,
            daily_pnls=np.array([]),
            max_daily_loss_observed=0.0,
            max_trailing_dd=0.0,
            n_days_exceed_daily_limit=0,
            n_days_exceed_trailing_dd=0,
            pct_days_profitable=0.0,
            avg_daily_pnl=0.0,
            is_compliant=True,
        )

    # Daily PnL: sum of trade PnLs by exit date
    exit_dates = bar_dates[trade_exit_bars]
    unique_dates = np.unique(exit_dates)

    daily_pnls = np.zeros(len(unique_dates))
    for i, d in enumerate(unique_dates):
        mask = exit_dates == d
        daily_pnls[i] = trade_pnls[mask].sum()

    # Max daily loss
    max_daily_loss = float(daily_pnls.min()) if len(daily_pnls) > 0 else 0.0
    n_exceed_daily = int((daily_pnls < -config.max_daily_loss).sum())

    # Trailing drawdown from equity curve
    running_max = np.maximum.accumulate(equity_curve)
    trailing_dd = equity_curve - running_max  # negative values
    max_trailing_dd = float(trailing_dd.min())
    n_exceed_trailing = int((trailing_dd < -config.trailing_max_dd).sum())

    # Summary
    n_days = len(unique_dates)
    pct_profitable = (
        float((daily_pnls > 0).sum() / n_days * 100) if n_days > 0 else 0.0
    )
    avg_daily = float(daily_pnls.mean()) if n_days > 0 else 0.0

    is_compliant = (n_exceed_daily == 0) and (
        max_trailing_dd >= -config.trailing_max_dd
    )

    return PropfirmResult(
        n_trading_days=n_days,
        daily_pnls=daily_pnls,
        max_daily_loss_observed=round(max_daily_loss, 2),
        max_trailing_dd=round(max_trailing_dd, 2),
        n_days_exceed_daily_limit=n_exceed_daily,
        n_days_exceed_trailing_dd=n_exceed_trailing,
        pct_days_profitable=round(pct_profitable, 1),
        avg_daily_pnl=round(avg_daily, 2),
        is_compliant=is_compliant,
    )
