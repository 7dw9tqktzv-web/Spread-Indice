"""Tests for performance metrics computation."""

import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, BacktestResult, Trade
from src.backtest.performance import compute_performance


def _make_trade(pnl_net, entry_bar=0, exit_bar=1):
    """Helper: create a minimal Trade with given PnL."""
    idx = pd.date_range("2024-01-02 18:00", periods=max(exit_bar + 1, 2), freq="5min")
    return Trade(
        entry_bar=entry_bar,
        exit_bar=exit_bar,
        entry_time=idx[entry_bar],
        exit_time=idx[exit_bar],
        side=1,
        entry_price_a=100.0,
        entry_price_b=200.0,
        exit_price_a=101.0,
        exit_price_b=200.0,
        n_a=1,
        n_b=1,
        pnl_gross=pnl_net,
        costs=0.0,
        pnl_net=pnl_net,
    )


def _make_equity(values):
    """Helper: wrap equity values in a Series."""
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name="equity")


def test_zero_trades_all_zeros():
    """With 0 trades, all metrics should be 0."""
    result = BacktestResult(
        trades=[],
        equity_curve=_make_equity([100_000.0] * 10),
        config=BacktestConfig(),
    )
    perf = compute_performance(result)

    assert perf.num_trades == 0
    assert perf.total_pnl == 0.0
    assert perf.win_rate == 0.0
    assert perf.profit_factor == 0.0
    assert perf.avg_pnl_per_trade == 0.0
    assert perf.sharpe_ratio == 0.0
    assert perf.max_drawdown_pct == 0.0
    assert perf.max_drawdown_duration == 0


def test_known_trades_metrics():
    """3 wins of $100 and 2 losses of $50 -> PF=3.0, WR=60%, total=$200."""
    trades = [
        _make_trade(100.0, entry_bar=0, exit_bar=1),
        _make_trade(100.0, entry_bar=2, exit_bar=3),
        _make_trade(-50.0, entry_bar=4, exit_bar=5),
        _make_trade(100.0, entry_bar=6, exit_bar=7),
        _make_trade(-50.0, entry_bar=8, exit_bar=9),
    ]

    # Build a simple equity curve reflecting the trades
    equity_vals = [100_000.0] * 10
    cumulative = 100_000.0
    pnls = [100.0, 100.0, -50.0, 100.0, -50.0]
    for i, pnl in enumerate(pnls):
        cumulative += pnl
        equity_vals[2 * i + 1] = cumulative

    # Fill forward so equity is monotonic between trades
    for i in range(1, len(equity_vals)):
        if equity_vals[i] == 100_000.0 and i > 0:
            equity_vals[i] = equity_vals[i - 1]

    result = BacktestResult(
        trades=trades,
        equity_curve=_make_equity(equity_vals),
        config=BacktestConfig(),
    )
    perf = compute_performance(result)

    assert perf.num_trades == 5
    assert perf.total_pnl == pytest.approx(200.0)
    assert perf.win_rate == pytest.approx(60.0)
    # PF = gross_gains / gross_losses = 300 / 100 = 3.0
    assert perf.profit_factor == pytest.approx(3.0)
    assert perf.avg_pnl_per_trade == pytest.approx(40.0)


def test_max_drawdown_pct_known_curve():
    """Equity [100000, 110000, 90000, 95000, 105000] -> DD from 110k to 90k = 18.18%."""
    equity_vals = [100_000.0, 110_000.0, 90_000.0, 95_000.0, 105_000.0]

    # Need at least one trade for metrics to compute (otherwise 0 trades path)
    trades = [_make_trade(5_000.0, entry_bar=0, exit_bar=4)]

    result = BacktestResult(
        trades=trades,
        equity_curve=_make_equity(equity_vals),
        config=BacktestConfig(),
    )
    perf = compute_performance(result)

    # Max drawdown: peak=110000, trough=90000 -> (90000-110000)/110000 = -18.18%
    expected_dd_pct = (110_000.0 - 90_000.0) / 110_000.0 * 100
    assert perf.max_drawdown_pct == pytest.approx(expected_dd_pct, rel=1e-3)


def test_all_winning_trades_infinite_pf():
    """If all trades are winners, profit_factor should be inf."""
    trades = [
        _make_trade(100.0, entry_bar=0, exit_bar=1),
        _make_trade(200.0, entry_bar=2, exit_bar=3),
    ]
    equity_vals = [100_000.0, 100_100.0, 100_100.0, 100_300.0]

    result = BacktestResult(
        trades=trades,
        equity_curve=_make_equity(equity_vals),
        config=BacktestConfig(),
    )
    perf = compute_performance(result)

    assert perf.profit_factor == float("inf")
    assert perf.win_rate == pytest.approx(100.0)


def test_max_drawdown_duration():
    """Equity curve that dips below peak for 3 bars then recovers."""
    equity_vals = [100_000.0, 110_000.0, 105_000.0, 104_000.0, 106_000.0, 115_000.0]

    trades = [_make_trade(15_000.0, entry_bar=0, exit_bar=5)]
    result = BacktestResult(
        trades=trades,
        equity_curve=_make_equity(equity_vals),
        config=BacktestConfig(),
    )
    perf = compute_performance(result)

    # Below peak of 110000 at bars 2, 3, 4 (3 bars), then new high at bar 5
    assert perf.max_drawdown_duration == 3
