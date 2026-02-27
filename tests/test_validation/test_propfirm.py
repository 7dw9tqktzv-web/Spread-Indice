"""Tests for propfirm compliance metrics."""

import numpy as np
import pytest

from src.validation.propfirm import PropfirmConfig, PropfirmResult, compute_propfirm_metrics


def _default_config():
    """Standard propfirm config: $150k account, $4500 daily, $5000 trailing DD."""
    return PropfirmConfig()


def test_empty_trades_compliant():
    """Empty trades should be compliant with 0 trading days."""
    config = _default_config()
    result = compute_propfirm_metrics(
        trade_entry_bars=np.array([], dtype=int),
        trade_exit_bars=np.array([], dtype=int),
        trade_pnls=np.array([]),
        bar_dates=np.array([]),
        equity_curve=np.array([150_000.0]),
        config=config,
    )

    assert result.is_compliant is True
    assert result.n_trading_days == 0
    assert result.max_daily_loss_observed == 0.0
    assert result.max_trailing_dd == 0.0


def test_compliant_scenario():
    """3 trades across 2 days, all within limits -> compliant."""
    config = _default_config()

    # 10 bars across 2 dates
    dates = np.array(
        ["2024-01-02"] * 5 + ["2024-01-03"] * 5,
        dtype="datetime64[D]",
    )
    entry_bars = np.array([1, 3, 6])
    exit_bars = np.array([2, 4, 8])
    pnls = np.array([500.0, -200.0, 300.0])

    # Equity curve: starts at 150k, moves within limits
    equity = np.array([
        150_000.0, 150_000.0, 150_500.0, 150_500.0, 150_300.0,
        150_300.0, 150_300.0, 150_300.0, 150_600.0, 150_600.0,
    ], dtype=float)

    result = compute_propfirm_metrics(
        trade_entry_bars=entry_bars,
        trade_exit_bars=exit_bars,
        trade_pnls=pnls,
        bar_dates=dates,
        equity_curve=equity,
        config=config,
    )

    assert result.is_compliant is True
    assert result.n_trading_days == 2
    assert result.n_days_exceed_daily_limit == 0
    assert result.n_days_exceed_trailing_dd == 0


def test_daily_loss_breach():
    """Single day loss > $4,500 -> not compliant."""
    config = _default_config()

    dates = np.array(["2024-01-02"] * 5, dtype="datetime64[D]")
    entry_bars = np.array([0, 2])
    exit_bars = np.array([1, 3])
    # Two big losses on same day: total = -5000
    pnls = np.array([-3_000.0, -2_000.0])

    equity = np.array([
        150_000.0, 147_000.0, 147_000.0, 145_000.0, 145_000.0,
    ], dtype=float)

    result = compute_propfirm_metrics(
        trade_entry_bars=entry_bars,
        trade_exit_bars=exit_bars,
        trade_pnls=pnls,
        bar_dates=dates,
        equity_curve=equity,
        config=config,
    )

    assert result.is_compliant is False
    assert result.n_days_exceed_daily_limit >= 1
    # Daily loss = -5000 which exceeds -4500 limit
    assert result.max_daily_loss_observed < -4_500.0


def test_trailing_dd_breach():
    """Equity curve that drops > $5,000 from peak -> not compliant."""
    config = _default_config()

    dates = np.array(["2024-01-02"] * 6, dtype="datetime64[D]")
    entry_bars = np.array([0])
    exit_bars = np.array([4])
    pnls = np.array([-4_000.0])

    # Equity rises to 155k then drops to 149k (DD = 6000 > 5000)
    equity = np.array([
        150_000.0, 155_000.0, 152_000.0, 150_000.0, 149_000.0, 149_000.0,
    ], dtype=float)

    result = compute_propfirm_metrics(
        trade_entry_bars=entry_bars,
        trade_exit_bars=exit_bars,
        trade_pnls=pnls,
        bar_dates=dates,
        equity_curve=equity,
        config=config,
    )

    assert result.is_compliant is False
    assert result.max_trailing_dd < -5_000.0
    assert result.n_days_exceed_trailing_dd >= 1


def test_trailing_dd_boundary_exactly_5000():
    """Trailing DD of exactly -$5,000 should be compliant (>= comparison)."""
    config = _default_config()

    dates = np.array(["2024-01-02"] * 4, dtype="datetime64[D]")
    entry_bars = np.array([0])
    exit_bars = np.array([2])
    pnls = np.array([-3_000.0])

    # Peak at 155k, trough at 150k -> DD = -5000 exactly
    equity = np.array([
        155_000.0, 152_000.0, 150_000.0, 150_000.0,
    ], dtype=float)

    result = compute_propfirm_metrics(
        trade_entry_bars=entry_bars,
        trade_exit_bars=exit_bars,
        trade_pnls=pnls,
        bar_dates=dates,
        equity_curve=equity,
        config=config,
    )

    # max_trailing_dd = -5000.0, config.trailing_max_dd = 5000.0
    # is_compliant checks: max_trailing_dd >= -config.trailing_max_dd
    # -5000.0 >= -5000.0 is True -> compliant
    assert result.is_compliant is True
    assert result.max_trailing_dd == pytest.approx(-5_000.0)
