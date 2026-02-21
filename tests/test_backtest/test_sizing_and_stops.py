"""Tests for find_optimal_multiplier and dollar stop loss."""

import numpy as np
import pandas as pd
import pytest

from src.sizing.position import find_optimal_multiplier
from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    InstrumentSpec,
    _apply_dollar_stop,
)


# ─── find_optimal_multiplier ─────────────────────────────────────────

class TestFindOptimalMultiplier:

    def test_integer_ratio_mult1(self):
        """n_b_raw=3.0 -> mult=1 is already perfect (0% error)."""
        m, na, nb, err = find_optimal_multiplier(3.0, max_multiplier=3)
        assert m == 1
        assert na == 1
        assert nb == 3
        assert err == 0.0

    def test_half_ratio_mult2(self):
        """n_b_raw=2.5 -> mult=2 gives n_b=5 (exact), error=0%."""
        m, na, nb, err = find_optimal_multiplier(2.5, max_multiplier=3)
        assert m == 2
        assert na == 2
        assert nb == 5
        assert err == 0.0

    def test_third_ratio_mult3(self):
        """n_b_raw=3.33 -> mult=3 gives n_b=10 (3.33*3=9.99->10), error ~0.1%."""
        m, na, nb, err = find_optimal_multiplier(3.33, max_multiplier=3)
        assert m == 3
        assert na == 3
        assert nb == 10
        # Effective ratio = 10/3 = 3.333..., raw = 3.33
        # Error should be small
        assert err < 1.0

    def test_mult1_when_already_close(self):
        """n_b_raw=3.01 -> mult=1 rounds to 3, error ~0.3%. Lower mults might also work."""
        m, na, nb, err = find_optimal_multiplier(3.01, max_multiplier=3)
        # mult=1: round(3.01)=3, error=|3-3.01|/3.01*100=0.33%
        # mult=2: round(6.02)=6, error=|3-3.01|/3.01*100=0.33%
        # mult=3: round(9.03)=9, error=|3-3.01|/3.01*100=0.33%
        # All equal, so first (m=1) wins
        assert m == 1
        assert nb == 3

    def test_zero_n_b_raw(self):
        """n_b_raw=0 -> returns (1, 1, 1, 0.0)."""
        m, na, nb, err = find_optimal_multiplier(0.0, max_multiplier=3)
        assert m == 1
        assert na == 1
        assert nb == 1
        assert err == 0.0

    def test_negative_n_b_raw(self):
        """Negative n_b_raw -> returns default."""
        m, na, nb, err = find_optimal_multiplier(-1.0, max_multiplier=3)
        assert m == 1

    def test_max_multiplier_1(self):
        """max_multiplier=1 -> always mult=1."""
        m, na, nb, err = find_optimal_multiplier(3.33, max_multiplier=1)
        assert m == 1
        assert nb == 3


# ─── _apply_dollar_stop ──────────────────────────────────────────────

class TestApplyDollarStop:

    def test_stop_advances_exit(self):
        """Trade with large loss should exit early."""
        n = 10
        px_a = np.array([100.0, 100.0, 100.0, 95.0, 90.0, 85.0, 100.0, 100.0, 100.0, 100.0])
        px_b = np.array([200.0] * n)

        te = np.array([1])       # entry at bar 1
        tx = np.array([8])       # original exit at bar 8
        sides = np.array([1], dtype=np.int8)  # long spread
        entry_px_a = np.array([100.0])
        entry_px_b = np.array([200.0])
        n_a = np.array([1])
        n_b = np.array([1])

        # unrealized at bar 3: 1 * (1*(95-100)*20 - 1*(200-200)*5) = -100
        # unrealized at bar 4: 1 * (1*(90-100)*20 - 1*(200-200)*5) = -200
        tx_mod = _apply_dollar_stop(
            px_a, px_b, te, tx, sides, entry_px_a, entry_px_b,
            n_a, n_b, 20.0, 5.0, 150.0  # dollar_stop=150
        )
        # Should exit at bar 4 where unrealized = -200 < -150
        assert tx_mod[0] == 4

    def test_no_stop_when_within_threshold(self):
        """Trade with small loss should not trigger stop."""
        n = 10
        px_a = np.array([100.0, 100.0, 99.0, 99.5, 100.5, 101.0, 100.0, 100.0, 100.0, 100.0])
        px_b = np.array([200.0] * n)

        te = np.array([1])
        tx = np.array([6])
        sides = np.array([1], dtype=np.int8)
        entry_px_a = np.array([100.0])
        entry_px_b = np.array([200.0])
        n_a = np.array([1])
        n_b = np.array([1])

        tx_mod = _apply_dollar_stop(
            px_a, px_b, te, tx, sides, entry_px_a, entry_px_b,
            n_a, n_b, 20.0, 5.0, 500.0  # dollar_stop=500 (high threshold)
        )
        # No bar reaches -500, so exit unchanged
        assert tx_mod[0] == 6

    def test_stop_at_first_breach(self):
        """Stop triggers at the first bar that breaches, not later ones."""
        n = 8
        px_a = np.array([100.0, 100.0, 90.0, 80.0, 70.0, 100.0, 100.0, 100.0])
        px_b = np.array([200.0] * n)

        te = np.array([1])
        tx = np.array([7])
        sides = np.array([1], dtype=np.int8)
        entry_px_a = np.array([100.0])
        entry_px_b = np.array([200.0])
        n_a = np.array([1])
        n_b = np.array([1])

        tx_mod = _apply_dollar_stop(
            px_a, px_b, te, tx, sides, entry_px_a, entry_px_b,
            n_a, n_b, 20.0, 5.0, 100.0  # dollar_stop=100
        )
        # Bar 2: unrealized = 1*(1*(90-100)*20) = -200 < -100 -> exit at bar 2
        assert tx_mod[0] == 2


# ─── Dollar stop in BacktestEngine ───────────────────────────────────

def _make_series(values, name="price"):
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name=name)


SPEC_SIMPLE = InstrumentSpec(multiplier=1.0, tick_size=0.25, tick_value=0.25)


class TestDollarStopEngine:

    def test_dollar_stop_forces_exit(self):
        """BacktestEngine with dollar_stop should exit when loss exceeds threshold."""
        config = BacktestConfig(
            commission_per_contract=0.0, slippage_ticks=0, dollar_stop=5.0,
        )
        engine = BacktestEngine(config=config)

        # Long spread: A drops from 100 to 90 (loss = 10 per contract)
        close_a = _make_series([100.0, 100.0, 90.0, 90.0, 105.0, 105.0])
        close_b = _make_series([200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
        signals = _make_series([0, 1, 1, 1, 1, 0], name="signal")
        beta = _make_series([1.0] * 6, name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        assert len(result.trades) == 1
        t = result.trades[0]
        # Should have exited early due to dollar stop (loss > $5)
        assert t.exit_bar == 2  # forced exit at bar 2

    def test_dollar_stop_no_reentry(self):
        """After dollar stop exit, signal still active should NOT trigger re-entry."""
        config = BacktestConfig(
            commission_per_contract=0.0, slippage_ticks=0, dollar_stop=5.0,
        )
        engine = BacktestEngine(config=config)

        # Long spread: A drops sharply then recovers
        close_a = _make_series([100.0, 100.0, 90.0, 95.0, 100.0, 100.0])
        close_b = _make_series([200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
        # Signal stays +1 the whole time (no normal exit)
        signals = _make_series([0, 1, 1, 1, 1, 0], name="signal")
        beta = _make_series([1.0] * 6, name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        # Only 1 trade (dollar stop exits, no re-entry because signal stays +1)
        assert len(result.trades) == 1
