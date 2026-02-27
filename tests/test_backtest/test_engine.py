"""Unit tests for backtest engine."""

import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    InstrumentSpec,
)


def _make_series(values: list[float], name: str = "price") -> pd.Series:
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name=name)


# Simple specs: $1/pt multiplier, tick=0.25, tick_value=$0.25
SPEC_SIMPLE = InstrumentSpec(multiplier=1.0, tick_size=0.25, tick_value=0.25)

# Realistic NQ-like and ES-like specs
SPEC_NQ = InstrumentSpec(multiplier=20.0, tick_size=0.25, tick_value=5.0)
SPEC_ES = InstrumentSpec(multiplier=50.0, tick_size=0.25, tick_value=12.50)


class TestSingleLongTrade:
    """Signal [0, +1, +1, 0] — one long trade."""

    def test_pnl_correct(self):
        # Prices: A goes 100→100→102→102, B goes 200→200→201→201
        # Entry at bar 1, exit at bar 3
        # With multiplier=1, n_a=1, slippage=0 for simplicity
        config = BacktestConfig(commission_per_contract=0.0, slippage_ticks=0)
        engine = BacktestEngine(config=config)

        close_a = _make_series([100.0, 100.0, 102.0, 102.0])
        close_b = _make_series([200.0, 200.0, 201.0, 201.0])
        signals = _make_series([0, 1, 1, 0], name="signal")
        beta = _make_series([1.0, 1.0, 1.0, 1.0], name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        assert len(result.trades) == 1
        t = result.trades[0]
        assert t.side == 1
        assert t.entry_bar == 1
        assert t.exit_bar == 3
        # PnL = side * [n_a * Δa * pv_a - n_b * Δb * pv_b]
        # = 1 * [1 * (102-100) * 1 - n_b * (201-200) * 1]
        # n_b = round((100*1)/(200*1) * 1.0 * 1) = round(0.5) = 1 (min 1)
        expected_pnl = 1 * (1 * 2.0 * 1.0 - 1 * 1.0 * 1.0)  # = 1.0
        assert t.pnl_net == pytest.approx(expected_pnl)


class TestSingleShortTrade:
    """Signal [0, -1, -1, 0] — one short trade."""

    def test_pnl_correct(self):
        config = BacktestConfig(commission_per_contract=0.0, slippage_ticks=0)
        engine = BacktestEngine(config=config)

        # A drops 100→100→98→98, B drops 200→200→199→199
        close_a = _make_series([100.0, 100.0, 98.0, 98.0])
        close_b = _make_series([200.0, 200.0, 199.0, 199.0])
        signals = _make_series([0, -1, -1, 0], name="signal")
        beta = _make_series([1.0, 1.0, 1.0, 1.0], name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        assert len(result.trades) == 1
        t = result.trades[0]
        assert t.side == -1
        # PnL = -1 * [1 * (98-100) * 1 - 1 * (199-200) * 1]
        #      = -1 * [-2 - (-1)] = -1 * (-1) = 1.0
        assert t.pnl_net == pytest.approx(1.0)


class TestCostsApplied:
    """Commission = 2 × commission_per_contract × (n_a + n_b)."""

    def test_costs(self):
        config = BacktestConfig(commission_per_contract=2.50, slippage_ticks=0)
        engine = BacktestEngine(config=config)

        close_a = _make_series([100.0, 100.0, 100.0, 100.0])
        close_b = _make_series([200.0, 200.0, 200.0, 200.0])
        signals = _make_series([0, 1, 1, 0], name="signal")
        beta = _make_series([1.0, 1.0, 1.0, 1.0], name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        t = result.trades[0]
        # n_a=1, n_b=1 (round(0.5)=0 but min 1) → costs = 2.50 * (1+1) * 2 = 10.0
        assert t.costs == pytest.approx(10.0)
        # No price movement → pnl_gross = 0, pnl_net = -10
        assert t.pnl_gross == pytest.approx(0.0)
        assert t.pnl_net == pytest.approx(-10.0)


class TestSlippageDirection:
    """LONG spread: entry_a +tick (worse buy), entry_b -tick (worse sell)."""

    def test_long_slippage(self):
        config = BacktestConfig(commission_per_contract=0.0, slippage_ticks=1)
        engine = BacktestEngine(config=config)

        close_a = _make_series([100.0, 100.0, 100.0, 100.0])
        close_b = _make_series([200.0, 200.0, 200.0, 200.0])
        signals = _make_series([0, 1, 1, 0], name="signal")
        beta = _make_series([1.0, 1.0, 1.0, 1.0], name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)
        t = result.trades[0]

        # LONG spread: buy A (+1 tick), sell B (-1 tick)
        assert t.entry_price_a == pytest.approx(100.0 + 0.25)  # 100.25
        assert t.entry_price_b == pytest.approx(200.0 - 0.25)  # 199.75

        # Exit LONG: sell A (-1 tick), buy B (+1 tick)
        assert t.exit_price_a == pytest.approx(100.0 - 0.25)   # 99.75
        assert t.exit_price_b == pytest.approx(200.0 + 0.25)   # 200.25

    def test_short_slippage(self):
        config = BacktestConfig(commission_per_contract=0.0, slippage_ticks=1)
        engine = BacktestEngine(config=config)

        close_a = _make_series([100.0, 100.0, 100.0, 100.0])
        close_b = _make_series([200.0, 200.0, 200.0, 200.0])
        signals = _make_series([0, -1, -1, 0], name="signal")
        beta = _make_series([1.0, 1.0, 1.0, 1.0], name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)
        t = result.trades[0]

        # SHORT spread: sell A (-1 tick), buy B (+1 tick)
        assert t.entry_price_a == pytest.approx(100.0 - 0.25)  # 99.75
        assert t.entry_price_b == pytest.approx(200.0 + 0.25)  # 200.25


class TestMarkToMarketEquity:
    """Position ouverte, prix bouge avant la sortie.
    Vérifier que l'equity_curve reflète le PnL latent avant la clôture."""

    def test_equity_reflects_unrealized_pnl(self):
        config = BacktestConfig(
            initial_capital=100_000.0, commission_per_contract=0.0, slippage_ticks=0
        )
        engine = BacktestEngine(config=config)

        # Bar 0: flat, Bar 1: entry long, Bar 2: price moves (still in position), Bar 3: exit
        close_a = _make_series([100.0, 100.0, 105.0, 103.0])
        close_b = _make_series([200.0, 200.0, 200.0, 200.0])
        signals = _make_series([0, 1, 1, 0], name="signal")
        beta = _make_series([1.0, 1.0, 1.0, 1.0], name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)
        eq = result.equity_curve

        # Bar 0: flat → 100_000
        assert eq.iloc[0] == pytest.approx(100_000.0)

        # Bar 1: just entered, price hasn't moved → unrealized = 0
        assert eq.iloc[1] == pytest.approx(100_000.0)

        # Bar 2: A moved +5, B unchanged → unrealized = 1*(1*5*1 - 1*0*1) = 5
        assert eq.iloc[2] == pytest.approx(100_005.0)

        # Bar 3: trade closed. realized PnL = 1*(1*3*1 - 1*0*1) = 3
        assert eq.iloc[3] == pytest.approx(100_003.0)

        # Key: equity at bar 2 != equity at bar 1 (proves mark-to-market works)
        assert eq.iloc[2] != eq.iloc[1]


class TestNoTradesWhenFlat:
    """All signals = 0 → no trades, equity constant."""

    def test_flat(self):
        engine = BacktestEngine(config=BacktestConfig())

        close_a = _make_series([100.0] * 10)
        close_b = _make_series([200.0] * 10)
        signals = _make_series([0] * 10, name="signal")
        beta = _make_series([1.0] * 10, name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        assert len(result.trades) == 0
        assert (result.equity_curve == 100_000.0).all()


class TestEquityCurveLength:
    """Equity curve has same length as input."""

    def test_length(self):
        engine = BacktestEngine(config=BacktestConfig())
        n = 50
        close_a = _make_series([100.0] * n)
        close_b = _make_series([200.0] * n)
        signals = _make_series([0] * n, name="signal")
        beta = _make_series([1.0] * n, name="beta")

        result = engine.run(close_a, close_b, signals, beta, SPEC_SIMPLE, SPEC_SIMPLE)

        assert len(result.equity_curve) == n
