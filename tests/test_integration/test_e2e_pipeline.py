"""End-to-end integration test: data → hedge → signals → backtest → performance.

Verifies the complete pipeline produces coherent results on synthetic data,
catching any data format mismatch between modules (e.g. column naming).
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    run_backtest_grid,
    run_backtest_vectorized,
)
from src.config.instruments import InstrumentSpec
from src.backtest.performance import compute_performance
from src.data.alignment import AlignedPair
from src.hedge.ols_rolling import OLSRollingEstimator
from src.hedge.kalman import KalmanEstimator
from src.signals.generator import SignalConfig, SignalGenerator
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument


def _make_mean_reverting_pair(n=5000, seed=42):
    """Create synthetic pair with known mean-reverting spread.

    log(A) = 1.5 * log(B) + noise, where noise is AR(1) mean-reverting.
    This ensures the spread is stationary and signals will be generated.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 18:00", periods=n, freq="5min")

    # B follows geometric random walk
    log_b = np.cumsum(rng.normal(0, 0.001, n)) + np.log(5000.0)
    close_b = np.exp(log_b)

    # Spread is AR(1) mean-reverting with occasional excursions
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.002)

    log_a = 1.5 * log_b + spread
    close_a = np.exp(log_a)

    df = pd.DataFrame({
        "close_a": close_a,
        "close_b": close_b,
        "volume_a": rng.integers(100, 10000, n).astype(float),
        "volume_b": rng.integers(100, 10000, n).astype(float),
    }, index=idx)

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.ES)
    return AlignedPair(pair=pair, df=df, timeframe="5min")


SPEC_A = InstrumentSpec(multiplier=20.0, tick_size=0.25, tick_value=5.0)
SPEC_B = InstrumentSpec(multiplier=50.0, tick_size=0.25, tick_value=12.50)


class TestE2EPipelineOLS:
    """Full pipeline with OLS rolling hedge ratio."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        aligned = _make_mean_reverting_pair()

        # Hedge ratio
        estimator = OLSRollingEstimator(window=500, zscore_window=20)
        hedge = estimator.estimate(aligned)

        # Signals
        sig_cfg = SignalConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0)
        gen = SignalGenerator(config=sig_cfg)
        signals = gen.generate(hedge.zscore)

        return aligned, hedge, signals

    def test_hedge_output_format(self, pipeline_result):
        aligned, hedge, _ = pipeline_result
        assert len(hedge.beta) == len(aligned.df)
        assert len(hedge.spread) == len(aligned.df)
        assert len(hedge.zscore) == len(aligned.df)

    def test_signals_generated(self, pipeline_result):
        _, _, signals = pipeline_result
        assert (signals != 0).any(), "No signals generated — synthetic data may need tuning"
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_backtest_engine_produces_trades(self, pipeline_result):
        aligned, hedge, signals = pipeline_result
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(
            aligned.df["close_a"], aligned.df["close_b"],
            signals, hedge.beta, SPEC_A, SPEC_B,
        )
        assert len(result.trades) > 0
        assert len(result.equity_curve) == len(aligned.df)
        # Equity curve starts at initial capital
        assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_vectorized_matches_engine(self, pipeline_result):
        aligned, hedge, signals = pipeline_result
        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values

        # Engine result
        engine = BacktestEngine(BacktestConfig())
        engine_result = engine.run(
            aligned.df["close_a"], aligned.df["close_b"],
            signals, hedge.beta, SPEC_A, SPEC_B,
        )
        engine_pnl = sum(t.pnl_net for t in engine_result.trades)

        # Vectorized result
        vec_result = run_backtest_vectorized(
            px_a, px_b, signals.values, hedge.beta.values,
            mult_a=20.0, mult_b=50.0, tick_a=0.25, tick_b=0.25,
        )

        assert vec_result["trades"] == len(engine_result.trades)
        assert vec_result["pnl"] == pytest.approx(engine_pnl, rel=1e-6)

    def test_grid_backtest_consistent(self, pipeline_result):
        aligned, hedge, signals = pipeline_result
        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values

        grid_result = run_backtest_grid(
            px_a, px_b, signals.values, hedge.beta.values,
            mult_a=20.0, mult_b=50.0, tick_a=0.25, tick_b=0.25,
        )

        vec_result = run_backtest_vectorized(
            px_a, px_b, signals.values, hedge.beta.values,
            mult_a=20.0, mult_b=50.0, tick_a=0.25, tick_b=0.25,
        )

        assert grid_result["trades"] == vec_result["trades"]
        assert grid_result["pnl"] == pytest.approx(vec_result["pnl"], rel=1e-6)
        assert grid_result["win_rate"] == vec_result["win_rate"]

    def test_performance_metrics(self, pipeline_result):
        aligned, hedge, signals = pipeline_result
        engine = BacktestEngine(BacktestConfig())
        bt_result = engine.run(
            aligned.df["close_a"], aligned.df["close_b"],
            signals, hedge.beta, SPEC_A, SPEC_B,
        )
        perf = compute_performance(bt_result)
        assert perf.num_trades > 0
        assert 0.0 <= perf.win_rate <= 100.0
        assert perf.profit_factor >= 0.0


class TestE2EPipelineKalman:
    """Full pipeline with Kalman hedge ratio."""

    def test_kalman_pipeline(self):
        aligned = _make_mean_reverting_pair()

        estimator = KalmanEstimator(alpha_ratio=1e-4, warmup=100)
        hedge = estimator.estimate(aligned)

        sig_cfg = SignalConfig(z_entry=1.5, z_exit=0.3, z_stop=3.0)
        gen = SignalGenerator(config=sig_cfg)
        signals = gen.generate(hedge.zscore)

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        result = run_backtest_grid(
            px_a, px_b, signals.values, hedge.beta.values,
            mult_a=20.0, mult_b=50.0, tick_a=0.25, tick_b=0.25,
        )

        assert result["trades"] > 0
        assert isinstance(result["pnl"], float)


class TestArrayLengthValidation:
    """Verify backtest engines reject mismatched arrays."""

    def test_vectorized_rejects_mismatch(self):
        with pytest.raises(ValueError, match="Array length mismatch"):
            run_backtest_vectorized(
                np.ones(100), np.ones(99), np.zeros(100), np.ones(100),
                mult_a=20.0, mult_b=50.0, tick_a=0.25, tick_b=0.25,
            )

    def test_grid_rejects_mismatch(self):
        with pytest.raises(ValueError, match="Array length mismatch"):
            run_backtest_grid(
                np.ones(100), np.ones(100), np.zeros(100), np.ones(99),
                mult_a=20.0, mult_b=50.0, tick_a=0.25, tick_b=0.25,
            )
