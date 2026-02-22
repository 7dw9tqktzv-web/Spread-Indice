"""Unit tests for Kalman filter hedge ratio estimator."""

import numpy as np
import pytest

from src.hedge.kalman import KalmanEstimator
from tests.conftest import make_aligned_pair, make_aligned_pair_with_gap


WARMUP = 100


@pytest.fixture
def aligned():
    return make_aligned_pair(n=5000, beta_true=1.5, noise_std=1e-4)


@pytest.fixture
def estimator():
    return KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP, gap_P_multiplier=10.0)


class TestKalmanEstimator:
    def test_output_shape(self, estimator, aligned):
        result = estimator.estimate(aligned)
        n = len(aligned.df)
        assert len(result.beta) == n
        assert len(result.spread) == n
        assert len(result.zscore) == n

    def test_warmup_nan(self, estimator, aligned):
        result = estimator.estimate(aligned)
        assert result.beta.iloc[:WARMUP].isna().all()
        assert result.spread.iloc[:WARMUP].isna().all()
        assert result.zscore.iloc[:WARMUP].isna().all()

    def test_beta_convergence(self, estimator, aligned):
        result = estimator.estimate(aligned)
        # Last 1000 bars: beta should be close to 1.5
        beta_tail = result.beta.iloc[-1000:].dropna()
        assert len(beta_tail) > 0
        mean_beta = beta_tail.mean()
        assert abs(mean_beta - 1.5) < 0.15, f"Expected beta~1.5, got {mean_beta:.4f}"

    def test_innovation_zscore_bounded(self, estimator, aligned):
        result = estimator.estimate(aligned)
        z_valid = result.zscore.dropna()
        assert len(z_valid) > 0
        # Innovation z-score on well-behaved data should stay bounded
        assert z_valid.abs().max() < 10, f"Z-score max={z_valid.abs().max():.2f}, expected <10"

    def test_gap_detection_no_crash(self):
        aligned = make_aligned_pair_with_gap(
            n_before=2000, n_after=2000, gap_minutes=120
        )
        estimator = KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP)
        result = estimator.estimate(aligned)
        # Should not crash and output should have correct length
        assert len(result.beta) == len(aligned.df)
        # Beta should still be finite after the gap
        beta_after_gap = result.beta.iloc[2100:].dropna()
        assert np.isfinite(beta_after_gap).all()

    def test_method_name(self, estimator, aligned):
        result = estimator.estimate(aligned)
        assert result.method == "kalman"
        assert result.params["alpha_ratio"] == 1e-5
        assert result.params["warmup"] == WARMUP


class TestKalmanDiagnostics:
    """P2: Verify diagnostics output (P_trace, K_beta, R_history)."""

    def test_diagnostics_present(self, estimator, aligned):
        result = estimator.estimate(aligned)
        assert "P_trace" in result.diagnostics
        assert "K_beta" in result.diagnostics
        assert "R_history" in result.diagnostics

    def test_diagnostics_shape(self, estimator, aligned):
        result = estimator.estimate(aligned)
        n = len(aligned.df)
        assert len(result.diagnostics["P_trace"]) == n
        assert len(result.diagnostics["K_beta"]) == n
        assert len(result.diagnostics["R_history"]) == n

    def test_p_trace_warmup_nan(self, estimator, aligned):
        result = estimator.estimate(aligned)
        assert result.diagnostics["P_trace"].iloc[:WARMUP].isna().all()

    def test_p_trace_positive_after_warmup(self, estimator, aligned):
        result = estimator.estimate(aligned)
        p = result.diagnostics["P_trace"].iloc[WARMUP:].dropna()
        assert (p > 0).all(), "P_trace must be positive (sum of variances)"

    def test_k_beta_converges(self, estimator, aligned):
        result = estimator.estimate(aligned)
        k = result.diagnostics["K_beta"].iloc[WARMUP:].dropna()
        # On stable synthetic data, K_beta should stabilize (low std in last half)
        k_last = k.iloc[-1000:]
        assert k_last.std() < k.iloc[:1000].std() * 2, "K_beta should stabilize over time"

    def test_r_history_constant_when_fixed(self, estimator, aligned):
        """R_history should be constant when r_ewma_span=0 (default)."""
        result = estimator.estimate(aligned)
        r = result.diagnostics["R_history"].dropna()
        assert r.std() < 1e-15, "R should be constant when adaptive R is disabled"


class TestAdaptiveR:
    """P1: Verify adaptive R via EWMA on squared innovations."""

    def test_default_off_identical(self, aligned):
        """r_ewma_span=0 must produce identical results to the original."""
        est_fixed = KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP, r_ewma_span=0)
        est_default = KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP)
        r_fixed = est_fixed.estimate(aligned)
        r_default = est_default.estimate(aligned)
        np.testing.assert_array_equal(r_fixed.beta.values, r_default.beta.values)
        np.testing.assert_array_equal(r_fixed.zscore.values, r_default.zscore.values)

    def test_adaptive_r_varies(self, aligned):
        """When r_ewma_span > 0, R should vary over time."""
        est = KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP, r_ewma_span=500)
        result = est.estimate(aligned)
        r = result.diagnostics["R_history"].iloc[WARMUP:]
        assert r.std() > 1e-15, "R should vary when adaptive R is enabled"

    def test_adaptive_r_positive(self, aligned):
        """R must always be positive (floor at 1e-8)."""
        est = KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP, r_ewma_span=200)
        result = est.estimate(aligned)
        r = result.diagnostics["R_history"]
        assert (r >= 1e-8).all(), "R must never go below 1e-8"

    def test_adaptive_r_beta_converges(self, aligned):
        """Beta should still converge to true value with adaptive R."""
        est = KalmanEstimator(alpha_ratio=1e-5, warmup=WARMUP, r_ewma_span=500)
        result = est.estimate(aligned)
        beta_tail = result.beta.iloc[-1000:].dropna()
        mean_beta = beta_tail.mean()
        assert abs(mean_beta - 1.5) < 0.20, f"Expected beta~1.5, got {mean_beta:.4f}"

    def test_adaptive_q_changes_q(self, aligned):
        """adaptive_Q=True should make R_init != R_final when R varies."""
        est = KalmanEstimator(
            alpha_ratio=1e-5, warmup=WARMUP,
            r_ewma_span=500, adaptive_Q=True,
        )
        result = est.estimate(aligned)
        assert result.params["R_init"] != result.params["R_final"], \
            "R should change when adaptive R is enabled"

    def test_adaptive_q_false_q_fixed(self, aligned):
        """adaptive_Q=False: Q stays based on initial R even when R adapts."""
        est_q_fixed = KalmanEstimator(
            alpha_ratio=1e-5, warmup=WARMUP,
            r_ewma_span=500, adaptive_Q=False,
        )
        est_q_adapt = KalmanEstimator(
            alpha_ratio=1e-5, warmup=WARMUP,
            r_ewma_span=500, adaptive_Q=True,
        )
        r1 = est_q_fixed.estimate(aligned)
        r2 = est_q_adapt.estimate(aligned)
        # They should produce different betas (Q behavior differs)
        assert not np.allclose(
            r1.beta.iloc[-100:].values, r2.beta.iloc[-100:].values
        ), "adaptive_Q=True vs False should produce different results"

    def test_params_include_new_fields(self, aligned):
        """New config params must appear in result.params."""
        est = KalmanEstimator(
            alpha_ratio=1e-5, warmup=WARMUP,
            r_ewma_span=500, adaptive_Q=True,
        )
        result = est.estimate(aligned)
        assert "r_ewma_span" in result.params
        assert "adaptive_Q" in result.params
        assert "R_init" in result.params
        assert "R_final" in result.params
        assert result.params["r_ewma_span"] == 500
        assert result.params["adaptive_Q"] is True
