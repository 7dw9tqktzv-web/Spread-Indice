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
        assert abs(mean_beta - 1.5) < 0.15, f"Expected β≈1.5, got {mean_beta:.4f}"

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
