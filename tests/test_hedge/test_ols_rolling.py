"""Unit tests for OLS Rolling hedge ratio estimator."""

import numpy as np
import pytest

from src.hedge.ols_rolling import OLSRollingEstimator
from tests.conftest import make_aligned_pair


WINDOW = 500  # smaller window for faster tests
ZSCORE_WINDOW = 12


@pytest.fixture
def aligned():
    return make_aligned_pair(n=2000, beta_true=1.5, noise_std=1e-4)


@pytest.fixture
def estimator():
    return OLSRollingEstimator(window=WINDOW, zscore_window=ZSCORE_WINDOW)


class TestOLSRollingEstimator:
    def test_output_shape(self, estimator, aligned):
        result = estimator.estimate(aligned)
        n = len(aligned.df)
        assert len(result.beta) == n
        assert len(result.spread) == n
        assert len(result.zscore) == n

    def test_beta_convergence(self, estimator, aligned):
        result = estimator.estimate(aligned)
        # OLS: log_a = α + β × log_b + ε → β = Cov(log_a, log_b) / Var(log_b)
        # With log_a = 1.5 * log_b, β ≈ 1.5
        beta_tail = result.beta.iloc[WINDOW + 100:].dropna()
        assert len(beta_tail) > 0
        mean_beta = beta_tail.mean()
        assert abs(mean_beta - 1.5) < 0.1, f"Expected β≈1.5, got {mean_beta:.4f}"

    def test_nan_warmup(self, estimator, aligned):
        result = estimator.estimate(aligned)
        # First (window-1) bars should be NaN for beta (rolling needs window obs)
        assert result.beta.iloc[:WINDOW - 1].isna().all()

    def test_zscore_centered(self, estimator, aligned):
        result = estimator.estimate(aligned)
        z_valid = result.zscore.dropna()
        assert len(z_valid) > 0
        # Z-score should be roughly centered around 0
        assert abs(z_valid.mean()) < 0.5, f"Z-score mean={z_valid.mean():.3f}, expected ~0"

    def test_method_name(self, estimator, aligned):
        result = estimator.estimate(aligned)
        assert result.method == "ols_rolling"
        assert result.params["window"] == WINDOW
        assert result.params["zscore_window"] == ZSCORE_WINDOW
