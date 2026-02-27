"""Tests for Deflated Sharpe Ratio module."""

import numpy as np

from src.validation.deflated_sharpe import (
    compute_dsr_for_config,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)


class TestExpectedMaxSharpe:
    def test_increases_with_trials(self):
        s1 = expected_max_sharpe(10)
        s2 = expected_max_sharpe(1000)
        s3 = expected_max_sharpe(16000)
        assert s1 < s2 < s3

    def test_single_trial(self):
        assert expected_max_sharpe(1) == 0.0

    def test_scales_with_std(self):
        s1 = expected_max_sharpe(100, std_sharpe=0.5)
        s2 = expected_max_sharpe(100, std_sharpe=1.0)
        assert s2 > s1


class TestDeflatedSharpeRatio:
    def test_high_sharpe_high_dsr(self):
        dsr = deflated_sharpe_ratio(3.0, 1.5, 200, 0.0, 3.0)
        assert dsr > 0.95

    def test_low_sharpe_low_dsr(self):
        dsr = deflated_sharpe_ratio(1.6, 1.5, 50, 0.0, 3.0)
        assert dsr < 0.95

    def test_few_trades_returns_zero(self):
        assert deflated_sharpe_ratio(2.0, 1.0, 2, 0.0, 3.0) == 0.0

    def test_below_benchmark_low_dsr(self):
        dsr = deflated_sharpe_ratio(0.5, 1.5, 200, 0.0, 3.0)
        assert dsr < 0.5

    def test_returns_probability(self):
        dsr = deflated_sharpe_ratio(2.0, 1.0, 100, 0.0, 3.0)
        assert 0.0 <= dsr <= 1.0


class TestComputeDSRForConfig:
    def test_basic(self):
        rng = np.random.default_rng(42)
        pnls = rng.normal(100, 200, 100)
        all_sharpes = rng.normal(0.5, 0.3, 16000)
        result = compute_dsr_for_config(1.5, pnls, 16000, all_sharpes)
        assert "dsr" in result
        assert 0 <= result["dsr"] <= 1
        assert result["n_trades"] == 100

    def test_empty_pnls(self):
        result = compute_dsr_for_config(0.5, np.array([]), 100, np.array([0.3, 0.4]))
        assert result["dsr"] == 0.0
        assert result["n_trades"] == 0
