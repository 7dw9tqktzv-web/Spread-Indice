"""Tests for CPCV module."""

import numpy as np

from src.validation.cpcv import (
    CPCVConfig,
    build_test_mask,
    compute_sharpe_from_pnls,
    filter_trades_by_mask,
    generate_cpcv_paths,
    generate_fold_boundaries,
    run_cpcv,
)


class TestGenerateFoldBoundaries:
    def test_basic_10_folds(self):
        bounds = generate_fold_boundaries(1000, 10)
        assert bounds.shape == (10, 2)
        assert bounds[0, 0] == 0
        assert bounds[-1, 1] == 1000

    def test_no_gaps(self):
        bounds = generate_fold_boundaries(1000, 10)
        for i in range(9):
            assert bounds[i, 1] == bounds[i + 1, 0]

    def test_uneven_division(self):
        bounds = generate_fold_boundaries(1003, 10)
        assert bounds[-1, 1] == 1003  # last fold absorbs remainder

    def test_covers_all_bars(self):
        bounds = generate_fold_boundaries(500, 7)
        assert bounds[0, 0] == 0
        assert bounds[-1, 1] == 500
        total = sum(bounds[i, 1] - bounds[i, 0] for i in range(7))
        assert total == 500


class TestGenerateCPCVPaths:
    def test_c10_2(self):
        paths = generate_cpcv_paths(10, 2)
        assert len(paths) == 45  # C(10,2)

    def test_c5_2(self):
        paths = generate_cpcv_paths(5, 2)
        assert len(paths) == 10  # C(5,2)

    def test_path_contents(self):
        paths = generate_cpcv_paths(5, 2)
        assert (0, 1) in paths
        assert (3, 4) in paths
        # Each path has exactly 2 elements
        for p in paths:
            assert len(p) == 2


class TestBuildTestMask:
    def test_single_fold_no_purge(self):
        bounds = generate_fold_boundaries(100, 10)  # each fold = 10 bars
        mask = build_test_mask(100, bounds, (0,), purge_bars=0)
        assert mask[:10].all()
        assert not mask[10:].any()

    def test_purge_zone_at_boundaries(self):
        bounds = generate_fold_boundaries(100, 10)
        # Fold 5 = bars 50-59, purge 2 bars each side
        mask = build_test_mask(100, bounds, (5,), purge_bars=2)
        # Purge before: bars 48-49 (outside test, already False)
        assert not mask[48]
        assert not mask[49]
        # Purge inside start: bars 50-51
        assert not mask[50]
        assert not mask[51]
        # Interior: bars 52-57 should be True
        assert mask[52]
        assert mask[57]
        # Purge inside end: bars 58-59
        assert not mask[58]
        assert not mask[59]
        # After test: bars 60-61 purged
        assert not mask[60]
        assert not mask[61]

    def test_non_contiguous_folds(self):
        bounds = generate_fold_boundaries(100, 10)
        mask = build_test_mask(100, bounds, (2, 7), purge_bars=1)
        # Fold 2 = 20-29, fold 7 = 70-79
        # Both should have interior bars True, boundaries purged
        assert mask[22]  # interior of fold 2
        assert mask[72]  # interior of fold 7
        assert not mask[35]  # train between folds

    def test_purge_at_dataset_start(self):
        bounds = generate_fold_boundaries(100, 10)
        mask = build_test_mask(100, bounds, (0,), purge_bars=3)
        # First 3 bars purged (inside start), last 3 of fold purged (inside end)
        assert not mask[0]
        assert not mask[2]
        assert mask[3]
        assert mask[6]
        assert not mask[7]

    def test_purge_at_dataset_end(self):
        bounds = generate_fold_boundaries(100, 10)
        mask = build_test_mask(100, bounds, (9,), purge_bars=3)
        # Fold 9 = 90-99. Purge inside start 90-92, purge inside end 97-99
        assert not mask[89]  # purge before
        assert not mask[90]
        assert not mask[92]
        assert mask[93]
        assert mask[96]
        assert not mask[97]


class TestFilterTradesByMask:
    def test_both_in_test(self):
        test_mask = np.array([False, False, True, True, True, False, False])
        entries = np.array([2, 5])
        exits = np.array([4, 6])
        result = filter_trades_by_mask(entries, exits, test_mask)
        assert result[0]  # entry=2 ok, exit=4 ok
        assert not result[1]  # entry=5 not in test

    def test_entry_in_test_exit_not(self):
        test_mask = np.array([False, True, True, False, False])
        entries = np.array([1])
        exits = np.array([3])
        result = filter_trades_by_mask(entries, exits, test_mask)
        assert not result[0]  # exit=3 not in test

    def test_empty_trades(self):
        mask = np.ones(10, dtype=bool)
        result = filter_trades_by_mask(
            np.array([], dtype=int), np.array([], dtype=int), mask
        )
        assert len(result) == 0


class TestComputeSharpe:
    def test_positive_pnls(self):
        pnls = np.array([100.0, 50.0, -20.0, 80.0, 60.0])
        s = compute_sharpe_from_pnls(pnls)
        assert s > 0

    def test_negative_pnls(self):
        pnls = np.array([-100.0, -50.0, 20.0, -80.0, -60.0])
        s = compute_sharpe_from_pnls(pnls)
        assert s < 0

    def test_single_trade(self):
        assert compute_sharpe_from_pnls(np.array([100.0])) == 0.0

    def test_zero_std(self):
        assert compute_sharpe_from_pnls(np.array([100.0, 100.0, 100.0])) == 0.0

    def test_no_sqrt_n_scaling(self):
        """Sharpe = mean/std, NOT mean/std * sqrt(N)."""
        pnls = np.array([100.0, 200.0, 50.0, 150.0])
        expected = pnls.mean() / pnls.std()
        result = compute_sharpe_from_pnls(pnls)
        assert abs(result - expected) < 1e-10


class TestRunCPCV:
    def test_basic_run(self):
        rng = np.random.default_rng(42)
        n_bars = 10000
        entries = np.sort(rng.choice(n_bars - 10, 50, replace=False))
        exits = entries + rng.integers(1, 10, 50)
        pnls = rng.normal(50, 200, 50)
        config = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=20, min_trades_per_path=2)
        result = run_cpcv(entries, exits, pnls, n_bars, config)
        assert result["n_paths"] == 45
        assert result["n_valid_paths"] > 0
        assert "median_sharpe" in result
        assert len(result["path_sharpes"]) == 45

    def test_no_trades(self):
        config = CPCVConfig(n_folds=5, n_test_folds=2, purge_bars=10)
        result = run_cpcv(
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([]),
            1000,
            config,
        )
        assert result["median_sharpe"] == 0.0
        assert result["n_valid_paths"] == 0

    def test_pct_positive_range(self):
        rng = np.random.default_rng(123)
        n_bars = 5000
        entries = np.sort(rng.choice(n_bars - 5, 30, replace=False))
        exits = entries + rng.integers(1, 5, 30)
        pnls = rng.normal(100, 50, 30)  # mostly positive
        config = CPCVConfig(n_folds=5, n_test_folds=2, purge_bars=10, min_trades_per_path=2)
        result = run_cpcv(entries, exits, pnls, n_bars, config)
        assert 0 <= result["pct_positive"] <= 100
