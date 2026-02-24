"""Combinatorially Purged Cross-Validation (CPCV) for backtest validation.

Implements CPCV(N, t) where N = number of folds, t = test folds per path.
Each path uses t folds as test, remaining N-t as train.
Purge zone of `purge_bars` at each train/test boundary prevents leakage.

Reference: Lopez de Prado, "Advances in Financial Machine Learning", Ch. 12.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class CPCVConfig:
    """Configuration for CPCV."""

    n_folds: int = 10
    n_test_folds: int = 2
    purge_bars: int = 100
    min_trades_per_path: int = 5


def generate_fold_boundaries(n_bars: int, n_folds: int) -> np.ndarray:
    """Generate fold start/end indices as (n_folds, 2) array.

    Returns
    -------
    np.ndarray of shape (n_folds, 2)
        Each row is (start_bar, end_bar) -- end is exclusive.
        No gaps between folds. Last fold absorbs remainder.
    """
    fold_size = n_bars // n_folds
    boundaries = np.zeros((n_folds, 2), dtype=np.int64)
    for i in range(n_folds):
        boundaries[i, 0] = i * fold_size
        boundaries[i, 1] = (i + 1) * fold_size if i < n_folds - 1 else n_bars
    return boundaries


def generate_cpcv_paths(n_folds: int, n_test_folds: int) -> list[tuple[int, ...]]:
    """Generate all CPCV paths as combinations of test fold indices.

    Returns C(n_folds, n_test_folds) paths.
    """
    return list(combinations(range(n_folds), n_test_folds))


def build_test_mask(
    n_bars: int,
    fold_boundaries: np.ndarray,
    test_fold_indices: tuple[int, ...],
    purge_bars: int,
) -> np.ndarray:
    """Build boolean mask: True for bars in test folds, False for purge/train.

    Purge zone: `purge_bars` on each side of every test/train boundary.
    Bars in purge zones are excluded from both test and train.
    """
    test_mask = np.zeros(n_bars, dtype=np.bool_)

    # Mark test fold bars
    for fi in test_fold_indices:
        start, end = fold_boundaries[fi]
        test_mask[start:end] = True

    # Purge at each test fold boundary
    for fi in test_fold_indices:
        t_start, t_end = fold_boundaries[fi]
        # Purge before test fold start
        purge_start = max(0, t_start - purge_bars)
        test_mask[purge_start:t_start] = False
        # Purge inside test fold at start (first purge_bars of test)
        test_mask[t_start : min(n_bars, t_start + purge_bars)] = False
        # Purge inside test fold at end (last purge_bars of test)
        test_mask[max(0, t_end - purge_bars) : t_end] = False
        # Purge after test fold end
        purge_end = min(n_bars, t_end + purge_bars)
        test_mask[t_end:purge_end] = False

    return test_mask


def filter_trades_by_mask(
    trade_entry_bars: np.ndarray,
    trade_exit_bars: np.ndarray,
    test_mask: np.ndarray,
) -> np.ndarray:
    """Return boolean array: True for trades whose entry AND exit are in test region.

    Both entry bar and exit bar must be within the test mask (not in purge zone).
    """
    if len(trade_entry_bars) == 0:
        return np.array([], dtype=np.bool_)
    entry_ok = test_mask[trade_entry_bars]
    exit_ok = test_mask[trade_exit_bars]
    return entry_ok & exit_ok


def compute_sharpe_from_pnls(pnls: np.ndarray) -> float:
    """Compute Sharpe ratio from trade PnLs.

    Sharpe = mean(pnl) / std(pnl).
    NOT annualized by sqrt(N) -- that would bias toward paths with few trades.
    Returns 0.0 if insufficient trades or zero std.
    """
    if len(pnls) < 2:
        return 0.0
    std = pnls.std()
    if std < 1e-12:
        return 0.0
    return float(pnls.mean() / std)


def run_cpcv(
    trade_entry_bars: np.ndarray,
    trade_exit_bars: np.ndarray,
    trade_pnls: np.ndarray,
    n_bars: int,
    config: CPCVConfig,
) -> dict:
    """Run full CPCV analysis on pre-computed trade arrays.

    For each of C(n_folds, n_test_folds) paths, filters trades to test folds
    (with purge) and computes Sharpe. Returns distribution statistics.
    """
    fold_boundaries = generate_fold_boundaries(n_bars, config.n_folds)
    paths = generate_cpcv_paths(config.n_folds, config.n_test_folds)

    path_sharpes = np.full(len(paths), np.nan)
    path_trade_counts = np.zeros(len(paths), dtype=np.int32)

    for i, test_folds in enumerate(paths):
        test_mask = build_test_mask(
            n_bars, fold_boundaries, test_folds, config.purge_bars
        )
        trade_mask = filter_trades_by_mask(
            trade_entry_bars, trade_exit_bars, test_mask
        )
        selected_pnls = trade_pnls[trade_mask]

        path_trade_counts[i] = len(selected_pnls)
        if len(selected_pnls) >= config.min_trades_per_path:
            path_sharpes[i] = compute_sharpe_from_pnls(selected_pnls)

    valid = ~np.isnan(path_sharpes)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return {
            "median_sharpe": 0.0,
            "mean_sharpe": 0.0,
            "std_sharpe": 0.0,
            "min_sharpe": 0.0,
            "max_sharpe": 0.0,
            "pct_positive": 0.0,
            "n_paths": len(paths),
            "n_valid_paths": 0,
            "path_sharpes": path_sharpes,
            "path_trade_counts": path_trade_counts,
        }

    valid_sharpes = path_sharpes[valid]
    return {
        "median_sharpe": float(np.median(valid_sharpes)),
        "mean_sharpe": float(np.mean(valid_sharpes)),
        "std_sharpe": float(np.std(valid_sharpes)),
        "min_sharpe": float(np.min(valid_sharpes)),
        "max_sharpe": float(np.max(valid_sharpes)),
        "pct_positive": float((valid_sharpes > 0).sum() / n_valid * 100),
        "n_paths": len(paths),
        "n_valid_paths": n_valid,
        "path_sharpes": path_sharpes,
        "path_trade_counts": path_trade_counts,
    }
