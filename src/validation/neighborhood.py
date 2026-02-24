"""Neighborhood robustness check for parameter stability.

For a given config, tests all immediate neighbors in the parameter grid
(+/- 1 step in each dimension). A robust config should have neighbors
that are also profitable with similar Sharpe ratios.

A config on an isolated peak (high Sharpe but neighbors are poor) is likely
overfitted to a specific parameter combination.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeighborhoodResult:
    """Result of a neighborhood robustness check."""

    center_sharpe: float
    n_neighbors: int
    n_profitable: int
    pct_profitable: float
    mean_neighbor_sharpe: float
    min_neighbor_sharpe: float
    sharpe_degradation_pct: float  # (center - mean_neighbor) / |center| * 100
    is_robust: bool  # >= 60% profitable AND degradation < 50%


def get_neighbor_configs(
    center: dict,
    grid_axes: dict[str, list],
) -> list[dict]:
    """Generate all immediate neighbors of center config in the grid.

    For each parameter, try +/- 1 step in the grid axis.
    Only one parameter changes at a time (L1 neighborhood).

    Parameters
    ----------
    center : dict
        Parameter values for the center config.
    grid_axes : dict
        Maps parameter name to sorted list of grid values.

    Returns
    -------
    list of dict : neighbor configs (excluding the center itself).
    """
    neighbors = []
    for param, values in grid_axes.items():
        if param not in center:
            continue
        current = center[param]
        try:
            idx = values.index(current)
        except ValueError:
            continue

        for delta in [-1, +1]:
            new_idx = idx + delta
            if 0 <= new_idx < len(values):
                neighbor = dict(center)
                neighbor[param] = values[new_idx]
                if neighbor != center:
                    neighbors.append(neighbor)

    return neighbors


def compute_neighborhood_robustness(
    center_sharpe: float,
    neighbor_sharpes: list[float],
    neighbor_pfs: list[float],
) -> NeighborhoodResult:
    """Compute robustness metrics from center and its neighbors.

    Parameters
    ----------
    center_sharpe : float
        CPCV median Sharpe of the center config.
    neighbor_sharpes : list of float
        CPCV median Sharpe of each neighbor.
    neighbor_pfs : list of float
        Profit factor of each neighbor (for profitability check).
    """
    n = len(neighbor_sharpes)
    if n == 0:
        return NeighborhoodResult(
            center_sharpe=center_sharpe,
            n_neighbors=0,
            n_profitable=0,
            pct_profitable=0.0,
            mean_neighbor_sharpe=0.0,
            min_neighbor_sharpe=0.0,
            sharpe_degradation_pct=100.0,
            is_robust=False,
        )

    n_profitable = sum(1 for pf in neighbor_pfs if pf > 1.0)
    pct_profitable = n_profitable / n * 100

    arr = np.array(neighbor_sharpes)
    mean_ns = float(arr.mean())
    min_ns = float(arr.min())

    degradation = 0.0
    if abs(center_sharpe) > 1e-8:
        degradation = (center_sharpe - mean_ns) / abs(center_sharpe) * 100

    is_robust = pct_profitable >= 60.0 and degradation < 50.0

    return NeighborhoodResult(
        center_sharpe=center_sharpe,
        n_neighbors=n,
        n_profitable=n_profitable,
        pct_profitable=round(pct_profitable, 1),
        mean_neighbor_sharpe=round(mean_ns, 4),
        min_neighbor_sharpe=round(min_ns, 4),
        sharpe_degradation_pct=round(degradation, 1),
        is_robust=is_robust,
    )
