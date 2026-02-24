"""Tests for neighborhood robustness module."""

import pytest

from src.validation.neighborhood import (
    get_neighbor_configs,
    compute_neighborhood_robustness,
    NeighborhoodResult,
)


class TestGetNeighborConfigs:
    def test_basic_two_params(self):
        center = {"z_entry": 2.75, "z_exit": 0.75}
        axes = {
            "z_entry": [2.25, 2.50, 2.75, 3.00, 3.25],
            "z_exit": [0.25, 0.50, 0.75, 1.00, 1.25],
        }
        neighbors = get_neighbor_configs(center, axes)
        assert len(neighbors) == 4  # 2 per param

    def test_edge_of_grid(self):
        center = {"z_entry": 2.25}
        axes = {"z_entry": [2.25, 2.50, 2.75]}
        neighbors = get_neighbor_configs(center, axes)
        assert len(neighbors) == 1  # only +1 step

    def test_center_not_in_grid(self):
        center = {"z_entry": 2.60}  # not in axes
        axes = {"z_entry": [2.25, 2.50, 2.75]}
        neighbors = get_neighbor_configs(center, axes)
        assert len(neighbors) == 0

    def test_l1_neighborhood(self):
        """Only one param changes at a time."""
        center = {"a": 2, "b": 5}
        axes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        neighbors = get_neighbor_configs(center, axes)
        for n in neighbors:
            changes = sum(1 for k in center if n[k] != center[k])
            assert changes == 1


class TestNeighborhoodRobustness:
    def test_robust_config(self):
        result = compute_neighborhood_robustness(
            center_sharpe=1.5,
            neighbor_sharpes=[1.4, 1.3, 1.2, 1.1],
            neighbor_pfs=[1.5, 1.3, 1.2, 1.1],
        )
        assert result.is_robust
        assert result.pct_profitable == 100.0
        assert result.sharpe_degradation_pct < 50.0

    def test_not_robust_config(self):
        result = compute_neighborhood_robustness(
            center_sharpe=2.0,
            neighbor_sharpes=[0.1, -0.2, 0.3, -0.1],
            neighbor_pfs=[0.8, 0.5, 1.1, 0.7],
        )
        assert not result.is_robust

    def test_no_neighbors(self):
        result = compute_neighborhood_robustness(1.5, [], [])
        assert not result.is_robust
        assert result.n_neighbors == 0
