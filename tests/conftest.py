"""Shared fixtures for spread trading tests."""

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import AlignedPair
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument


def make_aligned_pair(
    n: int = 10_000,
    beta_true: float = 1.5,
    noise_std: float = 1e-4,
    freq: str = "5min",
    seed: int = 42,
) -> AlignedPair:
    """Generate synthetic aligned pair where log(close_a) ≈ beta_true * log(close_b) + noise.

    close_b follows a geometric random walk. close_a = close_b^beta_true * exp(noise).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 18:00", periods=n, freq=freq)

    # Geometric random walk for close_b
    log_returns_b = rng.normal(0, 0.001, n)
    log_b = np.cumsum(log_returns_b) + np.log(5000.0)
    close_b = np.exp(log_b)

    # close_a = close_b^beta_true * exp(noise)
    noise = rng.normal(0, noise_std, n)
    log_a = beta_true * log_b + noise
    close_a = np.exp(log_a)

    df = pd.DataFrame(
        {
            "close_a": close_a,
            "close_b": close_b,
            "volume_a": rng.integers(100, 10000, n).astype(float),
            "volume_b": rng.integers(100, 10000, n).astype(float),
        },
        index=idx,
    )

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.ES)
    return AlignedPair(pair=pair, df=df, timeframe="5min")


def make_aligned_pair_with_gap(
    n_before: int = 5000,
    n_after: int = 5000,
    gap_minutes: int = 120,
    beta_true: float = 1.5,
    seed: int = 42,
) -> AlignedPair:
    """Generate aligned pair with a session gap > 30min in the middle."""
    rng = np.random.default_rng(seed)

    idx_before = pd.date_range("2024-01-02 18:00", periods=n_before, freq="5min")
    gap_start = idx_before[-1] + pd.Timedelta(minutes=gap_minutes)
    idx_after = pd.date_range(gap_start, periods=n_after, freq="5min")
    idx = idx_before.append(idx_after)

    n = len(idx)
    log_returns_b = rng.normal(0, 0.001, n)
    log_b = np.cumsum(log_returns_b) + np.log(5000.0)
    close_b = np.exp(log_b)

    noise = rng.normal(0, 1e-4, n)
    log_a = beta_true * log_b + noise
    close_a = np.exp(log_a)

    df = pd.DataFrame(
        {
            "close_a": close_a,
            "close_b": close_b,
            "volume_a": rng.integers(100, 10000, n).astype(float),
            "volume_b": rng.integers(100, 10000, n).astype(float),
        },
        index=idx,
    )

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.ES)
    return AlignedPair(pair=pair, df=df, timeframe="5min")
