"""Volatility Neutral hedge ratio estimator."""

import numpy as np
import pandas as pd

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


class VolatilityNeutralEstimator(HedgeRatioEstimator):
    """Equal risk contribution from both legs.

    β(t) = σ_A(t) / σ_B(t)   where σ = rolling std of returns.
    """

    def __init__(self, vol_window: int = 60, zscore_window: int = 60):
        self.vol_window = vol_window
        self.zscore_window = zscore_window

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        y = aligned.df["close_a"]
        x = aligned.df["close_b"]

        # Rolling realized volatility (std of returns)
        ret_a = y.pct_change()
        ret_b = x.pct_change()
        vol_a = ret_a.rolling(self.vol_window).std()
        vol_b = ret_b.rolling(self.vol_window).std()

        beta = (vol_a / vol_b).replace([np.inf, -np.inf], np.nan)

        # Spread
        spread = y - beta * x

        # Z-score (rolling)
        mu = spread.rolling(self.zscore_window).mean()
        sigma = spread.rolling(self.zscore_window).std()
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan)

        return HedgeResult(
            beta=beta,
            spread=spread,
            zscore=zscore,
            method="volatility_neutral",
            params={"vol_window": self.vol_window, "zscore_window": self.zscore_window},
        )
