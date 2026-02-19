"""Dollar Neutral hedge ratio estimator."""

import numpy as np
import pandas as pd

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


class DollarNeutralEstimator(HedgeRatioEstimator):
    """Equal notional value on both legs.

    β(t) = (Price_A × Multiplier_A) / (Price_B × Multiplier_B)
    """

    def __init__(
        self,
        multiplier_a: float,
        multiplier_b: float,
        zscore_window: int = 60,
    ):
        self.multiplier_a = multiplier_a
        self.multiplier_b = multiplier_b
        self.zscore_window = zscore_window

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        y = aligned.df["close_a"]
        x = aligned.df["close_b"]

        # Dynamic dollar-neutral beta
        beta = ((y * self.multiplier_a) / (x * self.multiplier_b)).replace([np.inf, -np.inf], np.nan)

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
            method="dollar_neutral",
            params={
                "multiplier_a": self.multiplier_a,
                "multiplier_b": self.multiplier_b,
                "zscore_window": self.zscore_window,
            },
        )
