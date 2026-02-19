"""OLS Rolling hedge ratio estimator."""

import numpy as np
import pandas as pd

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


class OLSRollingEstimator(HedgeRatioEstimator):
    """Rolling OLS regression for dynamic hedge ratio.

    β(t) = Cov(Y, X)[t-w:t] / Var(X)[t-w:t]
    """

    def __init__(self, window: int = 120, zscore_window: int = 60):
        self.window = window
        self.zscore_window = zscore_window

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        y = aligned.df["close_a"]
        x = aligned.df["close_b"]

        # Rolling beta
        cov = y.rolling(self.window).cov(x)
        var = x.rolling(self.window).var()
        beta = (cov / var).replace([np.inf, -np.inf], np.nan)

        # Spread
        spread = y - beta * x

        # Z-score (rolling mean/std on spread)
        mu = spread.rolling(self.zscore_window).mean()
        sigma = spread.rolling(self.zscore_window).std()
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan)

        return HedgeResult(
            beta=beta,
            spread=spread,
            zscore=zscore,
            method="ols_rolling",
            params={"window": self.window, "zscore_window": self.zscore_window},
        )
