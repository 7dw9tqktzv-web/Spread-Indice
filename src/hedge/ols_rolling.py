"""OLS Rolling hedge ratio estimator on log-prices."""

from dataclasses import dataclass

import numpy as np

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


@dataclass(frozen=True)
class OLSRollingConfig:
    """Configuration for OLS Rolling estimator."""
    window: int = 7200
    zscore_window: int = 12


class OLSRollingEstimator(HedgeRatioEstimator):
    """Rolling OLS regression on log-prices for dynamic hedge ratio.

    Convention: log_a = α + β × log_b + ε  (leg_a dependent, leg_b explanatory)
    β = Cov(log_a, log_b) / Var(log_b)
    α = mean(log_a) - β × mean(log_b)
    Spread = log_a - β × log_b - α  (OLS residual)
    Z-score = (Spread - μ) / σ
    """

    def __init__(self, config: OLSRollingConfig | None = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = OLSRollingConfig(**kwargs)
        self.window = self.config.window
        self.zscore_window = self.config.zscore_window

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        log_a = np.log(aligned.df["close_a"])
        log_b = np.log(aligned.df["close_b"])

        # Rolling OLS: log_a = α + β × log_b + ε
        # β = Cov(log_a, log_b) / Var(log_b)
        cov = log_a.rolling(self.window).cov(log_b)
        var = log_b.rolling(self.window).var()
        beta = (cov / var).replace([np.inf, -np.inf], np.nan)

        # Alpha (intercept)
        mean_a = log_a.rolling(self.window).mean()
        mean_b = log_b.rolling(self.window).mean()
        alpha = mean_a - beta * mean_b

        # Spread = OLS residual on log-prices
        spread = log_a - beta * log_b - alpha

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
