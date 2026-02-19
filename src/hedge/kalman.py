"""Kalman Filter hedge ratio estimator with innovation-based z-score."""

import numpy as np
import pandas as pd

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


class KalmanEstimator(HedgeRatioEstimator):
    """Dynamic hedge ratio via Kalman filter.

    State: [alpha, beta] with random walk transition.
    Z-score: innovation-based  z(t) = ν(t) / √F(t)  (auto-adaptive, no window).
    """

    def __init__(self, delta: float = 1e-4, obs_noise: float = 1e-3, warmup: int = 100):
        self.delta = delta
        self.obs_noise = obs_noise
        self.warmup = warmup

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        y = aligned.df["close_a"].values
        x = aligned.df["close_b"].values
        n = len(y)
        idx = aligned.df.index

        # State [alpha, beta]
        theta = np.array([0.0, 1.0])
        P = np.eye(2)
        Q = np.eye(2) * self.delta / (1.0 - self.delta)
        R = self.obs_noise

        betas = np.empty(n)
        spreads = np.empty(n)
        zscores = np.empty(n)

        for t in range(n):
            H = np.array([1.0, x[t]])

            # --- Predict ---
            # theta unchanged (random walk); P grows
            P = P + Q

            # --- Innovation ---
            y_pred = H @ theta
            nu = y[t] - y_pred                     # innovation
            F = H @ P @ H + R                      # innovation variance (scalar)

            # --- Z-score (innovation-based) ---
            zscores[t] = nu / np.sqrt(F) if F > 0 else 0.0

            # --- Update (Joseph form for numerical stability) ---
            K = P @ H / F                           # Kalman gain (2,)
            theta = theta + K * nu
            I_KH = np.eye(2) - np.outer(K, H)
            P = I_KH @ P @ I_KH.T + np.outer(K, K) * R

            betas[t] = theta[1]
            spreads[t] = y[t] - theta[1] * x[t]  # actual spread, not innovation

        # Warm-up: mask first N bars as unreliable
        betas[:self.warmup] = np.nan
        spreads[:self.warmup] = np.nan
        zscores[:self.warmup] = np.nan

        beta_s = pd.Series(betas, index=idx, name="beta")
        spread_s = pd.Series(spreads, index=idx, name="spread")
        zscore_s = pd.Series(zscores, index=idx, name="zscore")

        return HedgeResult(
            beta=beta_s,
            spread=spread_s,
            zscore=zscore_s,
            method="kalman",
            params={"delta": self.delta, "obs_noise": self.obs_noise, "warmup": self.warmup},
        )
