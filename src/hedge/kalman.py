"""Kalman Filter hedge ratio estimator on log-prices with innovation z-score.

State-space model:
    State:       θ(t) = [α(t), β(t)]  (random walk)
    Observation: ln(P_a) = α + β × ln(P_b) + ε

Z-score:  z(t) = ν(t) / √F(t)   (innovation-based, auto-adaptive)
Spread:   S(t) = ln(P_a) - β(t) × ln(P_b)  (residual after update)

Features:
    - Log-prices (consistent with OLS)
    - R estimated from OLS residual variance on first `warmup` bars
    - P enlarged at session boundaries (overnight gap handling)
    - Joseph form for numerical stability
"""

import numpy as np
import pandas as pd

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


def _detect_session_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """Detect bars where a new session begins (gap > 30min from previous bar)."""
    if len(index) < 2:
        return np.array([], dtype=int)
    diffs = np.diff(index.asi8) / 1e9 / 60  # minutes between bars
    # A gap > 30min indicates a new session
    session_starts = np.where(diffs > 30)[0] + 1  # +1 because diff shifts by 1
    return session_starts


class KalmanEstimator(HedgeRatioEstimator):
    """Dynamic hedge ratio via Kalman filter on log-prices.

    Parameters
    ----------
    alpha_ratio : float
        State noise ratio. Q = alpha_ratio × R × I.
        Controls adaptation speed of beta. Default 1e-4.
    warmup : int
        Number of initial bars to mask as unreliable. Default 100.
    gap_P_multiplier : float
        Factor to enlarge P at session boundaries. Default 10.0.
    """

    def __init__(self, alpha_ratio: float = 1e-5, warmup: int = 100, gap_P_multiplier: float = 10.0):
        self.alpha_ratio = alpha_ratio
        self.warmup = warmup
        self.gap_P_multiplier = gap_P_multiplier

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        log_a = np.log(aligned.df["close_a"].values)
        log_b = np.log(aligned.df["close_b"].values)
        n = len(log_a)
        idx = aligned.df.index

        # --- Estimate R from OLS residual variance ---
        # Use min 1000 bars (or all if fewer) for a stable estimate
        w_r = min(max(1000, self.warmup), n)
        if w_r > 20:
            x_init = np.column_stack([np.ones(w_r), log_b[:w_r]])
            try:
                params = np.linalg.lstsq(x_init, log_a[:w_r], rcond=None)[0]
                residuals = log_a[:w_r] - x_init @ params
                R = float(np.var(residuals))
                if R < 1e-8:
                    R = 1e-5
            except np.linalg.LinAlgError:
                R = 1e-5
        else:
            R = 1e-5

        # --- Detect session boundaries for P reset ---
        session_starts = set(_detect_session_starts(idx))

        # --- Kalman filter ---
        theta = np.array([0.0, 1.0])  # [alpha, beta]
        P = np.eye(2)
        Q = np.eye(2) * self.alpha_ratio * R  # scale-aware: Q proportional to R

        betas = np.empty(n)
        spreads = np.empty(n)
        zscores = np.empty(n)

        for t in range(n):
            # --- Reset P at session boundaries ---
            if t in session_starts:
                P = P * self.gap_P_multiplier

            H = np.array([1.0, log_b[t]])

            # --- Predict ---
            P = P + Q

            # --- Innovation ---
            y_pred = H @ theta
            nu = log_a[t] - y_pred
            F = H @ P @ H + R

            # --- Z-score (innovation-based) ---
            zscores[t] = nu / np.sqrt(F) if F > 0 else 0.0

            # --- Update (Joseph form) ---
            K = P @ H / F
            theta = theta + K * nu
            I_KH = np.eye(2) - np.outer(K, H)
            P = I_KH @ P @ I_KH.T + np.outer(K, K) * R

            betas[t] = theta[1]
            spreads[t] = log_a[t] - theta[1] * log_b[t]

        # Warm-up: mask first N bars
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
            params={
                "alpha_ratio": self.alpha_ratio,
                "R_estimated": R,
                "warmup": self.warmup,
                "gap_P_multiplier": self.gap_P_multiplier,
            },
        )
