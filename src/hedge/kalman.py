"""Kalman Filter hedge ratio estimator on log-prices with innovation z-score.

State-space model:
    State:       θ(t) = [α(t), β(t)]  (random walk)
    Observation: ln(P_a) = α + β × ln(P_b) + ε

Z-score:  z(t) = ν(t) / √F(t)   (innovation-based, auto-adaptive)
Spread:   S(t) = ln(P_a) - β(t) × ln(P_b)  (residual after update)

Features:
    - Log-prices (consistent with OLS)
    - R estimated from OLS residual variance on first `warmup` bars
    - Optional adaptive R via EWMA on squared innovations (r_ewma_span > 0)
    - Optional adaptive Q that tracks R changes (adaptive_Q=True)
    - P enlarged at session boundaries (overnight gap handling)
    - Joseph form for numerical stability (numba-compiled, scalar 2x2 math)
    - Diagnostics: P_trace(t), K_beta(t), R_history(t)
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from numba import njit

from src.data.alignment import AlignedPair
from src.hedge.base import HedgeRatioEstimator, HedgeResult


@dataclass(frozen=True)
class KalmanConfig:
    """Configuration for Kalman filter estimator."""
    alpha_ratio: float = 1e-5
    warmup: int = 100
    gap_P_multiplier: float = 10.0
    r_ewma_span: int = 0        # 0 = fixed R (current behavior), >0 = EWMA span in bars
    adaptive_Q: bool = False     # True = Q tracks R changes, False = Q fixed on initial R


def _detect_session_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """Detect bars where a new session begins (gap > 30min from previous bar)."""
    if len(index) < 2:
        return np.array([], dtype=int)
    diffs = np.diff(index.asi8) / 1e9 / 60  # minutes between bars
    # A gap > 30min indicates a new session
    session_starts = np.where(diffs > 30)[0] + 1  # +1 because diff shifts by 1
    return session_starts


@njit(cache=True)
def _kalman_loop(log_a, log_b, R_init, q_scalar_init, gap_P_mult,
                 session_starts_arr, r_adaptive, r_lambda, adaptive_Q,
                 alpha_ratio):
    """Numba-compiled Kalman filter loop with scalar 2x2 math.

    All 2x2 matrix operations are inlined as scalar ops for zero-allocation
    inner loop. ~50-200x faster than Python+numpy version.

    State: theta = [th0, th1] = [alpha, beta]
    Covariance P as 4 scalars: P00, P01, P10, P11
    """
    n = len(log_a)
    betas = np.empty(n)
    spreads = np.empty(n)
    zscores = np.empty(n)
    p_traces = np.empty(n)
    k_betas = np.empty(n)
    r_values = np.empty(n)

    # State initialization
    th0, th1 = 0.0, 1.0  # theta = [alpha, beta]
    P00, P01, P10, P11 = 1.0, 0.0, 0.0, 1.0
    R = R_init
    q_scalar = q_scalar_init

    # Convert session_starts to a set-like lookup (sorted array + binary search)
    n_sess = len(session_starts_arr)
    sess_idx = 0  # pointer into sorted session_starts_arr

    for t in range(n):
        # --- Reset P at session boundaries ---
        if sess_idx < n_sess and session_starts_arr[sess_idx] == t:
            P00 *= gap_P_mult
            P01 *= gap_P_mult
            P10 *= gap_P_mult
            P11 *= gap_P_mult
            sess_idx += 1

        # H = [1.0, log_b[t]]
        h0 = 1.0
        h1 = log_b[t]

        # --- Predict: P += Q (diagonal) ---
        P00 += q_scalar
        P11 += q_scalar

        # --- Innovation ---
        # y_pred = H @ theta = h0*th0 + h1*th1
        y_pred = h0 * th0 + h1 * th1
        nu = log_a[t] - y_pred

        # F = H @ P @ H^T + R (scalar)
        # = h0*(P00*h0 + P01*h1) + h1*(P10*h0 + P11*h1) + R
        F = h0 * (P00 * h0 + P01 * h1) + h1 * (P10 * h0 + P11 * h1) + R

        # --- Adaptive R ---
        if r_adaptive:
            R = r_lambda * R + (1.0 - r_lambda) * nu * nu
            if R < 1e-8:
                R = 1e-8
            if adaptive_Q:
                q_scalar = alpha_ratio * R

        # --- Z-score ---
        if F > 0.0:
            zscores[t] = nu / sqrt(F)
        else:
            zscores[t] = 0.0

        # --- Kalman gain: K = P @ H / F ---
        # K[0] = (P00*h0 + P01*h1) / F
        # K[1] = (P10*h0 + P11*h1) / F
        k0 = (P00 * h0 + P01 * h1) / F
        k1 = (P10 * h0 + P11 * h1) / F

        # --- Update theta ---
        th0 = th0 + k0 * nu
        th1 = th1 + k1 * nu

        # --- Joseph form update: P = (I - K@H) @ P @ (I - K@H)^T + K@K^T * R ---
        # I_KH = I - outer(K, H)
        # I_KH[0,0] = 1 - k0*h0, I_KH[0,1] = -k0*h1
        # I_KH[1,0] = -k1*h0,    I_KH[1,1] = 1 - k1*h1
        a00 = 1.0 - k0 * h0
        a01 = -k0 * h1
        a10 = -k1 * h0
        a11 = 1.0 - k1 * h1

        # temp = I_KH @ P
        t00 = a00 * P00 + a01 * P10
        t01 = a00 * P01 + a01 * P11
        t10 = a10 * P00 + a11 * P10
        t11 = a10 * P01 + a11 * P11

        # P_new = temp @ I_KH^T + outer(K,K) * R
        P00 = t00 * a00 + t01 * a01 + k0 * k0 * R
        P01 = t00 * a10 + t01 * a11 + k0 * k1 * R
        P10 = t10 * a00 + t11 * a01 + k1 * k0 * R
        P11 = t10 * a10 + t11 * a11 + k1 * k1 * R

        # --- Store outputs ---
        betas[t] = th1
        spreads[t] = log_a[t] - th1 * log_b[t]
        p_traces[t] = P00 + P11
        k_betas[t] = k1
        r_values[t] = R

    return betas, spreads, zscores, p_traces, k_betas, r_values


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

    def __init__(self, config: KalmanConfig | None = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = KalmanConfig(**kwargs)
        self.alpha_ratio = self.config.alpha_ratio
        self.warmup = self.config.warmup
        self.gap_P_multiplier = self.config.gap_P_multiplier
        self.r_ewma_span = self.config.r_ewma_span
        self.adaptive_Q = self.config.adaptive_Q

    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        log_a = np.log(aligned.df["close_a"].values).astype(np.float64)
        log_b = np.log(aligned.df["close_b"].values).astype(np.float64)
        n = len(log_a)
        idx = aligned.df.index

        # --- Estimate R from OLS residual variance ---
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

        R_init = R
        q_scalar = self.alpha_ratio * R

        # --- Detect session boundaries ---
        session_starts = _detect_session_starts(idx)

        # --- Adaptive R config ---
        r_adaptive = self.r_ewma_span > 0
        r_lambda = 1.0 - 2.0 / (self.r_ewma_span + 1) if r_adaptive else 0.0

        # --- Run numba-compiled Kalman loop ---
        betas, spreads, zscores, p_traces, k_betas, r_values = _kalman_loop(
            log_a, log_b, R, q_scalar, self.gap_P_multiplier,
            session_starts, r_adaptive, r_lambda, self.adaptive_Q,
            self.alpha_ratio,
        )

        # Warm-up: mask first N bars
        betas[:self.warmup] = np.nan
        spreads[:self.warmup] = np.nan
        zscores[:self.warmup] = np.nan
        p_traces[:self.warmup] = np.nan
        k_betas[:self.warmup] = np.nan

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
                "R_init": R_init,
                "R_final": float(r_values[-1]) if n > 0 else R_init,
                "warmup": self.warmup,
                "gap_P_multiplier": self.gap_P_multiplier,
                "r_ewma_span": self.r_ewma_span,
                "adaptive_Q": self.adaptive_Q,
            },
            diagnostics={
                "P_trace": pd.Series(p_traces, index=idx, name="P_trace"),
                "K_beta": pd.Series(k_betas, index=idx, name="K_beta"),
                "R_history": pd.Series(r_values, index=idx, name="R_history"),
            },
        )
