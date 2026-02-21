"""Dollar-neutral position sizing with beta adjustment.

Implements the ACSIL v1.4 sizing formula:
    Notional_A = Price_A × PointValue_A
    Notional_B = Price_B × PointValue_B
    N_b = round((Notional_A / Notional_B) × β × N_a)

Leg A is fixed at N_a contracts (default 1). Leg B is computed dynamically.
"""

import numpy as np
import pandas as pd


# --- Optimal lot multiplier ---

def find_optimal_multiplier(
    n_b_raw: float,
    max_multiplier: int = 3,
    max_hedge_error: float = 5.0,
) -> tuple[int, int, int, float]:
    """Find the multiplier (1..max) that minimizes rounding error on n_b.

    Parameters
    ----------
    n_b_raw : float
        Raw (fractional) contract count for leg B when n_a=1.
    max_multiplier : int
        Maximum multiplier to test (default 3).
    max_hedge_error : float
        Maximum acceptable hedge error in % (default 5%).

    Returns
    -------
    (multiplier, n_a, n_b, hedge_error_pct)
    """
    if n_b_raw <= 0:
        return 1, 1, 1, 0.0

    best_m = 1
    best_error = float("inf")
    best_nb = max(round(n_b_raw), 1)

    for m in range(1, max_multiplier + 1):
        nb = max(round(n_b_raw * m), 1)
        # Error = how far the effective ratio (nb/m) is from the ideal (n_b_raw)
        error = abs(nb / m - n_b_raw) / n_b_raw * 100
        if error < best_error:
            best_error = error
            best_m = m
            best_nb = nb

    return best_m, best_m, best_nb, round(best_error, 2)


# --- Scalar functions (single bar) ---

def calculate_position_size(
    price_a: float,
    price_b: float,
    beta: float,
    pv_a: float,
    pv_b: float,
    n_a: int = 1,
    max_multiplier: int = 1,
) -> tuple[int, int, float]:
    """Compute contract counts for a dollar-neutral beta-weighted spread.

    Parameters
    ----------
    max_multiplier : int
        If > 1, find optimal multiplier to minimize rounding error (for micros).
        Default 1 = original behavior (no multiplier).

    Returns (n_a, n_b, n_b_raw) where n_b is rounded to nearest integer (min 1).
    """
    notional_a = price_a * pv_a
    notional_b = price_b * pv_b
    n_b_raw = (notional_a / notional_b) * beta * n_a if notional_b != 0 else 0.0

    if max_multiplier > 1:
        _, n_a_out, n_b_out, _ = find_optimal_multiplier(n_b_raw, max_multiplier)
        return n_a_out, n_b_out, n_b_raw
    else:
        n_b = max(round(n_b_raw), 1)
        return n_a, n_b, n_b_raw


def calculate_spread_dollar(
    price_a: float,
    price_b: float,
    n_a: int,
    n_b: int,
    pv_a: float,
    pv_b: float,
) -> float:
    """Spread in dollar terms: Notional_A - Notional_B."""
    return price_a * pv_a * n_a - price_b * pv_b * n_b


def calculate_imbalance_pct(
    price_a: float,
    price_b: float,
    n_a: int,
    n_b: int,
    pv_a: float,
    pv_b: float,
) -> float:
    """Percentage imbalance between the two legs (0% = perfectly neutral)."""
    not_a = price_a * pv_a * n_a
    not_b = price_b * pv_b * n_b
    avg = (not_a + not_b) / 2
    return abs(not_a - not_b) / avg * 100 if avg > 0 else 0.0


# --- Vectorized functions (full Series for backtest) ---

def calculate_position_size_series(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: pd.Series,
    pv_a: float,
    pv_b: float,
    n_a: int = 1,
) -> pd.DataFrame:
    """Vectorized position sizing over a time series.

    Returns DataFrame with columns: n_a, n_b, n_b_raw, notional_a, notional_b, imbalance_pct.
    """
    notional_a = price_a * pv_a
    notional_b = price_b * pv_b

    n_b_raw = (notional_a / notional_b) * beta * n_a
    n_b_raw = n_b_raw.replace([np.inf, -np.inf], np.nan)
    n_b = n_b_raw.round().clip(lower=1).astype("Int64")

    # Actual notionals with contract counts
    total_a = notional_a * n_a
    total_b = notional_b * n_b
    avg_not = (total_a + total_b) / 2
    imbalance = ((total_a - total_b).abs() / avg_not * 100).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({
        "n_a": n_a,
        "n_b": n_b,
        "n_b_raw": n_b_raw,
        "notional_a": total_a,
        "notional_b": total_b,
        "imbalance_pct": imbalance,
    }, index=price_a.index)


def calculate_spread_dollar_series(
    price_a: pd.Series,
    price_b: pd.Series,
    n_a: int,
    n_b: pd.Series,
    pv_a: float,
    pv_b: float,
) -> pd.Series:
    """Vectorized dollar spread: P_a × PV_a × N_a - P_b × PV_b × N_b."""
    return price_a * pv_a * n_a - price_b * pv_b * n_b
