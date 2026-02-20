"""Dollar-neutral position sizing with beta adjustment.

Implements the ACSIL v1.4 sizing formula:
    Notional_A = Price_A × PointValue_A
    Notional_B = Price_B × PointValue_B
    N_b = round((Notional_A / Notional_B) × β × N_a)

Leg A is fixed at N_a contracts (default 1). Leg B is computed dynamically.
"""

import numpy as np
import pandas as pd


# --- Scalar functions (single bar) ---

def calculate_position_size(
    price_a: float,
    price_b: float,
    beta: float,
    pv_a: float,
    pv_b: float,
    n_a: int = 1,
) -> tuple[int, int, float]:
    """Compute contract counts for a dollar-neutral beta-weighted spread.

    Returns (n_a, n_b, n_b_raw) where n_b is rounded to nearest integer (min 1).
    """
    notional_a = price_a * pv_a
    notional_b = price_b * pv_b
    n_b_raw = (notional_a / notional_b) * beta * n_a if notional_b != 0 else 0.0
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
