"""Rolling correlation between two price series (on log-prices)."""

import numpy as np
import pandas as pd


def rolling_correlation(
    close_a: pd.Series,
    close_b: pd.Series,
    window: int = 12,
) -> pd.Series:
    """Rolling Pearson correlation between log-prices of two instruments."""
    log_a = np.log(close_a)
    log_b = np.log(close_b)
    return log_a.rolling(window).corr(log_b)
