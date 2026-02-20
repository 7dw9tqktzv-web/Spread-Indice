"""Aggregate all statistical metrics into a single DataFrame.

This is the metrics aggregation layer. Low-level computation lives in src/stats/.
"""

from dataclasses import dataclass

import pandas as pd

from src.stats.stationarity import adf_rolling
from src.stats.hurst import hurst_rolling
from src.stats.halflife import half_life_rolling
from src.stats.correlation import rolling_correlation


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for metrics computation windows."""
    adf_window: int = 24
    hurst_window: int = 64
    halflife_window: int = 24
    correlation_window: int = 12


def compute_all_metrics(
    spread: pd.Series,
    close_a: pd.Series,
    close_b: pd.Series,
    config: MetricsConfig,
) -> pd.DataFrame:
    """Compute all statistical metrics and return as a single DataFrame.

    Parameters
    ----------
    spread : pd.Series
        Spread residual series (from HedgeResult.spread).
    close_a, close_b : pd.Series
        Raw close prices for each leg (not log-prices — log transform done inside).
    config : MetricsConfig
        Window sizes for each metric.

    Returns
    -------
    pd.DataFrame
        Columns: adf_stat, hurst, half_life, correlation.
        Indexed by timestamp.
    """
    return pd.DataFrame(
        {
            "adf_stat": adf_rolling(spread, window=config.adf_window),
            "hurst": hurst_rolling(spread, window=config.hurst_window),
            "half_life": half_life_rolling(spread, window=config.halflife_window),
            "correlation": rolling_correlation(close_a, close_b, window=config.correlation_window),
        },
        index=spread.index,
    )
