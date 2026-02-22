"""Base interface and result dataclass for hedge ratio estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from src.data.alignment import AlignedPair


@dataclass
class HedgeResult:
    """Output of a hedge ratio estimation.

    Convention: log_a = α + β × log_b + ε  (leg_a dependent, leg_b explanatory)
    Spread S(t) = log_a(t) - β(t) × log_b(t)  (OLS residual on log-prices)
    """
    beta: pd.Series          # dynamic hedge ratio β(t)
    spread: pd.Series        # S(t) = log_a - β(t) × log_b
    zscore: pd.Series        # z-score of the spread
    method: str              # hedge method name
    params: dict = field(default_factory=dict)       # parameters used
    diagnostics: dict = field(default_factory=dict)  # optional time series (P_trace, K_beta, R_history)


class HedgeRatioEstimator(ABC):
    """Abstract base class for hedge ratio estimators."""

    @abstractmethod
    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        """Estimate hedge ratio, spread, and z-score from aligned pair data."""
        ...
