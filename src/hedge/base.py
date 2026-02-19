"""Base interface and result dataclass for hedge ratio estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from src.data.alignment import AlignedPair


@dataclass
class HedgeResult:
    """Output of a hedge ratio estimation."""
    beta: pd.Series          # dynamic hedge ratio β(t)
    spread: pd.Series        # S(t) = close_a - β(t) * close_b
    zscore: pd.Series        # z-score of the spread
    method: str              # hedge method name
    params: dict = field(default_factory=dict)  # parameters used


class HedgeRatioEstimator(ABC):
    """Abstract base class for hedge ratio estimators."""

    @abstractmethod
    def estimate(self, aligned: AlignedPair) -> HedgeResult:
        """Estimate hedge ratio, spread, and z-score from aligned pair data."""
        ...
