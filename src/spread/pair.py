"""SpreadPair dataclass — metadata for a trading pair."""

from dataclasses import dataclass

from src.utils.constants import Instrument


@dataclass(frozen=True)
class SpreadPair:
    """Defines a spread pair: leg_a - ratio * leg_b."""
    leg_a: Instrument
    leg_b: Instrument

    @property
    def name(self) -> str:
        return f"{self.leg_a.value}_{self.leg_b.value}"

    def __str__(self) -> str:
        return f"{self.leg_a.value}/{self.leg_b.value}"
