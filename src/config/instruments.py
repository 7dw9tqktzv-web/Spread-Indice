"""Instrument specifications loaded from config/instruments.yaml.

Replaces hardcoded MULT_A, MULT_B, TICK_A, TICK_B, COMMISSION constants
scattered across 25+ scripts.

Usage:
    from src.config.instruments import get_instrument_spec, get_pair_specs

    spec = get_instrument_spec("NQ")
    spec_a, spec_b = get_pair_specs("NQ", "YM")

    # Use in backtest:
    # spec_a.multiplier, spec_a.tick_size, spec_a.commission, etc.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract specification for one instrument."""

    name: str
    exchange: str
    multiplier: float  # $/point
    tick_size: float
    tick_value: float
    commission: float  # per side per contract
    margin: float


# Module-level cache
_SPECS: dict[str, InstrumentSpec] | None = None

# Default slippage (ticks) -- universal across all instruments
DEFAULT_SLIPPAGE_TICKS: int = 1
DEFAULT_INITIAL_CAPITAL: float = 100_000.0


def _load_specs() -> dict[str, InstrumentSpec]:
    """Load instrument specs from YAML (cached after first call)."""
    global _SPECS
    if _SPECS is not None:
        return _SPECS

    yaml_path = Path(__file__).resolve().parent.parent.parent / "config" / "instruments.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    _SPECS = {}
    for name, data in raw.items():
        _SPECS[name] = InstrumentSpec(
            name=name,
            exchange=data["exchange"],
            multiplier=float(data["multiplier"]),
            tick_size=float(data["tick_size"]),
            tick_value=float(data["tick_value"]),
            commission=float(data["commission"]),
            margin=float(data["margin"]),
        )
    return _SPECS


def get_instrument_spec(symbol: str) -> InstrumentSpec:
    """Get specification for a single instrument by symbol name.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g. "NQ", "YM", "RTY", "MNQ").

    Returns
    -------
    InstrumentSpec

    Raises
    ------
    KeyError
        If symbol not found in instruments.yaml.
    """
    specs = _load_specs()
    if symbol not in specs:
        raise KeyError(f"Unknown instrument '{symbol}'. Available: {list(specs.keys())}")
    return specs[symbol]


def get_pair_specs(symbol_a: str, symbol_b: str) -> tuple[InstrumentSpec, InstrumentSpec]:
    """Get specs for a pair of instruments.

    Parameters
    ----------
    symbol_a, symbol_b : str
        Instrument symbols (e.g. "NQ", "YM").

    Returns
    -------
    tuple[InstrumentSpec, InstrumentSpec]
        (spec_a, spec_b)
    """
    return get_instrument_spec(symbol_a), get_instrument_spec(symbol_b)
