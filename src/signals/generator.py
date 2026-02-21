"""Z-score threshold crossing signal generator (stateful loop)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class SignalConfig:
    """Thresholds for z-score based signal generation."""
    z_entry: float = 2.0    # |z| > z_entry → enter position
    z_exit: float = 0.5     # |z| < z_exit → exit position
    z_stop: float = 4.0     # |z| > z_stop → stop loss


# Position states
_FLAT = 0
_LONG = 1
_SHORT = -1
_COOLDOWN = 2  # after stop, wait for z to return to neutral


# ---------------------------------------------------------------------------
# Numba-compiled state machine (~100x faster than Python loop)
# ---------------------------------------------------------------------------

@njit(cache=True)
def generate_signals_numba(z: np.ndarray, z_entry: float, z_exit: float,
                            z_stop: float) -> np.ndarray:
    """Generate signals from z-score via 4-state machine (numba-compiled).

    States: FLAT (0), LONG (+1), SHORT (-1), COOLDOWN (2, emits 0)
    NaN in zscore forces FLAT.

    Returns np.ndarray of int8 {+1, 0, -1}.
    """
    n = len(z)
    signals = np.zeros(n, dtype=np.int8)
    state = 0  # FLAT

    for t in range(n):
        zt = z[t]

        # NaN → force flat
        if np.isnan(zt):
            state = 0
            signals[t] = 0
            continue

        if state == 0:  # FLAT
            if zt < -z_entry:
                state = 1   # LONG
            elif zt > z_entry:
                state = -1  # SHORT

        elif state == 1:  # LONG
            if zt > -z_exit:
                state = 0   # mean-reversion exit
            elif zt < -z_stop:
                state = 2   # COOLDOWN

        elif state == -1:  # SHORT
            if zt < z_exit:
                state = 0   # mean-reversion exit
            elif zt > z_stop:
                state = 2   # COOLDOWN

        elif state == 2:  # COOLDOWN
            if abs(zt) < z_exit:
                state = 0   # spread returned to neutral

        # COOLDOWN emits 0
        if state == 1:
            signals[t] = 1
        elif state == -1:
            signals[t] = -1
        else:
            signals[t] = 0

    return signals


class SignalGenerator:
    """Generate entry/exit/stop signals from z-score via stateful state machine.

    States: FLAT (0), LONG (+1), SHORT (-1), COOLDOWN (internal, emits 0)

    Transitions:
        FLAT     → LONG     if z < -z_entry
        FLAT     → SHORT    if z > +z_entry
        LONG     → FLAT     if z > -z_exit (mean-reversion exit)
        LONG     → COOLDOWN if z < -z_stop (stop loss)
        SHORT    → FLAT     if z < +z_exit (mean-reversion exit)
        SHORT    → COOLDOWN if z > +z_stop (stop loss)
        COOLDOWN → FLAT     if |z| < z_exit (spread returned to neutral)

    NaN in zscore forces FLAT (closes any open position, resets cooldown).
    """

    def __init__(self, config: SignalConfig | None = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = SignalConfig(**kwargs)

    def generate(self, zscore: pd.Series) -> pd.Series:
        """Generate signal series from z-score.

        Returns pd.Series of {+1, 0, -1} indexed like zscore.
        """
        signals = generate_signals_numba(
            zscore.values.astype(np.float64),
            self.config.z_entry,
            self.config.z_exit,
            self.config.z_stop,
        )
        return pd.Series(signals, index=zscore.index, name="signal")
