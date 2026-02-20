"""Z-score threshold crossing signal generator (stateful loop)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
        z = zscore.values
        n = len(z)
        signals = np.zeros(n, dtype=np.int8)
        state = _FLAT

        for t in range(n):
            zt = z[t]

            # NaN → force flat (reset cooldown too)
            if np.isnan(zt):
                state = _FLAT
                signals[t] = _FLAT
                continue

            if state == _FLAT:
                if zt < -self.config.z_entry:
                    state = _LONG
                elif zt > self.config.z_entry:
                    state = _SHORT

            elif state == _LONG:
                if zt > -self.config.z_exit:
                    state = _FLAT  # mean-reversion exit
                elif zt < -self.config.z_stop:
                    state = _COOLDOWN  # stop → cooldown

            elif state == _SHORT:
                if zt < self.config.z_exit:
                    state = _FLAT  # mean-reversion exit
                elif zt > self.config.z_stop:
                    state = _COOLDOWN  # stop → cooldown

            elif state == _COOLDOWN:
                if abs(zt) < self.config.z_exit:
                    state = _FLAT  # spread returned to neutral

            # COOLDOWN emits 0 (flat) — it's an internal state
            signals[t] = state if state in (_LONG, _SHORT) else _FLAT

        return pd.Series(signals, index=zscore.index, name="signal")
