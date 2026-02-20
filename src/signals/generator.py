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


class SignalGenerator:
    """Generate entry/exit/stop signals from z-score via stateful state machine.

    States: FLAT (0), LONG (+1), SHORT (-1)

    Transitions:
        FLAT  → LONG   if z < -z_entry
        FLAT  → SHORT  if z > +z_entry
        LONG  → FLAT   if z > -z_exit (mean-reversion exit) or z < -z_stop (stop)
        SHORT → FLAT   if z < +z_exit (mean-reversion exit) or z > +z_stop (stop)

    NaN in zscore forces FLAT (closes any open position).
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

            # NaN → force flat
            if np.isnan(zt):
                state = _FLAT
                signals[t] = _FLAT
                continue

            if state == _FLAT:
                if zt < -self.config.z_entry:
                    state = _LONG
                elif zt > self.config.z_entry:
                    state = _SHORT
                # else: stay flat

            elif state == _LONG:
                if zt > -self.config.z_exit:
                    state = _FLAT  # mean-reversion exit
                elif zt < -self.config.z_stop:
                    state = _FLAT  # stop loss

            elif state == _SHORT:
                if zt < self.config.z_exit:
                    state = _FLAT  # mean-reversion exit
                elif zt > self.config.z_stop:
                    state = _FLAT  # stop loss

            signals[t] = state

        return pd.Series(signals, index=zscore.index, name="signal")
