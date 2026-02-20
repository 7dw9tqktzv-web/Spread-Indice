"""Unit tests for signal generator (stateful z-score crossing)."""

import numpy as np
import pandas as pd
import pytest

from src.signals.generator import SignalConfig, SignalGenerator


def _make_zscore(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name="zscore")


@pytest.fixture
def gen():
    return SignalGenerator(config=SignalConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0))


class TestSignalGenerator:
    def test_long_entry_exit(self, gen):
        # z drops below -2 (entry long), then rises above -0.5 (exit)
        z = _make_zscore([0.0, -1.0, -2.5, -2.3, -1.0, -0.3, 0.0])
        sig = gen.generate(z)
        assert sig.iloc[0] == 0   # flat
        assert sig.iloc[1] == 0   # not yet below -2
        assert sig.iloc[2] == 1   # entry long
        assert sig.iloc[3] == 1   # hold long
        assert sig.iloc[4] == 1   # still below -0.5
        assert sig.iloc[5] == 0   # exit: z > -0.5
        assert sig.iloc[6] == 0   # flat

    def test_short_entry_exit(self, gen):
        # z rises above +2 (entry short), then drops below +0.5 (exit)
        z = _make_zscore([0.0, 1.0, 2.5, 2.3, 1.0, 0.3, 0.0])
        sig = gen.generate(z)
        assert sig.iloc[0] == 0
        assert sig.iloc[1] == 0
        assert sig.iloc[2] == -1  # entry short
        assert sig.iloc[3] == -1  # hold short
        assert sig.iloc[4] == -1  # still above +0.5
        assert sig.iloc[5] == 0   # exit: z < +0.5
        assert sig.iloc[6] == 0

    def test_stop_loss_long(self, gen):
        # z drops below -2 (entry long), then drops below -4 (stop)
        # After stop, z=-1.5 is above -2 so stays flat
        z = _make_zscore([0.0, -2.5, -3.0, -4.5, -1.5])
        sig = gen.generate(z)
        assert sig.iloc[1] == 1   # entry long
        assert sig.iloc[2] == 1   # hold
        assert sig.iloc[3] == 0   # stop: z < -4
        assert sig.iloc[4] == 0   # flat (z > -2, no re-entry)

    def test_stop_loss_short(self, gen):
        # z rises above +2 (entry short), then rises above +4 (stop)
        # After stop, z=+1.5 is below +2 so stays flat
        z = _make_zscore([0.0, 2.5, 3.0, 4.5, 1.5])
        sig = gen.generate(z)
        assert sig.iloc[1] == -1  # entry short
        assert sig.iloc[2] == -1  # hold
        assert sig.iloc[3] == 0   # stop: z > +4
        assert sig.iloc[4] == 0   # flat (z < +2, no re-entry)

    def test_no_double_entry(self, gen):
        # Already in position, stays in position (no re-entry signal)
        z = _make_zscore([-2.5, -2.8, -3.0, -2.5])
        sig = gen.generate(z)
        # All should be +1 (long), no signal change
        assert (sig.values == 1).all()

    def test_nan_closes_position(self, gen):
        z = _make_zscore([-2.5, -2.3, float("nan"), -2.5])
        sig = gen.generate(z)
        assert sig.iloc[0] == 1   # entry long
        assert sig.iloc[1] == 1   # hold
        assert sig.iloc[2] == 0   # NaN → flat
        assert sig.iloc[3] == 1   # re-entry (z < -2 again)

    def test_output_shape(self, gen):
        z = _make_zscore([0.0] * 100)
        sig = gen.generate(z)
        assert len(sig) == 100
        assert sig.name == "signal"
