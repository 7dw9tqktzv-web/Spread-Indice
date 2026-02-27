"""Unit tests for signal generator (stateful z-score crossing)."""

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

    def test_stop_loss_long_cooldown(self, gen):
        # Stop → cooldown → z still at -2.5 (can't re-enter) → z returns to 0 → re-entry
        z = _make_zscore([0.0, -2.5, -3.0, -4.5, -2.5, -1.0, 0.3, -2.5])
        sig = gen.generate(z)
        assert sig.iloc[1] == 1   # entry long
        assert sig.iloc[2] == 1   # hold
        assert sig.iloc[3] == 0   # stop → cooldown (emits 0)
        assert sig.iloc[4] == 0   # cooldown: |z|=2.5 > z_exit, blocked
        assert sig.iloc[5] == 0   # cooldown: |z|=1.0 > z_exit, blocked
        assert sig.iloc[6] == 0   # cooldown → flat: |z|=0.3 < z_exit, neutral reached
        assert sig.iloc[7] == 1   # re-entry allowed (flat + z < -2)

    def test_stop_loss_short_cooldown(self, gen):
        # Symmetric for short
        z = _make_zscore([0.0, 2.5, 3.0, 4.5, 2.5, 1.0, 0.3, 2.5])
        sig = gen.generate(z)
        assert sig.iloc[1] == -1  # entry short
        assert sig.iloc[2] == -1  # hold
        assert sig.iloc[3] == 0   # stop → cooldown
        assert sig.iloc[4] == 0   # cooldown: blocked
        assert sig.iloc[5] == 0   # cooldown: blocked
        assert sig.iloc[6] == 0   # cooldown → flat: neutral reached
        assert sig.iloc[7] == -1  # re-entry allowed

    def test_cooldown_blocks_reentry(self, gen):
        # After stop, z stays extreme → no re-entry even if z crosses entry threshold
        z = _make_zscore([0.0, -2.5, -4.5, -3.0, -2.5])
        sig = gen.generate(z)
        assert sig.iloc[1] == 1   # entry long
        assert sig.iloc[2] == 0   # stop → cooldown
        assert sig.iloc[3] == 0   # cooldown: |z|=3.0 > z_exit
        assert sig.iloc[4] == 0   # cooldown: |z|=2.5 > z_exit, still blocked

    def test_nan_resets_cooldown(self, gen):
        # NaN during cooldown resets to flat (not cooldown)
        z = _make_zscore([0.0, -2.5, -4.5, float("nan"), -2.5])
        sig = gen.generate(z)
        assert sig.iloc[2] == 0   # stop → cooldown
        assert sig.iloc[3] == 0   # NaN → flat (not cooldown)
        assert sig.iloc[4] == 1   # re-entry allowed (flat, not cooldown)

    def test_no_double_entry(self, gen):
        # Already in position, stays in position (no re-entry signal)
        z = _make_zscore([-2.5, -2.8, -3.0, -2.5])
        sig = gen.generate(z)
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
