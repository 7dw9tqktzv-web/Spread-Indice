"""Tests for binary gate filter module."""

import numpy as np

from src.validation.gates import apply_gate_filter_numba


class TestApplyGateFilter:
    def test_entry_blocked_when_gate_fails(self):
        sig = np.array([0, 1, 1, 0, -1, -1, 0], dtype=np.int8)
        gate = np.array([True, False, True, True, True, True, True])
        result = apply_gate_filter_numba(sig, gate)
        assert result[1] == 0  # entry blocked
        assert result[4] == -1  # entry allowed (gate True)

    def test_exit_never_blocked(self):
        sig = np.array([0, 1, 1, 0], dtype=np.int8)
        gate = np.array([True, True, True, False])
        result = apply_gate_filter_numba(sig, gate)
        assert result[1] == 1  # entry
        assert result[3] == 0  # exit preserved (not an entry)

    def test_all_gates_pass(self):
        sig = np.array([0, 1, 1, 0, -1, 0], dtype=np.int8)
        gate = np.ones(6, dtype=np.bool_)
        result = apply_gate_filter_numba(sig, gate)
        np.testing.assert_array_equal(sig, result)

    def test_all_gates_fail(self):
        sig = np.array([0, 1, 1, 0, -1, 0], dtype=np.int8)
        gate = np.zeros(6, dtype=np.bool_)
        result = apply_gate_filter_numba(sig, gate)
        np.testing.assert_array_equal(result, 0)

    def test_blocked_entry_does_not_create_phantom_exit(self):
        """If entry is blocked at t=1, t=2's value should not be treated as exit."""
        sig = np.array([0, 1, 0, -1, 0], dtype=np.int8)
        gate = np.array([True, False, True, True, True])
        result = apply_gate_filter_numba(sig, gate)
        # t=1 blocked (was entry), t=2 is 0 (no position)
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == -1  # new entry, gate passes
