"""Hybrid 1-second backtest engine: 5min indicators + 1s z-score scanning.

Architecture:
  Layer 1 (5min, Python wrapper): OLS -> beta, spread -> mu, sigma -> gates
  Layer 2 (1s, numba kernel):     z_live scanning, 4-state machine, dollar exits

The kernel uses the Config Vector pattern (fixed signature, extensible params).
All helpers are @njit(inline='always') for zero-overhead function calls.

Indicator timing:
  - beta/alpha/mu/sigma: bar [i] values used during bar i's 1s period.
    Minor look-ahead (~1/7000 for beta, ~1/30 for mu/sigma), keeps sanity check clean.
  - gate: bar [i-1] value (no look-ahead on binary decisions).
"""

import math  # noqa: F401 — used inside @njit functions

import numpy as np
import pandas as pd
from numba import njit

from src.data.alignment import AlignedPair
from src.hedge.factory import create_estimator
from src.validation.gates import GateConfig, compute_gate_mask

# =========================================================================
# Config Vector -- fixed-signature pattern
# =========================================================================

CFG_Z_ENTRY = 0
CFG_Z_EXIT = 1
CFG_Z_STOP = 2
CFG_PNL_TP = 3         # dollar TP (0 = disabled)
CFG_PNL_SL = 4         # dollar SL (0 = disabled)
CFG_ENTRY_START = 5    # minutes from midnight
CFG_ENTRY_END = 6
CFG_FLAT_TIME = 7
CFG_MULT_A = 8
CFG_MULT_B = 9
CFG_TICK_A = 10
CFG_TICK_B = 11
CFG_SLIPPAGE = 12
CFG_COMMISSION = 13
CFG_Z_COOLDOWN = 14     # z threshold to exit COOLDOWN (default = z_exit)
CFG_SIZE = 16


def pack_config(params: dict) -> np.ndarray:
    """Pack config dict into float64 array for numba kernel."""
    cfg = np.zeros(CFG_SIZE, dtype=np.float64)
    cfg[CFG_Z_ENTRY] = float(params["z_entry"])
    cfg[CFG_Z_EXIT] = float(params["z_exit"])
    cfg[CFG_Z_STOP] = float(params["z_stop"])
    cfg[CFG_PNL_TP] = float(params.get("dollar_tp", 0))
    cfg[CFG_PNL_SL] = float(params.get("dollar_sl", 0))
    cfg[CFG_ENTRY_START] = float(params.get("entry_start_min", 120))
    cfg[CFG_ENTRY_END] = float(params.get("entry_end_min", 840))
    cfg[CFG_FLAT_TIME] = float(params.get("flat_min", 930))
    cfg[CFG_MULT_A] = float(params["mult_a"])
    cfg[CFG_MULT_B] = float(params["mult_b"])
    cfg[CFG_TICK_A] = float(params["tick_a"])
    cfg[CFG_TICK_B] = float(params["tick_b"])
    cfg[CFG_SLIPPAGE] = float(params.get("slippage", 1))
    cfg[CFG_COMMISSION] = float(params.get("commission", 2.50))
    cfg[CFG_Z_COOLDOWN] = float(params.get("z_cooldown", params["z_exit"]))
    return cfg


# =========================================================================
# States and exit reasons (integers for numba)
# =========================================================================

_FLAT = 0
_LONG = 1
_SHORT = -1
_COOLDOWN = 2

_TP_ZSCORE = 0
_SL_ZSCORE = 1
_TP_DOLLAR = 2
_SL_DOLLAR = 3
_FLAT_EOD = 4

EXIT_REASON_MAP = {
    0: "TP_ZSCORE",
    1: "SL_ZSCORE",
    2: "TP_DOLLAR",
    3: "SL_DOLLAR",
    4: "FLAT_EOD",
}

# Result columns
_COL_ENTRY_TS = 0
_COL_EXIT_TS = 1
_COL_SIDE = 2
_COL_ENTRY_A = 3
_COL_ENTRY_B = 4
_COL_EXIT_A = 5
_COL_EXIT_B = 6
_COL_N_A = 7
_COL_N_B = 8
_COL_PNL_GROSS = 9
_COL_COSTS = 10
_COL_PNL_NET = 11
_COL_DURATION = 12
_COL_ENTRY_Z = 13
_COL_EXIT_Z = 14
_COL_EXIT_REASON = 15
_COL_MFE = 16
_COL_MAE = 17
_COL_ENTRY_BAR = 18
_COL_EXIT_BAR = 19
_N_COLS = 20
_MAX_TRADES = 5000


# =========================================================================
# Numba inline helpers
# =========================================================================

@njit(cache=True, inline="always")
def _calc_z_live(close_a, close_b, beta, alpha, mu, sigma):
    """Compute live z-score from 1s prices and 5min parameters."""
    spread = math.log(close_a) - beta * math.log(close_b) - alpha
    if sigma == 0.0:
        return 0.0
    return (spread - mu) / sigma


@njit(cache=True, inline="always")
def _calc_size(px_a, px_b, beta, mult_a, mult_b):
    """Dollar-neutral beta-weighted sizing. Returns (n_a, n_b)."""
    not_a = px_a * mult_a
    not_b = px_b * mult_b
    if not_b == 0.0:
        return 1, 1
    ratio = not_a / not_b * abs(beta)
    n_b = max(1, int(ratio + 0.5))  # round
    return 1, n_b


@njit(cache=True, inline="always")
def _calc_pnl_unrealized(side, entry_a, entry_b, curr_a, curr_b, n_a, n_b, cfg):
    """Raw unrealized PnL (entry slippage applied, no exit costs)."""
    da = curr_a - entry_a
    db = curr_b - entry_b
    return side * (n_a * da * cfg[CFG_MULT_A] - n_b * db * cfg[CFG_MULT_B])


@njit(cache=True, inline="always")
def _apply_entry_slippage(px_a, px_b, side, cfg):
    """Apply slippage at entry. Returns (adj_a, adj_b)."""
    slip = cfg[CFG_SLIPPAGE]
    adj_a = px_a + side * slip * cfg[CFG_TICK_A]
    adj_b = px_b - side * slip * cfg[CFG_TICK_B]
    return adj_a, adj_b


@njit(cache=True, inline="always")
def _apply_exit_slippage(px_a, px_b, side, cfg):
    """Apply slippage at exit. Returns (adj_a, adj_b)."""
    slip = cfg[CFG_SLIPPAGE]
    adj_a = px_a - side * slip * cfg[CFG_TICK_A]
    adj_b = px_b + side * slip * cfg[CFG_TICK_B]
    return adj_a, adj_b


@njit(cache=True, inline="always")
def _calc_trade_pnl(side, entry_a, entry_b, exit_a, exit_b, n_a, n_b, cfg):
    """Final PnL with exit slippage. Returns (pnl_gross, costs, pnl_net)."""
    da = exit_a - entry_a
    db = exit_b - entry_b
    pnl_gross = side * (n_a * da * cfg[CFG_MULT_A] - n_b * db * cfg[CFG_MULT_B])
    costs = cfg[CFG_COMMISSION] * (n_a + n_b) * 2.0
    return pnl_gross, costs, pnl_gross - costs


@njit(cache=True, inline="always")
def _is_force_flat(minute, cfg):
    """True if current time is in force-flat zone (outside trading session)."""
    return minute >= cfg[CFG_FLAT_TIME] or minute < cfg[CFG_ENTRY_START]


@njit(cache=True, inline="always")
def _is_entry_window(minute, cfg):
    """True if current time allows new entries."""
    return cfg[CFG_ENTRY_START] <= minute < cfg[CFG_ENTRY_END]


# =========================================================================
# Main kernel
# =========================================================================

@njit(cache=True)
def _hybrid_kernel(
    # Slow layer (5min) arrays
    slow_beta,       # float64[n_slow]
    slow_alpha,      # float64[n_slow]
    slow_mu,         # float64[n_slow]
    slow_sigma,      # float64[n_slow]
    slow_gate,       # bool[n_slow]
    slow_zscore,     # float64[n_slow] 5min z-score (for cooldown check)
    slow_ts_ns,      # int64[n_slow] timestamps in nanoseconds
    # Fast layer (1s) arrays
    fast_close_a,    # float64[n_fast]
    fast_close_b,    # float64[n_fast]
    fast_ts_ns,      # int64[n_fast] timestamps in nanoseconds
    fast_minutes,    # int32[n_fast] minutes from midnight
    # Config
    cfg,             # float64[CFG_SIZE]
):
    """Hybrid backtest kernel: 5min indicators + 1s z-score scanning.

    Returns (results, n_trades) where results is float64[MAX_TRADES, N_COLS].
    """
    n_slow = len(slow_beta)
    n_fast = len(fast_close_a)
    results = np.empty((_MAX_TRADES, _N_COLS), dtype=np.float64)
    trade_idx = 0

    # Unpack config
    z_entry = cfg[CFG_Z_ENTRY]
    z_exit = cfg[CFG_Z_EXIT]
    z_stop = cfg[CFG_Z_STOP]
    pnl_tp = cfg[CFG_PNL_TP]
    pnl_sl = cfg[CFG_PNL_SL]

    # State
    state = _FLAT
    side = 0
    entry_a = 0.0
    entry_b = 0.0
    entry_z = 0.0
    entry_ts = np.int64(0)
    entry_bar = 0
    n_a = 0
    n_b = 0
    mfe = 0.0
    mae = 0.0
    bars_in_trade = 0
    last_exit_bar = -1  # anti-cycling: no re-entry in same 5min bar after exit

    # Cursor for 1s bars (advances linearly, O(n_fast) total)
    cursor = 0

    for i in range(n_slow):
        # Period boundaries for this 5min bar
        t_start = slow_ts_ns[i]
        if i + 1 < n_slow:
            t_end = slow_ts_ns[i + 1]
        else:
            # Last bar: assume 5min = 300s = 300_000_000_000 ns
            t_end = slow_ts_ns[i] + 300_000_000_000

        # Skip bars with invalid indicators
        beta_i = slow_beta[i]
        alpha_i = slow_alpha[i]
        mu_i = slow_mu[i]
        sigma_i = slow_sigma[i]

        indicators_valid = (
            np.isfinite(beta_i)
            and np.isfinite(alpha_i)
            and np.isfinite(mu_i)
            and np.isfinite(sigma_i)
            and sigma_i != 0.0
        )

        if not indicators_valid:
            # Force flat if in position with invalid indicators
            if state == _LONG or state == _SHORT:
                # Find the first 1s bar in this period for exit price
                j = cursor
                while j < n_fast and fast_ts_ns[j] < t_start:
                    j += 1
                if j < n_fast and fast_ts_ns[j] < t_end:
                    exit_a, exit_b = _apply_exit_slippage(
                        fast_close_a[j], fast_close_b[j], side, cfg
                    )
                    pnl_g, costs, pnl_n = _calc_trade_pnl(
                        side, entry_a, entry_b, exit_a, exit_b, n_a, n_b, cfg
                    )
                    if trade_idx < _MAX_TRADES:
                        results[trade_idx, _COL_ENTRY_TS] = float(entry_ts)
                        results[trade_idx, _COL_EXIT_TS] = float(fast_ts_ns[j])
                        results[trade_idx, _COL_SIDE] = float(side)
                        results[trade_idx, _COL_ENTRY_A] = entry_a
                        results[trade_idx, _COL_ENTRY_B] = entry_b
                        results[trade_idx, _COL_EXIT_A] = exit_a
                        results[trade_idx, _COL_EXIT_B] = exit_b
                        results[trade_idx, _COL_N_A] = float(n_a)
                        results[trade_idx, _COL_N_B] = float(n_b)
                        results[trade_idx, _COL_PNL_GROSS] = pnl_g
                        results[trade_idx, _COL_COSTS] = costs
                        results[trade_idx, _COL_PNL_NET] = pnl_n
                        results[trade_idx, _COL_DURATION] = float(bars_in_trade)
                        results[trade_idx, _COL_ENTRY_Z] = entry_z
                        results[trade_idx, _COL_EXIT_Z] = 0.0
                        results[trade_idx, _COL_EXIT_REASON] = float(_FLAT_EOD)
                        results[trade_idx, _COL_MFE] = mfe
                        results[trade_idx, _COL_MAE] = mae
                        results[trade_idx, _COL_ENTRY_BAR] = float(entry_bar)
                        results[trade_idx, _COL_EXIT_BAR] = float(i)
                        trade_idx += 1
                state = _FLAT
                side = 0
            elif state == _COOLDOWN:
                state = _FLAT
            # Advance cursor past this period
            while cursor < n_fast and fast_ts_ns[cursor] < t_end:
                cursor += 1
            continue

        # COOLDOWN check on 5min z-score (bar [i-1] close, no look-ahead)
        if state == _COOLDOWN and i > 0:
            z_5min_prev = slow_zscore[i - 1]
            if np.isfinite(z_5min_prev) and abs(z_5min_prev) < cfg[CFG_Z_COOLDOWN]:
                state = _FLAT

        # Gate: use [i-1] for no look-ahead on binary decisions
        gate_ok = slow_gate[i - 1] if i > 0 else False

        # Advance cursor to start of this period
        while cursor < n_fast and fast_ts_ns[cursor] < t_start:
            cursor += 1

        # Scan 1s bars within [t_start, t_end)
        j = cursor
        while j < n_fast and fast_ts_ns[j] < t_end:
            minute = fast_minutes[j]
            z_live = _calc_z_live(
                fast_close_a[j], fast_close_b[j],
                beta_i, alpha_i, mu_i, sigma_i,
            )

            # --- Force flat zone ---
            if _is_force_flat(minute, cfg):
                if state == _LONG or state == _SHORT:
                    exit_a, exit_b = _apply_exit_slippage(
                        fast_close_a[j], fast_close_b[j], side, cfg
                    )
                    pnl_g, costs, pnl_n = _calc_trade_pnl(
                        side, entry_a, entry_b, exit_a, exit_b, n_a, n_b, cfg
                    )
                    if trade_idx < _MAX_TRADES:
                        results[trade_idx, _COL_ENTRY_TS] = float(entry_ts)
                        results[trade_idx, _COL_EXIT_TS] = float(fast_ts_ns[j])
                        results[trade_idx, _COL_SIDE] = float(side)
                        results[trade_idx, _COL_ENTRY_A] = entry_a
                        results[trade_idx, _COL_ENTRY_B] = entry_b
                        results[trade_idx, _COL_EXIT_A] = exit_a
                        results[trade_idx, _COL_EXIT_B] = exit_b
                        results[trade_idx, _COL_N_A] = float(n_a)
                        results[trade_idx, _COL_N_B] = float(n_b)
                        results[trade_idx, _COL_PNL_GROSS] = pnl_g
                        results[trade_idx, _COL_COSTS] = costs
                        results[trade_idx, _COL_PNL_NET] = pnl_n
                        results[trade_idx, _COL_DURATION] = float(bars_in_trade)
                        results[trade_idx, _COL_ENTRY_Z] = entry_z
                        results[trade_idx, _COL_EXIT_Z] = z_live
                        results[trade_idx, _COL_EXIT_REASON] = float(_FLAT_EOD)
                        results[trade_idx, _COL_MFE] = mfe
                        results[trade_idx, _COL_MAE] = mae
                        results[trade_idx, _COL_ENTRY_BAR] = float(entry_bar)
                        results[trade_idx, _COL_EXIT_BAR] = float(i)
                        trade_idx += 1
                    last_exit_bar = i
                    state = _FLAT
                    side = 0
                elif state == _COOLDOWN:
                    state = _FLAT
                j += 1
                continue

            # --- FLAT: check entries ---
            # Anti-cycling: no re-entry in same 5min bar after an exit
            if state == _FLAT:
                if i > last_exit_bar and gate_ok and _is_entry_window(minute, cfg):
                    if z_live < -z_entry:
                        # Enter LONG
                        side = 1
                        state = _LONG
                        n_a, n_b = _calc_size(
                            fast_close_a[j], fast_close_b[j],
                            beta_i, cfg[CFG_MULT_A], cfg[CFG_MULT_B],
                        )
                        entry_a, entry_b = _apply_entry_slippage(
                            fast_close_a[j], fast_close_b[j], side, cfg
                        )
                        entry_z = z_live
                        entry_ts = fast_ts_ns[j]
                        entry_bar = i
                        mfe = 0.0
                        mae = 0.0
                        bars_in_trade = 0

                    elif z_live > z_entry:
                        # Enter SHORT
                        side = -1
                        state = _SHORT
                        n_a, n_b = _calc_size(
                            fast_close_a[j], fast_close_b[j],
                            beta_i, cfg[CFG_MULT_A], cfg[CFG_MULT_B],
                        )
                        entry_a, entry_b = _apply_entry_slippage(
                            fast_close_a[j], fast_close_b[j], side, cfg
                        )
                        entry_z = z_live
                        entry_ts = fast_ts_ns[j]
                        entry_bar = i
                        mfe = 0.0
                        mae = 0.0
                        bars_in_trade = 0

            # --- LONG/SHORT: check exits ---
            elif state == _LONG or state == _SHORT:
                bars_in_trade += 1

                # Track unrealized PnL for dollar exits + MFE/MAE
                pnl_unr = _calc_pnl_unrealized(
                    side, entry_a, entry_b,
                    fast_close_a[j], fast_close_b[j],
                    n_a, n_b, cfg,
                )
                if pnl_unr > mfe:
                    mfe = pnl_unr
                if pnl_unr < mae:
                    mae = pnl_unr

                # Exit checks (priority order)
                exit_triggered = False
                exit_reason = -1
                next_state = _FLAT

                # 1. Dollar SL
                if pnl_sl > 0.0 and pnl_unr <= -pnl_sl:
                    exit_triggered = True
                    exit_reason = _SL_DOLLAR
                    next_state = _COOLDOWN

                # 2. Dollar TP
                elif pnl_tp > 0.0 and pnl_unr >= pnl_tp:
                    exit_triggered = True
                    exit_reason = _TP_DOLLAR
                    next_state = _COOLDOWN

                # 3. Z-score SL
                elif state == _LONG and z_live < -z_stop:
                    exit_triggered = True
                    exit_reason = _SL_ZSCORE
                    next_state = _COOLDOWN
                elif state == _SHORT and z_live > z_stop:
                    exit_triggered = True
                    exit_reason = _SL_ZSCORE
                    next_state = _COOLDOWN

                # 4. Z-score TP
                elif state == _LONG and z_live > -z_exit:
                    exit_triggered = True
                    exit_reason = _TP_ZSCORE
                    next_state = _COOLDOWN
                elif state == _SHORT and z_live < z_exit:
                    exit_triggered = True
                    exit_reason = _TP_ZSCORE
                    next_state = _COOLDOWN

                if exit_triggered:
                    exit_a, exit_b = _apply_exit_slippage(
                        fast_close_a[j], fast_close_b[j], side, cfg
                    )
                    pnl_g, costs, pnl_n = _calc_trade_pnl(
                        side, entry_a, entry_b, exit_a, exit_b, n_a, n_b, cfg
                    )
                    if trade_idx < _MAX_TRADES:
                        results[trade_idx, _COL_ENTRY_TS] = float(entry_ts)
                        results[trade_idx, _COL_EXIT_TS] = float(fast_ts_ns[j])
                        results[trade_idx, _COL_SIDE] = float(side)
                        results[trade_idx, _COL_ENTRY_A] = entry_a
                        results[trade_idx, _COL_ENTRY_B] = entry_b
                        results[trade_idx, _COL_EXIT_A] = exit_a
                        results[trade_idx, _COL_EXIT_B] = exit_b
                        results[trade_idx, _COL_N_A] = float(n_a)
                        results[trade_idx, _COL_N_B] = float(n_b)
                        results[trade_idx, _COL_PNL_GROSS] = pnl_g
                        results[trade_idx, _COL_COSTS] = costs
                        results[trade_idx, _COL_PNL_NET] = pnl_n
                        results[trade_idx, _COL_DURATION] = float(bars_in_trade)
                        results[trade_idx, _COL_ENTRY_Z] = entry_z
                        results[trade_idx, _COL_EXIT_Z] = z_live
                        results[trade_idx, _COL_EXIT_REASON] = float(exit_reason)
                        results[trade_idx, _COL_MFE] = mfe
                        results[trade_idx, _COL_MAE] = mae
                        results[trade_idx, _COL_ENTRY_BAR] = float(entry_bar)
                        results[trade_idx, _COL_EXIT_BAR] = float(i)
                        trade_idx += 1

                    last_exit_bar = i
                    state = next_state
                    side = 0 if next_state != _LONG and next_state != _SHORT else side

            # COOLDOWN: do nothing at 1s level, checked at 5min bar boundary
            # (see cooldown check at top of outer loop)

            j += 1

    return results[:trade_idx], trade_idx


# =========================================================================
# Python wrapper
# =========================================================================

def _prepare_slow_arrays(aligned_5min, config):
    """Compute 5min indicators and return arrays for the kernel."""
    log_a = np.log(aligned_5min.df["close_a"])
    log_b = np.log(aligned_5min.df["close_b"])

    est = create_estimator(
        "ols_rolling",
        window=config["ols_window"],
        zscore_window=config["zscore_window"],
    )
    hr = est.estimate(aligned_5min)
    beta = hr.beta.values
    spread = hr.spread

    # Alpha reconstruction: alpha = log_a - beta * log_b - spread
    alpha = (log_a - hr.beta * log_b - spread).values

    # Z-score components
    mu = spread.rolling(config["zscore_window"]).mean().values
    sigma = spread.rolling(config["zscore_window"]).std().values

    # 5min z-score (for cooldown check)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore_5min = (spread.values - mu) / sigma
    zscore_5min = np.where(np.isfinite(zscore_5min), zscore_5min, 0.0)

    # Binary gates
    gate_cfg = GateConfig(
        adf_threshold=config.get("gate_adf", -2.86),
        hurst_threshold=config.get("gate_hurst", 0.50),
        corr_threshold=config.get("gate_corr", 0.70),
        adf_window=config.get("gate_adf_window", 96),
        hurst_window=config.get("gate_hurst_window", 64),
        corr_window=config.get("gate_corr_window", 24),
    )
    gate_mask = compute_gate_mask(
        spread, aligned_5min.df["close_a"], aligned_5min.df["close_b"], gate_cfg
    )

    # Timestamps as int64 nanoseconds
    ts_ns = aligned_5min.df.index.values.astype("datetime64[ns]").astype(np.int64)

    return (
        np.ascontiguousarray(beta, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(mu, dtype=np.float64),
        np.ascontiguousarray(sigma, dtype=np.float64),
        np.ascontiguousarray(gate_mask, dtype=np.bool_),
        np.ascontiguousarray(zscore_5min, dtype=np.float64),
        ts_ns,
        hr,  # for reference
    )


def _prepare_fast_arrays(data_1s):
    """Convert 1s DataFrame to contiguous arrays for the kernel."""
    close_a = np.ascontiguousarray(data_1s["close_a"].values, dtype=np.float64)
    close_b = np.ascontiguousarray(data_1s["close_b"].values, dtype=np.float64)
    ts_ns = data_1s.index.values.astype("datetime64[ns]").astype(np.int64)
    minutes = (data_1s.index.hour * 60 + data_1s.index.minute).values.astype(np.int32)
    return close_a, close_b, ts_ns, minutes


def _results_to_dataframe(raw, n_trades):
    """Convert kernel results array to DataFrame with proper types."""
    if n_trades == 0:
        return pd.DataFrame()

    df = pd.DataFrame(raw[:n_trades], columns=[
        "entry_ts_ns", "exit_ts_ns", "side",
        "entry_a", "entry_b", "exit_a", "exit_b",
        "n_a", "n_b", "pnl_gross", "costs", "pnl_net",
        "duration_1s", "entry_z", "exit_z", "exit_reason_int",
        "mfe", "mae", "entry_bar_5min", "exit_bar_5min",
    ])

    # Convert timestamps
    df["entry_time"] = pd.to_datetime(df["entry_ts_ns"].astype(np.int64), unit="ns")
    df["exit_time"] = pd.to_datetime(df["exit_ts_ns"].astype(np.int64), unit="ns")

    # Convert types
    df["side"] = df["side"].astype(int)
    df["n_a"] = df["n_a"].astype(int)
    df["n_b"] = df["n_b"].astype(int)
    df["exit_reason"] = df["exit_reason_int"].astype(int).map(EXIT_REASON_MAP)
    df["entry_bar_5min"] = df["entry_bar_5min"].astype(int)
    df["exit_bar_5min"] = df["exit_bar_5min"].astype(int)

    return df


def _compute_summary(trades_df):
    """Compute summary stats from trades DataFrame."""
    if len(trades_df) == 0:
        return {
            "trades": 0, "win_rate": 0.0, "pnl": 0.0,
            "profit_factor": 0.0, "avg_pnl_trade": 0.0,
        }

    pnl = trades_df["pnl_net"].values
    n = len(pnl)
    total = float(pnl.sum())
    wins = pnl > 0
    wr = float(wins.sum() / n * 100) if n > 0 else 0.0
    gross_gains = float(pnl[wins].sum()) if wins.any() else 0.0
    gross_losses = float(abs(pnl[~wins].sum())) if (~wins).any() else 0.0
    pf = gross_gains / gross_losses if gross_losses > 0 else (
        float("inf") if gross_gains > 0 else 0.0
    )

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return {
        "trades": n,
        "win_rate": round(wr, 1),
        "pnl": round(total, 2),
        "profit_factor": round(pf, 2),
        "avg_pnl_trade": round(total / n, 2) if n > 0 else 0.0,
        "max_drawdown": round(max_dd, 2),
    }


def run_hybrid_backtest(
    aligned_5min: AlignedPair,
    data_1s: pd.DataFrame,
    config: dict,
) -> dict:
    """Run hybrid backtest: 5min indicators + 1s z-score scanning.

    Parameters
    ----------
    aligned_5min : AlignedPair
        5min aligned pair data (from cache).
    data_1s : pd.DataFrame
        1s aligned data with columns close_a, close_b (from loader_1s).
    config : dict
        All parameters: ols_window, zscore_window, z_entry, z_exit, z_stop,
        mult_a, mult_b, tick_a, tick_b, slippage, commission,
        entry_start_min, entry_end_min, flat_min, dollar_tp, dollar_sl,
        gate_adf, gate_hurst, gate_corr, gate_adf_window, ...

    Returns
    -------
    dict with keys: trades (int), win_rate, pnl, profit_factor,
    avg_pnl_trade, max_drawdown, trades_df (DataFrame), summary (dict)
    """
    # Prepare arrays
    slow_beta, slow_alpha, slow_mu, slow_sigma, slow_gate, slow_zsc, slow_ts, hr = \
        _prepare_slow_arrays(aligned_5min, config)
    fast_a, fast_b, fast_ts, fast_min = _prepare_fast_arrays(data_1s)
    cfg = pack_config(config)

    # Run kernel
    raw_results, n_trades = _hybrid_kernel(
        slow_beta, slow_alpha, slow_mu, slow_sigma,
        slow_gate, slow_zsc, slow_ts,
        fast_a, fast_b, fast_ts, fast_min,
        cfg,
    )

    # Convert to DataFrame
    trades_df = _results_to_dataframe(raw_results, n_trades)
    summary = _compute_summary(trades_df)

    return {
        **summary,
        "trades_df": trades_df,
    }


# =========================================================================
# Warmup (pre-compile kernel)
# =========================================================================

def warmup_hybrid():
    """Pre-compile the hybrid kernel with dummy data (~3-5s first time)."""
    n_slow = 10
    n_fast = 100
    cfg = pack_config({
        "z_entry": 2.0, "z_exit": 0.5, "z_stop": 4.0,
        "mult_a": 20.0, "mult_b": 5.0, "tick_a": 0.25, "tick_b": 1.0,
        "slippage": 1, "commission": 2.50,
        "entry_start_min": 0, "entry_end_min": 1440, "flat_min": 1440,
    })

    slow_beta = np.ones(n_slow, dtype=np.float64)
    slow_alpha = np.zeros(n_slow, dtype=np.float64)
    slow_mu = np.zeros(n_slow, dtype=np.float64)
    slow_sigma = np.ones(n_slow, dtype=np.float64)
    slow_gate = np.ones(n_slow, dtype=np.bool_)
    slow_zsc = np.zeros(n_slow, dtype=np.float64)
    slow_ts = np.arange(n_slow, dtype=np.int64) * 300_000_000_000

    fast_a = np.full(n_fast, 20000.0, dtype=np.float64)
    fast_b = np.full(n_fast, 40000.0, dtype=np.float64)
    fast_ts = np.arange(n_fast, dtype=np.int64) * 1_000_000_000
    fast_min = np.full(n_fast, 480, dtype=np.int32)

    _hybrid_kernel(
        slow_beta, slow_alpha, slow_mu, slow_sigma,
        slow_gate, slow_zsc, slow_ts,
        fast_a, fast_b, fast_ts, fast_min,
        cfg,
    )
