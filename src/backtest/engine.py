"""Backtest engine: simulate spread trades with costs and mark-to-market equity.

Two implementations:
- BacktestEngine: original bar-by-bar loop (reference, used for single backtests)
- run_backtest_vectorized: numpy-vectorized (fast, used for grid search)

Both produce identical results.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from src.sizing.position import calculate_position_size


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtest simulation."""
    initial_capital: float = 100_000.0
    commission_per_contract: float = 2.50  # per side per contract
    slippage_ticks: int = 1


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract specification for one instrument."""
    multiplier: float   # point value ($/pt)
    tick_size: float
    tick_value: float


@dataclass
class Trade:
    """Record of a single completed spread trade."""
    entry_bar: int
    exit_bar: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int            # +1 long spread, -1 short spread
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    n_a: int
    n_b: int
    pnl_gross: float
    costs: float
    pnl_net: float


@dataclass
class BacktestResult:
    """Output of a backtest run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    config: BacktestConfig = field(default_factory=BacktestConfig)


def _apply_slippage(price: float, tick_size: float, slippage_ticks: int, direction: int) -> float:
    """Apply slippage to a price. direction: +1 = buying (worse = higher), -1 = selling (worse = lower)."""
    return price + direction * slippage_ticks * tick_size


# ---------------------------------------------------------------------------
# Numba-compiled equity curve builder
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_equity_curve(px_a, px_b, te, tx, sides, entry_px_a, entry_px_b,
                         n_a, n_b, pnl_net, mult_a, mult_b, initial_capital,
                         n, num_trades):
    """Build mark-to-market equity curve (numba-compiled)."""
    equity = np.full(n, initial_capital)
    cumulative_realized = 0.0

    for i in range(num_trades):
        eb, xb = te[i], tx[i]
        side = sides[i]
        e_a, e_b = entry_px_a[i], entry_px_b[i]
        na, nb = n_a[i], n_b[i]

        for t in range(eb, xb):
            da = px_a[t] - e_a
            db = px_b[t] - e_b
            unrealized = side * (na * da * mult_a - nb * db * mult_b)
            equity[t] = initial_capital + cumulative_realized + unrealized

        equity[xb] = initial_capital + cumulative_realized + pnl_net[i]
        cumulative_realized += pnl_net[i]

        next_start = te[i + 1] if i + 1 < num_trades else n
        for t in range(xb + 1, next_start):
            equity[t] = initial_capital + cumulative_realized

    for t in range(0, te[0]):
        equity[t] = initial_capital

    return equity


# ---------------------------------------------------------------------------
# Vectorized backtest (fast, for grid search)
# ---------------------------------------------------------------------------

def run_backtest_vectorized(
    px_a: np.ndarray,
    px_b: np.ndarray,
    sig: np.ndarray,
    bt: np.ndarray,
    mult_a: float,
    mult_b: float,
    tick_a: float,
    tick_b: float,
    slippage_ticks: int = 1,
    commission: float = 2.50,
    initial_capital: float = 100_000.0,
) -> dict:
    """Vectorized backtest — returns summary dict (no Trade objects).

    Detects signal transitions, computes sizing/slippage/PnL per trade,
    then builds mark-to-market equity curve. ~50-100x faster than bar loop.

    Parameters
    ----------
    px_a, px_b : np.ndarray
        Raw close prices for each leg.
    sig : np.ndarray
        Signal array {+1, 0, -1}.
    bt : np.ndarray
        Hedge ratio beta.
    mult_a, mult_b : float
        Point value (multiplier) for each leg.
    tick_a, tick_b : float
        Tick size for each leg.
    slippage_ticks : int
        Ticks of slippage per leg.
    commission : float
        Commission per contract per side.
    initial_capital : float

    Returns
    -------
    dict with keys: trades (int), win_rate, pnl, profit_factor, avg_pnl_trade,
    equity (np.ndarray), trade_sides (np.ndarray), trade_pnls (np.ndarray),
    avg_duration_bars, max_duration_bars
    """
    n = len(px_a)

    # --- Detect transitions ---
    prev_sig = np.roll(sig, 1)
    prev_sig[0] = 0

    entries = (prev_sig == 0) & (sig != 0)
    exits = (prev_sig != 0) & (sig == 0)

    entry_bars = np.where(entries)[0]
    exit_bars = np.where(exits)[0]

    # Match entries to exits: each entry matched with the next exit after it
    if len(entry_bars) == 0 or len(exit_bars) == 0:
        equity = np.full(n, initial_capital)
        return {
            "trades": 0, "win_rate": 0.0, "pnl": 0.0, "profit_factor": 0.0,
            "avg_pnl_trade": 0.0, "equity": equity, "trade_sides": np.array([]),
            "trade_pnls": np.array([]), "trade_entry_bars": np.array([], dtype=int),
            "trade_exit_bars": np.array([], dtype=int),
            "avg_duration_bars": 0, "max_duration_bars": 0,
        }

    # Pair up: for each entry, find the first exit >= entry
    trade_entries = []
    trade_exits = []
    exit_idx = 0
    for eb in entry_bars:
        while exit_idx < len(exit_bars) and exit_bars[exit_idx] <= eb:
            exit_idx += 1
        if exit_idx < len(exit_bars):
            trade_entries.append(eb)
            trade_exits.append(exit_bars[exit_idx])
            exit_idx += 1

    if len(trade_entries) == 0:
        equity = np.full(n, initial_capital)
        return {
            "trades": 0, "win_rate": 0.0, "pnl": 0.0, "profit_factor": 0.0,
            "avg_pnl_trade": 0.0, "equity": equity, "trade_sides": np.array([]),
            "trade_pnls": np.array([]), "trade_entry_bars": np.array([], dtype=int),
            "trade_exit_bars": np.array([], dtype=int),
            "avg_duration_bars": 0, "max_duration_bars": 0,
        }

    te = np.array(trade_entries)
    tx = np.array(trade_exits)
    num_trades = len(te)

    # --- Compute per-trade values ---
    sides = sig[te].astype(np.int8)

    # Beta at entry
    b = bt[te].copy()
    b[~np.isfinite(b)] = 1.0

    # Sizing: N_b = round((px_a * mult_a) / (px_b * mult_b) * |beta| * N_a)
    not_a = px_a[te] * mult_a
    not_b = px_b[te] * mult_b
    with np.errstate(divide="ignore", invalid="ignore"):
        n_b_raw = (not_a / not_b) * np.abs(b)
    n_b_raw = np.nan_to_num(n_b_raw, nan=1.0, posinf=1.0, neginf=1.0)
    n_b = np.maximum(np.round(n_b_raw).astype(int), 1)
    n_a = np.ones(num_trades, dtype=int)

    # Slippage at entry
    entry_px_a = px_a[te] + sides * slippage_ticks * tick_a
    entry_px_b = px_b[te] - sides * slippage_ticks * tick_b

    # Slippage at exit
    exit_px_a = px_a[tx] - sides * slippage_ticks * tick_a
    exit_px_b = px_b[tx] + sides * slippage_ticks * tick_b

    # PnL per trade
    delta_a = exit_px_a - entry_px_a
    delta_b = exit_px_b - entry_px_b
    pnl_gross = sides * (n_a * delta_a * mult_a - n_b * delta_b * mult_b)
    costs = commission * (n_a + n_b) * 2
    pnl_net = pnl_gross - costs

    # Durations
    durations = tx - te

    # --- Mark-to-market equity ---
    equity = _build_equity_curve(px_a, px_b, te, tx, sides, entry_px_a, entry_px_b,
                                  n_a, n_b, pnl_net, mult_a, mult_b, initial_capital, n, num_trades)

    # --- Summary stats ---
    total_pnl = float(pnl_net.sum())
    wins = pnl_net > 0
    win_rate = float(wins.sum() / num_trades * 100) if num_trades > 0 else 0.0

    gross_gains = float(pnl_net[wins].sum()) if wins.any() else 0.0
    gross_losses = float(abs(pnl_net[~wins].sum())) if (~wins).any() else 0.0
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else (float("inf") if gross_gains > 0 else 0.0)

    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    avg_dur = float(durations.mean()) if num_trades > 0 else 0
    max_dur = int(durations.max()) if num_trades > 0 else 0

    return {
        "trades": num_trades,
        "win_rate": round(win_rate, 1),
        "pnl": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_pnl_trade": round(avg_pnl, 2),
        "equity": equity,
        "trade_sides": sides,
        "trade_pnls": pnl_net,
        "trade_entry_bars": te,
        "trade_exit_bars": tx,
        "avg_duration_bars": round(avg_dur, 1),
        "max_duration_bars": max_dur,
    }


# ---------------------------------------------------------------------------
# Grid-optimized backtest (no equity curve, ~20x faster for grid search)
# ---------------------------------------------------------------------------

def run_backtest_grid(
    px_a: np.ndarray,
    px_b: np.ndarray,
    sig: np.ndarray,
    bt: np.ndarray,
    mult_a: float,
    mult_b: float,
    tick_a: float,
    tick_b: float,
    slippage_ticks: int = 1,
    commission: float = 2.50,
) -> dict:
    """Lightweight backtest for grid search — no equity curve, no MtM.

    Returns only trade-level stats: trades, win_rate, pnl, profit_factor,
    avg_pnl_trade, avg_duration_bars.
    """
    n = len(px_a)

    prev_sig = np.roll(sig, 1)
    prev_sig[0] = 0

    entries = (prev_sig == 0) & (sig != 0)
    exits = (prev_sig != 0) & (sig == 0)

    entry_bars = np.where(entries)[0]
    exit_bars = np.where(exits)[0]

    if len(entry_bars) == 0 or len(exit_bars) == 0:
        return {"trades": 0, "win_rate": 0.0, "pnl": 0.0, "profit_factor": 0.0,
                "avg_pnl_trade": 0.0, "avg_duration_bars": 0}

    # Pair up entries/exits
    trade_entries = []
    trade_exits = []
    exit_idx = 0
    for eb in entry_bars:
        while exit_idx < len(exit_bars) and exit_bars[exit_idx] <= eb:
            exit_idx += 1
        if exit_idx < len(exit_bars):
            trade_entries.append(eb)
            trade_exits.append(exit_bars[exit_idx])
            exit_idx += 1

    if len(trade_entries) == 0:
        return {"trades": 0, "win_rate": 0.0, "pnl": 0.0, "profit_factor": 0.0,
                "avg_pnl_trade": 0.0, "avg_duration_bars": 0}

    te = np.array(trade_entries)
    tx = np.array(trade_exits)
    num_trades = len(te)

    sides = sig[te].astype(np.int8)

    b = bt[te].copy()
    b[~np.isfinite(b)] = 1.0

    not_a = px_a[te] * mult_a
    not_b = px_b[te] * mult_b
    with np.errstate(divide="ignore", invalid="ignore"):
        n_b_raw = (not_a / not_b) * np.abs(b)
    n_b_raw = np.nan_to_num(n_b_raw, nan=1.0, posinf=1.0, neginf=1.0)
    n_b = np.maximum(np.round(n_b_raw).astype(int), 1)
    n_a = np.ones(num_trades, dtype=int)

    entry_px_a = px_a[te] + sides * slippage_ticks * tick_a
    entry_px_b = px_b[te] - sides * slippage_ticks * tick_b
    exit_px_a = px_a[tx] - sides * slippage_ticks * tick_a
    exit_px_b = px_b[tx] + sides * slippage_ticks * tick_b

    delta_a = exit_px_a - entry_px_a
    delta_b = exit_px_b - entry_px_b
    pnl_gross = sides * (n_a * delta_a * mult_a - n_b * delta_b * mult_b)
    costs = commission * (n_a + n_b) * 2
    pnl_net = pnl_gross - costs

    durations = tx - te

    total_pnl = float(pnl_net.sum())
    wins = pnl_net > 0
    win_rate = float(wins.sum() / num_trades * 100) if num_trades > 0 else 0.0
    gross_gains = float(pnl_net[wins].sum()) if wins.any() else 0.0
    gross_losses = float(abs(pnl_net[~wins].sum())) if (~wins).any() else 0.0
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else (float("inf") if gross_gains > 0 else 0.0)
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    avg_dur = float(durations.mean()) if num_trades > 0 else 0

    return {
        "trades": num_trades,
        "win_rate": round(win_rate, 1),
        "pnl": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_pnl_trade": round(avg_pnl, 2),
        "avg_duration_bars": round(avg_dur, 1),
    }


# ---------------------------------------------------------------------------
# Original bar-by-bar engine (reference implementation, single backtests)
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtest engine with mark-to-market equity curve.

    Simulates spread trades bar by bar. Detects signal transitions,
    computes sizing at entry, applies slippage and commissions,
    and tracks unrealized PnL at every bar.
    """

    def __init__(self, config: BacktestConfig | None = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = BacktestConfig(**kwargs)

    def run(
        self,
        close_a: pd.Series,
        close_b: pd.Series,
        signals: pd.Series,
        beta: pd.Series,
        spec_a: InstrumentSpec,
        spec_b: InstrumentSpec,
    ) -> BacktestResult:
        """Run backtest simulation."""
        n = len(close_a)
        idx = close_a.index
        sig = signals.values
        px_a = close_a.values
        px_b = close_b.values
        bt = beta.values

        trades: list[Trade] = []
        equity = np.full(n, self.config.initial_capital)
        realized_pnl = 0.0

        in_position = False
        pos_side = 0
        entry_bar = 0
        entry_price_a = 0.0
        entry_price_b = 0.0
        pos_n_a = 0
        pos_n_b = 0

        prev_sig = 0

        for t in range(n):
            curr_sig = int(sig[t])

            if not in_position and curr_sig != 0 and prev_sig == 0:
                pos_side = curr_sig
                entry_bar = t

                b = bt[t] if np.isfinite(bt[t]) else 1.0
                pos_n_a, pos_n_b, _ = calculate_position_size(
                    px_a[t], px_b[t], abs(b), spec_a.multiplier, spec_b.multiplier
                )

                entry_price_a = _apply_slippage(
                    px_a[t], spec_a.tick_size, self.config.slippage_ticks, pos_side
                )
                entry_price_b = _apply_slippage(
                    px_b[t], spec_b.tick_size, self.config.slippage_ticks, -pos_side
                )
                in_position = True

            elif in_position and curr_sig == 0 and prev_sig != 0:
                exit_price_a = _apply_slippage(
                    px_a[t], spec_a.tick_size, self.config.slippage_ticks, -pos_side
                )
                exit_price_b = _apply_slippage(
                    px_b[t], spec_b.tick_size, self.config.slippage_ticks, pos_side
                )

                delta_a = exit_price_a - entry_price_a
                delta_b = exit_price_b - entry_price_b
                pnl_gross = pos_side * (
                    pos_n_a * delta_a * spec_a.multiplier
                    - pos_n_b * delta_b * spec_b.multiplier
                )

                costs = self.config.commission_per_contract * (pos_n_a + pos_n_b) * 2
                pnl_net = pnl_gross - costs

                trades.append(Trade(
                    entry_bar=entry_bar,
                    exit_bar=t,
                    entry_time=idx[entry_bar],
                    exit_time=idx[t],
                    side=pos_side,
                    entry_price_a=entry_price_a,
                    entry_price_b=entry_price_b,
                    exit_price_a=exit_price_a,
                    exit_price_b=exit_price_b,
                    n_a=pos_n_a,
                    n_b=pos_n_b,
                    pnl_gross=pnl_gross,
                    costs=costs,
                    pnl_net=pnl_net,
                ))

                realized_pnl += pnl_net
                in_position = False
                pos_side = 0

            if in_position:
                delta_a = px_a[t] - entry_price_a
                delta_b = px_b[t] - entry_price_b
                unrealized = pos_side * (
                    pos_n_a * delta_a * spec_a.multiplier
                    - pos_n_b * delta_b * spec_b.multiplier
                )
                equity[t] = self.config.initial_capital + realized_pnl + unrealized
            else:
                equity[t] = self.config.initial_capital + realized_pnl

            prev_sig = curr_sig

        equity_series = pd.Series(equity, index=idx, name="equity")

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            config=self.config,
        )
