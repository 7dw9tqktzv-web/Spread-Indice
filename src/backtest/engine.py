"""Backtest engine: simulate spread trades with costs and mark-to-market equity."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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
        """Run backtest simulation.

        Parameters
        ----------
        close_a, close_b : pd.Series
            Raw close prices (not log) for each leg.
        signals : pd.Series
            Signal series {+1, 0, -1} from SignalGenerator.
        beta : pd.Series
            Dynamic hedge ratio from HedgeResult.beta.
        spec_a, spec_b : InstrumentSpec
            Contract specs for each leg.

        Returns
        -------
        BacktestResult with trades list and mark-to-market equity curve.
        """
        n = len(close_a)
        idx = close_a.index
        sig = signals.values
        px_a = close_a.values
        px_b = close_b.values
        bt = beta.values

        trades: list[Trade] = []
        equity = np.full(n, self.config.initial_capital)
        realized_pnl = 0.0

        # Position state
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

            # --- OPEN position ---
            if not in_position and curr_sig != 0 and prev_sig == 0:
                pos_side = curr_sig
                entry_bar = t

                # Sizing at entry
                b = bt[t] if np.isfinite(bt[t]) else 1.0
                pos_n_a, pos_n_b, _ = calculate_position_size(
                    px_a[t], px_b[t], abs(b), spec_a.multiplier, spec_b.multiplier
                )

                # Slippage: LONG spread = buy A (+1), sell B (-1)
                #           SHORT spread = sell A (-1), buy B (+1)
                entry_price_a = _apply_slippage(
                    px_a[t], spec_a.tick_size, self.config.slippage_ticks, pos_side
                )
                entry_price_b = _apply_slippage(
                    px_b[t], spec_b.tick_size, self.config.slippage_ticks, -pos_side
                )
                in_position = True

            # --- CLOSE position ---
            elif in_position and curr_sig == 0 and prev_sig != 0:
                # Slippage at exit: opposite direction from entry
                exit_price_a = _apply_slippage(
                    px_a[t], spec_a.tick_size, self.config.slippage_ticks, -pos_side
                )
                exit_price_b = _apply_slippage(
                    px_b[t], spec_b.tick_size, self.config.slippage_ticks, pos_side
                )

                # PnL: side × [N_a × Δa × pv_a - N_b × Δb × pv_b]
                delta_a = exit_price_a - entry_price_a
                delta_b = exit_price_b - entry_price_b
                pnl_gross = pos_side * (
                    pos_n_a * delta_a * spec_a.multiplier
                    - pos_n_b * delta_b * spec_b.multiplier
                )

                # Costs: commission per contract per side × 2 sides
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

            # --- MARK-TO-MARKET ---
            if in_position:
                # Unrealized PnL at current bar prices (no slippage on mark)
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
