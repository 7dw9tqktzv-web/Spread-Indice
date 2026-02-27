"""Plot equity curves for validated OLS configs: NQ_YM Config E + NQ_RTY #8.

Usage:
    python scripts/plot_equity_curves.py
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import (
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

INITIAL_CAPITAL = 100_000.0
SLIPPAGE = 1
COMMISSION = 2.50
FLAT_MIN = 930  # 15:30 CT


def build_and_run(pair, aligned, cfg):
    """Build signal chain and run backtest for a config."""
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    zw = cfg["zw"]
    mu = spread.rolling(zw).mean()
    sigma = spread.rolling(zw).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics = compute_all_metrics(spread, aligned.df["close_a"],
                                  aligned.df["close_b"], cfg["metrics_cfg"])
    confidence = compute_confidence(metrics, cfg["conf_cfg"]).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])

    entry_start, entry_end = cfg["window_min"]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        cfg["mult_a"], cfg["mult_b"], cfg["tick_a"], cfg["tick_b"],
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    equity = bt["equity"]
    return equity, idx, bt


def main():
    # ==================================================================
    # Config E -- NQ_YM OLS (principale)
    # ==================================================================
    cfg_nq_ym = {
        "label": "NQ_YM Config E (OLS)",
        "ols": 3300, "zw": 30,
        "z_entry": 3.15, "z_exit": 1.00, "z_stop": 4.50,
        "conf": 67.0,
        "window_min": (120, 840),  # 02:00-14:00
        "metrics_cfg": MetricsConfig(adf_window=12, hurst_window=64,
                                      halflife_window=12, correlation_window=6),
        "conf_cfg": ConfidenceConfig(),  # default NQ_YM weights
        "mult_a": 20.0, "mult_b": 5.0,
        "tick_a": 0.25, "tick_b": 1.0,
    }

    # ==================================================================
    # Config #8 -- NQ_RTY OLS (principale)
    # ==================================================================
    cfg_nq_rty = {
        "label": "NQ_RTY #8 (OLS)",
        "ols": 9240, "zw": 20,
        "z_entry": 3.00, "z_exit": 0.75, "z_stop": 5.0,
        "conf": 75.0,
        "window_min": (360, 840),  # 06:00-14:00
        "metrics_cfg": MetricsConfig(adf_window=36, hurst_window=96,
                                      halflife_window=36, correlation_window=9),
        "conf_cfg": ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00),
        "mult_a": 20.0, "mult_b": 50.0,
        "tick_a": 0.25, "tick_b": 0.10,
    }

    # ==================================================================
    # Run backtests
    # ==================================================================
    results = {}

    for pair_name, pair_inst, cfg in [
        ("NQ_YM", SpreadPair(Instrument.NQ, Instrument.YM), cfg_nq_ym),
        ("NQ_RTY", SpreadPair(Instrument.NQ, Instrument.RTY), cfg_nq_rty),
    ]:
        print(f"\n{'='*60}")
        print(f"  {cfg['label']}")
        print(f"{'='*60}")

        aligned = load_aligned_pair_cache(pair_inst, "5min")
        if aligned is None:
            print(f"  ERREUR: pas de cache {pair_name}")
            continue

        equity, idx, bt = build_and_run(pair_name, aligned, cfg)

        # Compute MaxDD and Sharpe from raw arrays
        running_max = np.maximum.accumulate(equity)
        max_dd = float((equity - running_max).min())
        pnls = bt["trade_pnls"]
        n = bt["trades"]
        sharpe = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n > 1 and pnls.std() > 0 else 0.0

        bt["max_dd"] = max_dd
        bt["sharpe"] = round(sharpe, 2)
        results[pair_name] = {"equity": equity, "idx": idx, "bt": bt, "cfg": cfg}

        print(f"  Trades: {n}")
        print(f"  P&L: ${bt['pnl']:,.0f}")
        print(f"  PF: {bt['profit_factor']:.2f}")
        print(f"  WR: {bt['win_rate']:.1f}%")
        print(f"  MaxDD: ${max_dd:,.0f}")
        print(f"  Sharpe: {sharpe:.2f}")

    if not results:
        print("Aucun resultat. Verifier le cache.")
        return

    # ==================================================================
    # Plot equity curves
    # ==================================================================
    fig, axes = plt.subplots(len(results), 1, figsize=(16, 5 * len(results)),
                              squeeze=False)

    colors = {"NQ_YM": "#2196F3", "NQ_RTY": "#FF9800"}

    for i, (pair_name, r) in enumerate(results.items()):
        ax = axes[i, 0]
        equity = r["equity"]
        idx = r["idx"]
        bt = r["bt"]
        cfg = r["cfg"]

        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max

        # Equity curve
        color = colors.get(pair_name, "#4CAF50")
        ax.plot(idx, equity, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(idx, equity, INITIAL_CAPITAL,
                         where=equity >= INITIAL_CAPITAL,
                         color=color, alpha=0.08)

        # Drawdown shading
        ax.fill_between(idx, equity, running_max,
                         where=drawdown < 0,
                         color="red", alpha=0.15, label="Drawdown")

        # Reference line
        ax.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.4)

        # Annotations
        n = bt["trades"]
        pnl = bt["pnl"]
        pf = bt["profit_factor"]
        wr = bt["win_rate"]
        mdd = bt["max_dd"]

        title = f"{cfg['label']}  |  {n} trades  |  PnL {pnl:,.0f}  |  PF {pf:.2f}  |  WR {wr:.0f}%  |  MaxDD {mdd:,.0f}"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Equity", fontsize=10)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=9)

        # X-axis formatting
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()

    out_path = PROJECT_ROOT / "output" / "equity_curves_ols.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
