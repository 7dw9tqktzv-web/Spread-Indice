"""End-to-end backtest pipeline for spread trading.

Usage:
    python scripts/run_backtest.py --pair NQ_ES --method ols_rolling
    python scripts/run_backtest.py --all --method kalman
    python scripts/run_backtest.py --prepare-data
    python scripts/run_backtest.py --pair NQ_ES --method ols_rolling -v
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestConfig, BacktestEngine, InstrumentSpec
from src.backtest.performance import compute_performance
from src.data.alignment import align_pair
from src.data.cache import cache_aligned_pair, load_aligned_pair_cache
from src.data.cleaner import clean
from src.data.loader import load_sierra_csv
from src.data.resampler import resample_to_5min
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import (
    ConfidenceConfig,
    apply_confidence_filter,
    apply_trading_window_filter,
)
from src.signals.generator import SignalConfig, SignalGenerator
from src.spread.pair import SpreadPair
from src.utils.constants import RAW_FILE_PATTERN, Instrument
from src.utils.time_utils import SessionConfig, parse_session_config

log = logging.getLogger("backtest")

OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_backtest_yaml() -> dict:
    path = PROJECT_ROOT / "config" / "backtest.yaml"
    log.info(f"Loading config from {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_instruments_yaml() -> dict:
    path = PROJECT_ROOT / "config" / "instruments.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs_yaml() -> dict:
    path = PROJECT_ROOT / "config" / "pairs.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_session_config(cfg: dict) -> SessionConfig:
    return parse_session_config(cfg["session"])


def build_signal_config(cfg: dict) -> SignalConfig:
    s = cfg["signals"]
    return SignalConfig(z_entry=s["z_entry"], z_exit=s["z_exit"], z_stop=s["z_stop"])


def build_backtest_config(cfg: dict) -> BacktestConfig:
    b = cfg["backtest"]
    return BacktestConfig(
        initial_capital=b["initial_capital"],
        commission_per_contract=b["commission_per_contract"],
        slippage_ticks=b["slippage_ticks"],
    )


def build_metrics_config(cfg: dict) -> MetricsConfig:
    m = cfg["metrics"]
    return MetricsConfig(
        adf_window=m["adf_window"],
        hurst_window=m["hurst_window"],
        halflife_window=m["halflife_window"],
        correlation_window=m["correlation_window"],
    )


def build_confidence_config(cfg: dict) -> ConfidenceConfig:
    r = cfg.get("regime_filter", {})
    return ConfidenceConfig(
        min_confidence=r.get("min_confidence", 40.0),
        adf_gate=r.get("adf_gate", -1.00),
        w_adf=r.get("w_adf", 0.40),
        w_hurst=r.get("w_hurst", 0.25),
        w_corr=r.get("w_correlation", 0.20),
        w_hl=r.get("w_halflife", 0.15),
    )


def build_instrument_spec(instruments: dict, name: str) -> InstrumentSpec:
    s = instruments[name]
    return InstrumentSpec(
        multiplier=s["multiplier"],
        tick_size=s["tick_size"],
        tick_value=s["tick_value"],
    )


def build_hedge_kwargs(cfg: dict, method: str) -> dict:
    h = cfg["hedge"]
    if method == "ols_rolling":
        return {"window": h["ols_window"], "zscore_window": h["zscore_window"]}
    elif method == "kalman":
        return {
            "alpha_ratio": h["kalman_alpha_ratio"],
            "warmup": h["kalman_warmup"],
            "gap_P_multiplier": h["kalman_gap_P_mult"],
        }
    return {}


def resolve_pair(pairs_cfg: dict, pair_name: str) -> SpreadPair:
    p = pairs_cfg["pairs"][pair_name]
    return SpreadPair(leg_a=Instrument(p["leg_a"]), leg_b=Instrument(p["leg_b"]))


def all_pair_names(pairs_cfg: dict) -> list[str]:
    return list(pairs_cfg["pairs"].keys())


# ---------------------------------------------------------------------------
# Data preparation (cacheable)
# ---------------------------------------------------------------------------

def prepare_instrument(instrument: Instrument, session: SessionConfig) -> None:
    """Load, clean, resample a single instrument. Returns BarData 5min."""
    raw_path = PROJECT_ROOT / "raw" / RAW_FILE_PATTERN.format(symbol=instrument.value)
    log.info(f"[DATA] Loading {instrument.value} ({raw_path.name})")

    t0 = time.time()
    data = load_sierra_csv(raw_path, instrument)
    log.info(f"[DATA] {instrument.value}: {len(data.df):,} bars 1min loaded ({time.time()-t0:.1f}s)")

    data = clean(data, session)
    log.info(f"[DATA] {instrument.value}: {len(data.df):,} bars after clean")

    data = resample_to_5min(data)
    log.info(f"[DATA] {instrument.value}: {len(data.df):,} bars 5min after resample")

    return data


def prepare_pair_data(pair: SpreadPair, session: SessionConfig):
    """Prepare aligned pair data with cache."""
    cached = load_aligned_pair_cache(pair, "5min")
    if cached is not None:
        log.info(f"[DATA] Cache hit: {pair.name} ({len(cached.df):,} bars)")
        return cached

    data_a = prepare_instrument(pair.leg_a, session)
    data_b = prepare_instrument(pair.leg_b, session)

    aligned = align_pair(data_a, data_b, pair)
    log.info(f"[DATA] Aligned {pair.name}: {len(aligned.df):,} bars")

    path = cache_aligned_pair(aligned)
    log.info(f"[DATA] Cached -> {path}")

    return aligned


# ---------------------------------------------------------------------------
# Single pair backtest
# ---------------------------------------------------------------------------

def run_single_pair(
    pair_name: str,
    method: str,
    cfg: dict,
    instruments: dict,
    pairs_cfg: dict,
) -> dict | None:
    """Run full backtest pipeline on one pair. Returns summary dict."""
    pair = resolve_pair(pairs_cfg, pair_name)
    session = build_session_config(cfg)
    signal_cfg = build_signal_config(cfg)
    backtest_cfg = build_backtest_config(cfg)
    metrics_cfg = build_metrics_config(cfg)
    confidence_cfg = build_confidence_config(cfg)

    log.info(f"=== Pair {pair_name} | Method {method} ===")

    # --- 1. Data prep ---
    aligned = prepare_pair_data(pair, session)
    close_a = aligned.df["close_a"]
    close_b = aligned.df["close_b"]

    # --- 2. Hedge ratio ---
    hedge_kwargs = build_hedge_kwargs(cfg, method)
    log.info(f"[HEDGE] Computing {method} ({hedge_kwargs})")

    t0 = time.time()
    estimator = create_estimator(method, **hedge_kwargs)
    result_hedge = estimator.estimate(aligned)
    log.info(
        f"[HEDGE] Beta: mean={result_hedge.beta.dropna().mean():.4f}, "
        f"std={result_hedge.beta.dropna().std():.4f} ({time.time()-t0:.1f}s)"
    )

    # --- 3. Metrics ---
    log.info("[METRICS] Computing ADF/Hurst/HalfLife/Correlation")
    metrics = compute_all_metrics(result_hedge.spread, close_a, close_b, metrics_cfg)

    # --- 4. Signals ---
    gen = SignalGenerator(config=signal_cfg)
    raw_signals = gen.generate(result_hedge.zscore)
    n_raw = int((raw_signals != 0).sum())

    filtered_signals = apply_confidence_filter(raw_signals, metrics, confidence_cfg)
    n_regime = int((filtered_signals != 0).sum())

    final_signals = apply_trading_window_filter(filtered_signals, session)
    n_final = int((final_signals != 0).sum())

    log.info(f"[SIGNALS] Bars with signal: {n_raw} raw, {n_regime} after regime, {n_final} after trading window")

    # --- 5. Backtest ---
    spec_a = build_instrument_spec(instruments, pair.leg_a.value)
    spec_b = build_instrument_spec(instruments, pair.leg_b.value)

    log.info(
        f"[BACKTEST] Running engine (capital=${backtest_cfg.initial_capital:,.0f}, "
        f"slippage={backtest_cfg.slippage_ticks} tick, commission=${backtest_cfg.commission_per_contract})"
    )

    engine = BacktestEngine(config=backtest_cfg)
    bt_result = engine.run(close_a, close_b, final_signals, result_hedge.beta, spec_a, spec_b)

    log.info(f"[BACKTEST] {len(bt_result.trades)} trades executed")

    # --- 6. Performance ---
    perf = compute_performance(bt_result)

    # --- 7. Output ---
    _print_results(pair_name, method, perf)
    summary = _save_outputs(pair_name, method, bt_result, perf)

    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_results(pair_name: str, method: str, perf) -> None:
    log.info("[RESULT] " + "=" * 45)
    log.info(f"[RESULT]  {pair_name} | {method}")
    log.info(f"[RESULT]  Total PnL:      ${perf.total_pnl:>12,.2f}")
    log.info(f"[RESULT]  Trades:          {perf.num_trades:>8}")
    log.info(f"[RESULT]  Win Rate:        {perf.win_rate:>8.1f}%")
    log.info(f"[RESULT]  Profit Factor:   {perf.profit_factor:>8.2f}")
    log.info(f"[RESULT]  Avg PnL/Trade:  ${perf.avg_pnl_per_trade:>12,.2f}")
    log.info(f"[RESULT]  Sharpe:          {perf.sharpe_ratio:>8.2f}")
    log.info(f"[RESULT]  Max Drawdown:    {perf.max_drawdown_pct:>8.2f}%")
    log.info(f"[RESULT]  DD Duration:     {perf.max_drawdown_duration:>8} bars")
    log.info(f"[RESULT]  Calmar:          {perf.calmar_ratio:>8.2f}")
    log.info("[RESULT] " + "=" * 45)


def _save_outputs(pair_name: str, method: str, bt_result, perf) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{pair_name}_{method}"

    # Trades CSV
    if bt_result.trades:
        import pandas as pd

        trades_data = []
        for t in bt_result.trades:
            trades_data.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "entry_price_a": t.entry_price_a,
                "entry_price_b": t.entry_price_b,
                "exit_price_a": t.exit_price_a,
                "exit_price_b": t.exit_price_b,
                "n_a": t.n_a,
                "n_b": t.n_b,
                "pnl_gross": t.pnl_gross,
                "costs": t.costs,
                "pnl_net": t.pnl_net,
            })
        trades_path = OUTPUT_DIR / f"{prefix}_trades.csv"
        pd.DataFrame(trades_data).to_csv(trades_path, index=False)
        log.info(f"[OUTPUT] Trades -> {trades_path}")

    # Equity CSV
    equity_path = OUTPUT_DIR / f"{prefix}_equity.csv"
    bt_result.equity_curve.to_csv(equity_path, header=True)
    log.info(f"[OUTPUT] Equity -> {equity_path}")

    # Summary JSON
    summary = {
        "pair": pair_name,
        "method": method,
        **asdict(perf),
    }
    summary_path = OUTPUT_DIR / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[OUTPUT] Summary -> {summary_path}")

    return summary


def _print_recap(summaries: list[dict], method: str) -> None:
    log.info("")
    log.info("=" * 70)
    log.info(f" RECAP | {method} | {len(summaries)} paires")
    log.info("=" * 70)
    log.info(f" {'Pair':<10} | {'PnL':>10} | {'Trades':>6} | {'WinRate':>7} | {'Sharpe':>6} | {'MaxDD':>6}")
    log.info("-" * 70)
    for s in summaries:
        log.info(
            f" {s['pair']:<10} | ${s['total_pnl']:>9,.0f} | {s['num_trades']:>6} | "
            f"{s['win_rate']:>6.1f}% | {s['sharpe_ratio']:>6.2f} | {s['max_drawdown_pct']:>5.1f}%"
        )
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spread trading backtest pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_backtest.py --pair NQ_ES --method ols_rolling
  python scripts/run_backtest.py --all --method kalman
  python scripts/run_backtest.py --prepare-data
        """,
    )
    parser.add_argument("--pair", type=str, help="Pair to backtest (e.g. NQ_ES)")
    parser.add_argument("--all", action="store_true", help="Run all 6 pairs sequentially")
    parser.add_argument("--method", type=str, default="ols_rolling",
                        choices=["ols_rolling", "kalman"], help="Hedge ratio method")
    parser.add_argument("--prepare-data", action="store_true", help="Data prep only (cache Parquet)")
    parser.add_argument("--ols-window", type=int, default=None,
                        help="Override OLS window (bars). Default: from backtest.yaml")
    parser.add_argument("--z-entry", type=float, default=None, help="Override z_entry threshold")
    parser.add_argument("--z-exit", type=float, default=None, help="Override z_exit threshold")
    parser.add_argument("--z-stop", type=float, default=None, help="Override z_stop threshold")
    parser.add_argument("--alpha-ratio", type=float, default=None, help="Override Kalman alpha_ratio")
    parser.add_argument("--min-confidence", type=float, default=None, help="Override min_confidence (%%)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging (DEBUG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not (args.pair or args.all or args.prepare_data):
        log.error("Specify --pair, --all, or --prepare-data")
        sys.exit(1)

    cfg = load_backtest_yaml()
    instruments = load_instruments_yaml()
    pairs_cfg = load_pairs_yaml()
    session = build_session_config(cfg)

    t_start = time.time()

    # --- Prepare data mode ---
    if args.prepare_data:
        log.info("[DATA] Preparing all pairs (cache Parquet)")
        for pair_name in all_pair_names(pairs_cfg):
            pair = resolve_pair(pairs_cfg, pair_name)
            prepare_pair_data(pair, session)
        elapsed = time.time() - t_start
        log.info(f"[DATA] All pairs cached in {elapsed:.1f}s")
        return

    # --- CLI overrides ---
    if args.ols_window is not None:
        cfg["hedge"]["ols_window"] = args.ols_window
        log.info(f"[CONFIG] OLS window override: {args.ols_window} bars")
    if args.z_entry is not None:
        cfg["signals"]["z_entry"] = args.z_entry
        log.info(f"[CONFIG] z_entry override: {args.z_entry}")
    if args.z_exit is not None:
        cfg["signals"]["z_exit"] = args.z_exit
        log.info(f"[CONFIG] z_exit override: {args.z_exit}")
    if args.z_stop is not None:
        cfg["signals"]["z_stop"] = args.z_stop
        log.info(f"[CONFIG] z_stop override: {args.z_stop}")
    if args.alpha_ratio is not None:
        cfg["hedge"]["kalman_alpha_ratio"] = args.alpha_ratio
        log.info(f"[CONFIG] alpha_ratio override: {args.alpha_ratio}")
    if args.min_confidence is not None:
        cfg.setdefault("regime_filter", {})["min_confidence"] = args.min_confidence
        log.info(f"[CONFIG] min_confidence override: {args.min_confidence}%")

    # --- Determine pairs to run ---
    if args.all:
        pair_names = all_pair_names(pairs_cfg)
    else:
        if args.pair not in pairs_cfg["pairs"]:
            log.error(f"Unknown pair: {args.pair}. Available: {all_pair_names(pairs_cfg)}")
            sys.exit(1)
        pair_names = [args.pair]

    # --- Run backtests ---
    summaries = []
    for pair_name in pair_names:
        try:
            summary = run_single_pair(pair_name, args.method, cfg, instruments, pairs_cfg)
            if summary:
                summaries.append(summary)
        except Exception:
            log.exception(f"[ERROR] Failed on {pair_name}")

    # --- Recap ---
    if len(summaries) > 1:
        _print_recap(summaries, args.method)

    elapsed = time.time() - t_start
    log.info(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
