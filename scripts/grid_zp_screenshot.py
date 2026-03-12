"""Grid search TF x ZP anchored on screenshot prices.

Usage:
    python scripts/grid_zp_screenshot.py
    Edit the PARAMS dict below for each new screenshot.
"""
import yfinance as yf
import numpy as np
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# =====================================================================
# PARAMS -- edit these for each screenshot
# =====================================================================
PARAMS = {
    "pair": "ES / YM",
    "ticker_a": "ES=F",
    "ticker_b": "YM=F",
    "target_z": 2.33,
    "target_log": -1.9441,
    "target_price_a": 6726.3,
    "kalman_alpha": -0.0162,
    "kalman_beta": 0.8207,
    "ols_alpha": 3.6629,
    "ols_beta": 0.4796,
    # Set True if beta close to 1 (viable), False if beta >> 1
    "include_ln": True,
}

TF_MINUTES = {"1min": 1, "2min": 2, "5min": 5, "15min": 15, "30min": 30, "1h": 60}
TIMEFRAMES = {
    "1min": ("5d", "1m"),
    "2min": ("5d", "2m"),
    "5min": ("5d", "5m"),
    "15min": ("1mo", "15m"),
    "30min": ("1mo", "30m"),
    "1h": ("1mo", "60m"),
}


def fetch_data(ticker_a, ticker_b):
    all_data = {}
    for tf_name, (period, interval) in TIMEFRAMES.items():
        a = yf.download(ticker_a, period=period, interval=interval, progress=False)
        b = yf.download(ticker_b, period=period, interval=interval, progress=False)
        common = a.index.intersection(b.index)
        a_s = pd.Series(a.loc[common, "Close"].values.flatten(), index=common)
        b_s = pd.Series(b.loc[common, "Close"].values.flatten(), index=common)
        all_data[tf_name] = (a_s, b_s, common)
        print(f"{tf_name}: {len(common)} barres, {common[0]} -> {common[-1]}")
    return all_data


def find_anchors(all_data, p):
    anchor_indices = {}
    print(f"\nANCRAGE (cible: A={p['target_price_a']}, Log={p['target_log']}):")
    print(f"{'TF':>5} | {'Timestamp':>30} | {'A':>9} | {'B':>9} | {'Log':>8} | {'Bar#':>5}")
    print("-" * 85)

    for tf_name, (a_s, b_s, common) in all_data.items():
        log_r = np.log(a_s) - np.log(b_s)
        score = (log_r - p["target_log"]).abs() + 0.0001 * (a_s - p["target_price_a"]).abs()
        best_ts = score.idxmin()
        best_pos = list(common).index(best_ts)
        anchor_indices[tf_name] = best_pos
        print(
            f"{tf_name:>5} | {str(best_ts):>30} | {a_s[best_ts]:>9.2f} | "
            f"{b_s[best_ts]:>9.2f} | {log_r[best_ts]:>8.4f} | {best_pos:>5}"
        )
    return anchor_indices


def grid_search(all_data, anchor_indices, p):
    results = []
    models_cfg = {
        "Kalman": (p["kalman_alpha"], p["kalman_beta"]),
        "OLS": (p["ols_alpha"], p["ols_beta"]),
    }

    for tf_name, (a_s, b_s, common) in all_data.items():
        anchor = anchor_indices[tf_name]
        log_a = np.log(a_s.values)
        log_b = np.log(b_s.values)

        models = {}
        if p["include_ln"]:
            models["ln(A/B)"] = pd.Series(log_a - log_b)
        for name, (alpha, beta) in models_cfg.items():
            models[name] = pd.Series(log_a - alpha - beta * log_b)

        max_zp = min(anchor, 500)

        for model_name, series in models.items():
            for zp in range(5, max_zp + 1):
                window = series.iloc[anchor - zp + 1: anchor + 1]
                mu = window.mean()
                sigma = window.std()
                if sigma <= 1e-10:
                    continue
                z = (series.iloc[anchor] - mu) / sigma
                diff_z = abs(z - p["target_z"])

                hits = 0
                for offset in range(-5, 6):
                    idx = anchor + offset
                    if idx >= zp and idx < len(series):
                        w = series.iloc[idx - zp + 1: idx + 1]
                        mu_o, sigma_o = w.mean(), w.std()
                        if sigma_o > 1e-10:
                            z_o = (series.iloc[idx] - mu_o) / sigma_o
                            if abs(z_o - p["target_z"]) < 0.10:
                                hits += 1

                dur = zp * TF_MINUTES[tf_name]
                results.append({
                    "tf": tf_name, "model": model_name, "zp": zp,
                    "z": z, "diff": diff_z, "hits": hits, "dur_min": dur,
                })
    return pd.DataFrame(results)


def print_clusters_by_model(df, p):
    model_names = ["ln(A/B)", "Kalman", "OLS"] if p["include_ln"] else ["Kalman", "OLS"]
    for model in model_names:
        print(f"\n{'=' * 90}")
        print(f"  {model} -- Clusters ancres sur screenshot ({p['pair']})")
        print(f"{'=' * 90}")

        sub = df[(df["model"] == model) & ((df["diff"] < 0.15) | (df["hits"] >= 3))]
        sub = sub.sort_values(["tf", "zp"])

        for tf in TIMEFRAMES:
            tf_data = sub[sub["tf"] == tf]
            if tf_data.empty:
                print(f"\n  {tf}: aucun match")
                continue

            zps = tf_data["zp"].values
            hits_arr = tf_data["hits"].values
            z_arr = tf_data["z"].values
            dur_arr = tf_data["dur_min"].values
            diff_arr = tf_data["diff"].values

            clusters = _find_clusters(zps)

            print(f"\n  {tf}:")
            for start, end in clusters:
                sl = slice(start, end + 1)
                cl_zps, cl_h = zps[sl], hits_arr[sl]
                cl_z, cl_d, cl_diff = z_arr[sl], dur_arr[sl], diff_arr[sl]
                max_h = max(cl_h)
                best_i = np.argmin(cl_diff)
                avg_dur = np.mean(cl_d)
                dur_s = f"~{avg_dur:.0f}min" if avg_dur < 60 else f"~{avg_dur / 60:.1f}h"
                print(
                    f"    ZP {cl_zps[0]:>4}-{cl_zps[-1]:<4} ({dur_s:>7}) | "
                    f"{len(cl_zps):>2} vals | persist={max_h} | "
                    f"best ZP={cl_zps[best_i]} z={cl_z[best_i]:.3f}"
                )


def print_cross_tf_synthesis(df, p):
    print(f"\n{'=' * 90}")
    print("  SYNTHESE CROSS-TF normalisee en duree")
    print(f"{'=' * 90}")

    bins = [0, 15, 45, 90, 180, 360, 720, 1500, 3000]
    labels = ["<15min", "15-45min", "45-90min", "1.5-3h", "3-6h", "6-12h", "12-25h", "25h+"]

    model_names = ["ln(A/B)", "Kalman", "OLS"] if p["include_ln"] else ["Kalman", "OLS"]
    for model in model_names:
        sub = df[(df["model"] == model) & (df["diff"] < 0.12)].copy()
        if sub.empty:
            print(f"\n  {model}: aucun match fort (diff < 0.12)")
            continue

        sub["dur_bucket"] = pd.cut(sub["dur_min"], bins=bins, labels=labels)
        print(f"\n  {model}:")
        for bucket in labels:
            b_data = sub[sub["dur_bucket"] == bucket]
            if b_data.empty:
                continue
            n_tf = b_data["tf"].nunique()
            max_persist = b_data["hits"].max()
            best = b_data.loc[b_data["diff"].idxmin()]
            stars = "*" * n_tf
            print(
                f"    {bucket:>10} | {n_tf} TFs {stars:<5} | persist={max_persist} | "
                f"ex: {best['tf']} ZP={int(best['zp'])} z={best['z']:.3f}"
            )


def print_final_table(df, p):
    print(f"\n{'=' * 90}")
    tz = p["target_z"]
    print(f"  TABLE FINALE -- Meilleur ZP par TF (diff min au z={tz})")
    print(f"{'=' * 90}")

    model_names = ["ln(A/B)", "Kalman", "OLS"] if p["include_ln"] else ["Kalman", "OLS"]
    header = f"  {'TF':<6} |"
    for m in model_names:
        header += f" {'--- ' + m + ' ---':^24} |"
    print(header)
    sub_h = f"  {'':>6} |"
    for _ in model_names:
        sub_h += f" {'ZP':>4} {'dur':>7} {'z':>6} {'p':>3} |"
    print(sub_h)
    print(f"  {'-' * (24 + 3) * len(model_names)}")

    for tf in TIMEFRAMES:
        row = f"  {tf:<6} |"
        for model in model_names:
            sub = df[(df["model"] == model) & (df["tf"] == tf) & (df["diff"] < 0.5)]
            if sub.empty:
                row += f" {'--':>4} {'':>7} {'':>6} {'':>3} |"
            else:
                best = sub.loc[sub["diff"].idxmin()]
                dur = best["dur_min"]
                dur_s = f"{dur:.0f}m" if dur < 60 else f"{dur / 60:.1f}h"
                row += (
                    f" {int(best['zp']):>4} {dur_s:>7}"
                    f" {best['z']:>6.2f} {int(best['hits']):>3} |"
                )
        print(row)


def _find_clusters(zps):
    clusters = []
    cl_start = 0
    for i in range(1, len(zps)):
        if zps[i] - zps[i - 1] > 3:
            clusters.append((cl_start, i - 1))
            cl_start = i
    clusters.append((cl_start, len(zps) - 1))
    return clusters


if __name__ == "__main__":
    p = PARAMS
    all_data = fetch_data(p["ticker_a"], p["ticker_b"])
    anchors = find_anchors(all_data, p)
    df = grid_search(all_data, anchors, p)
    print_clusters_by_model(df, p)
    print_cross_tf_synthesis(df, p)
    print_final_table(df, p)
