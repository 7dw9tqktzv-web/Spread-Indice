"""Grid search TF x ZP anchored on screenshot prices."""
import yfinance as yf
import numpy as np
import pandas as pd
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGET_Z = 1.71
TARGET_LOG = 2.7166

# Kalman/OLS params from screenshot
KALMAN_ALPHA = 0.4582
KALMAN_BETA = 2.2584
OLS_ALPHA = -1.6893
OLS_BETA = 3.4537

TF_MINUTES = {"1min": 1, "2min": 2, "5min": 5, "15min": 15, "30min": 30, "1h": 60}
TIMEFRAMES = {
    "1min": ("5d", "1m"),
    "2min": ("5d", "2m"),
    "5min": ("5d", "5m"),
    "15min": ("1mo", "15m"),
    "30min": ("1mo", "30m"),
    "1h": ("1mo", "60m"),
}


def fetch_data():
    all_data = {}
    for tf_name, (period, interval) in TIMEFRAMES.items():
        si = yf.download("SI=F", period=period, interval=interval, progress=False)
        hg = yf.download("HG=F", period=period, interval=interval, progress=False)
        common = si.index.intersection(hg.index)
        si_s = pd.Series(si.loc[common, "Close"].values.flatten(), index=common)
        hg_s = pd.Series(hg.loc[common, "Close"].values.flatten(), index=common)
        all_data[tf_name] = (si_s, hg_s, common)
        print(f"{tf_name}: {len(common)} barres")
    return all_data


def find_anchors(all_data):
    """Find bar closest to screenshot prices for each TF."""
    anchor_indices = {}
    print("\nANCRAGE par TF:")
    print(f"{'TF':>5} | {'Timestamp':>30} | {'SI':>7} | {'HG':>7} | {'Log':>8} | {'Bar#':>5}")
    print("-" * 80)

    for tf_name, (si_s, hg_s, common) in all_data.items():
        log_r = np.log(si_s) - np.log(hg_s)
        score = (log_r - TARGET_LOG).abs() + 0.01 * (si_s - 88.4).abs()
        best_ts = score.idxmin()
        best_pos = list(common).index(best_ts)
        anchor_indices[tf_name] = best_pos
        print(
            f"{tf_name:>5} | {str(best_ts):>30} | {si_s[best_ts]:>7.2f} | "
            f"{hg_s[best_ts]:>7.4f} | {log_r[best_ts]:>8.4f} | {best_pos:>5}"
        )
    return anchor_indices


def grid_search(all_data, anchor_indices):
    results = []
    for tf_name, (si_s, hg_s, common) in all_data.items():
        anchor = anchor_indices[tf_name]
        log_si = np.log(si_s.values)
        log_hg = np.log(hg_s.values)

        kalman_sp = pd.Series(log_si - KALMAN_ALPHA - KALMAN_BETA * log_hg)
        ols_sp = pd.Series(log_si - OLS_ALPHA - OLS_BETA * log_hg)
        models = {"Kalman": kalman_sp, "OLS": ols_sp}

        max_zp = min(anchor, 500)

        for model_name, series in models.items():
            for zp in range(5, max_zp + 1):
                window = series.iloc[anchor - zp + 1 : anchor + 1]
                mu = window.mean()
                sigma = window.std()
                if sigma <= 1e-10:
                    continue
                z = (series.iloc[anchor] - mu) / sigma
                diff_z = abs(z - TARGET_Z)

                # Persistence: check +-5 bars around anchor
                hits = 0
                for offset in range(-5, 6):
                    idx = anchor + offset
                    if idx >= zp and idx < len(series):
                        w = series.iloc[idx - zp + 1 : idx + 1]
                        mu_o, sigma_o = w.mean(), w.std()
                        if sigma_o > 1e-10:
                            z_o = (series.iloc[idx] - mu_o) / sigma_o
                            if abs(z_o - TARGET_Z) < 0.10:
                                hits += 1

                dur = zp * TF_MINUTES[tf_name]
                results.append({
                    "tf": tf_name,
                    "model": model_name,
                    "zp": zp,
                    "z": z,
                    "diff": diff_z,
                    "hits": hits,
                    "dur_min": dur,
                })
    return pd.DataFrame(results)


def find_clusters(zps, hits_arr, z_arr, dur_arr, diff_arr):
    clusters = []
    cl_start = 0
    for i in range(1, len(zps)):
        if zps[i] - zps[i - 1] > 3:
            clusters.append((cl_start, i - 1))
            cl_start = i
    clusters.append((cl_start, len(zps) - 1))
    return clusters


def print_clusters_by_model(df):
    for model in ["Kalman", "OLS"]:
        print(f"\n{'=' * 90}")
        print(f"  {model} -- Clusters ancres sur screenshot (SI~88.4, HG~5.8)")
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

            clusters = find_clusters(zps, hits_arr, z_arr, dur_arr, diff_arr)

            print(f"\n  {tf}:")
            for start, end in clusters:
                sl = slice(start, end + 1)
                cl_zps = zps[sl]
                cl_h = hits_arr[sl]
                cl_z = z_arr[sl]
                cl_d = dur_arr[sl]
                cl_diff = diff_arr[sl]
                max_h = max(cl_h)
                best_i = np.argmin(cl_diff)
                avg_dur = np.mean(cl_d)
                dur_s = f"~{avg_dur:.0f}min" if avg_dur < 60 else f"~{avg_dur / 60:.1f}h"
                print(
                    f"    ZP {cl_zps[0]:>4}-{cl_zps[-1]:<4} ({dur_s:>7}) | "
                    f"{len(cl_zps):>2} vals | persist={max_h} | "
                    f"best ZP={cl_zps[best_i]} z={cl_z[best_i]:.3f}"
                )


def print_cross_tf_synthesis(df):
    print(f"\n{'=' * 90}")
    print("  SYNTHESE CROSS-TF normalisee en duree")
    print(f"{'=' * 90}")

    bins = [0, 15, 45, 90, 180, 360, 720, 1500, 3000]
    labels = ["<15min", "15-45min", "45-90min", "1.5-3h", "3-6h", "6-12h", "12-25h", "25h+"]

    for model in ["Kalman", "OLS"]:
        sub = df[(df["model"] == model) & (df["diff"] < 0.12)].copy()
        if sub.empty:
            print(f"\n  {model}: aucun match fort")
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


def print_final_table(df):
    print(f"\n{'=' * 90}")
    print("  TABLE FINALE -- Meilleur ZP par TF (diff minimum au z=1.71)")
    print(f"{'=' * 90}")
    print(f"  {'TF':<6} | {'--- Kalman ---':^30} | {'--- OLS ---':^30}")
    h = f"  {'':>6} | {'ZP':>5} {'Duree':>8} {'z':>7} {'pers':>5} | "
    h += f"{'ZP':>5} {'Duree':>8} {'z':>7} {'pers':>5}"
    print(h)
    print(f"  {'-' * 70}")

    for tf in TIMEFRAMES:
        row = f"  {tf:<6} |"
        for model in ["Kalman", "OLS"]:
            sub = df[(df["model"] == model) & (df["tf"] == tf) & (df["diff"] < 0.5)]
            if sub.empty:
                row += f" {'--':>5} {'':>8} {'':>7} {'':>5} |"
            else:
                best = sub.loc[sub["diff"].idxmin()]
                dur = best["dur_min"]
                dur_s = f"{dur:.0f}min" if dur < 60 else f"{dur / 60:.1f}h"
                row += (
                    f" {int(best['zp']):>5} {dur_s:>8} "
                    f"{best['z']:>7.3f} {int(best['hits']):>5} |"
                )
        print(row)


if __name__ == "__main__":
    all_data = fetch_data()
    anchors = find_anchors(all_data)
    df = grid_search(all_data, anchors)
    print_clusters_by_model(df)
    print_cross_tf_synthesis(df)
    print_final_table(df)
