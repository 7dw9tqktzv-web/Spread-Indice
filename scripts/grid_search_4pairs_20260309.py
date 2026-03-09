"""
Grid search TF x ZP for 4 spread pairs - 2026-03-09 morning screenshots
Both ln and beta methods for ALL pairs (no pre-filtering on R2)
"""
import numpy as np
import pandas as pd
import yfinance as yf

pairs = {
    "SI/HG": {"a": "SI=F", "b": "HG=F", "z_target": +1.65,
              "kalman_beta": 2.2768, "kalman_alpha": 0.426},
    "NQ/RTY": {"a": "NQ=F", "b": "RTY=F", "z_target": +1.94,
               "kalman_beta": 1.2794, "kalman_alpha": 0.0903},
    "CL/NG": {"a": "CL=F", "b": "NG=F", "z_target": +1.66,
              "kalman_beta": 0.5328, "kalman_alpha": 3.6667},
    "ZC/ZW": {"a": "ZC=F", "b": "ZW=F", "z_target": -1.83,
              "kalman_beta": 0.9151, "kalman_alpha": 0.2589},
}

timeframes = ["1m", "5m", "15m", "30m", "1h"]
tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def fetch_pair(ticker_a, ticker_b):
    data = {}
    for tf in timeframes:
        period = "7d" if tf == "1m" else "60d"
        try:
            da = yf.download(ticker_a, period=period, interval=tf, progress=False)
            db = yf.download(ticker_b, period=period, interval=tf, progress=False)
            if da.empty or db.empty:
                continue
            if isinstance(da.columns, pd.MultiIndex):
                da.columns = da.columns.get_level_values(0)
            if isinstance(db.columns, pd.MultiIndex):
                db.columns = db.columns.get_level_values(0)
            common = da.index.intersection(db.index)
            if len(common) < 50:
                continue
            data[tf] = pd.DataFrame({
                "close_a": da.loc[common, "Close"].values,
                "close_b": db.loc[common, "Close"].values,
            }, index=common)
            print(f"  {tf}: {len(common)} bars")
        except Exception as e:
            print(f"  ERROR {tf}: {e}")
    return data


def grid_search(df, z_target, kalman_beta, kalman_alpha):
    log_a = np.log(df["close_a"].values)
    log_b = np.log(df["close_b"].values)

    series_ln = log_a - log_b
    series_beta = log_a - kalman_alpha - kalman_beta * log_b

    max_zp = min(len(df) - 5, 2000)
    results = []

    for zp in range(10, max_zp + 1):
        for name, series in [("ln", series_ln), ("beta", series_beta)]:
            sma = pd.Series(series).rolling(zp).mean().values
            std = pd.Series(series).rolling(zp).std(ddof=0).values
            if std[-1] > 1e-12:
                z = (series[-1] - sma[-1]) / std[-1]
                results.append({"method": name, "zp": zp,
                              "z": round(z, 4), "diff": round(abs(z - z_target), 4)})

    return pd.DataFrame(results)


def show_results(res_df, tf, method):
    sub = res_df[res_df["method"] == method].sort_values("diff").head(5)
    if sub.empty:
        return
    mult = tf_minutes[tf]
    best_diff = sub.iloc[0]["diff"]
    print(f"  [{method.upper():>4s}] best diff={best_diff:.4f}  |  ", end="")
    for _, row in sub.iterrows():
        lb = row['zp'] * mult
        print(f"ZP={int(row['zp'])}({lb:.0f}m) z={row['z']:+.4f}  ", end="")
    print()

    # Cluster detection on top 10
    top10 = res_df[res_df["method"] == method].sort_values("diff").head(10)
    zps = sorted(top10["zp"].values)
    clusters = []
    cur = [zps[0]]
    for i in range(1, len(zps)):
        if zps[i] - zps[i-1] <= 5:
            cur.append(zps[i])
        else:
            if len(cur) >= 3:
                clusters.append(f"{int(cur[0])}-{int(cur[-1])}")
            cur = [zps[i]]
    if len(cur) >= 3:
        clusters.append(f"{int(cur[0])}-{int(cur[-1])}")
    if clusters:
        print(f"         clusters: [{', '.join(clusters)}]")


if __name__ == "__main__":
    for pair_name, cfg in pairs.items():
        print(f"\n{'='*70}")
        print(f"  {pair_name}  |  Z target: {cfg['z_target']}")
        print(f"  Kalman B={cfg['kalman_beta']}  A={cfg['kalman_alpha']}")
        print(f"{'='*70}")

        print(f"\nFetching {cfg['a']} / {cfg['b']}...")
        data = fetch_pair(cfg['a'], cfg['b'])

        for tf, df in data.items():
            print(f"\n  --- {tf} ({len(df)} bars, ZP max={min(len(df)-5, 2000)}) ---")
            res = grid_search(df, cfg['z_target'], cfg['kalman_beta'], cfg['kalman_alpha'])
            if res.empty:
                continue
            show_results(res, tf, "ln")
            show_results(res, tf, "beta")

            # Winner for this TF
            best_ln = res[res["method"] == "ln"]["diff"].min()
            best_beta = res[res["method"] == "beta"]["diff"].min()
            winner = "LN" if best_ln <= best_beta else "BETA"
            print(f"         >> {winner} wins ({min(best_ln, best_beta):.4f} vs {max(best_ln, best_beta):.4f})")
