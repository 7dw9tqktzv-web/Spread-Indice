"""
Grid search SI/HG using Sierra Chart 1min data (exact same data as dashboard)
Target: z=+1.65, Kalman B=2.2768, A=0.426
"""
import numpy as np
import pandas as pd

# Load Sierra data
si = pd.read_csv("raw/SIK26_FUT_CME.scid_BarData.txt")
si["datetime"] = pd.to_datetime(si["Date"] + " " + si[" Time"])
hg = pd.read_csv("raw/HGK26_FUT_CME.scid_BarData.txt")
hg["datetime"] = pd.to_datetime(hg["Date"] + " " + hg[" Time"])

si = si.rename(columns={" Open": "Open", " High": "High", " Low": "Low",
                        " Last": "Last", " Volume": "Volume"})
hg = hg.rename(columns={" Open": "Open", " High": "High", " Low": "Low",
                        " Last": "Last", " Volume": "Volume"})

print(f"SI: {len(si)} bars, {si['datetime'].iloc[0]} to {si['datetime'].iloc[-1]}")
print(f"HG: {len(hg)} bars, {hg['datetime'].iloc[0]} to {hg['datetime'].iloc[-1]}")

# Align on datetime
si = si.set_index("datetime")
hg = hg.set_index("datetime")
common = si.index.intersection(hg.index)
print(f"Common bars: {len(common)}")

df = pd.DataFrame({
    "si": si.loc[common, "Last"].values.astype(float),
    "hg": hg.loc[common, "Last"].values.astype(float),
}, index=common)

# Last values
print(f"\nLast bar: {df.index[-1]}")
print(f"SI={df['si'].iloc[-1]:.3f}, HG={df['hg'].iloc[-1]:.4f}")
print(f"Ratio={df['si'].iloc[-1]/df['hg'].iloc[-1]:.4f}")
print(f"Log ratio={np.log(df['si'].iloc[-1]/df['hg'].iloc[-1]):.4f}")

# Parameters from dashboard
z_target = +1.65
kalman_beta = 2.2768
kalman_alpha = 0.426
ols_beta = 3.408
ols_alpha = -1.6105

log_si = np.log(df["si"].values)
log_hg = np.log(df["hg"].values)

# 3 series
series_ln = log_si - log_hg
series_beta_k = log_si - kalman_alpha - kalman_beta * log_hg  # Kalman beta
series_beta_o = log_si - ols_alpha - ols_beta * log_hg        # OLS beta

# Also test reversed: log_hg - log_si (in case dashboard uses HG/SI)
series_ln_rev = log_hg - log_si
series_beta_k_rev = log_hg - kalman_alpha - kalman_beta * log_si
series_beta_o_rev = log_hg - ols_alpha - ols_beta * log_si

all_series = {
    "ln(SI/HG)": series_ln,
    "beta_kalman": series_beta_k,
    "beta_ols": series_beta_o,
    "ln(HG/SI)": series_ln_rev,
    "beta_kalman_rev": series_beta_k_rev,
    "beta_ols_rev": series_beta_o_rev,
}

# Timeframes to test via resampling
resample_map = {"1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60}

print(f"\n{'='*80}")
print(f"  SI/HG Grid Search | Z target: {z_target}")
print(f"  Kalman B={kalman_beta} A={kalman_alpha} | OLS B={ols_beta} A={ols_alpha}")
print(f"{'='*80}")

for tf_name, tf_mult in resample_map.items():
    # Resample by taking every Nth bar (simple)
    if tf_mult == 1:
        df_tf = df
    else:
        rule = f"{tf_mult}min" if tf_mult < 60 else "1h"
        df_tf = df.resample(rule).last().dropna()

    if len(df_tf) < 50:
        continue

    log_a = np.log(df_tf["si"].values)
    log_b = np.log(df_tf["hg"].values)

    local_series = {
        "ln": log_a - log_b,
        "beta_k": log_a - kalman_alpha - kalman_beta * log_b,
        "beta_o": log_a - ols_alpha - ols_beta * log_b,
        "ln_rev": log_b - log_a,
        "beta_k_rev": log_b - kalman_alpha - kalman_beta * log_a,
        "beta_o_rev": log_b - ols_alpha - ols_beta * log_a,
    }

    max_zp = min(len(df_tf) - 5, 3000)
    print(f"\n--- {tf_name} ({len(df_tf)} bars, ZP max={max_zp}) ---")

    for sname, series in local_series.items():
        best_diff = 999
        best_zp = 0
        best_z = 0
        top5 = []

        for zp in range(10, max_zp + 1):
            sma = pd.Series(series).rolling(zp).mean().values[-1]
            std = pd.Series(series).rolling(zp).std(ddof=0).values[-1]
            if std > 1e-12:
                z = (series[-1] - sma) / std
                diff = abs(z - z_target)
                if len(top5) < 5 or diff < top5[-1][0]:
                    top5.append((diff, zp, z))
                    top5.sort()
                    top5 = top5[:5]

        if top5:
            d, zp, z = top5[0]
            lb = zp * tf_mult
            marker = " <<<" if d < 0.01 else (" **" if d < 0.05 else "")
            print(f"  [{sname:>12s}] best: ZP={zp:>5d} ({lb:>6.0f}min/{lb/60:>5.1f}h) "
                  f"z={z:>+8.4f} diff={d:.4f}{marker}")
            if d < 0.05:
                # Show cluster
                items = [(dd, zpp, zz) for dd, zpp, zz in top5 if dd < 0.1]
                zps = [zpp for _, zpp, _ in items]
                print(f"             top: {['ZP='+str(z) for z in zps]}")
