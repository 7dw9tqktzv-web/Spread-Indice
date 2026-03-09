"""Compare Sierra vs yfinance data for SI and HG"""
import pandas as pd
import numpy as np
import yfinance as yf

# Load Sierra
si_sc = pd.read_csv("raw/SIK26_FUT_CME.scid_BarData.txt")
si_sc["datetime"] = pd.to_datetime(si_sc["Date"] + " " + si_sc[" Time"])
si_sc = si_sc.set_index("datetime")
si_sc = si_sc.rename(columns={" Open": "Open", " High": "High", " Low": "Low",
                              " Last": "Last", " Volume": "Volume"})

hg_sc = pd.read_csv("raw/HGK26_FUT_CME.scid_BarData.txt")
hg_sc["datetime"] = pd.to_datetime(hg_sc["Date"] + " " + hg_sc[" Time"])
hg_sc = hg_sc.set_index("datetime")
hg_sc = hg_sc.rename(columns={" Open": "Open", " High": "High", " Low": "Low",
                              " Last": "Last", " Volume": "Volume"})

# Load yfinance
si_yf = yf.download("SI=F", period="7d", interval="1m", progress=False)
hg_yf = yf.download("HG=F", period="7d", interval="1m", progress=False)
if isinstance(si_yf.columns, pd.MultiIndex):
    si_yf.columns = si_yf.columns.get_level_values(0)
if isinstance(hg_yf.columns, pd.MultiIndex):
    hg_yf.columns = hg_yf.columns.get_level_values(0)

print(f"Sierra SI: {si_sc.index[0]} to {si_sc.index[-1]} ({len(si_sc)} bars)")
print(f"Sierra HG: {hg_sc.index[0]} to {hg_sc.index[-1]} ({len(hg_sc)} bars)")
print(f"yfinance SI: {si_yf.index[0]} to {si_yf.index[-1]} ({len(si_yf)} bars)")
print(f"yfinance HG: {hg_yf.index[0]} to {hg_yf.index[-1]} ({len(hg_yf)} bars)")

# yfinance is UTC, Sierra is CT (UTC-6 in winter, UTC-5 in summer)
# March 2026: DST starts March 8 2026 in US -> after March 8 it's UTC-5
# Before March 8: UTC-6
# Let's check by comparing prices directly

# Strip timezone from yfinance
si_yf.index = si_yf.index.tz_localize(None)
hg_yf.index = hg_yf.index.tz_localize(None)

# Try offset -6h (CT = UTC-6 in winter) and -5h (CT = UTC-5 in summer/DST)
for offset_name, offset_hours in [("UTC-6 (CST)", -6), ("UTC-5 (CDT)", -5)]:
    si_yf_shifted = si_yf.copy()
    si_yf_shifted.index = si_yf_shifted.index + pd.Timedelta(hours=offset_hours)

    common = si_sc.index.intersection(si_yf_shifted.index)
    if len(common) > 0:
        print(f"\n--- SI overlap with {offset_name}: {len(common)} bars ---")
        # Compare Close/Last
        sc_vals = si_sc.loc[common, "Last"].astype(float)
        yf_vals = si_yf_shifted.loc[common, "Close"].astype(float)
        diff = (sc_vals - yf_vals).abs()
        print(f"  Mean abs diff: {diff.mean():.4f}")
        print(f"  Max abs diff:  {diff.max():.4f}")
        print(f"  Pct match (<0.01): {(diff < 0.01).sum()}/{len(diff)} ({(diff<0.01).mean()*100:.1f}%)")

        # Show first 10 comparisons
        sample = common[:10]
        print(f"\n  Sample comparison (first 10):")
        print(f"  {'Datetime':>22s}  {'Sierra':>10s}  {'yfinance':>10s}  {'diff':>8s}")
        for dt in sample:
            s = si_sc.loc[dt, "Last"]
            y = si_yf_shifted.loc[dt, "Close"]
            print(f"  {str(dt):>22s}  {float(s):>10.3f}  {float(y):>10.3f}  {abs(float(s)-float(y)):>8.4f}")

        # Show last 10
        sample = common[-10:]
        print(f"\n  Sample comparison (last 10):")
        print(f"  {'Datetime':>22s}  {'Sierra':>10s}  {'yfinance':>10s}  {'diff':>8s}")
        for dt in sample:
            s = si_sc.loc[dt, "Last"]
            y = si_yf_shifted.loc[dt, "Close"]
            print(f"  {str(dt):>22s}  {float(s):>10.3f}  {float(y):>10.3f}  {abs(float(s)-float(y)):>8.4f}")

# Same for HG
for offset_name, offset_hours in [("UTC-6 (CST)", -6), ("UTC-5 (CDT)", -5)]:
    hg_yf_shifted = hg_yf.copy()
    hg_yf_shifted.index = hg_yf_shifted.index + pd.Timedelta(hours=offset_hours)

    common = hg_sc.index.intersection(hg_yf_shifted.index)
    if len(common) > 0:
        print(f"\n--- HG overlap with {offset_name}: {len(common)} bars ---")
        sc_vals = hg_sc.loc[common, "Last"].astype(float)
        yf_vals = hg_yf_shifted.loc[common, "Close"].astype(float)
        diff = (sc_vals - yf_vals).abs()
        print(f"  Mean abs diff: {diff.mean():.4f}")
        print(f"  Max abs diff:  {diff.max():.4f}")
        print(f"  Pct match (<0.001): {(diff < 0.001).sum()}/{len(diff)} ({(diff<0.001).mean()*100:.1f}%)")

        sample = common[:10]
        print(f"\n  Sample comparison (first 10):")
        print(f"  {'Datetime':>22s}  {'Sierra':>10s}  {'yfinance':>10s}  {'diff':>8s}")
        for dt in sample:
            s = hg_sc.loc[dt, "Last"]
            y = hg_yf_shifted.loc[dt, "Close"]
            print(f"  {str(dt):>22s}  {float(s):>10.4f}  {float(y):>10.4f}  {abs(float(s)-float(y)):>8.5f}")

        sample = common[-10:]
        print(f"\n  Sample comparison (last 10):")
        print(f"  {'Datetime':>22s}  {'Sierra':>10s}  {'yfinance':>10s}  {'diff':>8s}")
        for dt in sample:
            s = hg_sc.loc[dt, "Last"]
            y = hg_yf_shifted.loc[dt, "Close"]
            print(f"  {str(dt):>22s}  {float(s):>10.4f}  {float(y):>10.4f}  {abs(float(s)-float(y)):>8.5f}")

# Also check: are yfinance SI=F and HG=F the right contracts?
print(f"\n--- Contract check ---")
print(f"Sierra SI contract: SIK26 (May 2026)")
print(f"Sierra HG contract: HGK26 (May 2026)")
print(f"yfinance SI=F: front month continuous (unknown exact contract)")
print(f"yfinance HG=F: front month continuous (unknown exact contract)")
print(f"\nSierra SI last price: {si_sc['Last'].iloc[-1]}")
print(f"yfinance SI last price: {si_yf['Close'].iloc[-1]:.3f}")
print(f"Sierra HG last price: {hg_sc['Last'].iloc[-1]}")
print(f"yfinance HG last price: {hg_yf['Close'].iloc[-1]:.4f}")
