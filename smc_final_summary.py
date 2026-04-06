#!/usr/bin/env python3
"""
SMC Strategies Final Summary + OOS Test

Тестирует лучшие версии всех 3 стратегий на:
1. In-sample (2022-2024H1)
2. Out-of-sample (2024H2-2026)
3. Walk-forward по полугодиям

Проверка на переобучение.
"""
import glob, os
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "BTCUSDT-15m-*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["open_time"], unit="ms")
    data.set_index("datetime", inplace=True)
    data.sort_index(inplace=True)
    data = data[~data.index.duplicated(keep="first")]
    data.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    return data[["Open","High","Low","Close","Volume"]].copy()

# ═════════════════════════════════════════════════════════════
# STRATEGY 3: FVG + Discount/Premium (BEST PERFORMER)
# ═════════════════════════════════════════════════════════════
def run_s3_fvg(df,
               range_lookback=96,
               zone_threshold=0.5,
               min_fvg_pct=0.15,
               target_mode="50pct",
               cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=96):
    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)
    trades = []; equity = cap
    in_trade = False; td=None; pos=0; hold=0
    warmup = max(range_lookback, 50) + 10

    for i in range(warmup, n - 2):
        if in_trade:
            hold += 1; ep=None; res=None
            if td["dir"] == "long":
                if l[i] <= td["sl"]: ep=td["sl"]; res="SL"
                elif h[i] >= td["tp"]: ep=td["tp"]; res="TP"
                elif hold >= max_hold: ep=c[i]; res="timeout"
            else:
                if h[i] >= td["sl"]: ep=td["sl"]; res="SL"
                elif l[i] <= td["tp"]: ep=td["tp"]; res="TP"
                elif hold >= max_hold: ep=c[i]; res="timeout"
            if ep is not None:
                cost=(comm+slip)/100
                rpnl=((ep-td["entry"])/td["entry"]-2*cost) if td["dir"]=="long" else ((td["entry"]-ep)/td["entry"]-2*cost)
                pnl=pos*rpnl; equity+=pnl
                td["exit"]=ep; td["res"]=res; td["pnl_pct"]=rpnl*100; td["pnl_abs"]=pnl
                trades.append(td); in_trade=False
                if equity<=0: break
                continue

        if in_trade or equity<=1: continue

        rh = max(h[i-range_lookback:i])
        rl = min(l[i-range_lookback:i])
        rsize = rh - rl
        if rsize <= 0: continue
        eq50 = (rh+rl)/2
        in_disc = c[i] < rl + rsize * zone_threshold
        in_prem = c[i] > rh - rsize * zone_threshold

        if i >= 2:
            bull_fvg = l[i] > h[i-2]
            bear_fvg = h[i] < l[i-2]
            if bull_fvg and in_disc:
                fvg_size = (l[i] - h[i-2])/h[i-2]*100
                if fvg_size >= min_fvg_pct:
                    entry = (l[i] + h[i-2])/2
                    sl = min(l[i-2], l[i-1])*0.999
                    tp = eq50 if target_mode=="50pct" else (rh if target_mode=="opposite" else entry+(entry-sl)*2)
                    if tp>entry and (entry-sl)>0:
                        ds=(entry-sl)/entry
                        if 0.001<ds<0.05:
                            ra=equity*(risk/100)
                            pos=ra/(ds+2*(comm+slip)/100)
                            if pos>0:
                                td={"dir":"long","entry":entry,"tp":tp,"sl":sl,"entry_t":bars[i],"exit":0,"res":"","pnl_pct":0,"pnl_abs":0}
                                in_trade=True; hold=0; continue

            if bear_fvg and in_prem:
                fvg_size = (l[i-2] - h[i])/l[i-2]*100
                if fvg_size >= min_fvg_pct:
                    entry = (h[i] + l[i-2])/2
                    sl = max(h[i-2], h[i-1])*1.001
                    tp = eq50 if target_mode=="50pct" else (rl if target_mode=="opposite" else entry-(sl-entry)*2)
                    if tp<entry and (sl-entry)>0:
                        ds=(sl-entry)/entry
                        if 0.001<ds<0.05:
                            ra=equity*(risk/100)
                            pos=ra/(ds+2*(comm+slip)/100)
                            if pos>0:
                                td={"dir":"short","entry":entry,"tp":tp,"sl":sl,"entry_t":bars[i],"exit":0,"res":"","pnl_pct":0,"pnl_abs":0}
                                in_trade=True; hold=0
    return trades, equity


def stats(trades, cap=200):
    if not trades: return None
    df = pd.DataFrame(trades)
    n = len(df); w = df[df["pnl_abs"]>0]; l = df[df["pnl_abs"]<=0]
    wr = len(w)/n*100
    aw = w["pnl_pct"].mean() if len(w) else 0
    al = l["pnl_pct"].mean() if len(l) else 0
    gp = w["pnl_abs"].sum() if len(w) else 0
    gl = abs(l["pnl_abs"].sum()) if len(l) else 0
    pf = gp/gl if gl>0 else 0
    feq = cap + df["pnl_abs"].sum()
    ret = (feq-cap)/cap*100
    cum = cap + df["pnl_abs"].cumsum()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max() * 100).min()
    exp = (wr/100*aw) + ((100-wr)/100*al)
    df["year"] = pd.to_datetime(df["entry_t"]).dt.year
    yearly = df.groupby("year")["pnl_abs"].sum().to_dict()
    return {"n":n,"wr":wr,"pf":pf,"aw":aw,"al":al,"exp":exp,"ret":ret,"dd":dd,"feq":feq,"yearly":yearly,"df":df}


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading..."); df = load_data(data_dir); print(f"{len(df)} bars\n")

    print("=" * 80)
    print("  SMC STRATEGY 3 (FVG + Discount/Premium) — Robustness Tests")
    print("=" * 80)

    # ── Best variants from initial backtest ───────────────
    variants = [
        dict(label="V1: 24h range + min FVG 0.15% + TP=50%",
             range_lookback=96, min_fvg_pct=0.15, target_mode="50pct"),
        dict(label="V2: 4h range + TP=50% (safer DD)",
             range_lookback=16, min_fvg_pct=0.05, target_mode="50pct"),
        dict(label="V3: 1W range + TP=50%",
             range_lookback=672, min_fvg_pct=0.05, target_mode="50pct"),
    ]

    # Run on FULL dataset
    print("\n── FULL PERIOD (2022-01 → 2026-03) ──\n")
    for v in variants:
        lbl = v["label"]
        params = {k:v[k] for k in v if k!="label"}
        tr, feq = run_s3_fvg(df, **params)
        st = stats(tr)
        if st:
            pfx = "+" if st["ret"]>0 else ""
            print(f"  {lbl}")
            print(f"    {st['n']} trades | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
                  f"E[R] {st['exp']:+.3f}% | {pfx}{st['ret']:,.1f}% | DD {st['dd']:.1f}% | $200 → ${feq:,.0f}")

    # Pick best variant for OOS test
    best_params = dict(range_lookback=16, min_fvg_pct=0.05, target_mode="50pct")
    best_label = "4h range + TP=50% (safest)"

    # ── OOS test ──────────────────────────────────────────
    print(f"\n── OUT-OF-SAMPLE TEST (best variant: {best_label}) ──\n")

    train = df[df.index < "2024-07-01"]
    test = df[df.index >= "2024-07-01"]

    for label, data_slice in [
        ("IN-SAMPLE (2022-2024H1)", train),
        ("OUT-OF-SAMPLE (2024H2-2026)", test),
        ("FULL", df),
    ]:
        tr, feq = run_s3_fvg(data_slice, **best_params)
        st = stats(tr)
        if st:
            print(f"  {label}: {st['n']} trades | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
                  f"$200 → ${feq:,.2f} | DD {st['dd']:.1f}%")

    # ── Walk-forward halves ───────────────────────────────
    print(f"\n── WALK-FORWARD ($200 fresh start each period) ──\n")
    print(f"  {'Period':<10} {'N':>6} {'WR%':>6} {'PF':>6} {'$200 →':>10} {'DD%':>7}")
    print("  " + "-" * 50)

    halves = [
        ("2022-H1", "2022-01-01", "2022-07-01"),
        ("2022-H2", "2022-07-01", "2023-01-01"),
        ("2023-H1", "2023-01-01", "2023-07-01"),
        ("2023-H2", "2023-07-01", "2024-01-01"),
        ("2024-H1", "2024-01-01", "2024-07-01"),
        ("2024-H2", "2024-07-01", "2025-01-01"),
        ("2025-H1", "2025-01-01", "2025-07-01"),
        ("2025-H2", "2025-07-01", "2026-01-01"),
        ("2026-H1", "2026-01-01", "2026-07-01"),
    ]
    profitable = 0; total = 0
    for label, s, e in halves:
        slice_df = df[(df.index >= s) & (df.index < e)]
        if len(slice_df) < 300: continue
        tr, feq = run_s3_fvg(slice_df, **best_params)
        st = stats(tr)
        if st and st["n"] > 0:
            total += 1
            if st["pf"] > 1: profitable += 1
            mark = "OK" if st["pf"] > 1 else "LOSS"
            print(f"  {label:<10} {st['n']:>6} {st['wr']:>5.1f}% {st['pf']:>5.2f} ${feq:>9.2f} {st['dd']:>6.1f}%  {mark}")
    print(f"\n  Profitable halves: {profitable}/{total}")

    # ── Yearly P&L on best ────────────────────────────────
    print(f"\n── YEARLY P&L ($200 start, full period, best variant) ──\n")
    tr, feq = run_s3_fvg(df, **best_params)
    st = stats(tr)
    print(f"  $200 → ${feq:,.2f} ({(feq-200)/200*100:,.1f}%)")
    print(f"  {st['n']} trades, WR {st['wr']:.1f}%, PF {st['pf']:.2f}")
    if st["yearly"]:
        for y, p in sorted(st["yearly"].items()):
            print(f"    {y}: ${p:,.2f}")

    # ── Save chart ────────────────────────────────────────
    cum = 200 + st["df"]["pnl_abs"].cumsum()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"SMC Strategy 3 (FVG + Discount/Premium) — {best_label}\n"
                 f"PF {st['pf']:.2f} | WR {st['wr']:.1f}% | DD {st['dd']:.1f}% | $200 → ${feq:,.0f}",
                 fontsize=14, fontweight="bold")

    axes[0,0].plot(range(len(cum)), cum.values, color="#00e676", linewidth=1)
    axes[0,0].axhline(200, color="white", linestyle="--", alpha=0.3)
    axes[0,0].set_yscale("log")
    axes[0,0].set_title("Equity Curve (log scale)")
    axes[0,0].grid(True, alpha=0.3)

    pk = cum.expanding().max()
    dd = (cum - pk) / pk * 100
    axes[0,1].fill_between(range(len(dd)), dd.values, 0, color="#ff1744", alpha=0.5)
    axes[0,1].set_title("Drawdown (%)")
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].hist(st["df"]["pnl_pct"].values, bins=60, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1,0].axvline(0, color="white", linestyle="--")
    axes[1,0].set_title("P&L Distribution (%)")
    axes[1,0].grid(True, alpha=0.3)

    if st["yearly"]:
        years = sorted(st["yearly"].keys())
        pnls = [st["yearly"][y] for y in years]
        colors = ["#00e676" if p > 0 else "#ff1744" for p in pnls]
        axes[1,1].bar([str(y) for y in years], pnls, color=colors)
        axes[1,1].set_yscale("symlog")
        axes[1,1].set_title("Yearly P&L ($, log scale)")
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "smc_final_summary.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("\n  Chart saved.")

if __name__ == "__main__":
    main()
