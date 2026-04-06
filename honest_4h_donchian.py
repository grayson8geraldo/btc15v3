#!/usr/bin/env python3
"""
HONEST 4h Donchian Breakout — OOS + Parameter Sweep + Walk-forward
The only strategy that showed promise. Must verify it's not lucky.
"""
import glob, os
from dataclasses import dataclass
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

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def resample_ohlc(df, tf):
    return df.resample(tf).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

@dataclass
class Trade:
    dir:str=""; entry:float=0; exit:float=0; tp:float=0; sl:float=0
    pnl_pct:float=0; pnl_abs:float=0; result:str=""; entry_t:object=None

def run(df, don_lb=20, trend_lb=200, atr_sl=2.0, atr_tp=4.0,
        cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=40, only_long=False):
    ema_trend = ema(df["Close"], trend_lb).shift(1).values
    atr14 = calc_atr(df["High"], df["Low"], df["Close"], 14).shift(1).values
    don_h = df["High"].rolling(don_lb).max().shift(1).values
    don_l = df["Low"].rolling(don_lb).min().shift(1).values

    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)
    trades = []
    equity = cap
    in_trade = False; t = None; pos = 0; hold = 0
    warmup = trend_lb + 20

    for i in range(warmup, n):
        if in_trade:
            hold += 1
            ep = None; res = None
            if t.dir == "long":
                if l[i] <= t.sl and h[i] >= t.tp:
                    ep = t.sl; res = "SL"
                elif l[i] <= t.sl: ep = t.sl; res = "SL"
                elif h[i] >= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = c[i]; res = "timeout"
            else:
                if h[i] >= t.sl and l[i] <= t.tp: ep = t.sl; res = "SL"
                elif h[i] >= t.sl: ep = t.sl; res = "SL"
                elif l[i] <= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = c[i]; res = "timeout"
            if ep is not None:
                cost = (comm+slip)/100
                rpnl = ((ep-t.entry)/t.entry-2*cost) if t.dir=="long" else ((t.entry-ep)/t.entry-2*cost)
                pnl = pos * rpnl; equity += pnl
                t.exit = ep; t.result = res; t.pnl_pct = rpnl*100; t.pnl_abs = pnl
                trades.append(t); in_trade = False
                if equity <= 0: break
                continue

        if in_trade or equity <= 1: continue
        if np.isnan(don_h[i]) or np.isnan(atr14[i]) or np.isnan(ema_trend[i]): continue

        trend_up = c[i-1] > ema_trend[i]
        trend_dn = c[i-1] < ema_trend[i]

        # LONG breakout
        if trend_up and h[i] >= don_h[i] and c[i-1] < don_h[i]:
            entry = don_h[i]
            if not (l[i] <= entry <= h[i]): continue
            sl = entry - atr14[i] * atr_sl
            tp = entry + atr14[i] * atr_tp
            ds = (entry - sl) / entry
            if 0.002 < ds < 0.08:
                ra = equity * (risk/100)
                p = ra / (ds + 2*(comm+slip)/100)
                if p > 0:
                    t = Trade(dir="long", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                    pos = p; in_trade = True; hold = 0
                    continue

        # SHORT breakout
        if not only_long and trend_dn and l[i] <= don_l[i] and c[i-1] > don_l[i]:
            entry = don_l[i]
            if not (l[i] <= entry <= h[i]): continue
            sl = entry + atr14[i] * atr_sl
            tp = entry - atr14[i] * atr_tp
            ds = (sl - entry) / entry
            if 0.002 < ds < 0.08:
                ra = equity * (risk/100)
                p = ra / (ds + 2*(comm+slip)/100)
                if p > 0:
                    t = Trade(dir="short", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                    pos = p; in_trade = True; hold = 0

    return trades, equity


def stats(trades, cap=200):
    if not trades: return None
    df = pd.DataFrame([t.__dict__ for t in trades])
    n = len(df); w = df[df["pnl_abs"]>0]; l = df[df["pnl_abs"]<=0]
    wr = len(w)/n*100
    aw = w["pnl_pct"].mean() if len(w) else 0
    al = l["pnl_pct"].mean() if len(l) else 0
    gp = w["pnl_abs"].sum() if len(w) else 0
    gl = abs(l["pnl_abs"].sum()) if len(l) else 0
    pf = gp/gl if gl>0 else 0
    feq = cap + df["pnl_abs"].sum()
    cum = cap + df["pnl_abs"].cumsum()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max() * 100).min()
    df["year"] = pd.to_datetime(df["entry_t"]).dt.year
    yearly = df.groupby("year")["pnl_abs"].sum().to_dict()
    return {"n":n,"wr":wr,"pf":pf,"aw":aw,"al":al,"ret":(feq-cap)/cap*100,"dd":dd,"feq":feq,"yearly":yearly}

df15 = load_data("/home/user/btc15v3")
df4h = resample_ohlc(df15, "4h")
print(f"4h bars: {len(df4h)}\n")

# ═══ PARAMETER SWEEP ═══
print("=" * 100)
print("  4h DONCHIAN PARAMETER SWEEP (HONEST, no look-ahead)")
print("=" * 100)
print(f"  {'Params':<40} {'N':>5} {'WR%':>6} {'PF':>6} {'Ret%':>9} {'DD%':>7} {'Status':>7}")
print("-" * 100)

results = {}
for don in [10, 15, 20, 25, 30, 40, 50]:
    for trend in [100, 200]:
        for atrsl, atrtp in [(2.0, 4.0), (1.5, 3.0), (2.5, 5.0), (2.0, 6.0)]:
            lbl = f"Don{don}/Trend{trend}/SL{atrsl}x/TP{atrtp}x"
            tr, feq = run(df4h, don_lb=don, trend_lb=trend, atr_sl=atrsl, atr_tp=atrtp)
            st = stats(tr)
            if st and st["n"] > 20:
                status = "OK" if st["pf"] > 1 else ""
                if st["pf"] > 1.05:
                    results[lbl] = st
                print(f"  {lbl:<40} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>5.2f} "
                      f"{st['ret']:>+8.1f}% {st['dd']:>6.1f}% {status:>7}")

# OOS test on top 5 profitable variants
print("\n" + "=" * 100)
print("  OOS TEST on profitable variants (Train: 2022-2024H1, Test: 2024H2-2026)")
print("=" * 100)

train = df4h[df4h.index < "2024-07-01"]
test = df4h[df4h.index >= "2024-07-01"]

top = sorted(results.items(), key=lambda x: x[1]["pf"], reverse=True)[:10]
for lbl, full_st in top:
    # Parse params
    parts = lbl.split("/")
    don = int(parts[0][3:])
    trend = int(parts[1][5:])
    atrsl = float(parts[2][2:-1])
    atrtp = float(parts[3][2:-1])

    tr_tr, _ = run(train, don_lb=don, trend_lb=trend, atr_sl=atrsl, atr_tp=atrtp)
    te_tr, _ = run(test, don_lb=don, trend_lb=trend, atr_sl=atrsl, atr_tp=atrtp)
    tr_st = stats(tr_tr)
    te_st = stats(te_tr)
    if tr_st and te_st and tr_st["n"] > 0 and te_st["n"] > 0:
        tr_ok = "OK" if tr_st["pf"] > 1 else "--"
        te_ok = "OK" if te_st["pf"] > 1 else "--"
        both = "BOTH PROFITABLE" if tr_st["pf"] > 1 and te_st["pf"] > 1 else ""
        print(f"\n  {lbl}")
        print(f"    Full:  n={full_st['n']:>4} WR {full_st['wr']:>4.1f}% PF {full_st['pf']:.2f} Ret {full_st['ret']:+.1f}% DD {full_st['dd']:.1f}%")
        print(f"    Train: n={tr_st['n']:>4} WR {tr_st['wr']:>4.1f}% PF {tr_st['pf']:.2f} Ret {tr_st['ret']:+.1f}% DD {tr_st['dd']:.1f}% {tr_ok}")
        print(f"    Test:  n={te_st['n']:>4} WR {te_st['wr']:>4.1f}% PF {te_st['pf']:.2f} Ret {te_st['ret']:+.1f}% DD {te_st['dd']:.1f}% {te_ok}  {both}")

# Walk-forward
print("\n" + "=" * 100)
print("  WALK-FORWARD TEST (best variant by OOS)")
print("=" * 100)
if top:
    # Find variant that's profitable on BOTH train and test
    best_oos = None
    for lbl, _ in top:
        parts = lbl.split("/")
        don = int(parts[0][3:])
        trend = int(parts[1][5:])
        atrsl = float(parts[2][2:-1])
        atrtp = float(parts[3][2:-1])
        tr_tr, _ = run(train, don_lb=don, trend_lb=trend, atr_sl=atrsl, atr_tp=atrtp)
        te_tr, _ = run(test, don_lb=don, trend_lb=trend, atr_sl=atrsl, atr_tp=atrtp)
        tr_st = stats(tr_tr)
        te_st = stats(te_tr)
        if tr_st and te_st and tr_st["pf"] > 1.05 and te_st["pf"] > 1.05:
            best_oos = (lbl, don, trend, atrsl, atrtp)
            print(f"\n  Best OOS-validated: {lbl}")
            break

    if best_oos:
        _, don, trend, atrsl, atrtp = best_oos
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
        print(f"  {'Period':<10} {'N':>4} {'WR%':>6} {'PF':>6} {'Ret%':>8} {'DD%':>6}")
        profitable_halves = 0
        total = 0
        for label, s, e in halves:
            sl_df = df4h[(df4h.index >= s) & (df4h.index < e)]
            if len(sl_df) < 200: continue
            total += 1
            tr, _ = run(sl_df, don_lb=don, trend_lb=trend, atr_sl=atrsl, atr_tp=atrtp)
            st = stats(tr)
            if st and st["n"] > 0:
                if st["pf"] > 1: profitable_halves += 1
                mark = "OK" if st["pf"] > 1 else "LOSS"
                print(f"  {label:<10} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>5.2f} {st['ret']:>+7.1f}% {st['dd']:>5.1f}%  {mark}")
            else:
                print(f"  {label:<10}  no trades")
        print(f"\n  Profitable halves: {profitable_halves}/{total}")
    else:
        print("\n  NO variant passes OOS validation")
