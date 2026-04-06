#!/usr/bin/env python3
"""
HONEST Mean Reversion 4σ backtest:
- dev_ma, stdev, bands computed on PREVIOUS bar data (shift(1))
- Limit order at band level (valid — if low[i] touches, order fills)
- TP and SL use previous bar's EMA (no current-bar dependency)
- Minimum RR 2:1 enforced
"""
import glob, os
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

def run(df, mult=4.0, sl_mult=1.0, min_rr=2.0, cap=200, risk=2.0,
        comm=0.04, slip=0.01, max_hold=96, use_shift=True):
    # Use .shift(1) so bands at bar i are computed from bars [0..i-1]
    if use_shift:
        dev_ma = df["Close"].ewm(span=20, adjust=False).mean().shift(1).values
        stdev = df["Close"].rolling(20).std().shift(1).values
    else:
        dev_ma = df["Close"].ewm(span=20, adjust=False).mean().values
        stdev = df["Close"].rolling(20).std().values

    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)

    trades = []
    equity = cap
    in_trade = False
    entry = tp = sl = pos = 0
    direction = ""
    hold = 0
    warmup = 260

    for i in range(warmup, n):
        if in_trade:
            hold += 1
            ep = None; res = None
            if direction == "long":
                if l[i] <= sl and h[i] >= tp:
                    ep = sl; res = "SL"  # conservative
                elif l[i] <= sl:
                    ep = sl; res = "SL"
                elif h[i] >= tp:
                    ep = tp; res = "TP"
                elif hold >= max_hold:
                    ep = c[i]; res = "timeout"
            else:
                if h[i] >= sl and l[i] <= tp:
                    ep = sl; res = "SL"
                elif h[i] >= sl:
                    ep = sl; res = "SL"
                elif l[i] <= tp:
                    ep = tp; res = "TP"
                elif hold >= max_hold:
                    ep = c[i]; res = "timeout"
            if ep is not None:
                cost = (comm+slip)/100
                rpnl = ((ep-entry)/entry-2*cost) if direction=="long" else ((entry-ep)/entry-2*cost)
                pnl = pos * rpnl
                equity += pnl
                trades.append({"time":bars[i], "dir":direction, "entry":entry, "exit":ep,
                              "tp":tp, "sl":sl, "pnl_pct":rpnl*100, "pnl_abs":pnl, "res":res})
                in_trade = False
                if equity <= 0: break
                continue

        if in_trade or equity <= 1: continue
        if np.isnan(dev_ma[i]) or np.isnan(stdev[i]) or stdev[i] == 0: continue

        lb = dev_ma[i] - stdev[i] * mult
        ub = dev_ma[i] + stdev[i] * mult

        # LONG: limit at lower band
        if l[i] <= lb:
            e = lb
            t = dev_ma[i]
            s = e - stdev[i] * sl_mult
            if t > e and (e-s) > 0:
                rr = (t - e) / (e - s)
                if rr >= min_rr:
                    ds = (e - s) / e
                    ra = equity * (risk/100)
                    pos = ra / (ds + 2*(comm+slip)/100)
                    if pos > 0 and ds < 0.05:
                        entry = e; tp = t; sl = s; direction = "long"
                        in_trade = True; hold = 0
                        continue

        # SHORT: limit at upper band
        if h[i] >= ub:
            e = ub
            t = dev_ma[i]
            s = e + stdev[i] * sl_mult
            if t < e and (s-e) > 0:
                rr = (e - t) / (s - e)
                if rr >= min_rr:
                    ds = (s - e) / e
                    ra = equity * (risk/100)
                    pos = ra / (ds + 2*(comm+slip)/100)
                    if pos > 0 and ds < 0.05:
                        entry = e; tp = t; sl = s; direction = "short"
                        in_trade = True; hold = 0

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
    cum = cap + df["pnl_abs"].cumsum()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max() * 100).min()
    exp = (wr/100*aw) + ((100-wr)/100*al)
    return {"n":n,"wr":wr,"pf":pf,"aw":aw,"al":al,"exp":exp,"ret":(feq-cap)/cap*100,"dd":dd,"feq":feq}

df = load_data("/home/user/btc15v3")
print(f"Loaded {len(df)} bars\n")

print("="*80)
print("  MEAN REVERSION 4σ — LOOK-AHEAD CHECK")
print("="*80)

# Test with and without shift, and with different RR requirements
tests = [
    ("OLD (no shift, no min RR)",         4.0, 1.0, 0.0, False),
    ("OLD (no shift) + min RR 2.0",       4.0, 1.0, 2.0, False),
    ("HONEST (shift=1, any RR)",          4.0, 1.0, 0.0, True),
    ("HONEST (shift=1) + min RR 2.0",     4.0, 1.0, 2.0, True),
    ("HONEST + min RR 2.5",               4.0, 1.0, 2.5, True),
    ("HONEST + min RR 3.0",               4.0, 1.0, 3.0, True),
    ("HONEST 3σ + min RR 2.0",            3.0, 1.0, 2.0, True),
    ("HONEST 5σ + min RR 2.0",            5.0, 1.0, 2.0, True),
    ("HONEST 4σ SL=1.5σ + RR>=2",         4.0, 1.5, 2.0, True),
    ("HONEST 4σ SL=0.5σ + RR>=2",         4.0, 0.5, 2.0, True),
]

print(f"\n{'Scenario':<42} {'N':>5} {'WR%':>6} {'PF':>6} {'AvgW%':>7} {'AvgL%':>7} {'Ret%':>9} {'DD%':>7}")
print("-"*105)
for label, mult, slm, rr, shift in tests:
    tr, feq = run(df, mult=mult, sl_mult=slm, min_rr=rr, use_shift=shift)
    st = stats(tr)
    if st and st["n"] > 0:
        pfx = "+" if st["ret"] > 0 else ""
        print(f"  {label:<40} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>5.2f} "
              f"{st['aw']:>+6.2f}% {st['al']:>+6.2f}% {pfx}{st['ret']:>7.1f}% {st['dd']:>6.1f}%")
    else:
        print(f"  {label:<40} NO TRADES")

# Full OOS test on best honest variant
print("\n" + "="*80)
print("  OOS TEST — HONEST variants")
print("="*80)

train = df[df.index < "2024-07-01"]
test = df[df.index >= "2024-07-01"]

for rr in [2.0, 2.5, 3.0]:
    print(f"\n  HONEST 4σ, SL=1σ, min RR {rr}:")
    for label, slice_df in [("  Train    ", train), ("  Test     ", test), ("  Full     ", df)]:
        tr, feq = run(slice_df, mult=4.0, sl_mult=1.0, min_rr=rr, use_shift=True)
        st = stats(tr)
        if st and st["n"] > 0:
            print(f"    {label}: {st['n']:>5} trades | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
                  f"$200 → ${feq:,.2f} | DD {st['dd']:.1f}%")
        else:
            print(f"    {label}: no trades")

