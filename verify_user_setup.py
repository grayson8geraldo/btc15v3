#!/usr/bin/env python3
"""
Verify EXACT user setup from bounce_indicator_v2.pine:
- Timeframe: 15m
- Tier 3 only (4.0 stdev)
- SL = band - 1.0 * stdev
- TP = Mean (EMA 20)
- Limit order at band level
- No RSI/ADX/Volume filters
- No trend filter

Test on LAST 3 MONTHS. Show each signal individually.
Also test with "proper" shift (where band at bar i uses prev bar data)
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

df = load_data("/home/user/btc15v3")
# Last 3 months
df = df[df.index >= "2026-01-01"]
print(f"Testing period: {df.index.min()} → {df.index.max()}")
print(f"Bars: {len(df)}\n")

# Compute indicators — realistic: band at bar i computed from bars [0..i-1]
# Using shift(1) is the safest/correct approach
dev_ma_shifted = df["Close"].ewm(span=20, adjust=False).mean().shift(1).values
stdev_shifted = df["Close"].rolling(20).std().shift(1).values

# Also try WITHOUT shift (as original indicator)
dev_ma_no_shift = df["Close"].ewm(span=20, adjust=False).mean().values
stdev_no_shift = df["Close"].rolling(20).std().values

h = df["High"].values
l = df["Low"].values
c = df["Close"].values
times = df.index

MULT = 4.0
SL_MULT = 1.0
COMM = 0.04
SLIP = 0.01
cost_pct = 2 * (COMM + SLIP) / 100  # total round-trip cost

def test_version(label, dev_ma, stdev):
    """Run backtest with given indicator arrays."""
    signals = []
    in_trade = False
    entry_price = tp = sl = 0
    direction = ""
    entry_idx = 0
    entry_time = None

    for i in range(30, len(df)):
        if np.isnan(dev_ma[i]) or np.isnan(stdev[i]) or stdev[i] == 0:
            continue

        lb = dev_ma[i] - stdev[i] * MULT
        ub = dev_ma[i] + stdev[i] * MULT

        # Check exit
        if in_trade:
            exit_price = None
            result = None
            if direction == "long":
                if l[i] <= sl:
                    exit_price = sl; result = "SL"
                elif h[i] >= tp:
                    exit_price = tp; result = "TP"
            else:
                if h[i] >= sl:
                    exit_price = sl; result = "SL"
                elif l[i] <= tp:
                    exit_price = tp; result = "TP"

            if exit_price is not None:
                if direction == "long":
                    pnl = (exit_price - entry_price) / entry_price * 100 - cost_pct * 100
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100 - cost_pct * 100
                signals.append({
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "bars": i - entry_idx,
                    "direction": direction,
                    "entry": entry_price,
                    "exit": exit_price,
                    "tp": tp,
                    "sl": sl,
                    "result": result,
                    "pnl": pnl,
                })
                in_trade = False
                continue

        if in_trade:
            continue

        # Check entry: limit at band level
        # LONG: if low touches lower band
        if l[i] <= lb:
            entry_price = lb
            tp_new = dev_ma[i]
            sl_new = lb - stdev[i] * SL_MULT
            if tp_new > entry_price and sl_new < entry_price:
                direction = "long"
                tp = tp_new
                sl = sl_new
                entry_idx = i
                entry_time = times[i]
                in_trade = True
                continue

        if h[i] >= ub:
            entry_price = ub
            tp_new = dev_ma[i]
            sl_new = ub + stdev[i] * SL_MULT
            if tp_new < entry_price and sl_new > entry_price:
                direction = "short"
                tp = tp_new
                sl = sl_new
                entry_idx = i
                entry_time = times[i]
                in_trade = True

    if not signals:
        print(f"\n── {label}: NO SIGNALS")
        return

    sdf = pd.DataFrame(signals)
    n = len(sdf)
    wins = sdf[sdf["pnl"] > 0]
    losses = sdf[sdf["pnl"] <= 0]
    wr = len(wins) / n * 100
    avg_w = wins["pnl"].mean() if len(wins) else 0
    avg_l = losses["pnl"].mean() if len(losses) else 0
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 0
    pf = gp / gl if gl > 0 else float("inf")
    total_pnl = sdf["pnl"].sum()

    print(f"\n── {label} ──")
    print(f"  Сигналов: {n}")
    print(f"  WR: {wr:.1f}% ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Avg Win: {avg_w:+.2f}% | Avg Loss: {avg_l:+.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Total P&L (sum of %): {total_pnl:+.2f}%")
    print(f"\n  Каждый сигнал:")
    print(f"  {'Entry time':<18} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'Result':>6} {'P&L':>8} {'Bars':>5}")
    for _, r in sdf.iterrows():
        pfx = "+" if r["pnl"] > 0 else ""
        print(f"  {str(r['entry_time'])[:16]:<18} {r['direction']:<5} "
              f"{r['entry']:>10.2f} {r['exit']:>10.2f} {r['result']:>6} "
              f"{pfx}{r['pnl']:>6.2f}% {r['bars']:>5}")

# Test both versions
print("=" * 80)
print("  USER SETUP VERIFICATION")
print("  Tier 3 (4σ) | SL 1σ | TP at Mean | Limit entry at band")
print("  NO filters (trend/RSI/ADX/vol OFF)")
print("=" * 80)

test_version("WITH shift(1) — strict no-lookahead", dev_ma_shifted, stdev_shifted)
test_version("WITHOUT shift — as indicator computes", dev_ma_no_shift, stdev_no_shift)

# Also test on larger period for statistical significance
print("\n" + "=" * 80)
print("  LARGER PERIOD (last 6 months) for more signals")
print("=" * 80)
df6 = load_data("/home/user/btc15v3")
df6 = df6[df6.index >= "2025-10-01"]
print(f"Bars: {len(df6)}\n")

dev_ma_s6 = df6["Close"].ewm(span=20, adjust=False).mean().shift(1).values
stdev_s6 = df6["Close"].rolling(20).std().shift(1).values
dev_ma_ns6 = df6["Close"].ewm(span=20, adjust=False).mean().values
stdev_ns6 = df6["Close"].rolling(20).std().values

# Rebind globals for test_version to work
df = df6; h = df["High"].values; l = df["Low"].values; c = df["Close"].values; times = df.index

def test_stats_only(label, dev_ma, stdev):
    signals = []
    in_trade = False
    entry_price = tp = sl = 0
    direction = ""
    entry_time = None
    for i in range(30, len(df)):
        if np.isnan(dev_ma[i]) or np.isnan(stdev[i]) or stdev[i] == 0: continue
        lb = dev_ma[i] - stdev[i] * MULT
        ub = dev_ma[i] + stdev[i] * MULT
        if in_trade:
            exit_price = None; result = None
            if direction == "long":
                if l[i] <= sl: exit_price = sl; result = "SL"
                elif h[i] >= tp: exit_price = tp; result = "TP"
            else:
                if h[i] >= sl: exit_price = sl; result = "SL"
                elif l[i] <= tp: exit_price = tp; result = "TP"
            if exit_price is not None:
                pnl = ((exit_price-entry_price)/entry_price if direction=="long" else (entry_price-exit_price)/entry_price)*100 - cost_pct*100
                signals.append(pnl)
                in_trade = False
                continue
        if in_trade: continue
        if l[i] <= lb:
            entry_price = lb; tp = dev_ma[i]; sl = lb - stdev[i]*SL_MULT
            if tp > entry_price and sl < entry_price:
                direction = "long"; in_trade = True; continue
        if h[i] >= ub:
            entry_price = ub; tp = dev_ma[i]; sl = ub + stdev[i]*SL_MULT
            if tp < entry_price and sl > entry_price:
                direction = "short"; in_trade = True

    if not signals: 
        print(f"\n  {label}: no signals"); return
    n = len(signals)
    wins = [s for s in signals if s > 0]
    wr = len(wins) / n * 100
    total = sum(signals)
    avg_w = np.mean(wins) if wins else 0
    losses = [s for s in signals if s <= 0]
    avg_l = np.mean(losses) if losses else 0
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = gp/gl if gl > 0 else float("inf")
    print(f"\n  {label}:")
    print(f"    {n} signals | WR {wr:.1f}% | PF {pf:.2f} | Total {total:+.1f}% | Avg W {avg_w:+.2f}% | Avg L {avg_l:+.2f}%")

test_stats_only("WITH shift(1)", dev_ma_s6, stdev_s6)
test_stats_only("WITHOUT shift", dev_ma_ns6, stdev_ns6)

