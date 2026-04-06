#!/usr/bin/env python3
"""
MOST REALISTIC test:
Limit order placed at end of bar i-1, at band level computed from bar i-1 data.
If during bar i, low <= band → fill at band (limit order).

This is what you'd actually do in live trading:
1. After each 15m bar closes, compute new band levels
2. Place/update limit orders at those levels for next bar
3. If next bar's low touches limit, order fills
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

df_full = load_data("/home/user/btc15v3")

def run(df, label, mult=4.0, sl_mult=1.0, cap=200, risk=2.0):
    """
    Limit order at band (shift(1) band level).
    Entry at shifted band, TP = shifted mean at signal time, SL = shifted band - 1σ.
    """
    # SHIFTED indicators: band at bar i uses data [0..i-1]
    dev_ma_s = df["Close"].ewm(span=20, adjust=False).mean().shift(1).values
    stdev_s = df["Close"].rolling(20).std().shift(1).values

    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    times = df.index
    n = len(df)

    signals = []
    equity = cap
    in_trade = False
    entry_price = tp = sl = 0
    direction = ""
    entry_t = None
    pos = 0
    COST = 0.001  # 0.1% round-trip

    for i in range(30, n):
        if np.isnan(dev_ma_s[i]) or np.isnan(stdev_s[i]) or stdev_s[i] == 0: continue

        # Band at bar i (computed from prev data)
        lb = dev_ma_s[i] - stdev_s[i] * mult
        ub = dev_ma_s[i] + stdev_s[i] * mult
        mean = dev_ma_s[i]

        # Exit
        if in_trade:
            ep = None; res = None
            if direction == "long":
                if l[i] <= sl: ep = sl; res = "SL"
                elif h[i] >= tp: ep = tp; res = "TP"
            else:
                if h[i] >= sl: ep = sl; res = "SL"
                elif l[i] <= tp: ep = tp; res = "TP"

            if ep is not None:
                if direction == "long":
                    rpnl = (ep - entry_price) / entry_price - COST
                else:
                    rpnl = (entry_price - ep) / entry_price - COST
                pnl_abs = pos * rpnl
                equity += pnl_abs
                signals.append({
                    "entry_t": entry_t, "exit_t": times[i], "dir": direction,
                    "entry": entry_price, "exit": ep, "tp": tp, "sl": sl,
                    "result": res, "pnl_pct": rpnl * 100, "pnl_abs": pnl_abs,
                    "equity": equity,
                })
                in_trade = False
                if equity <= 0: break
                continue

        if in_trade or equity <= 1: continue

        # LONG: limit at lower band (pre-computed from prev bar data)
        if l[i] <= lb:
            entry_price = lb
            tp = mean
            sl = lb - stdev_s[i] * sl_mult
            if tp > entry_price and sl < entry_price:
                direction = "long"
                entry_t = times[i]
                ds = (entry_price - sl) / entry_price
                if 0.001 < ds < 0.05:
                    pos = (equity * risk / 100) / ds
                    in_trade = True
                    continue

        # SHORT
        if h[i] >= ub:
            entry_price = ub
            tp = mean
            sl = ub + stdev_s[i] * sl_mult
            if tp < entry_price and sl > entry_price:
                direction = "short"
                entry_t = times[i]
                ds = (sl - entry_price) / entry_price
                if 0.001 < ds < 0.05:
                    pos = (equity * risk / 100) / ds
                    in_trade = True

    if not signals:
        print(f"\n  {label}: no signals")
        return None

    df_sig = pd.DataFrame(signals)
    n_sig = len(df_sig)
    wins = df_sig[df_sig["pnl_abs"] > 0]
    losses = df_sig[df_sig["pnl_abs"] <= 0]
    wr = len(wins) / n_sig * 100
    aw = wins["pnl_pct"].mean() if len(wins) else 0
    al = losses["pnl_pct"].mean() if len(losses) else 0
    gp = wins["pnl_abs"].sum() if len(wins) else 0
    gl = abs(losses["pnl_abs"].sum()) if len(losses) else 0
    pf = gp / gl if gl > 0 else float("inf")
    cum = cap + df_sig["pnl_abs"].cumsum()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max() * 100).min()
    feq = cap + df_sig["pnl_abs"].sum()

    print(f"\n  {label}:")
    print(f"    Signals: {n_sig} | WR {wr:.1f}% | PF {pf:.2f}")
    print(f"    Avg W {aw:+.2f}% | Avg L {al:+.2f}%")
    print(f"    ${cap} → ${feq:,.2f} ({(feq-cap)/cap*100:+.1f}%) | DD {dd:.1f}%")
    return df_sig

# Test on different periods
print("=" * 80)
print("  FINAL HONEST TEST — Limit order at shifted(1) band level")
print("  (Most realistic for live trading: place limit at band level,")
print("   updated every bar based on previous bar's indicator values)")
print("=" * 80)

for period_label, start in [
    ("Last 3 months (Jan-Mar 2026)", "2026-01-01"),
    ("Last 6 months", "2025-10-01"),
    ("Last year", "2025-04-01"),
    ("2025 full", "2025-01-01"),
    ("2024 full", "2024-01-01"),
    ("2023 full", "2023-01-01"),
    ("2022 full", "2022-01-01"),
    ("Full 4 years", "2022-01-01"),
]:
    print(f"\n── {period_label} ──")
    df = df_full[df_full.index >= start].copy()
    if "full" in period_label.lower() and period_label.startswith("2024"):
        df = df[df.index < "2025-01-01"].copy()
    if "full" in period_label.lower() and period_label.startswith("2023"):
        df = df[df.index < "2024-01-01"].copy()
    if "full" in period_label.lower() and period_label.startswith("2022"):
        df = df[df.index < "2023-01-01"].copy()
    run(df, f"4σ + SL 1σ, Tier3, no filters (LIMIT @ shifted band)")

# Also show last month's signals detailed
print("\n" + "=" * 80)
print("  DETAILED: Last month signals (Feb 25 - Mar 31 2026)")
print("=" * 80)
df_recent = df_full[df_full.index >= "2026-02-25"].copy()
sigs = run(df_recent, "Last month detailed")
if sigs is not None and len(sigs) > 0:
    print(f"\n  Each signal:")
    print(f"  {'Entry':<18} {'Dir':<6} {'Entry':>10} {'TP':>10} {'SL':>10} {'Exit':>10} {'Res':>5} {'PnL%':>7}")
    for _, r in sigs.iterrows():
        pfx = "+" if r["pnl_pct"] > 0 else ""
        print(f"  {str(r['entry_t'])[:16]:<18} {r['dir']:<6} "
              f"{r['entry']:>10.2f} {r['tp']:>10.2f} {r['sl']:>10.2f} "
              f"{r['exit']:>10.2f} {r['result']:>5} {pfx}{r['pnl_pct']:>6.2f}%")

