#!/usr/bin/env python3
"""
Most REALISTIC test of bounce_indicator_v2.pine user setup:
- User sees signal after bar closes
- Enters at NEXT bar open (realistic)
- TP at mean (dynamic, checks each bar)
- SL at band - 1σ stdev (fixed at signal time)
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

# Indicators computed with current bar data (as Pine Script does at close)
def run_realistic(df, label, entry_mode="band", fixed_tp=True):
    """
    entry_mode:
      "band"   - enter at band price (user sees level)
      "close"  - enter at close of signal bar (market order on signal)
      "open"   - enter at open of NEXT bar (most conservative realistic)

    fixed_tp: if True, TP is fixed at signal time (mean at that moment).
              If False, TP is dynamic — updates each bar to current mean.
    """
    dev_ma = df["Close"].ewm(span=20, adjust=False).mean().values
    stdev = df["Close"].rolling(20).std().values
    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    times = df.index
    n = len(df)

    MULT = 4.0
    SL_MULT = 1.0
    COST = 0.001  # 0.1% total round trip

    signals = []
    in_trade = False
    entry_price = tp = sl = 0
    direction = ""
    entry_idx = 0
    signal_bar = 0

    for i in range(30, n):
        if np.isnan(dev_ma[i]) or np.isnan(stdev[i]) or stdev[i] == 0: continue

        # Exit logic (dynamic TP if enabled)
        if in_trade:
            if not fixed_tp:
                tp = dev_ma[i]  # update to current mean
            exit_price = None; result = None
            if direction == "long":
                if l[i] <= sl: exit_price = sl; result = "SL"
                elif h[i] >= tp: exit_price = tp; result = "TP"
            else:
                if h[i] >= sl: exit_price = sl; result = "SL"
                elif l[i] <= tp: exit_price = tp; result = "TP"

            if exit_price is not None:
                if direction == "long":
                    pnl = (exit_price - entry_price) / entry_price - COST
                else:
                    pnl = (entry_price - exit_price) / entry_price - COST
                signals.append({
                    "entry_t": times[entry_idx],
                    "dir": direction, "entry": entry_price, "exit": exit_price,
                    "tp_init": tp, "sl": sl, "result": result, "pnl": pnl * 100,
                    "bars": i - entry_idx,
                })
                in_trade = False
                continue

        if in_trade: continue

        # Entry signal: price touched band on bar i
        lb = dev_ma[i] - stdev[i] * MULT
        ub = dev_ma[i] + stdev[i] * MULT

        long_sig = l[i] <= lb
        short_sig = h[i] >= ub

        if long_sig:
            if entry_mode == "band":
                entry_price = lb
            elif entry_mode == "close":
                entry_price = c[i]
            else:  # open next bar
                if i+1 >= n: continue
                entry_price = o[i+1]
                entry_idx_actual = i+1
            entry_idx = i+1 if entry_mode == "open" else i
            tp = dev_ma[i]
            sl = lb - stdev[i] * SL_MULT
            if tp > entry_price and sl < entry_price:
                direction = "long"
                in_trade = True
                continue

        if short_sig:
            if entry_mode == "band":
                entry_price = ub
            elif entry_mode == "close":
                entry_price = c[i]
            else:
                if i+1 >= n: continue
                entry_price = o[i+1]
            entry_idx = i+1 if entry_mode == "open" else i
            tp = dev_ma[i]
            sl = ub + stdev[i] * SL_MULT
            if tp < entry_price and sl > entry_price:
                direction = "short"
                in_trade = True

    if not signals:
        print(f"\n  {label}: no signals")
        return None

    df_sig = pd.DataFrame(signals)
    n_sig = len(df_sig)
    wins = df_sig[df_sig["pnl"] > 0]
    losses = df_sig[df_sig["pnl"] <= 0]
    wr = len(wins) / n_sig * 100
    aw = wins["pnl"].mean() if len(wins) else 0
    al = losses["pnl"].mean() if len(losses) else 0
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 0
    pf = gp / gl if gl > 0 else float("inf")
    total = df_sig["pnl"].sum()
    exp = total / n_sig

    # Compute equity with 2% risk per trade
    equity = 200.0
    for _, r in df_sig.iterrows():
        sl_dist_pct = abs(r["entry"] - r["sl"]) / r["entry"]
        risk_amt = equity * 0.02
        pos = risk_amt / sl_dist_pct if sl_dist_pct > 0 else 0
        equity += pos * (r["pnl"] / 100)
        if equity <= 0: equity = 0; break

    print(f"\n  {label}:")
    print(f"    Signals: {n_sig} | WR {wr:.1f}% | PF {pf:.2f}")
    print(f"    Avg W {aw:+.2f}% | Avg L {al:+.2f}% | Expect {exp:+.3f}%/trade")
    print(f"    Total sum: {total:+.1f}% | $200 → ${equity:.2f}")
    return df_sig

# Test on different periods
for period_label, start in [
    ("Last 3 months (Jan-Mar 2026)", "2026-01-01"),
    ("Last 6 months (Oct 2025 - Mar 2026)", "2025-10-01"),
    ("Last year (Apr 2025 - Mar 2026)", "2025-04-01"),
    ("Full 2025", "2025-01-01"),
    ("Full period (2022-2026)", "2022-01-01"),
]:
    df = df_full[df_full.index >= start].copy()
    print("\n" + "=" * 80)
    print(f"  {period_label}  ({len(df)} bars)")
    print("=" * 80)

    for mode in ["band", "close", "open"]:
        label = f"Entry: {mode:<6} | TP: fixed"
        run_realistic(df, label, entry_mode=mode, fixed_tp=True)

