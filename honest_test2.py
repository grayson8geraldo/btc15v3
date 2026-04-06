#!/usr/bin/env python3
"""
Additional honest tests:
1. Buy & Hold baseline
2. Higher timeframe (4h) strategies
3. Long-only dip buying in bull market
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

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def resample_ohlc(df, tf):
    return df.resample(tf).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

df_15m = load_data("/home/user/btc15v3")
print(f"Loaded {len(df_15m)} bars of 15m data")
print(f"Period: {df_15m.index.min()} → {df_15m.index.max()}\n")

# ═══════════════════════════════════════════════════════════════
# TEST 1: BUY & HOLD BASELINE
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  TEST 1: BUY & HOLD BASELINE")
print("=" * 70)

start_price = df_15m["Close"].iloc[0]
end_price = df_15m["Close"].iloc[-1]
buy_hold_ret = (end_price - start_price) / start_price * 100

# Simulate $200 buy & hold
shares = 200 / start_price
final_value = shares * end_price

# Max drawdown of buy & hold
cum_value = df_15m["Close"] * shares
peak = cum_value.expanding().max()
dd = ((cum_value - peak) / peak * 100).min()

# Yearly returns
df_15m_d = df_15m["Close"].resample("YE").last()
yearly_returns = df_15m_d.pct_change() * 100

print(f"  Start price: ${start_price:,.2f}")
print(f"  End price:   ${end_price:,.2f}")
print(f"  Return:      {buy_hold_ret:+.1f}%")
print(f"  $200 → ${final_value:,.2f}")
print(f"  Max DD:      {dd:.1f}%")
print(f"\n  Yearly returns:")
for date, ret in yearly_returns.items():
    if not np.isnan(ret):
        print(f"    {date.year}: {ret:+.1f}%")

# ═══════════════════════════════════════════════════════════════
# TEST 2: 4H DONCHIAN BREAKOUT (higher timeframe)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TEST 2: 4H TIMEFRAME STRATEGIES (HONEST)")
print("=" * 70)

df_4h = resample_ohlc(df_15m, "4h")
print(f"  Resampled to 4h: {len(df_4h)} bars")

@dataclass
class Trade:
    dir:str=""; entry:float=0; exit:float=0; tp:float=0; sl:float=0
    pnl_pct:float=0; pnl_abs:float=0; result:str=""; entry_t:object=None

def sim(df, sig_func, cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=50):
    h = df["High"].values; l = df["Low"].values
    c = df["Close"].values
    bars = df.index
    n = len(df)
    trades = []
    equity = cap
    in_trade = False; t = None; pos = 0; hold = 0
    warmup = 220

    for i in range(warmup, n):
        if in_trade:
            hold += 1
            ep = None; res = None
            if t.dir == "long":
                if l[i] <= t.sl: ep = t.sl; res = "SL"
                elif h[i] >= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = c[i]; res = "timeout"
            else:
                if h[i] >= t.sl: ep = t.sl; res = "SL"
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

        sig = sig_func(i)
        if sig is None: continue
        direction, entry, tp, sl = sig

        if not (l[i] <= entry <= h[i]): continue

        if direction == "long":
            if not (tp > entry > sl): continue
            ds = (entry - sl) / entry
        else:
            if not (tp < entry < sl): continue
            ds = (sl - entry) / entry

        if not (0.002 < ds < 0.08): continue
        ra = equity * (risk/100)
        p = ra / (ds + 2*(comm+slip)/100)
        if p > 0:
            t = Trade(dir=direction, entry=entry, tp=tp, sl=sl, entry_t=bars[i])
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
    return {"n":n,"wr":wr,"pf":pf,"aw":aw,"al":al,"ret":(feq-cap)/cap*100,"dd":dd,"feq":feq}

# Prepare 4h indicators with shift(1)
ema200_4h = ema(df_4h["Close"], 200).shift(1).values
ema50_4h = ema(df_4h["Close"], 50).shift(1).values
atr14_4h = calc_atr(df_4h["High"], df_4h["Low"], df_4h["Close"], 14).shift(1).values
rsi14_4h = calc_rsi(df_4h["Close"], 14).shift(1).values
don20_h = df_4h["High"].rolling(20).max().shift(1).values
don20_l = df_4h["Low"].rolling(20).min().shift(1).values

h4h = df_4h["High"].values
l4h = df_4h["Low"].values
c4h = df_4h["Close"].values

# 4h Donchian
def s_don_4h(i):
    if np.isnan(don20_h[i]) or np.isnan(atr14_4h[i]) or np.isnan(ema200_4h[i]): return None
    up = c4h[i-1] > ema200_4h[i]
    dn = c4h[i-1] < ema200_4h[i]
    if up and h4h[i] >= don20_h[i] and c4h[i-1] < don20_h[i]:
        entry = don20_h[i]
        sl = entry - atr14_4h[i] * 2.0
        tp = entry + atr14_4h[i] * 4.0
        return ("long", entry, tp, sl)
    if dn and l4h[i] <= don20_l[i] and c4h[i-1] > don20_l[i]:
        entry = don20_l[i]
        sl = entry + atr14_4h[i] * 2.0
        tp = entry - atr14_4h[i] * 4.0
        return ("short", entry, tp, sl)
    return None

# 4h EMA pullback
def s_pull_4h(i):
    if np.isnan(ema50_4h[i]) or np.isnan(ema200_4h[i]) or np.isnan(atr14_4h[i]): return None
    up = c4h[i-1] > ema200_4h[i]
    dn = c4h[i-1] < ema200_4h[i]
    if up and l4h[i] <= ema50_4h[i] and c4h[i-1] > ema50_4h[i]:
        entry = ema50_4h[i]
        sl = entry - atr14_4h[i] * 1.5
        tp = entry + atr14_4h[i] * 3.0
        return ("long", entry, tp, sl)
    if dn and h4h[i] >= ema50_4h[i] and c4h[i-1] < ema50_4h[i]:
        entry = ema50_4h[i]
        sl = entry + atr14_4h[i] * 1.5
        tp = entry - atr14_4h[i] * 3.0
        return ("short", entry, tp, sl)
    return None

# 4h RSI mean reversion
def s_rsi_4h(i):
    if np.isnan(rsi14_4h[i]) or np.isnan(atr14_4h[i]) or np.isnan(ema50_4h[i]): return None
    if rsi14_4h[i] < 30:
        entry = c4h[i-1]
        if not (l4h[i] <= entry <= h4h[i]): return None
        sl = entry - atr14_4h[i] * 2.0
        tp = ema50_4h[i]
        if tp <= entry or (tp - entry) / (entry - sl) < 1.5: return None
        return ("long", entry, tp, sl)
    if rsi14_4h[i] > 70:
        entry = c4h[i-1]
        if not (l4h[i] <= entry <= h4h[i]): return None
        sl = entry + atr14_4h[i] * 2.0
        tp = ema50_4h[i]
        if tp >= entry or (entry - tp) / (sl - entry) < 1.5: return None
        return ("short", entry, tp, sl)
    return None

for name, sf in [("4h Donchian20 + EMA200", s_don_4h),
                  ("4h EMA50 pullback", s_pull_4h),
                  ("4h RSI reversion", s_rsi_4h)]:
    trades, feq = sim(df_4h, sf, max_hold=40)
    st = stats(trades)
    if st and st["n"] > 0:
        status = "OK" if st["pf"] > 1 else ""
        print(f"  {name:<30} n={st['n']:>4} WR {st['wr']:>5.1f}% PF {st['pf']:>5.2f} "
              f"AvgW {st['aw']:>+5.2f}% AvgL {st['al']:>+5.2f}% Ret {st['ret']:>+7.1f}% DD {st['dd']:>6.1f}% {status}")

# ═══════════════════════════════════════════════════════════════
# TEST 3: LONG-ONLY DIP BUYING IN BULL MARKET (15m)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TEST 3: LONG-ONLY DIP BUY IN BULL MARKET")
print("=" * 70)

# Weekly trend filter (BTC in macro uptrend)
df_1w = resample_ohlc(df_15m, "W")
ema_weekly = ema(df_1w["Close"], 10).shift(1).reindex(df_15m.index, method="ffill").values
c1w = df_1w["Close"].shift(1).reindex(df_15m.index, method="ffill").values

# 15m indicators with shift
ema20_15 = ema(df_15m["Close"], 20).shift(1).values
ema50_15 = ema(df_15m["Close"], 50).shift(1).values
ema200_15 = ema(df_15m["Close"], 200).shift(1).values
atr14_15 = calc_atr(df_15m["High"], df_15m["Low"], df_15m["Close"], 14).shift(1).values
rsi14_15 = calc_rsi(df_15m["Close"], 14).shift(1).values

h15 = df_15m["High"].values
l15 = df_15m["Low"].values
c15 = df_15m["Close"].values

def s_dip_buy(i):
    """Long only. Buy dips only when weekly EMA trending up."""
    if np.isnan(ema_weekly[i]) or np.isnan(c1w[i]): return None
    if np.isnan(ema200_15[i]) or np.isnan(rsi14_15[i]) or np.isnan(atr14_15[i]): return None
    # Weekly uptrend
    if c1w[i] <= ema_weekly[i]: return None
    # 15m uptrend
    if c15[i-1] <= ema200_15[i]: return None
    # RSI pullback
    if rsi14_15[i] > 35: return None
    # Buy at close
    entry = c15[i-1]
    if not (l15[i] <= entry <= h15[i]): return None
    sl = entry - atr14_15[i] * 2.0
    tp = entry + atr14_15[i] * 4.0
    return ("long", entry, tp, sl)

trades, feq = sim(df_15m, s_dip_buy, max_hold=96)
st = stats(trades)
if st and st["n"] > 0:
    status = "OK" if st["pf"] > 1 else ""
    print(f"  Long-only dip buy (weekly uptrend + RSI<35)")
    print(f"  n={st['n']} | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
          f"AvgW {st['aw']:+.2f}% | AvgL {st['al']:+.2f}%")
    print(f"  Return: {st['ret']:+.1f}% | Max DD: {st['dd']:.1f}% | $200 → ${feq:.2f} {status}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  HONEST SUMMARY")
print("=" * 70)
print(f"  BUY & HOLD:    $200 → ${final_value:,.2f} ({buy_hold_ret:+.1f}%)")
print(f"                  ^^ No trading, no fees, just hold BTC")
print(f"                  Max DD was {dd:.1f}% but recovered")

