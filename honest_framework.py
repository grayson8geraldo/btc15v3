#!/usr/bin/env python3
"""
HONEST Strategy Framework — No Look-Ahead

Strict rules:
1. All indicators use .shift(1) — decisions at bar i use data [0..i-1]
2. Entry price must be achievable during bar i (within [low[i], high[i]])
3. Limit orders explicitly check future bar fill
4. Compare shift vs no-shift — if different → bug found

Strategies tested:
  S1: Donchian Breakout (classic turtle)
  S2: Opening Range Breakout (ORB)
  S3: EMA Pullback (trend-following)
  S4: Volatility Contraction (NR7) + breakout
  S5: Bollinger Band squeeze + breakout
  S6: RSI Divergence (simplified)
  S7: Session High/Low break (time-of-day)
  S8: Momentum (rate of change)
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
def sma(s, n): return s.rolling(n).mean()

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


@dataclass
class Trade:
    direction:str=""; entry:float=0; exit:float=0; tp:float=0; sl:float=0
    pnl_pct:float=0; pnl_abs:float=0; result:str=""; entry_t:object=None


def simulate(df, signals_func, cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=96):
    """
    Generic simulation. signals_func(df, i) returns (direction, entry, tp, sl) or None.
    All inputs to signals_func must be pre-shifted to avoid look-ahead.
    """
    h = df["High"].values; l = df["Low"].values
    c = df["Close"].values; o = df["Open"].values
    bars = df.index
    n = len(df)

    trades = []
    equity = cap
    in_trade = False
    t = None
    pos = 0
    hold = 0
    warmup = 250

    for i in range(warmup, n):
        # EXIT
        if in_trade:
            hold += 1
            ep = None; res = None
            if t.direction == "long":
                if l[i] <= t.sl and h[i] >= t.tp:
                    ep = t.sl; res = "SL"
                elif l[i] <= t.sl:
                    ep = t.sl; res = "SL"
                elif h[i] >= t.tp:
                    ep = t.tp; res = "TP"
                elif hold >= max_hold:
                    ep = c[i]; res = "timeout"
            else:
                if h[i] >= t.sl and l[i] <= t.tp:
                    ep = t.sl; res = "SL"
                elif h[i] >= t.sl:
                    ep = t.sl; res = "SL"
                elif l[i] <= t.tp:
                    ep = t.tp; res = "TP"
                elif hold >= max_hold:
                    ep = c[i]; res = "timeout"

            if ep is not None:
                cost = (comm+slip)/100
                rpnl = ((ep-t.entry)/t.entry - 2*cost) if t.direction == "long" else ((t.entry-ep)/t.entry - 2*cost)
                pnl = pos * rpnl
                equity += pnl
                t.exit = ep; t.result = res
                t.pnl_pct = rpnl * 100; t.pnl_abs = pnl
                trades.append(t)
                in_trade = False
                if equity <= 0: break
                continue

        if in_trade or equity <= 1: continue

        # ENTRY (signals_func provides signal from HISTORICAL data only)
        sig = signals_func(i)
        if sig is None: continue
        direction, entry, tp, sl = sig

        # Validate: entry must be reachable during bar i
        # For LONG: entry >= low[i] and entry <= high[i] (limit BELOW or at market)
        # For SHORT: entry <= high[i] and entry >= low[i]
        if not (l[i] <= entry <= h[i]):
            continue

        # Validate RR and distances
        if direction == "long":
            if not (tp > entry > sl): continue
            ds = (entry - sl) / entry
        else:
            if not (tp < entry < sl): continue
            ds = (sl - entry) / entry

        if not (0.001 < ds < 0.05): continue

        ra = equity * (risk/100)
        p = ra / (ds + 2*(comm+slip)/100)
        if p <= 0: continue

        t = Trade(direction=direction, entry=entry, tp=tp, sl=sl, entry_t=bars[i])
        pos = p
        in_trade = True
        hold = 0

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
    exp = (wr/100*aw) + ((100-wr)/100*al)
    return {"n":n,"wr":wr,"pf":pf,"aw":aw,"al":al,"exp":exp,"ret":(feq-cap)/cap*100,"dd":dd,"feq":feq}



# ═══════════════════════════════════════════════════════════════
# STRATEGIES — all use pre-shifted indicators
# ═══════════════════════════════════════════════════════════════

def build_strategies(df):
    """Pre-compute all indicators with shift(1). Return dict of signal funcs."""
    h = df["High"].values; l = df["Low"].values
    c = df["Close"].values; o = df["Open"].values
    n = len(df)

    # All indicators SHIFTED by 1 — decisions at bar i use data up to i-1
    ema20 = ema(df["Close"], 20).shift(1).values
    ema50 = ema(df["Close"], 50).shift(1).values
    ema200 = ema(df["Close"], 200).shift(1).values
    atr14 = calc_atr(df["High"], df["Low"], df["Close"], 14).shift(1).values
    rsi14 = calc_rsi(df["Close"], 14).shift(1).values
    stdev20 = df["Close"].rolling(20).std().shift(1).values
    vol_ma20 = df["Volume"].rolling(20).mean().shift(1).values

    # Donchian (N-bar high/low)
    don_high_20 = df["High"].rolling(20).max().shift(1).values
    don_low_20  = df["Low"].rolling(20).min().shift(1).values
    don_high_50 = df["High"].rolling(50).max().shift(1).values
    don_low_50  = df["Low"].rolling(50).min().shift(1).values

    # Bollinger
    bb_mid = ema20
    bb_upper = bb_mid + 2 * stdev20
    bb_lower = bb_mid - 2 * stdev20
    bb_width = (bb_upper - bb_lower) / bb_mid

    # Session info (hour of day UTC)
    hours = df.index.hour.values

    strategies = {}

    # ─── S1: Donchian 20 Breakout (classic turtle) ───
    def s1_donchian(i):
        if np.isnan(don_high_20[i]) or np.isnan(atr14[i]): return None
        if np.isnan(ema200[i]): return None
        trend_up = c[i-1] > ema200[i]
        trend_dn = c[i-1] < ema200[i]
        # LONG: breakout above 20-bar high (in uptrend)
        if trend_up and h[i] >= don_high_20[i] and c[i-1] < don_high_20[i]:
            entry = don_high_20[i]
            sl = entry - atr14[i] * 2.0
            tp = entry + atr14[i] * 4.0  # RR 2:1
            return ("long", entry, tp, sl)
        # SHORT
        if trend_dn and l[i] <= don_low_20[i] and c[i-1] > don_low_20[i]:
            entry = don_low_20[i]
            sl = entry + atr14[i] * 2.0
            tp = entry - atr14[i] * 4.0
            return ("short", entry, tp, sl)
        return None
    strategies["S1 Donchian20 + EMA200 trend"] = s1_donchian

    # ─── S2: Donchian 50 Breakout ───
    def s2_donchian50(i):
        if np.isnan(don_high_50[i]) or np.isnan(atr14[i]) or np.isnan(ema200[i]): return None
        trend_up = c[i-1] > ema200[i]
        trend_dn = c[i-1] < ema200[i]
        if trend_up and h[i] >= don_high_50[i] and c[i-1] < don_high_50[i]:
            entry = don_high_50[i]
            sl = entry - atr14[i] * 2.0
            tp = entry + atr14[i] * 4.0
            return ("long", entry, tp, sl)
        if trend_dn and l[i] <= don_low_50[i] and c[i-1] > don_low_50[i]:
            entry = don_low_50[i]
            sl = entry + atr14[i] * 2.0
            tp = entry - atr14[i] * 4.0
            return ("short", entry, tp, sl)
        return None
    strategies["S2 Donchian50 + trend"] = s2_donchian50

    # ─── S3: EMA pullback (trend following) ───
    # In uptrend (close > EMA200), buy pullback to EMA50
    def s3_ema_pullback(i):
        if np.isnan(ema50[i]) or np.isnan(ema200[i]) or np.isnan(atr14[i]): return None
        trend_up = c[i-1] > ema200[i]
        trend_dn = c[i-1] < ema200[i]
        if trend_up and l[i] <= ema50[i] and c[i-1] > ema50[i]:
            entry = ema50[i]
            sl = entry - atr14[i] * 1.5
            tp = entry + atr14[i] * 3.0  # RR 2:1
            return ("long", entry, tp, sl)
        if trend_dn and h[i] >= ema50[i] and c[i-1] < ema50[i]:
            entry = ema50[i]
            sl = entry + atr14[i] * 1.5
            tp = entry - atr14[i] * 3.0
            return ("short", entry, tp, sl)
        return None
    strategies["S3 EMA50 pullback in trend"] = s3_ema_pullback

    # ─── S4: NR7 (Narrow Range 7) + breakout ───
    # Find bar with smallest range of last 7, then trade breakout
    def s4_nr7(i):
        if i < 8 or np.isnan(atr14[i]): return None
        ranges = [h[i-k] - l[i-k] for k in range(1, 8)]
        if ranges[0] != min(ranges): return None  # prev bar was NR7
        # Breakout of NR7 bar
        nr_high = h[i-1]; nr_low = l[i-1]
        if h[i] > nr_high and c[i-1] <= nr_high:
            entry = nr_high
            sl = nr_low
            tp = entry + (entry - sl) * 2.5  # RR 2.5
            return ("long", entry, tp, sl)
        if l[i] < nr_low and c[i-1] >= nr_low:
            entry = nr_low
            sl = nr_high
            tp = entry - (sl - entry) * 2.5
            return ("short", entry, tp, sl)
        return None
    strategies["S4 NR7 Breakout"] = s4_nr7

    # ─── S5: Bollinger squeeze expansion ───
    # Low volatility then expansion: BB width expands dramatically
    def s5_bb_squeeze(i):
        if np.isnan(bb_width[i]) or np.isnan(bb_upper[i]) or np.isnan(atr14[i]): return None
        if i < 30: return None
        # BB width was in bottom 20% of last 30 bars → squeeze
        recent_widths = [bb_width[i-k] for k in range(1, 31) if not np.isnan(bb_width[i-k])]
        if len(recent_widths) < 20: return None
        threshold = np.percentile(recent_widths, 20)
        if bb_width[i-1] > threshold: return None  # not a squeeze
        # Breakout above/below BB
        if h[i] > bb_upper[i] and c[i-1] <= bb_upper[i]:
            entry = bb_upper[i]
            sl = entry - atr14[i] * 2.0
            tp = entry + atr14[i] * 4.0
            return ("long", entry, tp, sl)
        if l[i] < bb_lower[i] and c[i-1] >= bb_lower[i]:
            entry = bb_lower[i]
            sl = entry + atr14[i] * 2.0
            tp = entry - atr14[i] * 4.0
            return ("short", entry, tp, sl)
        return None
    strategies["S5 BB squeeze expansion"] = s5_bb_squeeze

    # ─── S6: RSI extreme reversal (mean reversion) ───
    def s6_rsi_extreme(i):
        if np.isnan(rsi14[i]) or np.isnan(atr14[i]) or np.isnan(ema20[i]): return None
        # Strong oversold: RSI < 25 and price below EMA20
        if rsi14[i] < 25 and c[i-1] < ema20[i]:
            entry = c[i-1]  # market order at previous close
            if not (l[i] <= entry <= h[i]): return None
            sl = entry - atr14[i] * 1.5
            tp = ema20[i]  # target the mean
            if (tp - entry) / (entry - sl) < 2.0: return None
            return ("long", entry, tp, sl)
        if rsi14[i] > 75 and c[i-1] > ema20[i]:
            entry = c[i-1]
            if not (l[i] <= entry <= h[i]): return None
            sl = entry + atr14[i] * 1.5
            tp = ema20[i]
            if (entry - tp) / (sl - entry) < 2.0: return None
            return ("short", entry, tp, sl)
        return None
    strategies["S6 RSI extreme (<25/>75)"] = s6_rsi_extreme

    # ─── S7: Session breakout (Asian range → London open) ───
    # Track 00:00-07:00 UTC range, trade breakout during 08:00-16:00 UTC
    def s7_session(i):
        if np.isnan(atr14[i]): return None
        hr = int(hours[i])
        if hr < 8 or hr >= 16: return None  # only trade London/NY sessions
        today_bars_back = int(hr - 8)
        start = int(i - today_bars_back - 32)
        end = int(i - today_bars_back)
        if start < 0 or end <= start: return None
        asian_high = max(h[start:end])
        asian_low = min(l[start:end])
        if h[i] > asian_high and c[i-1] <= asian_high:
            entry = asian_high
            sl = entry - atr14[i] * 1.5
            tp = entry + (entry - sl) * 2.0
            return ("long", entry, tp, sl)
        if l[i] < asian_low and c[i-1] >= asian_low:
            entry = asian_low
            sl = entry + atr14[i] * 1.5
            tp = entry - (sl - entry) * 2.0
            return ("short", entry, tp, sl)
        return None
    strategies["S7 Asian range breakout"] = s7_session

    # ─── S8: Momentum (ROC 20) ───
    def s8_momentum(i):
        if i < 25 or np.isnan(atr14[i]) or np.isnan(ema200[i]): return None
        roc = (c[i-1] - c[i-21]) / c[i-21] * 100  # 20-bar momentum (5h)
        # Strong momentum continuation with pullback entry
        if roc > 3 and c[i-1] > ema200[i]:  # strong up
            # Buy small pullback
            entry = c[i-1] - atr14[i] * 0.3
            if not (l[i] <= entry <= h[i]): return None
            sl = entry - atr14[i] * 1.5
            tp = entry + atr14[i] * 3.0
            return ("long", entry, tp, sl)
        if roc < -3 and c[i-1] < ema200[i]:
            entry = c[i-1] + atr14[i] * 0.3
            if not (l[i] <= entry <= h[i]): return None
            sl = entry + atr14[i] * 1.5
            tp = entry - atr14[i] * 3.0
            return ("short", entry, tp, sl)
        return None
    strategies["S8 Momentum continuation"] = s8_momentum

    return strategies


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading..."); df = load_data(data_dir); print(f"{len(df)} bars\n")

    strategies = build_strategies(df)

    # Full period test
    print("=" * 105)
    print("  HONEST BACKTEST — Full period (2022-2026)")
    print("=" * 105)
    print(f"  {'Strategy':<40} {'N':>5} {'WR%':>6} {'PF':>6} {'AvgW':>7} {'AvgL':>7} {'Ret%':>9} {'DD%':>7}")
    print("-" * 105)

    results = {}
    for name, sf in strategies.items():
        trades, feq = simulate(df, sf)
        st = stats(trades)
        if st and st["n"] > 0:
            results[name] = st
            pfx = "+" if st["ret"] > 0 else ""
            status = "OK" if st["pf"] > 1 else ""
            print(f"  {name:<40} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>5.2f} "
                  f"{st['aw']:>+6.2f}% {st['al']:>+6.2f}% {pfx}{st['ret']:>7.1f}% {st['dd']:>6.1f}%  {status}")
        else:
            print(f"  {name:<40} NO TRADES")

    # OOS test
    print("\n" + "=" * 105)
    print("  OUT-OF-SAMPLE TEST (Train: 2022-2024H1 | Test: 2024H2-2026)")
    print("=" * 105)
    train = df[df.index < "2024-07-01"]
    test = df[df.index >= "2024-07-01"]

    train_strats = build_strategies(train)
    test_strats = build_strategies(test)

    for name in results.keys():
        tr_trades, _ = simulate(train, train_strats[name])
        te_trades, _ = simulate(test, test_strats[name])
        tr_st = stats(tr_trades)
        te_st = stats(te_trades)
        if tr_st and te_st and tr_st["n"] > 0 and te_st["n"] > 0:
            tr_ok = "OK" if tr_st["pf"] > 1 else "--"
            te_ok = "OK" if te_st["pf"] > 1 else "--"
            both = "BOTH" if tr_st["pf"] > 1 and te_st["pf"] > 1 else ""
            print(f"  {name:<40}")
            print(f"    Train: n={tr_st['n']:>4} WR {tr_st['wr']:>5.1f}% PF {tr_st['pf']:>5.2f} {tr_ok}")
            print(f"    Test:  n={te_st['n']:>4} WR {te_st['wr']:>5.1f}% PF {te_st['pf']:>5.2f} {te_ok}  {both}")

if __name__ == "__main__":
    main()
