#!/usr/bin/env python3
"""
OB v2: добавляем Fair Value Gap (FVG) + HTF trend filter + импульс confirmation.

FVG — трёхсвечный паттерн где средняя свеча оставляет гэп между high[i-1] и low[i+1]
(для бычьего) или low[i-1] и high[i+1] (для медвежьего). 
Это показывает, что импульс был СИЛЬНЫЙ (институциональный).
Стратегия: ищем OB, подтверждённый FVG, в направлении HTF тренда.
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

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def resample_ohlc(df, tf):
    return df.resample(tf).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

@dataclass
class Trade:
    direction:str=""; entry:float=0; exit:float=0; tp:float=0; sl:float=0
    pnl_pct:float=0; pnl_abs:float=0; result:str=""; entry_t:object=None; exit_t:object=None

def run(df, label="", swing_lb=10, sweep_lb=30, ob_lb=10, ob_exp=30,
        min_sweep=0.15, rr=2.0, use_fvg=True, use_trend=True,
        trend_tf="4h", trend_len=50,
        initial_capital=200, risk_pct=2.0,
        commission=0.04, slippage=0.01, max_hold=96):

    opens = df["Open"].values; highs = df["High"].values
    lows = df["Low"].values; closes = df["Close"].values
    bars = df.index
    n = len(df)

    # Trend
    if use_trend:
        htf = resample_ohlc(df, trend_tf)
        tma = ema(htf["Close"], trend_len).shift(1).reindex(df.index, method="ffill").values
    else:
        tma = np.full(n, np.nan)

    # Swings
    swing_h = np.zeros(n, dtype=bool)
    swing_l = np.zeros(n, dtype=bool)
    for i in range(swing_lb, n - swing_lb):
        if highs[i] == max(highs[i-swing_lb:i+swing_lb+1]):
            swing_h[i] = True
        if lows[i] == min(lows[i-swing_lb:i+swing_lb+1]):
            swing_l[i] = True

    # FVG detection: bullish FVG at bar i means low[i+1] > high[i-1]
    # i.e. gap between i-1 and i+1. We'll check at bar i (middle).
    def has_bullish_fvg(idx):
        if idx < 1 or idx >= n - 1: return False
        return lows[idx + 1] > highs[idx - 1]

    def has_bearish_fvg(idx):
        if idx < 1 or idx >= n - 1: return False
        return highs[idx + 1] < lows[idx - 1]

    # Find OB: last opposite candle before impulse
    def find_ob(from_idx, direction, lb):
        for i in range(from_idx - 1, max(from_idx - lb, 0), -1):
            if direction == "long" and closes[i] < opens[i]:
                return i, lows[i], highs[i]
            if direction == "short" and closes[i] > opens[i]:
                return i, lows[i], highs[i]
        return None, None, None

    trades = []; equity = initial_capital
    in_trade = False; trade = None; pos = 0; hold = 0
    pending_long = []  # (ob_lo, ob_hi, sl, tp, bar)
    pending_short = []

    warmup = max(swing_lb + 50, 300)

    for i in range(warmup, n - 2):
        pending_long = [o for o in pending_long if i - o[4] < ob_exp]
        pending_short = [o for o in pending_short if i - o[4] < ob_exp]

        # EXIT
        if in_trade:
            hold += 1
            t = trade; ep = None; res = None
            if t.direction == "long":
                if lows[i] <= t.sl: ep = t.sl; res = "SL"
                elif highs[i] >= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = closes[i]; res = "timeout"
            else:
                if highs[i] >= t.sl: ep = t.sl; res = "SL"
                elif lows[i] <= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = closes[i]; res = "timeout"

            if ep is not None:
                cost = (commission + slippage)/100
                rpnl = ((ep - t.entry)/t.entry - 2*cost) if t.direction == "long" else ((t.entry - ep)/t.entry - 2*cost)
                pnl = pos * rpnl; equity += pnl
                t.exit = ep; t.exit_t = bars[i]; t.result = res
                t.pnl_pct = rpnl * 100; t.pnl_abs = pnl
                trades.append(t); in_trade = False
                if equity <= 0: break
                continue

        # DETECT SWEEP + OB + FVG + TREND
        if i < sweep_lb + 2: continue

        # Recent swing low (for long setup)
        rsl_idx = -1; rsl_price = np.inf
        for j in range(i - 1, max(i - sweep_lb, 0), -1):
            if swing_l[j]:
                rsl_idx = j; rsl_price = lows[j]; break

        # Recent swing high (for short setup)
        rsh_idx = -1; rsh_price = -np.inf
        for j in range(i - 1, max(i - sweep_lb, 0), -1):
            if swing_h[j]:
                rsh_idx = j; rsh_price = highs[j]; break

        # BULLISH setup: sweep low + close above + bullish FVG + uptrend
        if rsl_idx > 0:
            sweep_size = (rsl_price - lows[i]) / rsl_price * 100
            if (lows[i] < rsl_price and closes[i] > rsl_price and sweep_size >= min_sweep):
                # Trend check
                trend_ok = (not use_trend) or (not np.isnan(tma[i]) and closes[i] > tma[i])
                # FVG check (within last 3 bars)
                fvg_ok = (not use_fvg) or has_bullish_fvg(i-1) or has_bullish_fvg(i)
                if trend_ok and fvg_ok:
                    ob_i, ob_lo, ob_hi = find_ob(rsl_idx, "long", ob_lb)
                    if ob_i is not None and ob_hi > lows[i] and ob_hi < closes[i]:
                        sl = lows[i] * 0.999  # just below sweep
                        entry = ob_hi
                        tp = entry + (entry - sl) * rr
                        if entry > sl and tp > entry:
                            pending_long.append((ob_lo, ob_hi, sl, tp, i))

        # BEARISH setup
        if rsh_idx > 0:
            sweep_size = (highs[i] - rsh_price) / rsh_price * 100
            if (highs[i] > rsh_price and closes[i] < rsh_price and sweep_size >= min_sweep):
                trend_ok = (not use_trend) or (not np.isnan(tma[i]) and closes[i] < tma[i])
                fvg_ok = (not use_fvg) or has_bearish_fvg(i-1) or has_bearish_fvg(i)
                if trend_ok and fvg_ok:
                    ob_i, ob_lo, ob_hi = find_ob(rsh_idx, "short", ob_lb)
                    if ob_i is not None and ob_lo < highs[i] and ob_lo > closes[i]:
                        sl = highs[i] * 1.001
                        entry = ob_lo
                        tp = entry - (sl - entry) * rr
                        if entry < sl and tp < entry:
                            pending_short.append((ob_lo, ob_hi, sl, tp, i))

        # ENTRY
        if not in_trade and equity > 1:
            for ob in pending_long[:]:
                ob_lo, ob_hi, sl, tp, formed = ob
                if lows[i] <= ob_hi and lows[i] >= ob_lo:
                    entry = ob_hi
                    ds = (entry - sl) / entry
                    dt = (tp - entry) / entry
                    if ds > 0 and dt > 0:
                        ra = equity * (risk_pct/100)
                        pos = ra / (ds + 2*(commission+slippage)/100)
                        if pos > 0:
                            trade = Trade(direction="long", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                            in_trade = True; hold = 0
                            pending_long.remove(ob); break
            if not in_trade:
                for ob in pending_short[:]:
                    ob_lo, ob_hi, sl, tp, formed = ob
                    if highs[i] >= ob_lo and highs[i] <= ob_hi:
                        entry = ob_lo
                        ds = (sl - entry) / entry
                        dt = (entry - tp) / entry
                        if ds > 0 and dt > 0:
                            ra = equity * (risk_pct/100)
                            pos = ra / (ds + 2*(commission+slippage)/100)
                            if pos > 0:
                                trade = Trade(direction="short", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                                in_trade = True; hold = 0
                                pending_short.remove(ob); break

    return trades, equity, label


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

    scenarios = [
        dict(label="OB+FVG+Trend (base)",                    use_fvg=True,  use_trend=True, rr=2.0),
        dict(label="OB+FVG only (no trend)",                 use_fvg=True,  use_trend=False, rr=2.0),
        dict(label="OB+Trend only (no FVG)",                 use_fvg=False, use_trend=True, rr=2.0),
        dict(label="OB+FVG+Trend, RR=1.5",                   use_fvg=True,  use_trend=True, rr=1.5),
        dict(label="OB+FVG+Trend, RR=3",                     use_fvg=True,  use_trend=True, rr=3.0),
        dict(label="OB+FVG+Trend 1h/200",                    use_fvg=True,  use_trend=True, trend_tf="1h", trend_len=200, rr=2.0),
        dict(label="OB+FVG+Trend D/20",                      use_fvg=True,  use_trend=True, trend_tf="1D", trend_len=20, rr=2.0),
        dict(label="OB+FVG+Trend, min_sweep=0.25",           use_fvg=True,  use_trend=True, min_sweep=0.25, rr=2.0),
        dict(label="OB+FVG+Trend, swing=5",                  use_fvg=True,  use_trend=True, swing_lb=5, sweep_lb=20, rr=2.0),
        dict(label="OB+FVG+Trend, swing=20",                 use_fvg=True,  use_trend=True, swing_lb=20, sweep_lb=60, rr=2.0),
        dict(label="OB+FVG+Trend Conservative",              use_fvg=True,  use_trend=True, swing_lb=15, sweep_lb=50, min_sweep=0.25, rr=2.5),
        dict(label="OB+FVG+Trend Aggressive",                use_fvg=True,  use_trend=True, swing_lb=5, sweep_lb=15, min_sweep=0.08, rr=1.8),
    ]

    print("Running...\n")
    results = []
    for s in scenarios:
        lbl = s.pop("label")
        tr, feq, _ = run(df, label=lbl, **s)
        st = stats(tr)
        if st and st["n"] > 0:
            st["label"] = lbl
            results.append(st)
            pfx = "+" if st["ret"] > 0 else ""
            status = "OK" if st["pf"] > 1 else "LOSS"
            print(f"  {lbl:<42} {st['n']:>5} | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
                  f"E[R] {st['exp']:+.3f}% | {pfx}{st['ret']:>7.1f}% | DD {st['dd']:.1f}%  {status}")
        else:
            print(f"  {lbl:<42} No trades")

    if not results:
        print("\nNo valid scenarios."); return

    print("\n" + "=" * 125)
    print("  OB v2 COMPARISON")
    print("=" * 125)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["ret"] > 0 else ""
        print(f"  {s['label']:<42} {s['n']:>5} WR {s['wr']:>5.1f}% PF {s['pf']:>5.2f} "
              f"AvgW {s['aw']:>+5.2f}% AvgL {s['al']:>+5.2f}% E[R] {s['exp']:>+.3f}% "
              f"{pfx}{s['ret']:>7.1f}% DD {s['dd']:>5.1f}% ${s['feq']:>8.2f}")

    best = max(results, key=lambda x: x["pf"])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['label']}")
    print(f"{'='*70}")
    print(f"  Trades: {best['n']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | E[R]: {best['exp']:.3f}%")
    print(f"  Return: {best['ret']:.1f}% | DD: {best['dd']:.1f}% | $200 → ${best['feq']:,.2f}")

    # Charts
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("OB v2: Order Block + FVG + Trend Filter", fontsize=16, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [x["label"][:35] for x in sr]
    for ax, m, t, v in [(axes[0,0],"pf","Profit Factor",1.0), (axes[0,1],"ret","Return (%)",0),
                         (axes[1,0],"wr","Win Rate (%)",50), (axes[1,1],"exp","Expectancy (%)",0)]:
        vals = [x[m] for x in sr]
        colors = ["#00e676" if v2 >= v else "#ff1744" for v2 in vals]
        ax.barh(labels, vals, color=colors); ax.axvline(v, color="white", linestyle="--", alpha=0.5)
        ax.set_title(t); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "backtest_ob_v2.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nChart saved.")

if __name__ == "__main__":
    main()
