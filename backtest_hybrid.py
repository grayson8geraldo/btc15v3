#!/usr/bin/env python3
"""
HYBRID Strategy: Mean Reversion Bands + Order Block confluence

Идея: брать наши рабочие 4σ сигналы mean reversion, но ТОЛЬКО когда
на уровне банды есть подтверждение Order Block (institutional zone).
Это должно увеличить WR и уменьшить ложные сигналы.
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

@dataclass
class Trade:
    direction:str=""; entry:float=0; exit:float=0; tp:float=0; sl:float=0
    pnl_pct:float=0; pnl_abs:float=0; result:str=""; entry_t:object=None
    has_ob:bool=False

def run(df, label="", mult=4.0, sl_mult=1.0, rr_mode="mean",
        use_ob=True, ob_lookback=30, ob_zone_pct=0.15,
        use_fvg=False,
        cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=96):

    dev_ma = ema(df["Close"], 20).values
    stdev = df["Close"].rolling(20).std().values
    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)

    # Find OB bullish zones (last bearish candle before strong up move)
    # Simpler: any red candle that preceded 3+ consecutive green bars
    def is_bullish_ob(idx):
        if idx < 0 or idx >= n - 3: return False
        if c[idx] >= o[idx]: return False  # must be red
        # next 3 bars must be green and higher
        return c[idx+1] > o[idx+1] and c[idx+2] > o[idx+2] and c[idx+3] > h[idx]

    def is_bearish_ob(idx):
        if idx < 0 or idx >= n - 3: return False
        if c[idx] <= o[idx]: return False
        return c[idx+1] < o[idx+1] and c[idx+2] < o[idx+2] and c[idx+3] < l[idx]

    # Pre-compute OB arrays (avoid recalc in hot loop)
    bull_ob_zones = []  # (idx, low, high)
    bear_ob_zones = []
    for i in range(n - 4):
        if is_bullish_ob(i):
            bull_ob_zones.append((i, l[i], h[i]))
        if is_bearish_ob(i):
            bear_ob_zones.append((i, l[i], h[i]))

    # FVG detection
    def bull_fvg_near(idx, lookback=5):
        for k in range(max(idx - lookback, 1), min(idx + 1, n - 1)):
            if l[k + 1] > h[k - 1]:
                return True
        return False

    def bear_fvg_near(idx, lookback=5):
        for k in range(max(idx - lookback, 1), min(idx + 1, n - 1)):
            if h[k + 1] < l[k - 1]:
                return True
        return False

    trades = []; equity = cap
    in_trade = False; t = None; pos = 0; hold = 0
    warmup = 300

    for i in range(warmup, n):
        if in_trade:
            hold += 1; ep = None; res = None
            if t.direction == "long":
                if l[i] <= t.sl: ep = t.sl; res = "SL"
                elif h[i] >= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = c[i]; res = "timeout"
            else:
                if h[i] >= t.sl: ep = t.sl; res = "SL"
                elif l[i] <= t.tp: ep = t.tp; res = "TP"
                elif hold >= max_hold: ep = c[i]; res = "timeout"
            if ep is not None:
                cost = (comm+slip)/100
                rpnl = ((ep - t.entry)/t.entry - 2*cost) if t.direction == "long" else ((t.entry - ep)/t.entry - 2*cost)
                pnl = pos * rpnl; equity += pnl
                t.exit = ep; t.result = res; t.pnl_pct = rpnl*100; t.pnl_abs = pnl
                trades.append(t); in_trade = False
                if equity <= 0: break
                continue

        if in_trade or equity <= 1: continue
        if np.isnan(dev_ma[i]) or np.isnan(stdev[i]) or stdev[i] == 0: continue

        lb = dev_ma[i] - stdev[i] * mult
        ub = dev_ma[i] + stdev[i] * mult

        # Long signal: low touches lower band
        if l[i] <= lb:
            entry = lb
            tp = dev_ma[i]
            sl = entry - stdev[i] * sl_mult
            zone_half = stdev[i] * ob_zone_pct * mult  # tolerance around band

            has_ob = False
            if use_ob:
                # Check if any bull OB from recent past is near the band
                for ob_i, ob_lo, ob_hi in reversed(bull_ob_zones):
                    if i - ob_i > ob_lookback: break
                    if ob_i >= i: continue
                    # OB zone overlaps with band area
                    if ob_lo <= lb + zone_half and ob_hi >= lb - zone_half:
                        has_ob = True
                        break

            fvg_ok = (not use_fvg) or bull_fvg_near(i)
            cond = (has_ob or not use_ob) and fvg_ok

            if cond and tp > entry and (entry - sl) > 0:
                ds = (entry - sl) / entry
                ra = equity * (risk/100)
                pos = ra / (ds + 2*(comm+slip)/100)
                if pos > 0:
                    t = Trade(direction="long", entry=entry, tp=tp, sl=sl, entry_t=bars[i], has_ob=has_ob)
                    in_trade = True; hold = 0
                    continue

        # Short signal
        if h[i] >= ub:
            entry = ub
            tp = dev_ma[i]
            sl = entry + stdev[i] * sl_mult
            zone_half = stdev[i] * ob_zone_pct * mult

            has_ob = False
            if use_ob:
                for ob_i, ob_lo, ob_hi in reversed(bear_ob_zones):
                    if i - ob_i > ob_lookback: break
                    if ob_i >= i: continue
                    if ob_lo <= ub + zone_half and ob_hi >= ub - zone_half:
                        has_ob = True
                        break

            fvg_ok = (not use_fvg) or bear_fvg_near(i)
            cond = (has_ob or not use_ob) and fvg_ok

            if cond and tp < entry and (sl - entry) > 0:
                ds = (sl - entry) / entry
                ra = equity * (risk/100)
                pos = ra / (ds + 2*(comm+slip)/100)
                if pos > 0:
                    t = Trade(direction="short", entry=entry, tp=tp, sl=sl, entry_t=bars[i], has_ob=has_ob)
                    in_trade = True; hold = 0

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
        dict(label="4σ baseline (no OB, no FVG)",        mult=4.0, use_ob=False, use_fvg=False),
        dict(label="4σ + OB confluence",                 mult=4.0, use_ob=True,  use_fvg=False, ob_lookback=30),
        dict(label="4σ + OB (long lookback 100)",        mult=4.0, use_ob=True,  use_fvg=False, ob_lookback=100),
        dict(label="4σ + OB (short lookback 15)",        mult=4.0, use_ob=True,  use_fvg=False, ob_lookback=15),
        dict(label="4σ + OB + FVG",                      mult=4.0, use_ob=True,  use_fvg=True,  ob_lookback=30),
        dict(label="4σ + FVG only",                      mult=4.0, use_ob=False, use_fvg=True),
        dict(label="3σ + OB confluence",                 mult=3.0, use_ob=True,  use_fvg=False, ob_lookback=30),
        dict(label="3σ baseline",                        mult=3.0, use_ob=False, use_fvg=False),
        dict(label="4σ + OB, SL=1.5σ",                   mult=4.0, sl_mult=1.5, use_ob=True, ob_lookback=30),
        dict(label="4σ + OB, zone wider",                mult=4.0, use_ob=True,  ob_zone_pct=0.3, ob_lookback=30),
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
            # Count trades with OB
            ob_count = st["df"]["has_ob"].sum() if "has_ob" in st["df"].columns else 0
            print(f"  {lbl:<38} {st['n']:>5} | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
                  f"E[R] {st['exp']:+.3f}% | {pfx}{st['ret']:>8.1f}% | DD {st['dd']:.1f}% | OB={ob_count}  {status}")
        else:
            print(f"  {lbl:<38} No trades")

    if not results:
        print("No valid scenarios"); return

    print("\n" + "=" * 125)
    print("  HYBRID STRATEGY COMPARISON")
    print("=" * 125)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["ret"] > 0 else ""
        print(f"  {s['label']:<38} {s['n']:>5} WR {s['wr']:>5.1f}% PF {s['pf']:>5.2f} "
              f"E[R] {s['exp']:>+.3f}% {pfx}{s['ret']:>9.1f}% DD {s['dd']:>6.1f}% ${s['feq']:>10,.0f}")

    best = max(results, key=lambda x: x["pf"])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['label']}")
    print(f"  {best['n']} trades | WR {best['wr']:.1f}% | PF {best['pf']:.2f}")
    print(f"  $200 → ${best['feq']:,.2f} | DD {best['dd']:.1f}%")

    # Charts
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Hybrid: Mean Reversion + Order Block Confluence", fontsize=16, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [x["label"][:35] for x in sr]
    for ax, m, t, v in [(axes[0,0],"pf","Profit Factor",1.0), (axes[0,1],"ret","Return (%)",0),
                         (axes[1,0],"wr","Win Rate (%)",50), (axes[1,1],"exp","Expectancy (%)",0)]:
        vals = [x[m] for x in sr]
        colors = ["#00e676" if vl >= v else "#ff1744" for vl in vals]
        ax.barh(labels, vals, color=colors); ax.axvline(v, color="white", linestyle="--", alpha=0.5)
        ax.set_title(t); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "backtest_hybrid.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("\nChart saved.")

if __name__ == "__main__":
    main()
