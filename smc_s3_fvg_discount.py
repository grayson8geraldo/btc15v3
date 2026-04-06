#!/usr/bin/env python3
"""
SMC Strategy 3: FVG + OB in Discount/Premium zones

Логика:
1. Определяем dealing range — последние swing high и swing low (HTF)
2. 50% range делит зоны:
   - Discount zone (нижняя половина) — только LONG (зона спроса)
   - Premium zone (верхняя половина) — только SHORT (зона предложения)
3. Ищем FVG (Fair Value Gap) — гэп между свечами i-1 и i+1
4. Если FVG в Discount + есть OB рядом → LONG на ретесте
5. Если FVG в Premium + есть OB рядом → SHORT на ретесте
6. SL за свипом, TP — обычно 50% линия или противоположный экстремум
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

@dataclass
class Trade:
    direction:str=""; entry:float=0; exit:float=0; tp:float=0; sl:float=0
    pnl_pct:float=0; pnl_abs:float=0; result:str=""; entry_t:object=None


def run(df, label="",
        range_lookback=96,    # bars to define dealing range (96 * 15m = 24h)
        zone_threshold=0.5,   # 0.5 = strict 50%, 0.4 = wider Discount
        min_fvg_pct=0.05,     # min FVG size in %
        rr_target=2.0,
        target_mode="50pct",  # "50pct" / "opposite" / "rr"
        cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=96):

    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)

    trades = []; equity = cap
    in_trade = False; t = None; pos = 0; hold = 0
    warmup = max(range_lookback, 50) + 10

    for i in range(warmup, n - 2):
        if in_trade:
            hold += 1
            ep = None; res = None
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
                rpnl = ((ep-t.entry)/t.entry-2*cost) if t.direction == "long" else ((t.entry-ep)/t.entry-2*cost)
                pnl = pos * rpnl; equity += pnl
                t.exit = ep; t.result = res; t.pnl_pct = rpnl*100; t.pnl_abs = pnl
                trades.append(t); in_trade = False
                if equity <= 0: break
                continue

        if in_trade or equity <= 1: continue

        # Define dealing range
        range_high = max(h[i-range_lookback:i])
        range_low = min(l[i-range_lookback:i])
        range_size = range_high - range_low
        if range_size <= 0: continue
        eq50 = (range_high + range_low) / 2

        # Discount: lower half. Premium: upper half
        in_discount = c[i] < range_low + range_size * zone_threshold
        in_premium = c[i] > range_high - range_size * zone_threshold

        # Detect FVG on previous 3 bars (i-2, i-1, i): middle is i-1
        # Bullish FVG: low[i] > high[i-2]
        # Bearish FVG: high[i] < low[i-2]

        # Check bullish FVG completed at bar i (gap between i-2 and i)
        if i >= 2:
            bull_fvg = l[i] > h[i-2]
            bear_fvg = h[i] < l[i-2]

            # LONG: bullish FVG in discount zone
            if bull_fvg and in_discount:
                fvg_size = (l[i] - h[i-2]) / h[i-2] * 100
                if fvg_size >= min_fvg_pct:
                    # OB: the bullish candle that created the FVG (bar i-1)
                    # Entry at the FVG midpoint or top of i-1 candle (limit retracement)
                    entry = (l[i] + h[i-2]) / 2  # FVG midpoint
                    sl = min(l[i-2], l[i-1]) * 0.999
                    if target_mode == "50pct":
                        tp = eq50
                    elif target_mode == "opposite":
                        tp = range_high
                    else:
                        tp = entry + (entry - sl) * rr_target

                    if tp > entry and (entry - sl) > 0:
                        ds = (entry - sl) / entry
                        if 0.001 < ds < 0.05:
                            ra = equity * (risk/100)
                            pos = ra / (ds + 2*(comm+slip)/100)
                            if pos > 0:
                                t = Trade(direction="long", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                                in_trade = True; hold = 0
                                continue

            # SHORT: bearish FVG in premium zone
            if bear_fvg and in_premium:
                fvg_size = (l[i-2] - h[i]) / l[i-2] * 100
                if fvg_size >= min_fvg_pct:
                    entry = (h[i] + l[i-2]) / 2
                    sl = max(h[i-2], h[i-1]) * 1.001
                    if target_mode == "50pct":
                        tp = eq50
                    elif target_mode == "opposite":
                        tp = range_low
                    else:
                        tp = entry - (sl - entry) * rr_target

                    if tp < entry and (sl - entry) > 0:
                        ds = (sl - entry) / entry
                        if 0.001 < ds < 0.05:
                            ra = equity * (risk/100)
                            pos = ra / (ds + 2*(comm+slip)/100)
                            if pos > 0:
                                t = Trade(direction="short", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
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
        # Range 24h (96 bars)
        dict(label="24h range, 50%, TP=50pct",          range_lookback=96, zone_threshold=0.5, target_mode="50pct"),
        dict(label="24h range, 50%, TP=opposite",       range_lookback=96, zone_threshold=0.5, target_mode="opposite"),
        dict(label="24h range, 50%, TP=RR2",            range_lookback=96, zone_threshold=0.5, target_mode="rr", rr_target=2.0),
        dict(label="24h range, 40%, TP=50pct",          range_lookback=96, zone_threshold=0.4, target_mode="50pct"),
        # Range 4h (16 bars)
        dict(label="4h range, 50%, TP=50pct",           range_lookback=16, zone_threshold=0.5, target_mode="50pct"),
        dict(label="4h range, 50%, TP=opposite",        range_lookback=16, zone_threshold=0.5, target_mode="opposite"),
        # Range 3 days
        dict(label="3D range, 50%, TP=50pct",           range_lookback=288, zone_threshold=0.5, target_mode="50pct"),
        dict(label="3D range, 50%, TP=opposite",        range_lookback=288, zone_threshold=0.5, target_mode="opposite"),
        # Range 1 week
        dict(label="1W range, 50%, TP=50pct",           range_lookback=672, zone_threshold=0.5, target_mode="50pct"),
        dict(label="1W range, 50%, TP=opposite",        range_lookback=672, zone_threshold=0.5, target_mode="opposite"),
        # Tight zones (0.3 threshold)
        dict(label="24h range, 30% zones, TP=50pct",    range_lookback=96, zone_threshold=0.3, target_mode="50pct"),
        dict(label="24h range, 30% zones, TP=opposite", range_lookback=96, zone_threshold=0.3, target_mode="opposite"),
        # Larger min FVG
        dict(label="24h, min FVG 0.15%, TP=50pct",      range_lookback=96, min_fvg_pct=0.15, target_mode="50pct"),
        dict(label="24h, min FVG 0.15%, TP=opp",        range_lookback=96, min_fvg_pct=0.15, target_mode="opposite"),
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
                  f"E[R] {st['exp']:+.3f}% | {pfx}{st['ret']:>9.1f}% | DD {st['dd']:.1f}%  {status}")
        else:
            print(f"  {lbl:<42} No trades")

    if not results:
        print("No valid scenarios"); return

    print("\n" + "=" * 130)
    print("  STRATEGY 3: FVG + Discount/Premium Zones (sorted by PF)")
    print("=" * 130)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["ret"] > 0 else ""
        print(f"  {s['label']:<42} {s['n']:>5} WR {s['wr']:>5.1f}% PF {s['pf']:>5.2f} "
              f"E[R] {s['exp']:>+.3f}% {pfx}{s['ret']:>9.1f}% DD {s['dd']:>6.1f}% ${s['feq']:>10,.0f}")

    best = max(results, key=lambda x: x["pf"])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['label']}")
    print(f"  {best['n']} trades | WR {best['wr']:.1f}% | PF {best['pf']:.2f} | $200 → ${best['feq']:,.2f} | DD {best['dd']:.1f}%")
    if best["yearly"]:
        for y, p in sorted(best["yearly"].items()):
            print(f"    {y}: ${p:,.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Strategy 3: FVG + Discount/Premium Zones", fontsize=16, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [x["label"][:38] for x in sr]
    for ax, m, t, v in [(axes[0,0],"pf","Profit Factor",1.0), (axes[0,1],"ret","Return (%)",0),
                         (axes[1,0],"wr","Win Rate (%)",50), (axes[1,1],"exp","Expectancy (%)",0)]:
        vals = [x[m] for x in sr]
        colors = ["#00e676" if vl >= v else "#ff1744" for vl in vals]
        ax.barh(labels, vals, color=colors); ax.axvline(v, color="white", linestyle="--", alpha=0.5)
        ax.set_title(t); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "smc_s3_fvg_discount.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

if __name__ == "__main__":
    main()
