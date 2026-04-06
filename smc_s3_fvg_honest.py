#!/usr/bin/env python3
"""
SMC Strategy 3 HONEST: FVG + Discount/Premium — NO LOOK-AHEAD BIAS

Исправления:
1. FVG детектится в конце бара i (по закрытию)
2. Лимитный ордер ставится на midpoint
3. Исполняется ТОЛЬКО если будущий бар (i+1, i+2, ...) коснётся этого уровня
4. Ордер истекает после N баров
5. Минимум RR 2:1 (как просил пользователь)
6. Все индикаторы shift(1) где применимо
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
    fvg_bar:int=0  # bar when FVG was detected


def run(df, label="",
        range_lookback=16,
        zone_threshold=0.5,
        min_fvg_pct=0.05,
        min_rr=2.0,           # MINIMUM Risk:Reward requirement
        target_mode="50pct",
        fvg_expiry=30,        # order valid for N bars after FVG formed
        cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=96):

    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)

    trades = []; equity = cap
    in_trade = False; t = None; pos = 0; hold = 0

    # Pending limit orders (from FVG detection)
    # Each: (direction, entry_price, tp, sl, fvg_bar)
    pending_long = []
    pending_short = []

    warmup = max(range_lookback, 50) + 10

    for i in range(warmup, n):
        # ── Expire old pending orders ────────────────────────
        pending_long = [o_ for o_ in pending_long if i - o_[4] <= fvg_expiry]
        pending_short = [o_ for o_ in pending_short if i - o_[4] <= fvg_expiry]

        # ── EXIT LOGIC ──────────────────────────────────────
        if in_trade:
            hold += 1
            ep = None; res = None
            if t.direction == "long":
                # Conservative: SL first if both hit
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
                rpnl = ((ep-t.entry)/t.entry-2*cost) if t.direction == "long" else ((t.entry-ep)/t.entry-2*cost)
                pnl = pos * rpnl; equity += pnl
                t.exit = ep; t.result = res; t.pnl_pct = rpnl*100; t.pnl_abs = pnl
                trades.append(t); in_trade = False
                if equity <= 0: break
                continue

        # ── CHECK PENDING ORDER FILLS ────────────────────────
        if not in_trade and equity > 1:
            # Long limit fills: price comes DOWN to entry
            for ord_ in pending_long[:]:
                direction, e_price, tp, sl, fvg_bar = ord_
                if l[i] <= e_price:
                    # Fill
                    ds = (e_price - sl) / e_price
                    if ds > 0:
                        ra = equity * (risk/100)
                        pos = ra / (ds + 2*(comm+slip)/100)
                        if pos > 0:
                            t = Trade(direction="long", entry=e_price, tp=tp, sl=sl,
                                     entry_t=bars[i], fvg_bar=fvg_bar)
                            in_trade = True
                            hold = 0
                            pending_long.remove(ord_)
                            break

            if not in_trade:
                for ord_ in pending_short[:]:
                    direction, e_price, tp, sl, fvg_bar = ord_
                    if h[i] >= e_price:
                        ds = (sl - e_price) / e_price
                        if ds > 0:
                            ra = equity * (risk/100)
                            pos = ra / (ds + 2*(comm+slip)/100)
                            if pos > 0:
                                t = Trade(direction="short", entry=e_price, tp=tp, sl=sl,
                                         entry_t=bars[i], fvg_bar=fvg_bar)
                                in_trade = True
                                hold = 0
                                pending_short.remove(ord_)
                                break

        # ── DETECT NEW FVG & PLACE PENDING ORDER ────────────
        # Use PREVIOUS bar data to avoid look-ahead
        # At bar i, check if FVG formed AT bar i-1 (so we use confirmed data)
        if i < 3: continue

        # Dealing range from bars [i-range_lookback-1 : i-1] (no current bar)
        range_high = max(h[i-range_lookback-1:i-1])
        range_low = min(l[i-range_lookback-1:i-1])
        range_size = range_high - range_low
        if range_size <= 0: continue
        eq50 = (range_high + range_low) / 2

        # Use previous bar close for zone check
        ref_close = c[i-1]
        in_discount = ref_close < range_low + range_size * zone_threshold
        in_premium = ref_close > range_high - range_size * zone_threshold

        # FVG formed at bar i-1: gap between i-3 and i-1
        # Bullish: l[i-1] > h[i-3]
        bull_fvg = l[i-1] > h[i-3]
        bear_fvg = h[i-1] < l[i-3]

        # LONG: bullish FVG in discount
        if bull_fvg and in_discount:
            fvg_size = (l[i-1] - h[i-3]) / h[i-3] * 100
            if fvg_size >= min_fvg_pct:
                entry_p = (l[i-1] + h[i-3]) / 2  # FVG midpoint — will be filled on retest
                sl = min(l[i-3], l[i-2]) * 0.999

                if target_mode == "50pct":
                    tp = eq50
                elif target_mode == "opposite":
                    tp = range_high
                else:
                    tp = entry_p + (entry_p - sl) * 2

                # Validate RR
                if tp > entry_p and (entry_p - sl) > 0:
                    rr = (tp - entry_p) / (entry_p - sl)
                    ds = (entry_p - sl) / entry_p
                    if rr >= min_rr and 0.001 < ds < 0.05:
                        pending_long.append(("long", entry_p, tp, sl, i))

        if bear_fvg and in_premium:
            fvg_size = (l[i-3] - h[i-1]) / l[i-3] * 100
            if fvg_size >= min_fvg_pct:
                entry_p = (h[i-1] + l[i-3]) / 2
                sl = max(h[i-3], h[i-2]) * 1.001

                if target_mode == "50pct":
                    tp = eq50
                elif target_mode == "opposite":
                    tp = range_low
                else:
                    tp = entry_p - (sl - entry_p) * 2

                if tp < entry_p and (sl - entry_p) > 0:
                    rr = (entry_p - tp) / (sl - entry_p)
                    ds = (sl - entry_p) / entry_p
                    if rr >= min_rr and 0.001 < ds < 0.05:
                        pending_short.append(("short", entry_p, tp, sl, i))

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

    print("=" * 80)
    print("  HONEST BACKTEST (no look-ahead, min RR 2:1)")
    print("=" * 80)

    scenarios = [
        dict(label="4h range, RR>=2, TP=50%",           range_lookback=16,  target_mode="50pct", min_rr=2.0),
        dict(label="4h range, RR>=2, TP=opposite",      range_lookback=16,  target_mode="opposite", min_rr=2.0),
        dict(label="4h range, RR>=2.5, TP=50%",         range_lookback=16,  target_mode="50pct", min_rr=2.5),
        dict(label="4h range, RR>=3, TP=50%",           range_lookback=16,  target_mode="50pct", min_rr=3.0),
        dict(label="24h range, RR>=2, TP=50%",          range_lookback=96,  target_mode="50pct", min_rr=2.0),
        dict(label="24h range, RR>=2, TP=opposite",     range_lookback=96,  target_mode="opposite", min_rr=2.0),
        dict(label="24h range, RR>=3, TP=50%",          range_lookback=96,  target_mode="50pct", min_rr=3.0),
        dict(label="24h, min FVG 0.15%, RR>=2",         range_lookback=96,  target_mode="50pct", min_rr=2.0, min_fvg_pct=0.15),
        dict(label="24h, min FVG 0.15%, RR>=3",         range_lookback=96,  target_mode="50pct", min_rr=3.0, min_fvg_pct=0.15),
        dict(label="1W range, RR>=2, TP=50%",           range_lookback=672, target_mode="50pct", min_rr=2.0),
        dict(label="1W range, RR>=2, TP=opposite",      range_lookback=672, target_mode="opposite", min_rr=2.0),
        dict(label="3D range, RR>=2, TP=50%",           range_lookback=288, target_mode="50pct", min_rr=2.0),
        # Test fvg_expiry variations
        dict(label="24h, RR>=2, expire 10 bars",        range_lookback=96,  target_mode="50pct", min_rr=2.0, fvg_expiry=10),
        dict(label="24h, RR>=2, expire 60 bars",        range_lookback=96,  target_mode="50pct", min_rr=2.0, fvg_expiry=60),
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
    print("  HONEST RESULTS (sorted by PF)")
    print("=" * 130)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["ret"] > 0 else ""
        print(f"  {s['label']:<42} {s['n']:>5} WR {s['wr']:>5.1f}% PF {s['pf']:>5.2f} "
              f"E[R] {s['exp']:>+.3f}% {pfx}{s['ret']:>9.1f}% DD {s['dd']:>6.1f}% ${s['feq']:>10,.0f}")

    best = max(results, key=lambda x: x["pf"])
    print(f"\n{'='*70}")
    print(f"  BEST (honest): {best['label']}")
    print(f"{'='*70}")
    print(f"  Trades: {best['n']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | E[R]: {best['exp']:.3f}%")
    print(f"  Avg Win: {best['aw']:.2f}% | Avg Loss: {best['al']:.2f}%")
    print(f"  $200 → ${best['feq']:,.2f} ({best['ret']:.1f}%) | Max DD: {best['dd']:.1f}%")
    if best["yearly"]:
        print("  YEARLY:")
        for y, p in sorted(best["yearly"].items()):
            print(f"    {y}: ${p:,.2f}")

    # OOS test on best
    print(f"\n{'='*70}")
    print(f"  OOS TEST (best honest variant)")
    print(f"{'='*70}")
    best_params = {k:v for k,v in [("range_lookback", 96), ("target_mode", "50pct"),
                                     ("min_rr", 2.0), ("min_fvg_pct", 0.15)]}
    train = df[df.index < "2024-07-01"]
    test = df[df.index >= "2024-07-01"]
    for label, slice_df in [("IN-SAMPLE", train), ("OUT-OF-SAMPLE", test), ("FULL", df)]:
        tr, feq, _ = run(slice_df, **best_params)
        st = stats(tr)
        if st and st["n"] > 0:
            print(f"  {label}: {st['n']} trades | WR {st['wr']:.1f}% | PF {st['pf']:.2f} | "
                  f"$200 → ${feq:,.2f} | DD {st['dd']:.1f}%")
        else:
            print(f"  {label}: no trades")

    # Save chart
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("HONEST Backtest: FVG + Discount/Premium (no look-ahead, min RR 2:1)",
                 fontsize=14, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [x["label"][:38] for x in sr]
    for ax, m, t, v in [(axes[0,0],"pf","Profit Factor",1.0), (axes[0,1],"ret","Return (%)",0),
                         (axes[1,0],"wr","Win Rate (%)",50), (axes[1,1],"exp","Expectancy (%)",0)]:
        vals = [x[m] for x in sr]
        colors = ["#00e676" if vl >= v else "#ff1744" for vl in vals]
        ax.barh(labels, vals, color=colors); ax.axvline(v, color="white", linestyle="--", alpha=0.5)
        ax.set_title(t); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "smc_s3_fvg_honest.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

if __name__ == "__main__":
    main()
