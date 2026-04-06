#!/usr/bin/env python3
"""
SMC Strategy 1: Equal Highs/Lows Liquidity Sweeps

Логика:
1. Найти кластеры равных максимумов/минимумов (Equal Highs/Lows)
   — это очевидные пулы ликвидности где скапливаются стопы
2. Дождаться ложного пробоя (sweep) этого уровня
3. Цена должна вернуться обратно (закрытие внутри структуры)
4. Вход против движения, SL за свипом, TP до противоположной ликвидности
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
        swing_lb=5,            # bars to confirm swing point
        eq_lookback=50,        # lookback to find equal levels
        eq_tolerance_pct=0.1,  # max % difference to be "equal"
        min_eq_count=2,        # minimum highs/lows needed to form pool
        sweep_min_pct=0.05,    # min sweep size %
        sweep_max_bars=3,      # close back within N bars
        rr_target=2.0,
        cap=200, risk=2.0, comm=0.04, slip=0.01, max_hold=96):

    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values; c = df["Close"].values
    bars = df.index
    n = len(df)

    # Find swing points (fractal-like)
    swing_h = np.zeros(n, dtype=bool)
    swing_l = np.zeros(n, dtype=bool)
    for i in range(swing_lb, n - swing_lb):
        if h[i] == max(h[i-swing_lb:i+swing_lb+1]):
            swing_h[i] = True
        if l[i] == min(l[i-swing_lb:i+swing_lb+1]):
            swing_l[i] = True

    # For each bar, find nearest "equal high" and "equal low" liquidity pool
    # Equal Highs: at least 2 swing highs within tolerance %
    def find_equal_pool(idx, is_high):
        """Look back for equal levels. Returns (pool_price, count) or (None, 0)."""
        swings = swing_h if is_high else swing_l
        prices = h if is_high else l
        # Get all swings in lookback window
        swing_prices = []
        for j in range(max(idx - eq_lookback, 0), idx):
            if swings[j]:
                swing_prices.append(prices[j])
        if len(swing_prices) < min_eq_count:
            return None, 0
        # Cluster them: find any group within tolerance%
        # Sort and look for densest cluster near the most recent swing
        most_recent = swing_prices[-1]
        cluster = [most_recent]
        for sp in swing_prices[:-1]:
            if abs(sp - most_recent) / most_recent * 100 <= eq_tolerance_pct:
                cluster.append(sp)
        if len(cluster) >= min_eq_count:
            return max(cluster) if is_high else min(cluster), len(cluster)
        return None, 0

    trades = []; equity = cap
    in_trade = False; t = None; pos = 0; hold = 0
    warmup = max(eq_lookback, 50) + 10

    for i in range(warmup, n):
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

        # Find Equal Highs (short setup) and Equal Lows (long setup)
        eh_pool, eh_count = find_equal_pool(i, True)
        el_pool, el_count = find_equal_pool(i, False)

        # LONG: check if low swept Equal Lows and closed back
        if el_pool is not None and l[i] < el_pool:
            sweep_size = (el_pool - l[i]) / el_pool * 100
            if sweep_size >= sweep_min_pct and c[i] > el_pool:
                # Sweep + close back inside — valid setup
                entry = c[i]  # market entry on sweep candle close
                sl = l[i] * 0.999  # below sweep wick
                # TP: opposite Equal Highs or fixed RR
                if eh_pool is not None and eh_pool > entry:
                    tp = eh_pool
                    # Validate RR
                    if (tp - entry) / (entry - sl) >= 1.5:
                        pass
                    else:
                        tp = entry + (entry - sl) * rr_target
                else:
                    tp = entry + (entry - sl) * rr_target

                if tp > entry and (entry - sl) > 0:
                    ds = (entry - sl) / entry
                    ra = equity * (risk/100)
                    pos = ra / (ds + 2*(comm+slip)/100)
                    if pos > 0:
                        t = Trade(direction="long", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                        in_trade = True; hold = 0
                        continue

        # SHORT: check if high swept Equal Highs and closed back
        if eh_pool is not None and h[i] > eh_pool:
            sweep_size = (h[i] - eh_pool) / eh_pool * 100
            if sweep_size >= sweep_min_pct and c[i] < eh_pool:
                entry = c[i]
                sl = h[i] * 1.001
                if el_pool is not None and el_pool < entry:
                    tp = el_pool
                    if (entry - tp) / (sl - entry) >= 1.5:
                        pass
                    else:
                        tp = entry - (sl - entry) * rr_target
                else:
                    tp = entry - (sl - entry) * rr_target

                if tp < entry and (sl - entry) > 0:
                    ds = (sl - entry) / entry
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
        dict(label="EH/EL base (sw=5, lb=50, tol=0.1%)",   swing_lb=5, eq_lookback=50, eq_tolerance_pct=0.1, rr_target=2.0),
        dict(label="EH/EL strict tolerance (0.05%)",       swing_lb=5, eq_lookback=50, eq_tolerance_pct=0.05),
        dict(label="EH/EL loose tolerance (0.2%)",         swing_lb=5, eq_lookback=50, eq_tolerance_pct=0.2),
        dict(label="EH/EL min 3 levels",                   swing_lb=5, eq_lookback=50, min_eq_count=3),
        dict(label="EH/EL bigger lookback (100)",          swing_lb=5, eq_lookback=100),
        dict(label="EH/EL swing 10",                       swing_lb=10, eq_lookback=80),
        dict(label="EH/EL swing 3 fast",                   swing_lb=3, eq_lookback=30),
        dict(label="EH/EL RR=3",                           rr_target=3.0),
        dict(label="EH/EL RR=1.5",                         rr_target=1.5),
        dict(label="EH/EL min sweep 0.15%",                sweep_min_pct=0.15),
        dict(label="EH/EL min sweep 0.02%",                sweep_min_pct=0.02),
        dict(label="EH/EL strict (sw=10, tol=0.05, min=3)", swing_lb=10, eq_tolerance_pct=0.05, min_eq_count=3, eq_lookback=80),
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
    print("  STRATEGY 1: EQUAL HIGHS/LOWS LIQUIDITY SWEEPS — Comparison")
    print("=" * 130)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["ret"] > 0 else ""
        print(f"  {s['label']:<42} {s['n']:>5} WR {s['wr']:>5.1f}% PF {s['pf']:>5.2f} "
              f"E[R] {s['exp']:>+.3f}% {pfx}{s['ret']:>9.1f}% DD {s['dd']:>6.1f}% ${s['feq']:>10,.0f}")

    best = max(results, key=lambda x: x["pf"])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['label']}")
    print(f"  {best['n']} trades | WR {best['wr']:.1f}% | PF {best['pf']:.2f}")
    print(f"  $200 → ${best['feq']:,.2f} | DD {best['dd']:.1f}%")
    if best["yearly"]:
        print("  YEARLY:")
        for y, p in sorted(best["yearly"].items()):
            print(f"    {y}: ${p:,.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Strategy 1: Equal Highs/Lows Liquidity Sweeps", fontsize=16, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [x["label"][:38] for x in sr]
    for ax, m, t, v in [(axes[0,0],"pf","Profit Factor",1.0), (axes[0,1],"ret","Return (%)",0),
                         (axes[1,0],"wr","Win Rate (%)",50), (axes[1,1],"exp","Expectancy (%)",0)]:
        vals = [x[m] for x in sr]
        colors = ["#00e676" if vl >= v else "#ff1744" for vl in vals]
        ax.barh(labels, vals, color=colors); ax.axvline(v, color="white", linestyle="--", alpha=0.5)
        ax.set_title(t); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "smc_s1_equal_hl.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nChart saved.")

if __name__ == "__main__":
    main()
