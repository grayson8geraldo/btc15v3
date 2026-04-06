#!/usr/bin/env python3
"""
Backtest: Liquidity Sweep + Order Block Strategy
BTC/USDT 15m

Концепция (Smart Money / ICT):
1. Swing High/Low формируется (локальный экстремум)
2. Sweep: цена делает новый high/low за swing и закрывается обратно
   → Это ликвидация стопов (liquidity grab)
3. Order Block: последняя противоположная свеча перед импульсом
4. Вход при ретесте OB
5. SL: за экстремумом свипа
6. TP: противоположный swing или фиксированный RR
"""

import glob, os
from dataclasses import dataclass
from datetime import datetime
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
    direction: str = ""
    entry_price: float = 0
    exit_price: float = 0
    tp_price: float = 0
    sl_price: float = 0
    pnl_pct: float = 0
    pnl_abs: float = 0
    result: str = ""
    entry_time: object = None
    exit_time: object = None
    ob_low: float = 0
    ob_high: float = 0
    sweep_level: float = 0


def find_swings(highs, lows, lookback=10):
    """Find swing highs and swing lows using fractal-like logic."""
    n = len(highs)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_high[i] = True
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_low[i] = True

    return swing_high, swing_low


def find_order_block(opens, closes, highs, lows, direction, from_idx, lookback=20):
    """
    Find the order block before an impulse move.

    For bullish move (after bearish sweep): find last bearish candle before the up move.
    For bearish move (after bullish sweep): find last bullish candle before the down move.
    """
    # Search backward from from_idx for opposite colored candle
    for i in range(from_idx - 1, max(from_idx - lookback, 0), -1):
        if direction == "long":
            # Looking for last bearish candle (red) before up move
            if closes[i] < opens[i]:
                return i, lows[i], highs[i]
        else:
            # Looking for last bullish candle (green) before down move
            if closes[i] > opens[i]:
                return i, lows[i], highs[i]
    return None, None, None


def run_backtest(df, label="",
                 swing_lookback=10,    # bars for swing detection
                 sweep_lookback=30,    # look back for swing to sweep
                 ob_lookback=10,       # search for OB within N bars
                 ob_expiry=30,         # OB valid for N bars after formation
                 rr_target=2.0,        # RR ratio for TP
                 use_opposite_swing=True,  # TP at opposite swing or fixed RR
                 min_sweep_pct=0.1,    # min sweep size in %
                 commission=0.04, slippage=0.01,
                 initial_capital=200, risk_pct=2.0,
                 max_hold=96):

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    bars = df.index

    # Find swings
    swing_high, swing_low = find_swings(highs, lows, swing_lookback)

    trades = []
    equity = initial_capital
    in_trade = False
    trade = None
    pos_size = 0
    hold = 0

    # Pending Order Blocks waiting for retest
    pending_obs_long = []   # (ob_low, ob_high, sl_level, tp_level, formed_bar)
    pending_obs_short = []

    warmup = swing_lookback + 50

    for i in range(warmup, len(df)):
        # ── Update pending OBs: remove expired ────────────────────
        pending_obs_long = [ob for ob in pending_obs_long if i - ob[4] < ob_expiry]
        pending_obs_short = [ob for ob in pending_obs_short if i - ob[4] < ob_expiry]

        # ── EXIT LOGIC ──────────────────────────────────────
        if in_trade:
            hold += 1
            t = trade
            exit_price = None
            result = None

            if t.direction == "long":
                if lows[i] <= t.sl_price:
                    exit_price = t.sl_price
                    result = "SL"
                elif highs[i] >= t.tp_price:
                    exit_price = t.tp_price
                    result = "TP"
                elif hold >= max_hold:
                    exit_price = closes[i]
                    result = "timeout"
            else:
                if highs[i] >= t.sl_price:
                    exit_price = t.sl_price
                    result = "SL"
                elif lows[i] <= t.tp_price:
                    exit_price = t.tp_price
                    result = "TP"
                elif hold >= max_hold:
                    exit_price = closes[i]
                    result = "timeout"

            if exit_price is not None:
                cost = (commission + slippage) / 100
                if t.direction == "long":
                    rpnl = (exit_price - t.entry_price) / t.entry_price - 2 * cost
                else:
                    rpnl = (t.entry_price - exit_price) / t.entry_price - 2 * cost
                pnl = pos_size * rpnl
                equity += pnl
                t.exit_price = exit_price
                t.exit_time = bars[i]
                t.result = result
                t.pnl_pct = rpnl * 100
                t.pnl_abs = pnl
                trades.append(t)
                in_trade = False
                if equity <= 0:
                    break
                continue

        # ── DETECT LIQUIDITY SWEEP ──────────────────────────
        # Look for: recent swing high/low was violated then closed back inside
        if i >= sweep_lookback:
            # Find most recent swing high in lookback window (excluding current bar)
            recent_swing_high_idx = -1
            recent_swing_high_price = -np.inf
            for j in range(i - 1, max(i - sweep_lookback, 0), -1):
                if swing_high[j] and highs[j] > recent_swing_high_price:
                    recent_swing_high_idx = j
                    recent_swing_high_price = highs[j]
                    break

            recent_swing_low_idx = -1
            recent_swing_low_price = np.inf
            for j in range(i - 1, max(i - sweep_lookback, 0), -1):
                if swing_low[j] and lows[j] < recent_swing_low_price:
                    recent_swing_low_idx = j
                    recent_swing_low_price = lows[j]
                    break

            # BEARISH SWEEP (bullish setup):
            # Current bar makes new low below swing low, but closes back above
            if recent_swing_low_idx > 0:
                sweep_size = (recent_swing_low_price - lows[i]) / recent_swing_low_price * 100
                if (lows[i] < recent_swing_low_price and
                    closes[i] > recent_swing_low_price and
                    sweep_size >= min_sweep_pct):
                    # Find order block — last bearish candle before the up move
                    # The move that created the swing low came from above
                    # So OB is the bearish candle right before the down-move to the swing low
                    ob_idx, ob_lo, ob_hi = find_order_block(opens, closes, highs, lows,
                                                             "long", recent_swing_low_idx, ob_lookback)
                    if ob_idx is not None and ob_hi > lows[i]:
                        sl = lows[i] - (highs[i] - lows[i]) * 0.1  # below sweep low
                        # TP: opposite swing or fixed RR
                        if use_opposite_swing and recent_swing_high_idx > 0:
                            tp = recent_swing_high_price
                            # Verify TP gives decent RR
                            entry_mid = (ob_lo + ob_hi) / 2
                            if tp > entry_mid:
                                rr = (tp - entry_mid) / (entry_mid - sl)
                                if rr >= 1.5:
                                    pending_obs_long.append((ob_lo, ob_hi, sl, tp, i))
                        else:
                            entry_mid = ob_hi
                            tp = entry_mid + (entry_mid - sl) * rr_target
                            pending_obs_long.append((ob_lo, ob_hi, sl, tp, i))

            # BULLISH SWEEP (bearish setup):
            if recent_swing_high_idx > 0:
                sweep_size = (highs[i] - recent_swing_high_price) / recent_swing_high_price * 100
                if (highs[i] > recent_swing_high_price and
                    closes[i] < recent_swing_high_price and
                    sweep_size >= min_sweep_pct):
                    ob_idx, ob_lo, ob_hi = find_order_block(opens, closes, highs, lows,
                                                             "short", recent_swing_high_idx, ob_lookback)
                    if ob_idx is not None and ob_lo < highs[i]:
                        sl = highs[i] + (highs[i] - lows[i]) * 0.1
                        if use_opposite_swing and recent_swing_low_idx > 0:
                            tp = recent_swing_low_price
                            entry_mid = (ob_lo + ob_hi) / 2
                            if tp < entry_mid:
                                rr = (entry_mid - tp) / (sl - entry_mid)
                                if rr >= 1.5:
                                    pending_obs_short.append((ob_lo, ob_hi, sl, tp, i))
                        else:
                            entry_mid = ob_lo
                            tp = entry_mid - (sl - entry_mid) * rr_target
                            pending_obs_short.append((ob_lo, ob_hi, sl, tp, i))

        # ── ENTRY: check if price retests any pending OB ────
        if not in_trade and equity > 1:
            # Long OBs: entry when price comes back DOWN to OB range
            for ob in pending_obs_long[:]:
                ob_lo, ob_hi, sl, tp, formed = ob
                if lows[i] <= ob_hi and lows[i] >= ob_lo:
                    # Price tagged the OB
                    entry = ob_hi  # enter at top of OB (limit order style)
                    dist_sl = (entry - sl) / entry
                    dist_tp = (tp - entry) / entry
                    if dist_sl > 0 and dist_tp > 0:
                        risk_amt = equity * (risk_pct / 100)
                        pos_size = risk_amt / (dist_sl + 2*(commission+slippage)/100)
                        if pos_size > 0:
                            trade = Trade(direction="long", entry_price=entry,
                                         tp_price=tp, sl_price=sl, entry_time=bars[i],
                                         ob_low=ob_lo, ob_high=ob_hi)
                            in_trade = True
                            hold = 0
                            pending_obs_long.remove(ob)
                            break

            if not in_trade:
                for ob in pending_obs_short[:]:
                    ob_lo, ob_hi, sl, tp, formed = ob
                    if highs[i] >= ob_lo and highs[i] <= ob_hi:
                        entry = ob_lo
                        dist_sl = (sl - entry) / entry
                        dist_tp = (entry - tp) / entry
                        if dist_sl > 0 and dist_tp > 0:
                            risk_amt = equity * (risk_pct / 100)
                            pos_size = risk_amt / (dist_sl + 2*(commission+slippage)/100)
                            if pos_size > 0:
                                trade = Trade(direction="short", entry_price=entry,
                                             tp_price=tp, sl_price=sl, entry_time=bars[i],
                                             ob_low=ob_lo, ob_high=ob_hi)
                                in_trade = True
                                hold = 0
                                pending_obs_short.remove(ob)
                                break

    return trades, equity, label


def calc_stats(trades, cap=200):
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

    dir_i = {}
    for d in ["long","short"]:
        dt = df[df["direction"]==d]
        if len(dt):
            dw = dt[dt["pnl_abs"]>0]
            dir_i[d] = {"n":len(dt),"wr":len(dw)/len(dt)*100,"avg":dt["pnl_pct"].mean(),"total":dt["pnl_abs"].sum()}

    res_i = {}
    for r in df["result"].unique():
        rt = df[df["result"]==r]
        res_i[r] = {"n":len(rt),"avg":rt["pnl_pct"].mean(),"total":rt["pnl_abs"].sum()}

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    yearly = df.groupby("year")["pnl_abs"].sum().to_dict()

    return {"total":n,"wr":wr,"avg_w":aw,"avg_l":al,"pf":pf,"feq":feq,"ret":ret,
            "dd":dd,"exp":exp,"dir":dir_i,"result":res_i,"yearly":yearly,"df":df,"label":""}


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading BTC/USDT 15m data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df)} bars\n")

    scenarios = [
        dict(label="OB Base (sw=10, OB=10, RR=2)",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=30, use_opposite_swing=False, rr_target=2.0),
        dict(label="OB sw=5, fast signals",
             swing_lookback=5, sweep_lookback=20, ob_lookback=10, ob_expiry=20, use_opposite_swing=False, rr_target=2.0),
        dict(label="OB sw=15, slow quality",
             swing_lookback=15, sweep_lookback=50, ob_lookback=15, ob_expiry=40, use_opposite_swing=False, rr_target=2.0),
        dict(label="OB RR=1.5",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=30, use_opposite_swing=False, rr_target=1.5),
        dict(label="OB RR=3",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=30, use_opposite_swing=False, rr_target=3.0),
        dict(label="OB TP at Opposite Swing",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=30, use_opposite_swing=True),
        dict(label="OB min_sweep=0.2%",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=30, use_opposite_swing=False, rr_target=2.0, min_sweep_pct=0.2),
        dict(label="OB min_sweep=0.05%",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=30, use_opposite_swing=False, rr_target=2.0, min_sweep_pct=0.05),
        dict(label="OB expiry=10 (fresh only)",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=10, use_opposite_swing=False, rr_target=2.0),
        dict(label="OB expiry=50 (long hold)",
             swing_lookback=10, sweep_lookback=30, ob_lookback=10, ob_expiry=50, use_opposite_swing=False, rr_target=2.0),
        dict(label="OB sw=20 RR=2.5",
             swing_lookback=20, sweep_lookback=60, ob_lookback=15, ob_expiry=40, use_opposite_swing=False, rr_target=2.5),
        dict(label="OB Conservative (sw=15, RR=3)",
             swing_lookback=15, sweep_lookback=50, ob_lookback=15, ob_expiry=30, use_opposite_swing=False, rr_target=3.0, min_sweep_pct=0.15),
    ]

    print("Running scenarios...\n")
    results = []
    for s in scenarios:
        lbl = s.pop("label")
        trades, eq, _ = run_backtest(df, label=lbl, initial_capital=200, **s)
        stats = calc_stats(trades)
        if stats:
            stats["label"] = lbl
            results.append(stats)
            pfx = "+" if stats["ret"] > 0 else ""
            status = "OK" if stats["pf"] > 1 else "LOSS"
            print(f"  {lbl:<40} {stats['total']:>5} | WR {stats['wr']:.1f}% | PF {stats['pf']:.2f} | "
                  f"E[R] {stats['exp']:+.3f}% | {pfx}{stats['ret']:>7.1f}% | DD {stats['dd']:.1f}%  {status}")
        else:
            print(f"  {lbl:<40} No trades")

    if not results:
        print("No valid scenarios")
        return

    print("\n" + "=" * 130)
    print("  ORDER BLOCK STRATEGY COMPARISON (sorted by Profit Factor)")
    print("=" * 130)
    h = f"{'Scenario':<42} {'N':>5} {'WR%':>6} {'PF':>6} {'AvgW%':>7} {'AvgL%':>7} {'E[R]%':>7} {'Ret%':>9} {'DD%':>7} {'Final$':>10}"
    print(h)
    print("-" * 130)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["ret"] > 0 else ""
        print(f"  {s['label']:<40} {s['total']:>5} {s['wr']:>6.1f} {s['pf']:>6.2f} "
              f"{s['avg_w']:>7.2f} {s['avg_l']:>7.2f} {s['exp']:>7.3f} "
              f"{pfx}{s['ret']:>8.1f} {s['dd']:>6.1f}% ${s['feq']:>9.2f}")
    print("=" * 130)

    best = max(results, key=lambda x: x["pf"])
    s = best
    print(f"\n{'=' * 70}")
    print(f"  BEST: {s['label']}")
    print(f"{'=' * 70}")
    print(f"  Trades: {s['total']}  |  WR: {s['wr']:.1f}%  |  PF: {s['pf']:.2f}  |  E[R]: {s['exp']:.3f}%")
    print(f"  Avg Win: {s['avg_w']:.2f}%  |  Avg Loss: {s['avg_l']:.2f}%")
    print(f"  Return: {s['ret']:.2f}%  |  Max DD: {s['dd']:.2f}%  |  $200 → ${s['feq']:,.2f}")

    for label, d in [("DIR", s["dir"]), ("EXIT", s["result"])]:
        if d:
            print(f"\n  {label}:")
            for k, v in d.items():
                wr_str = f"WR {v['wr']:.1f}% | " if 'wr' in v else ""
                print(f"    {str(k).upper()}: {v['n']} | {wr_str}Avg {v['avg']:.2f}% | ${v['total']:,.2f}")

    if s["yearly"]:
        print(f"\n  YEARLY:")
        for y, p in sorted(s["yearly"].items()):
            print(f"    {y}: ${p:,.2f}")

    # Save charts
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Order Block + Liquidity Sweep Strategy", fontsize=16, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [x["label"][:35] for x in sr]
    for ax, metric, title, vline in [(axes[0,0],"pf","Profit Factor",1.0),
                                      (axes[0,1],"ret","Total Return (%)",0),
                                      (axes[1,0],"wr","Win Rate (%)",50),
                                      (axes[1,1],"exp","Expectancy (%)",0)]:
        vals = [x[metric] for x in sr]
        colors = ["#00e676" if v >= vline else "#ff1744" for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(vline, color="white", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "backtest_ob_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # Best equity
    bdf = best["df"]
    cum = 200 + bdf["pnl_abs"].cumsum()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"BEST OB: {best['label']} | PF {best['pf']:.2f} | Return {best['ret']:.1f}%",
                 fontsize=16, fontweight="bold")
    axes[0,0].plot(range(len(cum)), cum.values, color="#00e676", linewidth=1.2)
    axes[0,0].axhline(200, color="white", linestyle="--", alpha=0.3)
    axes[0,0].set_title("Equity Curve ($200 start)")
    axes[0,0].grid(True, alpha=0.3)

    pk = cum.expanding().max()
    dd = (cum - pk) / pk * 100
    axes[0,1].fill_between(range(len(dd)), dd.values, 0, color="#ff1744", alpha=0.5)
    axes[0,1].set_title("Drawdown (%)")
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].hist(bdf["pnl_pct"].values, bins=60, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1,0].axvline(0, color="white", linestyle="--")
    axes[1,0].set_title("P&L Distribution (%)")
    axes[1,0].grid(True, alpha=0.3)

    if best["yearly"]:
        years = sorted(best["yearly"].keys())
        pnls = [best["yearly"][y] for y in years]
        colors = ["#00e676" if p > 0 else "#ff1744" for p in pnls]
        axes[1,1].bar([str(y) for y in years], pnls, color=colors)
        axes[1,1].set_title("Yearly P&L ($)")
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "backtest_ob_best.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    cols = ["entry_time","exit_time","direction","entry_price","tp_price","sl_price",
            "exit_price","result","pnl_pct","pnl_abs","ob_low","ob_high"]
    best["df"][cols].to_csv(os.path.join(data_dir, "backtest_ob_trades.csv"), index=False)
    print("\nCharts and trades saved.")


if __name__ == "__main__":
    main()
