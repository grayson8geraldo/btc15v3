#!/usr/bin/env python3
"""
Optimized Backtest: Mean Reversion Bounce Strategy on BTC 15m data.

Key improvements over baseline:
1. Parameter grid search across tier multipliers, trend filters, lookbacks
2. Volume confirmation filter
3. ATR-based dynamic SL (trailing)
4. Cooldown between trades
5. Multi-scenario comparison report
"""

import glob
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "BTCUSDT-15m-*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["open_time"], unit="ms")
    data.set_index("datetime", inplace=True)
    data.sort_index(inplace=True)
    data = data[~data.index.duplicated(keep="first")]
    data.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    return data[["Open","High","Low","Close","Volume"]]


# ═══════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def sma(s, n):
    return s.rolling(n).mean()

def calc_ma(s, n, t="EMA"):
    return ema(s, n) if t == "EMA" else sma(s, n)

def resample_ohlc(df, tf):
    return df.resample(tf).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

def calc_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    direction: str = ""
    tier: int = 0
    entry_price: float = 0
    exit_price: float = 0
    tp_price: float = 0
    sl_price: float = 0
    pnl_pct: float = 0
    pnl_abs: float = 0
    result: str = ""
    entry_time: object = None
    exit_time: object = None


def run_scenario(df, trend_tf="1h", trend_len=200, trend_type="EMA",
                 dev_ma_len=20, dev_lookback=20, dev_ma_type="EMA",
                 mult1=1.5, mult2=2.5, mult3=3.5,
                 liq_tf1="1D", liq_tf2="4h", liq_lookback=20,
                 trade_tiers=(1,2,3),
                 use_volume_filter=False, vol_mult=1.2,
                 use_atr_sl=False, atr_sl_mult=2.0,
                 cooldown_bars=0,
                 commission=0.04, slippage=0.01,
                 initial_capital=10000, risk_pct=1.0,
                 max_hold=96, label=""):

    # --- Trend ---
    htf_trend = resample_ohlc(df, trend_tf)
    trend_ma = calc_ma(htf_trend["Close"], trend_len, trend_type).shift(1)
    trend_ma = trend_ma.reindex(df.index, method="ffill")
    is_bull = (df["Close"] > trend_ma).values
    is_bear = (df["Close"] < trend_ma).values

    # --- Deviation Bands ---
    dev_ma = calc_ma(df["Close"], dev_ma_len, dev_ma_type)
    stdev = df["Close"].rolling(dev_lookback).std()

    bands = {}
    for tier, m in [(1, mult1), (2, mult2), (3, mult3)]:
        bands[f"u{tier}"] = (dev_ma + stdev * m).values
        bands[f"l{tier}"] = (dev_ma - stdev * m).values

    # --- Liquidity ---
    liq_data = {}
    for tag, tf, lb in [("a", liq_tf1, liq_lookback), ("b", liq_tf2, liq_lookback)]:
        htf = resample_ohlc(df, tf)
        liq_data[f"{tag}_h"] = htf["High"].rolling(lb).max().shift(1).reindex(df.index, method="ffill").values
        liq_data[f"{tag}_l"] = htf["Low"].rolling(lb).min().shift(1).reindex(df.index, method="ffill").values

    # --- Volume filter ---
    vol_ma = df["Volume"].rolling(20).mean().values if use_volume_filter else None

    # --- ATR ---
    atr = calc_atr(df, 14).values if use_atr_sl else None

    # --- Arrays ---
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    # --- Simulation ---
    trades = []
    equity = initial_capital
    in_trade = False
    hold = 0
    cooldown = 0
    trade = None

    warmup = max(trend_len, dev_ma_len, dev_lookback) + 50

    for i in range(warmup, len(df)):
        if cooldown > 0:
            cooldown -= 1

        # --- Exit ---
        if in_trade:
            hold += 1
            t = trade
            hit_tp = hit_sl = False

            if t.direction == "long":
                hit_sl = lows[i] <= t.sl_price
                hit_tp = highs[i] >= t.tp_price
            else:
                hit_sl = highs[i] >= t.sl_price
                hit_tp = lows[i] <= t.tp_price

            ep = None
            res = None
            if hit_tp and not hit_sl:
                ep, res = t.tp_price, "TP"
            elif hit_sl:
                ep, res = t.sl_price, "SL"
            elif hold >= max_hold:
                ep, res = closes[i], "timeout"

            if ep is not None:
                cost = (commission + slippage) / 100
                if t.direction == "long":
                    rpnl = (ep - t.entry_price) / t.entry_price
                else:
                    rpnl = (t.entry_price - ep) / t.entry_price
                npnl = rpnl - 2 * cost
                pabs = t.pnl_abs  # position_size stored here temporarily
                t.exit_price = ep
                t.pnl_pct = npnl * 100
                t.pnl_abs = pabs * npnl
                t.result = res
                t.exit_time = df.index[i]
                equity += t.pnl_abs
                trades.append(t)
                in_trade = False
                cooldown = cooldown_bars
                continue

        # --- Entry ---
        if not in_trade and cooldown <= 0:
            for tier in sorted(trade_tiers, reverse=True):
                if tier not in trade_tiers:
                    continue
                ub = bands[f"u{tier}"]
                lb = bands[f"l{tier}"]

                long_sig = (lows[i] <= lb[i] and closes[i] > lb[i] and
                           i > 0 and lows[i-1] > lb[i-1])
                short_sig = (highs[i] >= ub[i] and closes[i] < ub[i] and
                            i > 0 and highs[i-1] < ub[i-1])

                # Volume filter
                if use_volume_filter and vol_ma is not None:
                    if not np.isnan(vol_ma[i]) and df["Volume"].values[i] < vol_ma[i] * vol_mult:
                        long_sig = short_sig = False

                # Best liquidity targets
                def best_above():
                    cands = []
                    for k in ["a_h", "b_h"]:
                        v = liq_data[k][i]
                        if not np.isnan(v) and v > closes[i]:
                            cands.append(v)
                    return min(cands) if cands else np.nan

                def best_below():
                    cands = []
                    for k in ["a_l", "b_l"]:
                        v = liq_data[k][i]
                        if not np.isnan(v) and v < closes[i]:
                            cands.append(v)
                    return max(cands) if cands else np.nan

                entered = False

                if long_sig and is_bull[i]:
                    ba = best_above()
                    if not np.isnan(ba):
                        entry = closes[i]
                        tp = ba
                        dist = tp - entry
                        if use_atr_sl and atr is not None and not np.isnan(atr[i]):
                            sl = entry - atr[i] * atr_sl_mult
                        else:
                            sl = entry - dist
                        if dist / entry >= 0.0005:
                            risk_amt = equity * (risk_pct / 100)
                            pos = risk_amt / ((entry - sl) / entry + 2 * (commission + slippage) / 100)
                            trade = TradeResult(direction="long", tier=tier,
                                entry_price=entry, tp_price=tp, sl_price=sl,
                                pnl_abs=pos, entry_time=df.index[i])
                            in_trade = True
                            hold = 0
                            entered = True

                if not entered and short_sig and is_bear[i]:
                    bb = best_below()
                    if not np.isnan(bb):
                        entry = closes[i]
                        tp = bb
                        dist = entry - tp
                        if use_atr_sl and atr is not None and not np.isnan(atr[i]):
                            sl = entry + atr[i] * atr_sl_mult
                        else:
                            sl = entry + dist
                        if dist / entry >= 0.0005:
                            risk_amt = equity * (risk_pct / 100)
                            pos = risk_amt / ((sl - entry) / entry + 2 * (commission + slippage) / 100)
                            trade = TradeResult(direction="short", tier=tier,
                                entry_price=entry, tp_price=tp, sl_price=sl,
                                pnl_abs=pos, entry_time=df.index[i])
                            in_trade = True
                            hold = 0
                            entered = True

                if entered:
                    break

    return trades, equity, label


# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

def calc_stats(trades, initial_capital=10000):
    if not trades:
        return None
    df = pd.DataFrame([t.__dict__ for t in trades])
    total = len(df)
    wins = df[df["pnl_abs"] > 0]
    losses = df[df["pnl_abs"] <= 0]
    wr = len(wins) / total * 100
    avg_w = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_l = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    gp = wins["pnl_abs"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_abs"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")
    final_eq = initial_capital + df["pnl_abs"].sum()
    total_ret = (final_eq - initial_capital) / initial_capital * 100

    # Max drawdown from cumulative equity
    cum = initial_capital + df["pnl_abs"].cumsum()
    peak = cum.expanding().max()
    dd = ((cum - peak) / peak * 100).min()

    # By tier
    tier_info = {}
    for tier in [1, 2, 3]:
        tt = df[df["tier"] == tier]
        if len(tt) > 0:
            tw = tt[tt["pnl_abs"] > 0]
            tier_info[tier] = {"n": len(tt), "wr": len(tw)/len(tt)*100,
                              "avg": tt["pnl_pct"].mean(), "total": tt["pnl_abs"].sum()}

    # By direction
    dir_info = {}
    for d in ["long", "short"]:
        dt = df[df["direction"] == d]
        if len(dt) > 0:
            dw = dt[dt["pnl_abs"] > 0]
            dir_info[d] = {"n": len(dt), "wr": len(dw)/len(dt)*100,
                          "avg": dt["pnl_pct"].mean(), "total": dt["pnl_abs"].sum()}

    # By result type
    result_info = {}
    for r in df["result"].unique():
        rt = df[df["result"] == r]
        result_info[r] = {"n": len(rt), "avg": rt["pnl_pct"].mean(), "total": rt["pnl_abs"].sum()}

    # Yearly
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    yearly = df.groupby("year")["pnl_abs"].sum().to_dict()

    return {
        "total": total, "wr": wr, "avg_w": avg_w, "avg_l": avg_l,
        "pf": pf, "final_eq": final_eq, "total_ret": total_ret,
        "max_dd": dd, "tier": tier_info, "dir": dir_info,
        "result": result_info, "yearly": yearly, "trades_df": df,
    }


# ═══════════════════════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════════════════════

def run_all_scenarios(df):
    scenarios = [
        # Baseline (original)
        dict(label="Baseline (all tiers, 1:1 RR)",
             trade_tiers=(1,2,3)),

        # Only Tier 2+3 (stronger signals)
        dict(label="Tier 2+3 only",
             trade_tiers=(2,3)),

        # Only Tier 3 (strongest)
        dict(label="Tier 3 only",
             trade_tiers=(3,)),

        # Wider bands: 2.0/3.0/4.0
        dict(label="Wider bands (2.0/3.0/4.0)",
             mult1=2.0, mult2=3.0, mult3=4.0, trade_tiers=(1,2,3)),

        # Volume filter
        dict(label="Volume filter (>1.2x avg)",
             trade_tiers=(1,2,3), use_volume_filter=True, vol_mult=1.2),

        # ATR stop loss (2x ATR)
        dict(label="ATR SL (2x ATR) + all tiers",
             trade_tiers=(1,2,3), use_atr_sl=True, atr_sl_mult=2.0),

        # ATR SL + Tier 2+3
        dict(label="ATR SL (2x) + Tier 2+3",
             trade_tiers=(2,3), use_atr_sl=True, atr_sl_mult=2.0),

        # ATR SL + Volume + Tier 2+3
        dict(label="ATR SL + Volume + Tier 2+3",
             trade_tiers=(2,3), use_atr_sl=True, atr_sl_mult=2.0,
             use_volume_filter=True, vol_mult=1.2),

        # Cooldown 4 bars (1 hour)
        dict(label="Cooldown 4 bars + Tier 2+3",
             trade_tiers=(2,3), cooldown_bars=4),

        # Wider bands + ATR SL + Volume
        dict(label="Wide bands + ATR SL + Volume",
             mult1=2.0, mult2=3.0, mult3=4.0, trade_tiers=(1,2,3),
             use_atr_sl=True, atr_sl_mult=2.0, use_volume_filter=True, vol_mult=1.2),

        # HMA trend
        dict(label="HMA trend + Tier 2+3",
             trend_type="HMA", trend_len=200, trade_tiers=(2,3)),

        # Shorter trend (100 EMA)
        dict(label="EMA 100 trend + Tier 2+3",
             trend_len=100, trade_tiers=(2,3)),

        # Longer dev lookback
        dict(label="Dev lookback 50 + Tier 2+3",
             dev_lookback=50, dev_ma_len=50, trade_tiers=(2,3)),

        # ATR SL 3x + Tier 2+3 (wider SL)
        dict(label="ATR SL (3x) + Tier 2+3",
             trade_tiers=(2,3), use_atr_sl=True, atr_sl_mult=3.0),

        # Tight ATR + Volume + Wide bands + Tier 2+3
        dict(label="Best combo attempt",
             mult1=2.0, mult2=3.0, mult3=4.0, trade_tiers=(2,3),
             use_atr_sl=True, atr_sl_mult=2.5,
             use_volume_filter=True, vol_mult=1.5, cooldown_bars=4),
    ]

    results = []
    for s in scenarios:
        lbl = s.pop("label")
        print(f"  Running: {lbl}...", end=" ", flush=True)
        trades, final_eq, _ = run_scenario(df, label=lbl, **s)
        stats = calc_stats(trades)
        if stats:
            stats["label"] = lbl
            results.append(stats)
            print(f"{stats['total']} trades | WR {stats['wr']:.1f}% | "
                  f"PF {stats['pf']:.2f} | Ret {stats['total_ret']:.1f}% | "
                  f"DD {stats['max_dd']:.1f}%")
        else:
            print("No trades")
    return results


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def print_comparison(results):
    print("\n" + "=" * 120)
    print("  SCENARIO COMPARISON")
    print("=" * 120)
    header = f"{'Scenario':<40} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Avg W%':>8} {'Avg L%':>8} {'Return%':>9} {'Max DD%':>9} {'Final$':>10}"
    print(header)
    print("-" * 120)

    # Sort by profit factor
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        print(f"{s['label']:<40} {s['total']:>7} {s['wr']:>7.1f} {s['pf']:>7.2f} "
              f"{s['avg_w']:>8.2f} {s['avg_l']:>8.2f} {s['total_ret']:>9.2f} "
              f"{s['max_dd']:>9.2f} {s['final_eq']:>10,.0f}")

    print("=" * 120)


def print_best_detail(results):
    best = max(results, key=lambda x: x["pf"])
    s = best

    print(f"\n{'=' * 70}")
    print(f"  BEST SCENARIO: {s['label']}")
    print(f"{'=' * 70}")
    print(f"\n  Total Trades:    {s['total']}")
    print(f"  Win Rate:        {s['wr']:.1f}%")
    print(f"  Profit Factor:   {s['pf']:.2f}")
    print(f"  Avg Win:         {s['avg_w']:.2f}%")
    print(f"  Avg Loss:        {s['avg_l']:.2f}%")
    print(f"  Total Return:    {s['total_ret']:.2f}%")
    print(f"  Max Drawdown:    {s['max_dd']:.2f}%")
    print(f"  Final Equity:    ${s['final_eq']:,.2f}")

    print(f"\n  BY TIER:")
    for t, info in s["tier"].items():
        print(f"    Tier {t}: {info['n']} trades | WR {info['wr']:.1f}% | Avg {info['avg']:.2f}% | ${info['total']:,.2f}")

    print(f"\n  BY DIRECTION:")
    for d, info in s["dir"].items():
        print(f"    {d.upper()}: {info['n']} trades | WR {info['wr']:.1f}% | Avg {info['avg']:.2f}% | ${info['total']:,.2f}")

    print(f"\n  BY EXIT TYPE:")
    for r, info in s["result"].items():
        print(f"    {r.upper()}: {info['n']} trades | Avg {info['avg']:.2f}% | ${info['total']:,.2f}")

    print(f"\n  YEARLY P&L:")
    for y, pnl in sorted(s["yearly"].items()):
        print(f"    {y}: ${pnl:,.2f}")

    return best


def plot_comparison(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Mean Reversion Bounce — Scenario Comparison", fontsize=16, fontweight="bold")

    sorted_r = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [s["label"][:30] for s in sorted_r]

    # 1) Profit factor
    ax = axes[0, 0]
    pfs = [s["pf"] for s in sorted_r]
    colors = ["#00e676" if p >= 1 else "#ff1744" for p in pfs]
    ax.barh(labels, pfs, color=colors)
    ax.axvline(1.0, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Profit Factor")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # 2) Total Return
    ax = axes[0, 1]
    rets = [s["total_ret"] for s in sorted_r]
    colors = ["#00e676" if r > 0 else "#ff1744" for r in rets]
    ax.barh(labels, rets, color=colors)
    ax.axvline(0, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Total Return (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # 3) Win Rate
    ax = axes[1, 0]
    wrs = [s["wr"] for s in sorted_r]
    ax.barh(labels, wrs, color="#42a5f5")
    ax.axvline(50, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Win Rate (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # 4) Max Drawdown
    ax = axes[1, 1]
    dds = [abs(s["max_dd"]) for s in sorted_r]
    ax.barh(labels, dds, color="#ff9800")
    ax.set_title("Max Drawdown (% abs)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "backtest_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nComparison chart saved to {path}")


def plot_best_equity(best, output_dir):
    df = best["trades_df"]
    cum = 10000 + df["pnl_abs"].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Best Scenario: {best['label']}", fontsize=16, fontweight="bold")

    # Equity curve
    ax = axes[0, 0]
    ax.plot(range(len(cum)), cum.values, color="#00e676", linewidth=1)
    ax.fill_between(range(len(cum)), cum.values, cum.min(), alpha=0.15, color="#00e676")
    ax.set_title("Equity Curve (by trade)")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)

    # Drawdown by trade
    ax = axes[0, 1]
    peak = cum.expanding().max()
    dd = (cum - peak) / peak * 100
    ax.fill_between(range(len(dd)), dd.values, 0, color="#ff1744", alpha=0.5)
    ax.set_title("Drawdown by Trade (%)")
    ax.grid(True, alpha=0.3)

    # PnL distribution
    ax = axes[1, 0]
    ax.hist(df["pnl_pct"].values, bins=50, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="white", linestyle="--")
    ax.set_title("P&L Distribution (%)")
    ax.grid(True, alpha=0.3)

    # Win rate by tier
    ax = axes[1, 1]
    tiers = list(best["tier"].keys())
    wr = [best["tier"][t]["wr"] for t in tiers]
    ns = [best["tier"][t]["n"] for t in tiers]
    bars = ax.bar([f"Tier {t}\n({n} trades)" for t, n in zip(tiers, ns)], wr,
                  color=["#42a5f5", "#ab47bc", "#ffa726"][:len(tiers)])
    for b, w in zip(bars, wr):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{w:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Win Rate by Tier")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "backtest_best_scenario.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Best scenario chart saved to {path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading BTC/USDT 15m data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")

    print(f"\nRunning {15} scenarios...\n")
    results = run_all_scenarios(df)

    print_comparison(results)
    best = print_best_detail(results)

    plot_comparison(results, data_dir)
    plot_best_equity(best, data_dir)

    # Save all trades from best scenario
    path = os.path.join(data_dir, "backtest_best_trades.csv")
    cols = ["entry_time","exit_time","direction","tier","entry_price","tp_price","sl_price","exit_price","result","pnl_pct","pnl_abs"]
    best["trades_df"][cols].to_csv(path, index=False)
    print(f"Best scenario trades saved to {path}")


if __name__ == "__main__":
    main()
