#!/usr/bin/env python3
"""Backtest V2: Improved Mean Reversion Bounce Strategy."""

import glob, os
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    return data[["Open","High","Low","Close","Volume"]]


# ── Indicators ──────────────────────────────────────────────────

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def sma(s, n):
    return s.rolling(n).mean()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def resample_ohlc(df, tf):
    return df.resample(tf).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()



# ── Backtest Engine ─────────────────────────────────────────────

@dataclass
class Trade:
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


def run_scenario(df, label="",
                 # Trend
                 trend_tf="1h", trend_len=200, use_trend=True,
                 # Deviation
                 dev_ma_len=20, dev_lookback=20,
                 mult1=1.5, mult2=2.5, mult3=3.5,
                 # Filters
                 use_rsi=True, rsi_period=14, rsi_os=30, rsi_ob=70,
                 use_adx=True, adx_period=14, adx_thresh=25,
                 use_vol=True, vol_ma_len=20, vol_mult=1.5,
                 cooldown=8,
                 # TP/SL mode
                 tp_mode="mean",  # "mean" or "liquidity"
                 sl_mode="next_band",  # "next_band", "atr", "rr11"
                 use_partial=True, partial_pct=0.6,
                 trail_atr_mult=1.5, sl_atr_mult=2.0,
                 # Tiers
                 trade_tiers=(1,2,3),
                 # Execution
                 commission=0.04, slippage=0.01,
                 initial_capital=10000, risk_pct=1.0,
                 max_hold=192):  # 48h

    # ── Pre-compute indicators ──────────────────────────────
    # Trend MA on HTF
    htf = resample_ohlc(df, trend_tf)
    trend_ma = ema(htf["Close"], trend_len).shift(1).reindex(df.index, method="ffill").values

    # Dev bands
    dev_ma = ema(df["Close"], dev_ma_len).values
    stdev = df["Close"].rolling(dev_lookback).std().values

    ub1 = dev_ma + stdev * mult1
    ub2 = dev_ma + stdev * mult2
    ub3 = dev_ma + stdev * mult3
    lb1 = dev_ma - stdev * mult1
    lb2 = dev_ma - stdev * mult2
    lb3 = dev_ma - stdev * mult3

    # Extended band for T3 SL
    mult4 = mult3 + (mult3 - mult2)
    ub4 = dev_ma + stdev * mult4
    lb4 = dev_ma - stdev * mult4

    # RSI
    rsi = calc_rsi(df["Close"], rsi_period).values

    # ADX
    adx = calc_adx(df["High"], df["Low"], df["Close"], adx_period).values

    # ATR
    atr = calc_atr(df["High"], df["Low"], df["Close"], 14).values

    # Volume
    vol_ma = sma(df["Volume"], vol_ma_len).values

    # Price arrays
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    bars = df.index

    # ── Simulation ──────────────────────────────────────────
    trades = []
    equity = initial_capital
    in_trade = False
    trade = None
    cooldown_counter = 0
    hold = 0
    # For partial TP tracking
    partial_closed = False
    pos_size = 0
    remaining_size = 0
    trail_stop = 0
    best_price = 0  # best price since entry (for trailing)
    warmup = max(trend_len, dev_ma_len, dev_lookback, 50) + 10

    for i in range(warmup, len(df)):
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # ── EXIT LOGIC ──────────────────────────────────────
        if in_trade:
            hold += 1
            t = trade
            exit_price = None
            result = None

            if not partial_closed:
                # Check primary TP (mean) and SL
                if t.direction == "long":
                    # Update trailing best
                    if highs[i] > best_price:
                        best_price = highs[i]

                    if highs[i] >= t.tp_price:  # TP hit
                        if use_partial:
                            # Close partial_pct at TP
                            partial_exit = t.tp_price
                            cost = (commission + slippage) / 100
                            rpnl = (partial_exit - t.entry_price) / t.entry_price - 2 * cost
                            partial_size = pos_size * partial_pct
                            equity += partial_size * rpnl
                            remaining_size = pos_size * (1 - partial_pct)
                            partial_closed = True
                            # Set trailing stop for remainder
                            if not np.isnan(atr[i]):
                                trail_stop = best_price - atr[i] * trail_atr_mult
                            else:
                                trail_stop = t.sl_price
                            continue
                        else:
                            exit_price = t.tp_price
                            result = "TP"
                    elif lows[i] <= t.sl_price:
                        exit_price = t.sl_price
                        result = "SL"
                    elif hold >= max_hold:
                        exit_price = closes[i]
                        result = "timeout"

                else:  # short
                    if lows[i] < best_price:
                        best_price = lows[i]

                    if lows[i] <= t.tp_price:
                        if use_partial:
                            partial_exit = t.tp_price
                            cost = (commission + slippage) / 100
                            rpnl = (t.entry_price - partial_exit) / t.entry_price - 2 * cost
                            partial_size = pos_size * partial_pct
                            equity += partial_size * rpnl
                            remaining_size = pos_size * (1 - partial_pct)
                            partial_closed = True
                            if not np.isnan(atr[i]):
                                trail_stop = best_price + atr[i] * trail_atr_mult
                            else:
                                trail_stop = t.sl_price
                            continue
                        else:
                            exit_price = t.tp_price
                            result = "TP"
                    elif highs[i] >= t.sl_price:
                        exit_price = t.sl_price
                        result = "SL"
                    elif hold >= max_hold:
                        exit_price = closes[i]
                        result = "timeout"

            else:
                # Trailing stop for remaining portion
                if t.direction == "long":
                    if highs[i] > best_price:
                        best_price = highs[i]
                        if not np.isnan(atr[i]):
                            trail_stop = max(trail_stop, best_price - atr[i] * trail_atr_mult)

                    if lows[i] <= trail_stop:
                        exit_price = trail_stop
                        result = "trail"
                    elif hold >= max_hold:
                        exit_price = closes[i]
                        result = "timeout_trail"
                else:
                    if lows[i] < best_price:
                        best_price = lows[i]
                        if not np.isnan(atr[i]):
                            trail_stop = min(trail_stop, best_price + atr[i] * trail_atr_mult)

                    if highs[i] >= trail_stop:
                        exit_price = trail_stop
                        result = "trail"
                    elif hold >= max_hold:
                        exit_price = closes[i]
                        result = "timeout_trail"

            if exit_price is not None:
                cost = (commission + slippage) / 100
                if t.direction == "long":
                    rpnl = (exit_price - t.entry_price) / t.entry_price - 2 * cost
                else:
                    rpnl = (t.entry_price - exit_price) / t.entry_price - 2 * cost

                if partial_closed:
                    pnl_abs = remaining_size * rpnl
                else:
                    pnl_abs = pos_size * rpnl

                equity += pnl_abs
                t.exit_price = exit_price
                t.exit_time = bars[i]
                t.result = result
                # Total PnL for this trade
                if partial_closed:
                    # Already added partial pnl, compute total
                    partial_pnl_abs = pos_size * partial_pct * ((t.tp_price - t.entry_price) / t.entry_price - 2 * cost) if t.direction == "long" else pos_size * partial_pct * ((t.entry_price - t.tp_price) / t.entry_price - 2 * cost)
                    t.pnl_abs = partial_pnl_abs + pnl_abs
                    t.pnl_pct = t.pnl_abs / pos_size * 100
                else:
                    t.pnl_abs = pnl_abs
                    t.pnl_pct = rpnl * 100

                trades.append(t)
                in_trade = False
                partial_closed = False
                cooldown_counter = cooldown
                continue

        # ── ENTRY LOGIC ─────────────────────────────────────
        if not in_trade and cooldown_counter <= 0:
            # Trend filter
            if use_trend:
                is_bull = closes[i] > trend_ma[i] if not np.isnan(trend_ma[i]) else False
                is_bear = closes[i] < trend_ma[i] if not np.isnan(trend_ma[i]) else False
            else:
                is_bull = is_bear = True

            # RSI filter
            rsi_long_ok = (not use_rsi) or (not np.isnan(rsi[i]) and rsi[i] < rsi_os)
            rsi_short_ok = (not use_rsi) or (not np.isnan(rsi[i]) and rsi[i] > rsi_ob)

            # ADX filter
            adx_ok = (not use_adx) or (not np.isnan(adx[i]) and adx[i] < adx_thresh)

            # Volume filter
            vol_ok = (not use_vol) or (not np.isnan(vol_ma[i]) and vol_ma[i] > 0 and volumes[i] > vol_ma[i] * vol_mult)

            for tier in sorted(trade_tiers, reverse=True):
                if tier == 1:
                    ub_t, lb_t = ub1, lb1
                    sl_ub, sl_lb = ub2, lb2
                elif tier == 2:
                    ub_t, lb_t = ub2, lb2
                    sl_ub, sl_lb = ub3, lb3
                else:
                    ub_t, lb_t = ub3, lb3
                    sl_ub, sl_lb = ub4, lb4

                if np.isnan(ub_t[i]) or np.isnan(lb_t[i]):
                    continue

                # Long signal: touch lower band and bounce
                long_sig = (lows[i] <= lb_t[i] and closes[i] > lb_t[i] and
                           i > 0 and lows[i-1] > lb_t[i-1])
                # Short signal
                short_sig = (highs[i] >= ub_t[i] and closes[i] < ub_t[i] and
                            i > 0 and highs[i-1] < ub_t[i-1])

                entered = False

                if long_sig and is_bull and rsi_long_ok and adx_ok and vol_ok:
                    entry = closes[i]
                    # TP: mean or liquidity
                    tp = dev_ma[i] if tp_mode == "mean" else entry + (entry - lb_t[i])

                    # SL based on mode
                    if sl_mode == "next_band":
                        sl = sl_lb[i] if not np.isnan(sl_lb[i]) else entry - (entry - lb_t[i]) * 1.5
                    elif sl_mode == "atr" and not np.isnan(atr[i]):
                        sl = entry - atr[i] * sl_atr_mult
                    else:  # rr11
                        sl = entry - (tp - entry)

                    dist_tp = tp - entry
                    dist_sl = entry - sl
                    if dist_tp > 0 and dist_sl > 0 and dist_tp / entry > 0.0003:
                        risk_amt = equity * (risk_pct / 100)
                        pos = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                        trade = Trade(direction="long", tier=tier, entry_price=entry,
                                     tp_price=tp, sl_price=sl, entry_time=bars[i])
                        pos_size = pos
                        in_trade = True
                        hold = 0
                        partial_closed = False
                        best_price = entry
                        entered = True

                if not entered and short_sig and is_bear and rsi_short_ok and adx_ok and vol_ok:
                    entry = closes[i]
                    tp = dev_ma[i] if tp_mode == "mean" else entry - (ub_t[i] - entry)

                    if sl_mode == "next_band":
                        sl = sl_ub[i] if not np.isnan(sl_ub[i]) else entry + (ub_t[i] - entry) * 1.5
                    elif sl_mode == "atr" and not np.isnan(atr[i]):
                        sl = entry + atr[i] * sl_atr_mult
                    else:
                        sl = entry + (entry - tp)

                    dist_tp = entry - tp
                    dist_sl = sl - entry
                    if dist_tp > 0 and dist_sl > 0 and dist_tp / entry > 0.0003:
                        risk_amt = equity * (risk_pct / 100)
                        pos = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                        trade = Trade(direction="short", tier=tier, entry_price=entry,
                                     tp_price=tp, sl_price=sl, entry_time=bars[i])
                        pos_size = pos
                        in_trade = True
                        hold = 0
                        partial_closed = False
                        best_price = entry
                        entered = True

                if entered:
                    break

    return trades, equity, label


# ── Analysis ────────────────────────────────────────────────────

def calc_stats(trades, initial_capital=10000):
    if not trades:
        return None
    df = pd.DataFrame([t.__dict__ for t in trades])
    total = len(df)
    wins = df[df["pnl_abs"] > 0]
    losses = df[df["pnl_abs"] <= 0]
    wr = len(wins) / total * 100
    avg_w = wins["pnl_pct"].mean() if len(wins) else 0
    avg_l = losses["pnl_pct"].mean() if len(losses) else 0
    gp = wins["pnl_abs"].sum() if len(wins) else 0
    gl = abs(losses["pnl_abs"].sum()) if len(losses) else 0
    pf = gp / gl if gl > 0 else float("inf")
    final_eq = initial_capital + df["pnl_abs"].sum()
    ret = (final_eq - initial_capital) / initial_capital * 100
    cum = initial_capital + df["pnl_abs"].cumsum()
    peak = cum.expanding().max()
    dd = ((cum - peak) / peak * 100).min()

    tier_info = {}
    for t in [1, 2, 3]:
        tt = df[df["tier"] == t]
        if len(tt):
            tw = tt[tt["pnl_abs"] > 0]
            tier_info[t] = {"n":len(tt), "wr":len(tw)/len(tt)*100, "avg":tt["pnl_pct"].mean(), "total":tt["pnl_abs"].sum()}

    dir_info = {}
    for d in ["long", "short"]:
        dt = df[df["direction"] == d]
        if len(dt):
            dw = dt[dt["pnl_abs"] > 0]
            dir_info[d] = {"n":len(dt), "wr":len(dw)/len(dt)*100, "avg":dt["pnl_pct"].mean(), "total":dt["pnl_abs"].sum()}

    result_info = {}
    for r in df["result"].unique():
        rt = df[df["result"] == r]
        result_info[r] = {"n":len(rt), "avg":rt["pnl_pct"].mean(), "total":rt["pnl_abs"].sum()}

    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    yearly = df.groupby("year")["pnl_abs"].sum().to_dict()

    # Expectancy
    expectancy = (wr/100 * avg_w) + ((100-wr)/100 * avg_l)

    return {"total":total, "wr":wr, "avg_w":avg_w, "avg_l":avg_l,
            "pf":pf, "final_eq":final_eq, "total_ret":ret, "max_dd":dd,
            "tier":tier_info, "dir":dir_info, "result":result_info,
            "yearly":yearly, "trades_df":df, "expectancy": expectancy, "label":""}


# ── Scenarios ───────────────────────────────────────────────────

def run_all(df):
    scenarios = [
        # V1 Baseline for comparison
        dict(label="V1 Baseline (liq TP, 1:1 RR, no filters)",
             tp_mode="liquidity", sl_mode="rr11",
             use_rsi=False, use_adx=False, use_vol=False,
             use_partial=False, cooldown=0, trade_tiers=(1,2,3)),

        # V2 core: mean TP + next band SL + all filters
        dict(label="V2 Full (mean TP, band SL, RSI+ADX+Vol)",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=True, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 without trend filter
        dict(label="V2 No Trend Filter",
             tp_mode="mean", sl_mode="next_band", use_trend=False,
             use_rsi=True, use_adx=True, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 Tier 2+3 only
        dict(label="V2 Tier 2+3",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=True, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(2,3)),

        # V2 Strict RSI
        dict(label="V2 Strict RSI (25/75)",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, rsi_os=25, rsi_ob=75,
             use_adx=True, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 No ADX
        dict(label="V2 No ADX Filter",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=False, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 No Volume
        dict(label="V2 No Volume Filter",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=True, use_vol=False,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 No Partial TP
        dict(label="V2 No Partial TP (full close at mean)",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=True, use_vol=True,
             use_partial=False, cooldown=8, trade_tiers=(1,2,3)),

        # V2 Conservative
        dict(label="V2 Conservative (T2+3, strict RSI)",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, rsi_os=25, rsi_ob=75,
             use_adx=True, adx_thresh=22, use_vol=True, vol_mult=1.8,
             use_partial=True, cooldown=12, trade_tiers=(2,3)),

        # V2 with ATR SL
        dict(label="V2 ATR SL (2x ATR)",
             tp_mode="mean", sl_mode="atr", sl_atr_mult=2.0,
             use_rsi=True, use_adx=True, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 wider bands
        dict(label="V2 Wider Bands (2.0/3.0/4.0)",
             tp_mode="mean", sl_mode="next_band",
             mult1=2.0, mult2=3.0, mult3=4.0,
             use_rsi=True, use_adx=True, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),

        # V2 Aggressive
        dict(label="V2 Aggressive (RSI 35/65, no ADX)",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, rsi_os=35, rsi_ob=65,
             use_adx=False, use_vol=False,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V2 Mean TP + no partial + Tier 2+3
        dict(label="V2 Simple (mean TP, T2+3, RSI only)",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=False, use_vol=False,
             use_partial=False, cooldown=8, trade_tiers=(2,3)),

        # V2 with looser ADX
        dict(label="V2 Loose ADX (30) + all filters",
             tp_mode="mean", sl_mode="next_band",
             use_rsi=True, use_adx=True, adx_thresh=30, use_vol=True,
             use_partial=True, cooldown=8, trade_tiers=(1,2,3)),
    ]

    results = []
    for s in scenarios:
        lbl = s.pop("label")
        print(f"  {lbl}...", end=" ", flush=True)
        trades, final_eq, _ = run_scenario(df, label=lbl, **s)
        stats = calc_stats(trades)
        if stats:
            stats["label"] = lbl
            results.append(stats)
            print(f"{stats['total']} trades | WR {stats['wr']:.1f}% | PF {stats['pf']:.2f} | "
                  f"Ret {stats['total_ret']:.1f}% | DD {stats['max_dd']:.1f}% | E[R] {stats['expectancy']:.3f}%")
        else:
            print("No trades")
    return results


def print_comparison(results):
    print("\n" + "=" * 130)
    print("  SCENARIO COMPARISON (sorted by Profit Factor)")
    print("=" * 130)
    h = f"{'Scenario':<45} {'Trades':>6} {'WR%':>6} {'PF':>6} {'AvgW%':>7} {'AvgL%':>7} {'E[R]%':>7} {'Ret%':>8} {'DD%':>8} {'Final$':>10}"
    print(h)
    print("-" * 130)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        print(f"{s['label']:<45} {s['total']:>6} {s['wr']:>6.1f} {s['pf']:>6.2f} "
              f"{s['avg_w']:>7.2f} {s['avg_l']:>7.2f} {s['expectancy']:>7.3f} {s['total_ret']:>8.1f} "
              f"{s['max_dd']:>8.1f} {s['final_eq']:>10,.0f}")
    print("=" * 130)


def print_best(results):
    best = max(results, key=lambda x: x["pf"])
    s = best
    print(f"\n{'=' * 70}")
    print(f"  BEST: {s['label']}")
    print(f"{'=' * 70}")
    print(f"  Trades: {s['total']}  |  WR: {s['wr']:.1f}%  |  PF: {s['pf']:.2f}")
    print(f"  Avg Win: {s['avg_w']:.2f}%  |  Avg Loss: {s['avg_l']:.2f}%  |  Expectancy: {s['expectancy']:.3f}%")
    print(f"  Return: {s['total_ret']:.2f}%  |  Max DD: {s['max_dd']:.2f}%  |  Final: ${s['final_eq']:,.2f}")

    if s["tier"]:
        print(f"\n  BY TIER:")
        for t, info in s["tier"].items():
            print(f"    T{t}: {info['n']} trades | WR {info['wr']:.1f}% | Avg {info['avg']:.2f}% | ${info['total']:,.2f}")

    if s["dir"]:
        print(f"\n  BY DIRECTION:")
        for d, info in s["dir"].items():
            print(f"    {d.upper()}: {info['n']} trades | WR {info['wr']:.1f}% | Avg {info['avg']:.2f}% | ${info['total']:,.2f}")

    if s["result"]:
        print(f"\n  BY EXIT:")
        for r, info in s["result"].items():
            print(f"    {r.upper()}: {info['n']} trades | Avg {info['avg']:.2f}% | ${info['total']:,.2f}")

    if s["yearly"]:
        print(f"\n  YEARLY P&L:")
        for y, pnl in sorted(s["yearly"].items()):
            print(f"    {y}: ${pnl:,.2f}")

    return best


def plot_results(results, best, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Mean Reversion Bounce V2 — Scenario Comparison", fontsize=16, fontweight="bold")

    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [s["label"][:35] for s in sr]

    # PF
    ax = axes[0, 0]
    pfs = [s["pf"] for s in sr]
    colors = ["#00e676" if p >= 1 else "#ff1744" for p in pfs]
    ax.barh(labels, pfs, color=colors)
    ax.axvline(1.0, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Profit Factor")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Return
    ax = axes[0, 1]
    rets = [s["total_ret"] for s in sr]
    colors = ["#00e676" if r > 0 else "#ff1744" for r in rets]
    ax.barh(labels, rets, color=colors)
    ax.axvline(0, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Total Return (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # WR
    ax = axes[1, 0]
    wrs = [s["wr"] for s in sr]
    ax.barh(labels, wrs, color="#42a5f5")
    ax.axvline(50, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Win Rate (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Expectancy
    ax = axes[1, 1]
    exps = [s["expectancy"] for s in sr]
    colors = ["#00e676" if e > 0 else "#ff1744" for e in exps]
    ax.barh(labels, exps, color=colors)
    ax.axvline(0, color="white", linestyle="--", alpha=0.5)
    ax.set_title("Expectancy per Trade (%)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_v2_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # Best scenario detail
    df = best["trades_df"]
    cum = 10000 + df["pnl_abs"].cumsum()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Best: {best['label']}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(range(len(cum)), cum.values, color="#00e676", linewidth=1)
    ax.fill_between(range(len(cum)), cum.values, cum.min(), alpha=0.15, color="#00e676")
    ax.set_title("Equity Curve")
    ax.set_ylabel("$")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    peak = cum.expanding().max()
    dd = (cum - peak) / peak * 100
    ax.fill_between(range(len(dd)), dd.values, 0, color="#ff1744", alpha=0.5)
    ax.set_title("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(df["pnl_pct"].values, bins=50, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="white", linestyle="--")
    ax.set_title("P&L Distribution (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if best["tier"]:
        tiers = list(best["tier"].keys())
        wr = [best["tier"][t]["wr"] for t in tiers]
        ns = [best["tier"][t]["n"] for t in tiers]
        bars = ax.bar([f"T{t}\n({n})" for t, n in zip(tiers, ns)], wr,
                      color=["#42a5f5", "#ab47bc", "#ffa726"][:len(tiers)])
        for b, w in zip(bars, wr):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{w:.1f}%", ha="center", fontweight="bold")
        ax.set_ylim(0, 100)
    ax.set_title("Win Rate by Tier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_v2_best.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nCharts saved.")


# ── Main ────────────────────────────────────────────────────────

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df)} bars: {df.index.min()} to {df.index.max()}")
    print(f"\nRunning 14 scenarios...\n")
    results = run_all(df)
    print_comparison(results)
    best = print_best(results)
    plot_results(results, best, data_dir)
    cols = ["entry_time","exit_time","direction","tier","entry_price","tp_price","sl_price","exit_price","result","pnl_pct","pnl_abs"]
    best["trades_df"][cols].to_csv(os.path.join(data_dir, "backtest_v2_trades.csv"), index=False)
    print(f"Trades saved to backtest_v2_trades.csv")

if __name__ == "__main__":
    main()
