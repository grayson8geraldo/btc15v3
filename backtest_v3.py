#!/usr/bin/env python3
"""
Backtest V3: Mean Reversion with Limit Orders at Bands.

Key insight: Enter via LIMIT ORDER at the band price, not at market close.
This gives much better RR since entry is at the extreme, TP at mean.
- Entry: Limit order at band level  
- TP: Deviation MA (mean)
- SL: Tight — 0.5x stdev below entry band
- RR: ~5:1 for Tier 2, ~7:1 for Tier 3
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
    return data[["Open","High","Low","Close","Volume"]]

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

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def resample_ohlc(df, tf):
    return df.resample(tf).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()


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
                 trend_tf="1h", trend_len=200, use_trend=True,
                 dev_ma_len=20, dev_lookback=20,
                 mult1=1.5, mult2=2.5, mult3=3.5,
                 sl_stdev_mult=0.5,
                 use_rsi=True, rsi_period=14, rsi_os=30, rsi_ob=70,
                 use_adx=True, adx_period=14, adx_thresh=25,
                 use_vol=False, vol_ma_len=20, vol_mult=1.5,
                 cooldown=4,
                 use_partial=True, partial_pct=0.5,
                 trail_atr_mult=1.5,
                 trade_tiers=(1,2,3),
                 use_limit_order=True,
                 commission=0.04, slippage=0.01,
                 initial_capital=10000, risk_pct=1.0,
                 max_hold=96):

    # Pre-compute
    htf = resample_ohlc(df, trend_tf)
    trend_ma = ema(htf["Close"], trend_len).shift(1).reindex(df.index, method="ffill").values
    dev_ma = ema(df["Close"], dev_ma_len).values
    stdev = df["Close"].rolling(dev_lookback).std().values
    rsi = calc_rsi(df["Close"], rsi_period).values
    adx = calc_adx(df["High"], df["Low"], df["Close"], adx_period).values
    atr = calc_atr(df["High"], df["Low"], df["Close"], 14).values
    vol_ma_arr = sma(df["Volume"], vol_ma_len).values

    ub = {t: dev_ma + stdev * m for t, m in [(1, mult1), (2, mult2), (3, mult3)]}
    lb = {t: dev_ma - stdev * m for t, m in [(1, mult1), (2, mult2), (3, mult3)]}

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    volumes = df["Volume"].values
    bars = df.index

    trades = []
    equity = initial_capital
    in_trade = False
    pending_order = None  # (direction, tier, band_price, tp, sl, bar_placed)
    trade = None
    pos_size = 0
    cooldown_counter = 0
    hold = 0
    partial_closed = False
    remaining_size = 0
    trail_stop = 0
    best_price = 0
    warmup = max(trend_len, dev_ma_len, dev_lookback, 50) + 10

    for i in range(warmup, len(df)):
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # ── CHECK PENDING LIMIT ORDER FILL ──────────────────
        if pending_order is not None and not in_trade:
            pdir, ptier, pband, ptp, psl, pbar = pending_order
            # Order expires after 4 bars
            if i - pbar > 4:
                pending_order = None
            else:
                filled = False
                if pdir == "long" and lows[i] <= pband:
                    entry = pband
                    filled = True
                elif pdir == "short" and highs[i] >= pband:
                    entry = pband
                    filled = True

                if filled:
                    # Recalculate TP/SL with current values
                    tp = dev_ma[i]  # current mean
                    if pdir == "long":
                        sl = entry - stdev[i] * sl_stdev_mult if not np.isnan(stdev[i]) else psl
                        dist_tp = tp - entry
                        dist_sl = entry - sl
                    else:
                        sl = entry + stdev[i] * sl_stdev_mult if not np.isnan(stdev[i]) else psl
                        dist_tp = entry - tp
                        dist_sl = sl - entry

                    if dist_tp > 0 and dist_sl > 0 and dist_tp / entry > 0.0002:
                        risk_amt = equity * (risk_pct / 100)
                        pos_size = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                        trade = Trade(direction=pdir, tier=ptier, entry_price=entry,
                                     tp_price=tp, sl_price=sl, entry_time=bars[i])
                        in_trade = True
                        hold = 0
                        partial_closed = False
                        best_price = entry
                    pending_order = None

        # ── EXIT LOGIC ──────────────────────────────────────
        if in_trade:
            hold += 1
            t = trade
            exit_price = None
            result = None

            if not partial_closed:
                if t.direction == "long":
                    if highs[i] > best_price:
                        best_price = highs[i]
                    # Check SL first (conservative for non-partial)
                    if lows[i] <= t.sl_price:
                        exit_price = t.sl_price
                        result = "SL"
                    elif highs[i] >= t.tp_price:
                        if use_partial:
                            cost = (commission + slippage) / 100
                            rpnl = (t.tp_price - t.entry_price) / t.entry_price - 2 * cost
                            partial_size = pos_size * partial_pct
                            equity += partial_size * rpnl
                            remaining_size = pos_size * (1 - partial_pct)
                            partial_closed = True
                            trail_stop = best_price - atr[i] * trail_atr_mult if not np.isnan(atr[i]) else t.sl_price
                            continue
                        else:
                            exit_price = t.tp_price
                            result = "TP"
                    elif hold >= max_hold:
                        exit_price = closes[i]
                        result = "timeout"
                else:  # short
                    if lows[i] < best_price:
                        best_price = lows[i]
                    if highs[i] >= t.sl_price:
                        exit_price = t.sl_price
                        result = "SL"
                    elif lows[i] <= t.tp_price:
                        if use_partial:
                            cost = (commission + slippage) / 100
                            rpnl = (t.entry_price - t.tp_price) / t.entry_price - 2 * cost
                            partial_size = pos_size * partial_pct
                            equity += partial_size * rpnl
                            remaining_size = pos_size * (1 - partial_pct)
                            partial_closed = True
                            trail_stop = best_price + atr[i] * trail_atr_mult if not np.isnan(atr[i]) else t.sl_price
                            continue
                        else:
                            exit_price = t.tp_price
                            result = "TP"
                    elif hold >= max_hold:
                        exit_price = closes[i]
                        result = "timeout"
            else:
                # Trailing for remaining portion
                if t.direction == "long":
                    if highs[i] > best_price:
                        best_price = highs[i]
                        if not np.isnan(atr[i]):
                            trail_stop = max(trail_stop, best_price - atr[i] * trail_atr_mult)
                    if lows[i] <= trail_stop:
                        exit_price = max(trail_stop, lows[i])
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
                        exit_price = min(trail_stop, highs[i])
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
                    # Recompute total
                    pcost = (commission + slippage) / 100
                    if t.direction == "long":
                        partial_rpnl = (t.tp_price - t.entry_price) / t.entry_price - 2 * pcost
                    else:
                        partial_rpnl = (t.entry_price - t.tp_price) / t.entry_price - 2 * pcost
                    total_pnl = pos_size * partial_pct * partial_rpnl + remaining_size * rpnl
                    t.pnl_abs = total_pnl
                    t.pnl_pct = total_pnl / pos_size * 100
                else:
                    pnl_abs = pos_size * rpnl
                    t.pnl_abs = pnl_abs
                    t.pnl_pct = rpnl * 100

                equity += pnl_abs
                t.exit_price = exit_price
                t.exit_time = bars[i]
                t.result = result
                trades.append(t)
                in_trade = False
                partial_closed = False
                cooldown_counter = cooldown
                continue

        # ── ENTRY SIGNAL + PLACE LIMIT ORDER ────────────────
        if not in_trade and pending_order is None and cooldown_counter <= 0:
            if use_trend:
                is_bull = not np.isnan(trend_ma[i]) and closes[i] > trend_ma[i]
                is_bear = not np.isnan(trend_ma[i]) and closes[i] < trend_ma[i]
            else:
                is_bull = is_bear = True

            rsi_long_ok = (not use_rsi) or (not np.isnan(rsi[i]) and rsi[i] < rsi_os)
            rsi_short_ok = (not use_rsi) or (not np.isnan(rsi[i]) and rsi[i] > rsi_ob)
            adx_ok = (not use_adx) or (not np.isnan(adx[i]) and adx[i] < adx_thresh)
            vol_ok = (not use_vol) or (not np.isnan(vol_ma_arr[i]) and vol_ma_arr[i] > 0 and volumes[i] > vol_ma_arr[i] * vol_mult)

            for tier in sorted(trade_tiers, reverse=True):
                if np.isnan(lb[tier][i]) or np.isnan(ub[tier][i]) or np.isnan(dev_ma[i]) or np.isnan(stdev[i]):
                    continue

                if use_limit_order:
                    # Place limit order when price APPROACHES band (within 0.5 stdev)
                    approach_long = closes[i] < lb[tier][i] + stdev[i] * 0.5 and closes[i] > lb[tier][i]
                    approach_short = closes[i] > ub[tier][i] - stdev[i] * 0.5 and closes[i] < ub[tier][i]

                    # Or: price already touched band on this bar
                    touch_long = lows[i] <= lb[tier][i]
                    touch_short = highs[i] >= ub[tier][i]

                    if (approach_long or touch_long) and is_bull and rsi_long_ok and adx_ok and vol_ok:
                        band_price = lb[tier][i]
                        tp = dev_ma[i]
                        sl = band_price - stdev[i] * sl_stdev_mult
                        if tp > band_price:
                            if touch_long:
                                # Already touched — fill immediately
                                entry = band_price
                                dist_sl = entry - sl
                                dist_tp = tp - entry
                                if dist_tp > 0 and dist_sl > 0:
                                    risk_amt = equity * (risk_pct / 100)
                                    pos_size = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                                    trade = Trade(direction="long", tier=tier, entry_price=entry,
                                                 tp_price=tp, sl_price=sl, entry_time=bars[i])
                                    in_trade = True
                                    hold = 0
                                    partial_closed = False
                                    best_price = entry
                            else:
                                pending_order = ("long", tier, band_price, tp, sl, i)
                            break

                    if not in_trade and pending_order is None and (approach_short or touch_short) and is_bear and rsi_short_ok and adx_ok and vol_ok:
                        band_price = ub[tier][i]
                        tp = dev_ma[i]
                        sl = band_price + stdev[i] * sl_stdev_mult
                        if tp < band_price:
                            if touch_short:
                                entry = band_price
                                dist_sl = sl - entry
                                dist_tp = entry - tp
                                if dist_tp > 0 and dist_sl > 0:
                                    risk_amt = equity * (risk_pct / 100)
                                    pos_size = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                                    trade = Trade(direction="short", tier=tier, entry_price=entry,
                                                 tp_price=tp, sl_price=sl, entry_time=bars[i])
                                    in_trade = True
                                    hold = 0
                                    partial_closed = False
                                    best_price = entry
                            else:
                                pending_order = ("short", tier, band_price, tp, sl, i)
                            break

                else:
                    # Market order (V1 style)
                    long_sig = lows[i] <= lb[tier][i] and closes[i] > lb[tier][i] and i > 0 and lows[i-1] > lb[tier][i-1]
                    short_sig = highs[i] >= ub[tier][i] and closes[i] < ub[tier][i] and i > 0 and highs[i-1] < ub[tier][i-1]

                    if long_sig and is_bull and rsi_long_ok and adx_ok and vol_ok:
                        entry = closes[i]
                        tp = dev_ma[i]
                        sl = entry - stdev[i] * sl_stdev_mult if not np.isnan(stdev[i]) else lb[tier][i] - abs(closes[i] - lb[tier][i])
                        dist_tp = tp - entry
                        dist_sl = entry - sl
                        if dist_tp > 0 and dist_sl > 0:
                            risk_amt = equity * (risk_pct / 100)
                            pos_size = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                            trade = Trade(direction="long", tier=tier, entry_price=entry,
                                         tp_price=tp, sl_price=sl, entry_time=bars[i])
                            in_trade = True; hold = 0; partial_closed = False; best_price = entry
                            break

                    if not in_trade and short_sig and is_bear and rsi_short_ok and adx_ok and vol_ok:
                        entry = closes[i]
                        tp = dev_ma[i]
                        sl = entry + stdev[i] * sl_stdev_mult if not np.isnan(stdev[i]) else ub[tier][i] + abs(ub[tier][i] - closes[i])
                        dist_tp = entry - tp
                        dist_sl = sl - entry
                        if dist_tp > 0 and dist_sl > 0:
                            risk_amt = equity * (risk_pct / 100)
                            pos_size = risk_amt / (dist_sl / entry + 2 * (commission + slippage) / 100)
                            trade = Trade(direction="short", tier=tier, entry_price=entry,
                                         tp_price=tp, sl_price=sl, entry_time=bars[i])
                            in_trade = True; hold = 0; partial_closed = False; best_price = entry
                            break

    return trades, equity, label


def calc_stats(trades, initial_capital=10000):
    if not trades: return None
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
    expectancy = (wr/100 * avg_w) + ((100-wr)/100 * avg_l)

    tier_info = {}
    for t in [1,2,3]:
        tt = df[df["tier"] == t]
        if len(tt):
            tw = tt[tt["pnl_abs"] > 0]
            tier_info[t] = {"n":len(tt), "wr":len(tw)/len(tt)*100, "avg":tt["pnl_pct"].mean(), "total":tt["pnl_abs"].sum()}

    dir_info = {}
    for d in ["long","short"]:
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

    return {"total":total, "wr":wr, "avg_w":avg_w, "avg_l":avg_l, "pf":pf,
            "final_eq":final_eq, "total_ret":ret, "max_dd":dd, "expectancy":expectancy,
            "tier":tier_info, "dir":dir_info, "result":result_info,
            "yearly":yearly, "trades_df":df, "label":""}


def run_all(df):
    scenarios = [
        # V1 baseline
        dict(label="V1 Baseline (market order, no filters)",
             use_limit_order=False, use_rsi=False, use_adx=False,
             sl_stdev_mult=1.0, use_partial=False, cooldown=0, trade_tiers=(1,2,3)),

        # V3 core: limit order + tight SL + RSI + ADX
        dict(label="V3 Limit + Tight SL(0.5) + RSI + ADX",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 Tier 2+3
        dict(label="V3 Limit T2+3 + RSI + ADX",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(2,3)),

        # V3 no ADX
        dict(label="V3 Limit + RSI only",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=False,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 wider SL (0.75 stdev)
        dict(label="V3 Limit + SL 0.75 stdev",
             use_limit_order=True, sl_stdev_mult=0.75,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 tighter SL (0.3 stdev)
        dict(label="V3 Limit + SL 0.3 stdev",
             use_limit_order=True, sl_stdev_mult=0.3,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 no partial TP
        dict(label="V3 Limit + Full TP at mean",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True,
             use_partial=False, cooldown=4, trade_tiers=(1,2,3)),

        # V3 no trend filter
        dict(label="V3 Limit + No Trend Filter",
             use_limit_order=True, sl_stdev_mult=0.5, use_trend=False,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 strict RSI (25/75)
        dict(label="V3 Limit + Strict RSI (25/75)",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, rsi_os=25, rsi_ob=75, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 relaxed RSI (35/65) + no ADX
        dict(label="V3 Limit + Relaxed (RSI 35/65, no ADX)",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, rsi_os=35, rsi_ob=65, use_adx=False,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 wider bands (2/3/4)
        dict(label="V3 Limit + Wide bands (2/3/4)",
             use_limit_order=True, sl_stdev_mult=0.5,
             mult1=2.0, mult2=3.0, mult3=4.0,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 market order + tight SL (compare limit vs market)
        dict(label="V3 Market Order + Tight SL + Filters",
             use_limit_order=False, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 Tier 3 only (strongest signals)
        dict(label="V3 Limit T3 only",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(3,)),

        # V3 longer dev lookback
        dict(label="V3 Limit + Dev LB 50",
             use_limit_order=True, sl_stdev_mult=0.5,
             dev_ma_len=50, dev_lookback=50,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),

        # V3 no cooldown
        dict(label="V3 Limit + No Cooldown",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True,
             use_partial=True, cooldown=0, trade_tiers=(1,2,3)),

        # V3 ADX thresh 30
        dict(label="V3 Limit + ADX<30 + RSI",
             use_limit_order=True, sl_stdev_mult=0.5,
             use_rsi=True, use_adx=True, adx_thresh=30,
             use_partial=True, cooldown=4, trade_tiers=(1,2,3)),
    ]

    results = []
    for s in scenarios:
        lbl = s.pop("label")
        print(f"  {lbl}...", end=" ", flush=True)
        trades, eq, _ = run_scenario(df, label=lbl, **s)
        stats = calc_stats(trades)
        if stats:
            stats["label"] = lbl
            results.append(stats)
            pfx = "+" if stats["total_ret"] > 0 else ""
            print(f"{stats['total']} | WR {stats['wr']:.1f}% | PF {stats['pf']:.2f} | "
                  f"E[R] {stats['expectancy']:.3f}% | {pfx}{stats['total_ret']:.1f}% | DD {stats['max_dd']:.1f}%")
        else:
            print("No trades")
    return results


def print_all(results):
    print("\n" + "=" * 140)
    print("  SCENARIO COMPARISON (by Profit Factor)")
    print("=" * 140)
    h = f"{'Scenario':<45} {'N':>5} {'WR%':>6} {'PF':>6} {'AvgW%':>7} {'AvgL%':>7} {'E[R]%':>7} {'Ret%':>9} {'DD%':>8} {'Final$':>10}"
    print(h)
    print("-" * 140)
    for s in sorted(results, key=lambda x: x["pf"], reverse=True):
        pfx = "+" if s["total_ret"] > 0 else ""
        print(f"{s['label']:<45} {s['total']:>5} {s['wr']:>6.1f} {s['pf']:>6.2f} "
              f"{s['avg_w']:>7.2f} {s['avg_l']:>7.2f} {s['expectancy']:>7.3f} "
              f"{pfx}{s['total_ret']:>8.1f} {s['max_dd']:>8.1f} {s['final_eq']:>10,.0f}")
    print("=" * 140)

    best = max(results, key=lambda x: x["pf"])
    s = best
    print(f"\n{'=' * 70}")
    print(f"  BEST: {s['label']}")
    print(f"{'=' * 70}")
    print(f"  Trades: {s['total']}  |  WR: {s['wr']:.1f}%  |  PF: {s['pf']:.2f}  |  E[R]: {s['expectancy']:.3f}%")
    print(f"  Avg Win: {s['avg_w']:.2f}%  |  Avg Loss: {s['avg_l']:.2f}%")
    print(f"  Return: {s['total_ret']:.2f}%  |  Max DD: {s['max_dd']:.2f}%  |  Final: ${s['final_eq']:,.2f}")

    for label, info_dict in [("TIER", s["tier"]), ("DIRECTION", s["dir"]), ("EXIT", s["result"])]:
        if info_dict:
            print(f"\n  BY {label}:")
            for k, v in info_dict.items():
                key = f"T{k}" if label == "TIER" else str(k).upper()
                print(f"    {key}: {v['n']} | WR {v.get('wr', 0):.1f}% | Avg {v['avg']:.2f}% | ${v['total']:,.2f}")

    if s["yearly"]:
        print(f"\n  YEARLY P&L:")
        for y, pnl in sorted(s["yearly"].items()):
            print(f"    {y}: ${pnl:,.2f}")

    return best


def plot_results(results, best, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Mean Reversion V3 — Limit Orders at Bands", fontsize=16, fontweight="bold")
    sr = sorted(results, key=lambda x: x["pf"], reverse=True)
    labels = [s["label"][:35] for s in sr]

    for ax, metric, title, vline in [
        (axes[0,0], "pf", "Profit Factor", 1.0),
        (axes[0,1], "total_ret", "Total Return (%)", 0),
        (axes[1,0], "wr", "Win Rate (%)", 50),
        (axes[1,1], "expectancy", "Expectancy per Trade (%)", 0)]:
        vals = [s[metric] for s in sr]
        colors = ["#00e676" if (v >= vline if metric != "wr" else v >= 50) else "#ff1744" for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(vline, color="white", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_v3_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    df = best["trades_df"]
    cum = 10000 + df["pnl_abs"].cumsum()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Best: {best['label']}", fontsize=16, fontweight="bold")

    axes[0,0].plot(range(len(cum)), cum.values, color="#00e676", linewidth=1)
    axes[0,0].fill_between(range(len(cum)), cum.values, cum.min(), alpha=0.15, color="#00e676")
    axes[0,0].set_title("Equity Curve"); axes[0,0].grid(True, alpha=0.3)

    peak = cum.expanding().max()
    dd = (cum - peak) / peak * 100
    axes[0,1].fill_between(range(len(dd)), dd.values, 0, color="#ff1744", alpha=0.5)
    axes[0,1].set_title("Drawdown (%)"); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].hist(df["pnl_pct"].values, bins=60, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1,0].axvline(0, color="white", linestyle="--")
    axes[1,0].set_title("P&L Distribution (%)"); axes[1,0].grid(True, alpha=0.3)

    if best["tier"]:
        tiers = list(best["tier"].keys())
        wr = [best["tier"][t]["wr"] for t in tiers]
        ns = [best["tier"][t]["n"] for t in tiers]
        bars = axes[1,1].bar([f"T{t}\n({n})" for t, n in zip(tiers, ns)], wr,
                      color=["#42a5f5","#ab47bc","#ffa726"][:len(tiers)])
        for b, w in zip(bars, wr):
            axes[1,1].text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{w:.1f}%", ha="center", fontweight="bold")
        axes[1,1].set_ylim(0, 100)
    axes[1,1].set_title("Win Rate by Tier"); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_v3_best.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nCharts saved.")


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading data...")
    df = load_data(data_dir)
    print(f"{len(df)} bars: {df.index.min()} to {df.index.max()}")
    print(f"\nRunning scenarios...\n")
    results = run_all(df)
    print_all(results)
    best = max(results, key=lambda x: x["pf"])
    plot_results(results, best, data_dir)
    cols = ["entry_time","exit_time","direction","tier","entry_price","tp_price","sl_price","exit_price","result","pnl_pct","pnl_abs"]
    best["trades_df"][cols].to_csv(os.path.join(data_dir, "backtest_v3_trades.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    main()
