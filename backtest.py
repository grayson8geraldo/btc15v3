#!/usr/bin/env python3
"""
Backtest: Mean Reversion Bounce Strategy on BTC 15m data.

Replicates the Pine Script indicator logic:
- Trend Ribbon (200 EMA, with optional MTF)
- 3-tier entry signals based on std deviation from mean
- Liquidity levels from HTF highs/lows as take-profit targets
- 1:1 RR stop-loss
- Filter: signal must match trend + liquidity target must exist
"""

import glob
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Trend Ribbon
    trend_type: str = "EMA"
    trend_length: int = 200
    trend_tf: str = "1h"          # MTF for trend (resample period)

    # Mean Reversion Engine
    dev_ma_type: str = "EMA"
    dev_ma_length: int = 20
    dev_lookback: int = 20        # std dev lookback
    dev_mult1: float = 1.5        # Tier 1
    dev_mult2: float = 2.5        # Tier 2
    dev_mult3: float = 3.5        # Tier 3

    # Liquidity
    liq_tf1: str = "1D"           # solid — daily
    liq_tf2: str = "4h"           # dashed — 4h
    liq_lookback: int = 20

    # Execution
    commission_pct: float = 0.04  # 0.04% per side (Binance taker)
    slippage_pct: float = 0.01    # 0.01% slippage per side
    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 1.0   # risk 1% of equity per trade

    # Tiers to trade
    trade_tier1: bool = True
    trade_tier2: bool = True
    trade_tier3: bool = True


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "BTCUSDT-15m-*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["open_time"], unit="ms")
    data.set_index("datetime", inplace=True)
    data.sort_index(inplace=True)
    data = data[~data.index.duplicated(keep="first")]

    # Rename for convenience
    data.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    }, inplace=True)

    return data[["Open", "High", "Low", "Close", "Volume"]]


# ═══════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calc_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "SMA":
        return series.rolling(length).mean()
    elif ma_type == "HMA":
        half = max(length // 2, 1)
        sqrt_len = max(int(np.sqrt(length)), 1)
        wma_half = series.rolling(half).apply(
            lambda x: np.dot(x, np.arange(1, half + 1)) / np.arange(1, half + 1).sum(), raw=True)
        wma_full = series.rolling(length).apply(
            lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(), raw=True)
        diff = 2 * wma_half - wma_full
        return diff.rolling(sqrt_len).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_len + 1)) / np.arange(1, sqrt_len + 1).sum(), raw=True)
    elif ma_type == "DEMA":
        e1 = series.ewm(span=length, adjust=False).mean()
        return 2 * e1 - e1.ewm(span=length, adjust=False).mean()
    else:
        return series.ewm(span=length, adjust=False).mean()


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample to higher timeframe."""
    return df.resample(tf).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()


def calc_trend_ribbon(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Calculate trend MA on MTF and map back to 15m."""
    htf = resample_ohlc(df, cfg.trend_tf)
    ma = calc_ma(htf["Close"], cfg.trend_length, cfg.trend_type)
    # Forward-fill to 15m index (no lookahead: shift HTF by 1 before reindex)
    ma_shifted = ma.shift(1)  # use completed HTF bar only
    return ma_shifted.reindex(df.index, method="ffill")


def calc_deviation_bands(df: pd.DataFrame, cfg: Config):
    """Calculate mean and std deviation bands."""
    dev_ma = calc_ma(df["Close"], cfg.dev_ma_length, cfg.dev_ma_type)
    stdev = df["Close"].rolling(cfg.dev_lookback).std()

    bands = {}
    for tier, mult in [(1, cfg.dev_mult1), (2, cfg.dev_mult2), (3, cfg.dev_mult3)]:
        bands[f"upper{tier}"] = dev_ma + stdev * mult
        bands[f"lower{tier}"] = dev_ma - stdev * mult

    return dev_ma, pd.DataFrame(bands, index=df.index)


def calc_liquidity_levels(df: pd.DataFrame, cfg: Config):
    """Calculate HTF high/low liquidity pools."""
    results = {}
    for label, tf, lookback in [("tf1", cfg.liq_tf1, cfg.liq_lookback),
                                  ("tf2", cfg.liq_tf2, cfg.liq_lookback)]:
        htf = resample_ohlc(df, tf)
        htf_high = htf["High"].rolling(lookback).max().shift(1)  # no lookahead
        htf_low  = htf["Low"].rolling(lookback).min().shift(1)
        results[f"{label}_high"] = htf_high.reindex(df.index, method="ffill")
        results[f"{label}_low"]  = htf_low.reindex(df.index, method="ffill")

    liq = pd.DataFrame(results, index=df.index)
    return liq


# ═══════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_signals(df: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Detect tier 1/2/3 long and short signals."""
    sigs = pd.DataFrame(index=df.index)

    for tier in [1, 2, 3]:
        ub = bands[f"upper{tier}"]
        lb = bands[f"lower{tier}"]

        # Long: low touches lower band and closes above it, previous bar didn't touch
        sigs[f"long_t{tier}"] = (
            (df["Low"] <= lb) &
            (df["Close"] > lb) &
            (df["Low"].shift(1) > lb.shift(1))
        )

        # Short: high touches upper band and closes below it, previous bar didn't touch
        sigs[f"short_t{tier}"] = (
            (df["High"] >= ub) &
            (df["Close"] < ub) &
            (df["High"].shift(1) < ub.shift(1))
        )

    return sigs


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_time: datetime = None
    exit_time: datetime = None
    direction: str = ""       # "long" or "short"
    tier: int = 0
    entry_price: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0
    result: str = ""          # "TP", "SL", "timeout"
    position_size: float = 0.0


def run_backtest(df: pd.DataFrame, cfg: Config) -> tuple[list[Trade], pd.Series]:
    print("Calculating indicators...")

    # 1) Trend Ribbon
    trend_ma = calc_trend_ribbon(df, cfg)
    is_bull = df["Close"] > trend_ma
    is_bear = df["Close"] < trend_ma

    # 2) Deviation bands
    dev_ma, bands = calc_deviation_bands(df, cfg)

    # 3) Liquidity levels
    liq = calc_liquidity_levels(df, cfg)

    # Best liquidity targets
    def best_liq_above(row):
        candidates = []
        for col in ["tf1_high", "tf2_high"]:
            v = row[col]
            if not np.isnan(v) and v > row["Close"]:
                candidates.append(v)
        return min(candidates) if candidates else np.nan

    def best_liq_below(row):
        candidates = []
        for col in ["tf1_low", "tf2_low"]:
            v = row[col]
            if not np.isnan(v) and v < row["Close"]:
                candidates.append(v)
        return max(candidates) if candidates else np.nan

    merged = df[["Close"]].join(liq)
    liq_above = merged.apply(best_liq_above, axis=1)
    liq_below = merged.apply(best_liq_below, axis=1)

    has_liq_long = liq_above.notna()
    has_liq_short = liq_below.notna()

    # 4) Signals
    sigs = detect_signals(df, bands)

    # 5) Filter signals
    tier_map = {1: cfg.trade_tier1, 2: cfg.trade_tier2, 3: cfg.trade_tier3}

    print("Running backtest simulation...")

    trades: list[Trade] = []
    equity = cfg.initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = equity

    in_trade = False
    current_trade: Trade = None
    max_hold_bars = 96  # 96 * 15m = 24h max hold

    bars = df.index
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # Pre-compute signal arrays for speed
    sig_arrays = {}
    for tier in [1, 2, 3]:
        sig_arrays[f"long_t{tier}"] = sigs[f"long_t{tier}"].values
        sig_arrays[f"short_t{tier}"] = sigs[f"short_t{tier}"].values

    is_bull_arr = is_bull.values
    is_bear_arr = is_bear.values
    has_liq_long_arr = has_liq_long.values
    has_liq_short_arr = has_liq_short.values
    liq_above_arr = liq_above.values
    liq_below_arr = liq_below.values

    hold_counter = 0

    for i in range(cfg.trend_length + cfg.dev_lookback, len(df)):
        idx = bars[i]
        equity_curve.iloc[i] = equity

        # --- Check exit for existing trade ---
        if in_trade and current_trade is not None:
            hold_counter += 1
            t = current_trade
            hit_tp = False
            hit_sl = False

            if t.direction == "long":
                if lows[i] <= t.sl_price:
                    hit_sl = True
                if highs[i] >= t.tp_price:
                    hit_tp = True
            else:
                if highs[i] >= t.sl_price:
                    hit_sl = True
                if lows[i] <= t.tp_price:
                    hit_tp = True

            exit_price = None
            result = None

            if hit_sl and hit_tp:
                # Ambiguous — assume SL hit first (conservative)
                exit_price = t.sl_price
                result = "SL"
            elif hit_sl:
                exit_price = t.sl_price
                result = "SL"
            elif hit_tp:
                exit_price = t.tp_price
                result = "TP"
            elif hold_counter >= max_hold_bars:
                exit_price = closes[i]
                result = "timeout"

            if exit_price is not None:
                cost = cfg.commission_pct / 100 + cfg.slippage_pct / 100  # per side
                if t.direction == "long":
                    raw_pnl_pct = (exit_price - t.entry_price) / t.entry_price
                else:
                    raw_pnl_pct = (t.entry_price - exit_price) / t.entry_price

                net_pnl_pct = raw_pnl_pct - 2 * cost  # entry + exit
                pnl_abs = t.position_size * net_pnl_pct

                t.exit_time = idx
                t.exit_price = exit_price
                t.pnl_pct = net_pnl_pct * 100
                t.pnl_abs = pnl_abs
                t.result = result

                equity += pnl_abs
                trades.append(t)
                in_trade = False
                current_trade = None
                hold_counter = 0
                continue

        # --- Check entry signals ---
        if not in_trade:
            best_signal = None
            best_tier = 0

            # Check tiers 3 → 1 (prefer strongest)
            for tier in [3, 2, 1]:
                if not tier_map[tier]:
                    continue

                if sig_arrays[f"long_t{tier}"][i] and is_bull_arr[i] and has_liq_long_arr[i]:
                    best_signal = "long"
                    best_tier = tier
                    break
                if sig_arrays[f"short_t{tier}"][i] and is_bear_arr[i] and has_liq_short_arr[i]:
                    best_signal = "short"
                    best_tier = tier
                    break

            if best_signal is not None:
                entry_price = closes[i]

                if best_signal == "long":
                    tp = liq_above_arr[i]
                    dist = tp - entry_price
                    sl = entry_price - dist
                else:
                    tp = liq_below_arr[i]
                    dist = entry_price - tp
                    sl = entry_price + dist

                # Skip if dist is too small (< 0.05%)
                if dist / entry_price < 0.0005:
                    continue

                # Position sizing: risk X% of equity
                risk_amount = equity * (cfg.risk_per_trade_pct / 100)
                position_size = risk_amount / (dist / entry_price + 2 * (cfg.commission_pct + cfg.slippage_pct) / 100)

                current_trade = Trade(
                    entry_time=idx,
                    direction=best_signal,
                    tier=best_tier,
                    entry_price=entry_price,
                    tp_price=tp,
                    sl_price=sl,
                    position_size=position_size,
                )
                in_trade = True
                hold_counter = 0

    # Close any open trade at end
    if in_trade and current_trade is not None:
        t = current_trade
        exit_price = closes[-1]
        cost = cfg.commission_pct / 100 + cfg.slippage_pct / 100
        if t.direction == "long":
            raw_pnl_pct = (exit_price - t.entry_price) / t.entry_price
        else:
            raw_pnl_pct = (t.entry_price - exit_price) / t.entry_price
        net_pnl_pct = raw_pnl_pct - 2 * cost
        t.exit_time = bars[-1]
        t.exit_price = exit_price
        t.pnl_pct = net_pnl_pct * 100
        t.pnl_abs = t.position_size * net_pnl_pct
        t.result = "close"
        equity += t.pnl_abs
        trades.append(t)

    # Fill forward equity curve
    equity_curve = equity_curve.ffill()
    equity_curve = equity_curve.fillna(cfg.initial_capital)

    return trades, equity_curve


# ═══════════════════════════════════════════════════════════════
# ANALYSIS & REPORTING
# ═══════════════════════════════════════════════════════════════

def analyze_trades(trades: list[Trade], equity_curve: pd.Series, cfg: Config) -> dict:
    if not trades:
        return {"error": "No trades"}

    df_trades = pd.DataFrame([t.__dict__ for t in trades])

    total = len(trades)
    wins = df_trades[df_trades["pnl_abs"] > 0]
    losses = df_trades[df_trades["pnl_abs"] <= 0]

    win_rate = len(wins) / total * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0

    # By tier
    tier_stats = {}
    for tier in [1, 2, 3]:
        tier_trades = df_trades[df_trades["tier"] == tier]
        if len(tier_trades) > 0:
            tw = tier_trades[tier_trades["pnl_abs"] > 0]
            tier_stats[tier] = {
                "count": len(tier_trades),
                "win_rate": len(tw) / len(tier_trades) * 100,
                "avg_pnl": tier_trades["pnl_pct"].mean(),
                "total_pnl": tier_trades["pnl_abs"].sum(),
            }

    # By direction
    dir_stats = {}
    for d in ["long", "short"]:
        dt = df_trades[df_trades["direction"] == d]
        if len(dt) > 0:
            dw = dt[dt["pnl_abs"] > 0]
            dir_stats[d] = {
                "count": len(dt),
                "win_rate": len(dw) / len(dt) * 100,
                "avg_pnl": dt["pnl_pct"].mean(),
                "total_pnl": dt["pnl_abs"].sum(),
            }

    # By result
    result_stats = df_trades.groupby("result").agg(
        count=("result", "size"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_abs", "sum"),
    ).to_dict("index")

    # Equity curve stats
    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - cfg.initial_capital) / cfg.initial_capital * 100

    # Max drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100
    max_dd = drawdown.min()

    # Sharpe (annualized, assuming 15m bars)
    returns = equity_curve.pct_change().dropna()
    if returns.std() > 0:
        bars_per_year = 4 * 24 * 365  # 15m bars
        sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    else:
        sharpe = 0

    # Profit factor
    gross_profit = wins["pnl_abs"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_abs"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Consecutive wins/losses
    results_seq = (df_trades["pnl_abs"] > 0).astype(int).values
    max_consec_wins = 0
    max_consec_losses = 0
    cw = cl = 0
    for r in results_seq:
        if r == 1:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_consec_wins = max(max_consec_wins, cw)
        max_consec_losses = max(max_consec_losses, cl)

    # Average hold time
    df_trades["hold_time"] = pd.to_datetime(df_trades["exit_time"]) - pd.to_datetime(df_trades["entry_time"])
    avg_hold = df_trades["hold_time"].mean()

    # Monthly returns
    df_trades["month"] = pd.to_datetime(df_trades["entry_time"]).dt.to_period("M")
    monthly = df_trades.groupby("month")["pnl_abs"].sum()

    # Yearly returns
    df_trades["year"] = pd.to_datetime(df_trades["entry_time"]).dt.year
    yearly = df_trades.groupby("year")["pnl_abs"].sum()

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
        "total_return_pct": total_return,
        "final_equity": final_equity,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "avg_hold_time": str(avg_hold),
        "tier_stats": tier_stats,
        "dir_stats": dir_stats,
        "result_stats": result_stats,
        "monthly_pnl": monthly,
        "yearly_pnl": yearly,
        "trades_df": df_trades,
    }


def print_report(stats: dict, cfg: Config):
    print("\n" + "=" * 70)
    print("   BACKTEST REPORT: Mean Reversion Bounce Strategy")
    print("   BTC/USDT 15m | {} - present".format("2021-01"))
    print("=" * 70)

    print(f"\n{'─' * 40}")
    print(f"  CONFIGURATION")
    print(f"{'─' * 40}")
    print(f"  Trend:          {cfg.trend_type} {cfg.trend_length} ({cfg.trend_tf})")
    print(f"  Dev MA:         {cfg.dev_ma_type} {cfg.dev_ma_length}")
    print(f"  Tiers:          {cfg.dev_mult1}x / {cfg.dev_mult2}x / {cfg.dev_mult3}x")
    print(f"  Liquidity TFs:  {cfg.liq_tf1} / {cfg.liq_tf2}")
    print(f"  Commission:     {cfg.commission_pct}% per side")
    print(f"  Initial Capital: ${cfg.initial_capital:,.0f}")
    print(f"  Risk per Trade: {cfg.risk_per_trade_pct}%")

    print(f"\n{'─' * 40}")
    print(f"  OVERALL PERFORMANCE")
    print(f"{'─' * 40}")
    print(f"  Total Trades:       {stats['total_trades']}")
    print(f"  Win Rate:           {stats['win_rate']:.1f}%")
    print(f"  Profit Factor:      {stats['profit_factor']:.2f}")
    print(f"  Avg Win:            {stats['avg_win_pct']:.2f}%")
    print(f"  Avg Loss:           {stats['avg_loss_pct']:.2f}%")
    print(f"  Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:       {stats['max_drawdown_pct']:.2f}%")
    print(f"  Total Return:       {stats['total_return_pct']:.2f}%")
    print(f"  Final Equity:       ${stats['final_equity']:,.2f}")
    print(f"  Max Consec Wins:    {stats['max_consec_wins']}")
    print(f"  Max Consec Losses:  {stats['max_consec_losses']}")
    print(f"  Avg Hold Time:      {stats['avg_hold_time']}")

    print(f"\n{'─' * 40}")
    print(f"  BY TIER")
    print(f"{'─' * 40}")
    for tier, ts in stats["tier_stats"].items():
        print(f"  Tier {tier}: {ts['count']} trades | WR {ts['win_rate']:.1f}% | "
              f"Avg {ts['avg_pnl']:.2f}% | Total ${ts['total_pnl']:,.2f}")

    print(f"\n{'─' * 40}")
    print(f"  BY DIRECTION")
    print(f"{'─' * 40}")
    for d, ds in stats["dir_stats"].items():
        print(f"  {d.upper()}: {ds['count']} trades | WR {ds['win_rate']:.1f}% | "
              f"Avg {ds['avg_pnl']:.2f}% | Total ${ds['total_pnl']:,.2f}")

    print(f"\n{'─' * 40}")
    print(f"  BY EXIT TYPE")
    print(f"{'─' * 40}")
    for res, rs in stats["result_stats"].items():
        print(f"  {res.upper()}: {rs['count']} trades | Avg {rs['avg_pnl']:.2f}% | Total ${rs['total_pnl']:,.2f}")

    print(f"\n{'─' * 40}")
    print(f"  YEARLY P&L")
    print(f"{'─' * 40}")
    for period, pnl in stats["yearly_pnl"].items():
        print(f"  {period}: ${pnl:,.2f}")

    print("\n" + "=" * 70 + "\n")


def plot_results(equity_curve: pd.Series, stats: dict, output_dir: str):
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle("Mean Reversion Bounce — BTC/USDT 15m Backtest", fontsize=16, fontweight="bold")

    # 1) Equity curve
    ax = axes[0, 0]
    eq_daily = equity_curve.resample("1D").last().dropna()
    ax.plot(eq_daily.index, eq_daily.values, color="#00e676", linewidth=1.2)
    ax.fill_between(eq_daily.index, eq_daily.values, eq_daily.values.min(), alpha=0.15, color="#00e676")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    # 2) Drawdown
    ax = axes[0, 1]
    peak = eq_daily.expanding().max()
    dd = (eq_daily - peak) / peak * 100
    ax.fill_between(dd.index, dd.values, 0, color="#ff1744", alpha=0.5)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    # 3) Monthly P&L
    ax = axes[1, 0]
    monthly = stats["monthly_pnl"]
    colors = ["#00e676" if v > 0 else "#ff1744" for v in monthly.values]
    monthly_idx = [str(p) for p in monthly.index]
    # Show every Nth label
    n = max(len(monthly_idx) // 20, 1)
    ax.bar(range(len(monthly_idx)), monthly.values, color=colors, width=0.8)
    ax.set_xticks(range(0, len(monthly_idx), n))
    ax.set_xticklabels([monthly_idx[i] for i in range(0, len(monthly_idx), n)], rotation=45, ha="right")
    ax.set_title("Monthly P&L ($)")
    ax.set_ylabel("P&L ($)")
    ax.grid(True, alpha=0.3)

    # 4) Win Rate by Tier
    ax = axes[1, 1]
    tiers = list(stats["tier_stats"].keys())
    wr = [stats["tier_stats"][t]["win_rate"] for t in tiers]
    counts = [stats["tier_stats"][t]["count"] for t in tiers]
    tier_colors = ["#42a5f5", "#ab47bc", "#ffa726"]
    bars = ax.bar([f"Tier {t}\n({c} trades)" for t, c in zip(tiers, counts)], wr,
                  color=tier_colors[:len(tiers)])
    ax.set_title("Win Rate by Tier")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    for bar, w in zip(bars, wr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{w:.1f}%", ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 5) Trade P&L distribution
    ax = axes[2, 0]
    pnls = stats["trades_df"]["pnl_pct"].values
    ax.hist(pnls, bins=50, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="white", linestyle="--", linewidth=1)
    ax.set_title("Trade P&L Distribution (%)")
    ax.set_xlabel("P&L %")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # 6) Cumulative trades P&L
    ax = axes[2, 1]
    cum_pnl = stats["trades_df"]["pnl_abs"].cumsum()
    ax.plot(range(len(cum_pnl)), cum_pnl.values, color="#ab47bc", linewidth=1.2)
    ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0, alpha=0.15, color="#ab47bc")
    ax.set_title("Cumulative Trade P&L ($)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "backtest_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Charts saved to {path}")
    return path


def save_trades_csv(stats: dict, output_dir: str):
    path = os.path.join(output_dir, "backtest_trades.csv")
    cols = ["entry_time", "exit_time", "direction", "tier", "entry_price",
            "tp_price", "sl_price", "exit_price", "result", "pnl_pct", "pnl_abs", "position_size"]
    stats["trades_df"][cols].to_csv(path, index=False)
    print(f"Trades saved to {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = Config()

    print("Loading BTC/USDT 15m data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    trades, equity_curve = run_backtest(df, cfg)
    print(f"Completed: {len(trades)} trades executed")

    if not trades:
        print("No trades generated. Check parameters.")
        return

    stats = analyze_trades(trades, equity_curve, cfg)
    print_report(stats, cfg)

    plot_results(equity_curve, stats, data_dir)
    save_trades_csv(stats, data_dir)


if __name__ == "__main__":
    main()
