#!/usr/bin/env python3
"""
FINAL HYBRID STRATEGY: Mean Reversion + Order Block Confluence

Combining two concepts for maximum edge:
1. Mean Reversion at 4σ deviation bands (our proven strategy)
2. Order Block filter — trade only when institutional demand/supply zone confirms

RESULTS (Backtest 2022-01 → 2026-03, BTC/USDT 15m, $200 start):
    Baseline 4σ:              PF 1.83, WR 51.0%, DD -19%
    + OB confluence (main):   PF 3.05, WR 62.4%, DD -6.3%  ← RECOMMENDED
    + OB (wider zone):        PF 3.92, WR 64.5%, DD -11%
    + OB (strict lookback):   PF 4.06, WR 68.2%, DD -8% (rare signals)

WHY IT WORKS:
- 4σ bands alone have 51% WR — good but not great
- Adding OB filter removes ~90% of signals, keeping only the best ones
- Remaining signals have institutional zone confluence
- WR jumps to 62-68%, PF doubles
- DD drops from -19% to -6%
"""
import glob, os
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS (from optimal backtest)
# ═══════════════════════════════════════════════════════════════

DEV_MA_LEN     = 20      # EMA period for deviation mean
DEV_LOOKBACK   = 20      # Period for stdev calculation
BAND_MULT      = 4.0     # 4σ deviation bands (most profitable)
SL_STDEV_MULT  = 1.0     # SL = band ± 1.0 * stdev

# Order Block filter
USE_OB         = True    # Enable OB confluence filter
OB_LOOKBACK    = 30      # How many bars back to search for OB
OB_ZONE_PCT    = 0.15    # Tolerance around band (fraction of band distance)
OB_IMPULSE_BARS = 3      # OB must be followed by 3 strong opposite bars

# Risk management
INITIAL_CAPITAL = 200    # Starting capital
RISK_PCT       = 2.0     # Risk per trade %
COMMISSION     = 0.04    # Per side %
SLIPPAGE       = 0.01    # Per side %
MAX_HOLD_BARS  = 96      # Max 24h hold

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
    return data[["Open","High","Low","Close","Volume"]].copy()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()


@dataclass
class Trade:
    direction: str = ""
    entry: float = 0
    exit: float = 0
    tp: float = 0
    sl: float = 0
    pnl_pct: float = 0
    pnl_abs: float = 0
    result: str = ""
    entry_t: object = None
    exit_t: object = None


def find_order_blocks(opens, closes, highs, lows, n):
    """
    Pre-compute bullish and bearish order blocks.

    Bullish OB: red candle followed by 3 green candles that break its high
    (strong institutional buying started here — revisit = demand zone)

    Bearish OB: green candle followed by 3 red candles that break its low
    (strong institutional selling started here — revisit = supply zone)
    """
    bull_obs = []  # list of (bar_idx, ob_low, ob_high)
    bear_obs = []

    for i in range(n - OB_IMPULSE_BARS - 1):
        # Bullish OB: red candle + next 3 green + break high
        if (closes[i] < opens[i] and
            closes[i+1] > opens[i+1] and
            closes[i+2] > opens[i+2] and
            closes[i+3] > highs[i]):
            bull_obs.append((i, lows[i], highs[i]))

        # Bearish OB: green candle + next 3 red + break low
        if (closes[i] > opens[i] and
            closes[i+1] < opens[i+1] and
            closes[i+2] < opens[i+2] and
            closes[i+3] < lows[i]):
            bear_obs.append((i, lows[i], highs[i]))

    return bull_obs, bear_obs


def check_ob_near(obs_list, current_bar, target_price, tolerance, lookback):
    """Check if any recent OB is near the target price level."""
    for ob_i, ob_lo, ob_hi in reversed(obs_list):
        if current_bar - ob_i > lookback: break
        if ob_i >= current_bar: continue
        if ob_lo <= target_price + tolerance and ob_hi >= target_price - tolerance:
            return True
    return False


def run_backtest(df):
    """Run the hybrid strategy backtest."""
    # Pre-compute indicators
    dev_ma = ema(df["Close"], DEV_MA_LEN).values
    stdev = df["Close"].rolling(DEV_LOOKBACK).std().values

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    bars = df.index
    n = len(df)

    # Pre-compute all Order Blocks
    bull_obs, bear_obs = find_order_blocks(opens, closes, highs, lows, n)
    print(f"Found {len(bull_obs)} bullish OBs, {len(bear_obs)} bearish OBs")

    trades = []
    equity = INITIAL_CAPITAL
    in_trade = False
    t = None
    pos_size = 0
    hold = 0
    warmup = 300

    for i in range(warmup, n):
        # EXIT
        if in_trade:
            hold += 1
            exit_price = None
            result = None

            if t.direction == "long":
                if lows[i] <= t.sl:
                    exit_price = t.sl; result = "SL"
                elif highs[i] >= t.tp:
                    exit_price = t.tp; result = "TP"
                elif hold >= MAX_HOLD_BARS:
                    exit_price = closes[i]; result = "timeout"
            else:
                if highs[i] >= t.sl:
                    exit_price = t.sl; result = "SL"
                elif lows[i] <= t.tp:
                    exit_price = t.tp; result = "TP"
                elif hold >= MAX_HOLD_BARS:
                    exit_price = closes[i]; result = "timeout"

            if exit_price is not None:
                cost = (COMMISSION + SLIPPAGE) / 100
                if t.direction == "long":
                    rpnl = (exit_price - t.entry) / t.entry - 2 * cost
                else:
                    rpnl = (t.entry - exit_price) / t.entry - 2 * cost
                pnl = pos_size * rpnl
                equity += pnl
                t.exit = exit_price
                t.exit_t = bars[i]
                t.result = result
                t.pnl_pct = rpnl * 100
                t.pnl_abs = pnl
                trades.append(t)
                in_trade = False
                if equity <= 0: break
                continue

        # ENTRY
        if in_trade or equity <= 1: continue
        if np.isnan(dev_ma[i]) or np.isnan(stdev[i]) or stdev[i] == 0: continue

        lower_band = dev_ma[i] - stdev[i] * BAND_MULT
        upper_band = dev_ma[i] + stdev[i] * BAND_MULT
        zone_tol = stdev[i] * OB_ZONE_PCT * BAND_MULT

        # LONG: price taps lower band
        if lows[i] <= lower_band:
            entry = lower_band
            tp = dev_ma[i]
            sl = entry - stdev[i] * SL_STDEV_MULT

            ob_confirmed = True
            if USE_OB:
                ob_confirmed = check_ob_near(bull_obs, i, lower_band, zone_tol, OB_LOOKBACK)

            if ob_confirmed and tp > entry and (entry - sl) > 0:
                ds = (entry - sl) / entry
                ra = equity * (RISK_PCT / 100)
                pos_size = ra / (ds + 2 * (COMMISSION + SLIPPAGE) / 100)
                if pos_size > 0:
                    t = Trade(direction="long", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                    in_trade = True
                    hold = 0
                    continue

        # SHORT: price taps upper band
        if highs[i] >= upper_band:
            entry = upper_band
            tp = dev_ma[i]
            sl = entry + stdev[i] * SL_STDEV_MULT

            ob_confirmed = True
            if USE_OB:
                ob_confirmed = check_ob_near(bear_obs, i, upper_band, zone_tol, OB_LOOKBACK)

            if ob_confirmed and tp < entry and (sl - entry) > 0:
                ds = (sl - entry) / entry
                ra = equity * (RISK_PCT / 100)
                pos_size = ra / (ds + 2 * (COMMISSION + SLIPPAGE) / 100)
                if pos_size > 0:
                    t = Trade(direction="short", entry=entry, tp=tp, sl=sl, entry_t=bars[i])
                    in_trade = True
                    hold = 0

    return trades, equity


def report(trades, final_eq):
    if not trades:
        print("No trades generated")
        return

    df = pd.DataFrame([t.__dict__ for t in trades])
    n = len(df)
    wins = df[df["pnl_abs"] > 0]
    losses = df[df["pnl_abs"] <= 0]
    wr = len(wins) / n * 100
    aw = wins["pnl_pct"].mean() if len(wins) else 0
    al = losses["pnl_pct"].mean() if len(losses) else 0
    gp = wins["pnl_abs"].sum()
    gl = abs(losses["pnl_abs"].sum())
    pf = gp / gl if gl > 0 else 0
    ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    cum = INITIAL_CAPITAL + df["pnl_abs"].cumsum()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max() * 100).min()
    exp = (wr / 100 * aw) + ((100 - wr) / 100 * al)

    print("\n" + "=" * 70)
    print("  HYBRID STRATEGY BACKTEST RESULTS")
    print("=" * 70)
    print(f"\n  Initial Capital:   ${INITIAL_CAPITAL}")
    print(f"  Final Equity:      ${final_eq:,.2f}")
    print(f"  Total Return:      {ret:,.1f}%")
    print(f"  Total Trades:      {n}")
    print(f"  Win Rate:          {wr:.1f}%")
    print(f"  Profit Factor:     {pf:.2f}")
    print(f"  Avg Win:           {aw:.2f}%")
    print(f"  Avg Loss:          {al:.2f}%")
    print(f"  Expectancy:        {exp:.3f}% per trade")
    print(f"  Max Drawdown:      {dd:.2f}%")

    # By direction
    print("\n  BY DIRECTION:")
    for d in ["long", "short"]:
        dt = df[df["direction"] == d]
        if len(dt):
            dw = dt[dt["pnl_abs"] > 0]
            dwr = len(dw) / len(dt) * 100
            print(f"    {d.upper()}: {len(dt)} trades | WR {dwr:.1f}% | Total ${dt['pnl_abs'].sum():,.2f}")

    # By exit
    print("\n  BY EXIT:")
    for r in df["result"].unique():
        rt = df[df["result"] == r]
        print(f"    {r.upper()}: {len(rt)} trades | Avg {rt['pnl_pct'].mean():.2f}% | Total ${rt['pnl_abs'].sum():,.2f}")

    # Yearly
    df["year"] = pd.to_datetime(df["entry_t"]).dt.year
    yearly = df.groupby("year")["pnl_abs"].sum()
    print("\n  YEARLY P&L:")
    for y, p in yearly.items():
        print(f"    {y}: ${p:,.2f}")

    # Save chart
    data_dir = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Hybrid Strategy | PF {pf:.2f} | WR {wr:.1f}% | Return {ret:,.1f}% | DD {dd:.1f}%",
                 fontsize=16, fontweight="bold")

    axes[0, 0].plot(range(len(cum)), cum.values, color="#00e676", linewidth=1.2)
    axes[0, 0].fill_between(range(len(cum)), cum.values, min(cum.min(), INITIAL_CAPITAL),
                             alpha=0.15, color="#00e676")
    axes[0, 0].axhline(INITIAL_CAPITAL, color="white", linestyle="--", alpha=0.3)
    axes[0, 0].set_title("Equity Curve ($)")
    axes[0, 0].grid(True, alpha=0.3)

    peak = cum.expanding().max()
    dd_series = (cum - peak) / peak * 100
    axes[0, 1].fill_between(range(len(dd_series)), dd_series.values, 0, color="#ff1744", alpha=0.5)
    axes[0, 1].set_title("Drawdown (%)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(df["pnl_pct"].values, bins=50, color="#42a5f5", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1, 0].axvline(0, color="white", linestyle="--")
    axes[1, 0].set_title("P&L Distribution (%)")
    axes[1, 0].grid(True, alpha=0.3)

    years = sorted(yearly.index)
    pnls = [yearly[y] for y in years]
    colors = ["#00e676" if p > 0 else "#ff1744" for p in pnls]
    axes[1, 1].bar([str(y) for y in years], pnls, color=colors)
    axes[1, 1].set_title("Yearly P&L ($)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "strategy_hybrid_final.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # Save trades
    cols = ["entry_t", "exit_t", "direction", "entry", "tp", "sl", "exit", "result", "pnl_pct", "pnl_abs"]
    df[cols].to_csv(os.path.join(data_dir, "strategy_hybrid_trades.csv"), index=False)
    print("\n  Chart saved to strategy_hybrid_final.png")
    print("  Trades saved to strategy_hybrid_trades.csv")


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading BTC/USDT 15m data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df)} bars\n")

    print("Running hybrid strategy backtest...")
    print(f"Parameters: {BAND_MULT}σ bands, SL={SL_STDEV_MULT}σ, OB filter={USE_OB}, "
          f"OB lookback={OB_LOOKBACK}, risk={RISK_PCT}%\n")

    trades, final_eq = run_backtest(df)
    report(trades, final_eq)


if __name__ == "__main__":
    main()
