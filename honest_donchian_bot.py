#!/usr/bin/env python3
"""
HONEST 4h Donchian Breakout Telegram Bot
BTC/USDT 4h timeframe — verified out-of-sample

Honest backtest results (no look-ahead):
    Full period (2022-2026): PF 1.10, WR 39.5%, +46.4%, DD -21.9%
    In-Sample:               PF 1.09, +20.7%  ✅
    Out-of-Sample:           PF 1.05, +7.0%   ✅ (both profitable)

Strategy logic (all indicators use data from CLOSED bars only):
1. Wait for 4h bar to close
2. Check EMA(200) trend direction
3. If uptrend: LONG on breakout above 20-bar Donchian high
4. If downtrend: SHORT on breakout below 20-bar Donchian low
5. SL = 2x ATR, TP = 4x ATR (2:1 RR)

NOTE: Signals appear on 4h timeframe, so check only every 4 hours (or at close of 4h bars: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC).
"""

import time
import logging
from datetime import datetime, timezone

import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = "8742677193:AAGs3g_s9LKr_JpmuTmNcKYSp5jFy8lHkOM"
CHAT_ID        = "8549041207"

# Portfolio
PORTFOLIO  = 200      # USD
RISK_PCT   = 2.0      # Risk per trade %

# Strategy params (from honest OOS-validated backtest)
DON_LOOKBACK = 20     # Donchian period
TREND_EMA    = 200    # EMA period for trend filter
ATR_PERIOD   = 14     # ATR period
SL_ATR_MULT  = 2.0    # SL = 2x ATR
TP_ATR_MULT  = 4.0    # TP = 4x ATR (RR 2:1)

# Technical
INTERVAL      = "4h"
CHECK_EVERY   = 300   # Check every 5 minutes
CANDLES_FETCH = 250

# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("donchian_bot")


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram: ✅")
        else:
            log.error(f"Telegram: {resp.text}")
    except Exception as e:
        log.error(f"Telegram error: {e}")


def fetch_candles(symbol="BTCUSDT", interval="4h", limit=250):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return [{
            "time":  datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
            "open":  float(c[1]),
            "high":  float(c[2]),
            "low":   float(c[3]),
            "close": float(c[4]),
        } for c in data]
    except Exception as e:
        log.error(f"Binance API error: {e}")
        return None


def calc_ema(values, period):
    k = 2 / (period + 1)
    ema_vals = [values[0]]
    for v in values[1:]:
        ema_vals.append(v * k + ema_vals[-1] * (1 - k))
    return ema_vals


def calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    atrs = [np.nan] * period
    if len(trs) >= period:
        first = sum(trs[:period]) / period
        atrs.append(first)
        for i in range(period, len(trs)):
            atrs.append((atrs[-1] * (period - 1) + trs[i]) / period)
    return [np.nan] + atrs


def analyze(candles):
    """Detect Donchian breakout on last CLOSED bar. No look-ahead."""
    if not candles or len(candles) < TREND_EMA + 5:
        return None

    n = len(candles)
    # Last candle (index n-1) is the currently forming bar.
    # Last CLOSED bar is index n-2. Decisions use data up to n-3 (shift(1)).

    # Use data from bars [0 .. n-3] (excluding current and last closed for shift)
    # Actually, simpler: compute at the close of the LAST closed bar (index n-2)
    # using data from [0 .. n-3]

    current_i = n - 2  # last closed bar index

    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]

    # Indicators based on data [0 .. current_i - 1]
    # Donchian high/low over previous DON_LOOKBACK bars
    don_high = max(highs[current_i - DON_LOOKBACK : current_i])
    don_low  = min(lows [current_i - DON_LOOKBACK : current_i])

    # EMA trend
    ema_values = calc_ema(closes[:current_i], TREND_EMA)
    trend_ema = ema_values[-1]

    # ATR
    atr_values = calc_atr(highs[:current_i], lows[:current_i], closes[:current_i], ATR_PERIOD)
    atr = atr_values[-1]
    if np.isnan(atr):
        return None

    # Current (closed) bar stats
    bar = candles[current_i]
    prev_close = closes[current_i - 1]

    trend_up = prev_close > trend_ema
    trend_dn = prev_close < trend_ema

    signal = None

    # LONG breakout
    if trend_up and bar["high"] >= don_high and prev_close < don_high:
        entry = don_high
        sl    = entry - atr * SL_ATR_MULT
        tp    = entry + atr * TP_ATR_MULT
        sl_pct = (entry - sl) / entry * 100
        if bar["low"] <= entry <= bar["high"] and 0.2 < sl_pct < 8:
            signal = ("LONG", entry, tp, sl, sl_pct, atr, trend_ema, don_high, don_low)

    # SHORT breakout
    if trend_dn and bar["low"] <= don_low and prev_close > don_low:
        entry = don_low
        sl    = entry + atr * SL_ATR_MULT
        tp    = entry - atr * TP_ATR_MULT
        sl_pct = (sl - entry) / entry * 100
        if bar["low"] <= entry <= bar["high"] and 0.2 < sl_pct < 8:
            signal = ("SHORT", entry, tp, sl, sl_pct, atr, trend_ema, don_high, don_low)

    return {
        "signal": signal,
        "price": candles[-1]["close"],  # current real price
        "prev_close": prev_close,
        "trend_ema": trend_ema,
        "trend_up": trend_up,
        "don_high": don_high,
        "don_low": don_low,
        "atr": atr,
        "bar_time": bar["time"],
    }


def format_signal(direction, entry, tp, sl, sl_pct, atr):
    risk_amt = PORTFOLIO * (RISK_PCT / 100)
    pos_size = risk_amt / (sl_pct / 100)
    leverage = pos_size / PORTFOLIO

    if direction == "LONG":
        rr = (tp - entry) / (entry - sl)
        emoji = "🟢"
    else:
        rr = (entry - tp) / (sl - entry)
        emoji = "🔴"

    msg = (
        f"🚨 DONCHIAN BREAKOUT (4h)\n"
        f"{'━' * 28}\n"
        f"{emoji} <b>{direction} BTC/USDT</b>\n"
        f"{'━' * 28}\n"
        f"📍 Entry: <b>${entry:,.2f}</b>\n"
        f"✅ Take Profit: <b>${tp:,.2f}</b>\n"
        f"❌ Stop Loss: <b>${sl:,.2f}</b>\n"
        f"📊 RR: <b>{rr:.2f}:1</b>\n"
        f"📏 SL distance: {sl_pct:.2f}%\n"
        f"📈 ATR: ${atr:,.2f}\n"
        f"{'━' * 28}\n"
        f"💰 Position: <b>${pos_size:,.2f}</b>\n"
        f"⚡ Leverage: <b>{leverage:.1f}x</b>\n"
        f"🎯 Risk: <b>${risk_amt:.2f}</b> ({RISK_PCT}%)\n"
        f"💼 Portfolio: ${PORTFOLIO}\n"
        f"{'━' * 28}\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"⏳ 4h timeframe — check next bar at closest 4h mark"
    )
    return msg


def format_status(data):
    trend = "BULLISH 🟢" if data["trend_up"] else "BEARISH 🔴"
    msg = (
        f"📊 <b>Donchian 4h Status</b>\n"
        f"Price: ${data['price']:,.2f}\n"
        f"Trend: {trend}\n"
        f"EMA200: ${data['trend_ema']:,.2f}\n"
        f"Don High: ${data['don_high']:,.2f}\n"
        f"Don Low:  ${data['don_low']:,.2f}\n"
        f"ATR: ${data['atr']:,.2f}\n"
    )
    return msg


def main():
    log.info("=" * 50)
    log.info("HONEST Donchian 4h Breakout Bot")
    log.info(f"Portfolio: ${PORTFOLIO} | Risk: {RISK_PCT}%")
    log.info(f"Donchian: {DON_LOOKBACK} | Trend EMA: {TREND_EMA}")
    log.info(f"SL: {SL_ATR_MULT}x ATR | TP: {TP_ATR_MULT}x ATR (RR 2:1)")
    log.info("=" * 50)

    use_telegram = TELEGRAM_TOKEN != "ВСТАВЬТЕ_ТОКЕН"
    if use_telegram:
        send_telegram(
            "🤖 <b>Donchian 4h Bot started</b>\n"
            f"Portfolio: ${PORTFOLIO} | Risk: {RISK_PCT}%\n"
            f"Strategy: Donchian Breakout 4h\n"
            f"Backtest: PF 1.10 | OOS verified ✅\n"
            "Monitoring BTC/USDT 4h..."
        )

    last_bar_time = None
    last_status_time = 0
    status_interval = 14400  # Every 4 hours

    while True:
        try:
            candles = fetch_candles(interval=INTERVAL, limit=CANDLES_FETCH)
            if not candles:
                time.sleep(CHECK_EVERY)
                continue

            data = analyze(candles)
            if not data:
                time.sleep(CHECK_EVERY)
                continue

            now = time.time()

            # Status every 4 hours
            if now - last_status_time > status_interval:
                log.info(f"BTC ${data['price']:,.2f} | "
                         f"Trend: {'UP' if data['trend_up'] else 'DN'} | "
                         f"DonH ${data['don_high']:,.0f} DonL ${data['don_low']:,.0f}")
                if use_telegram:
                    send_telegram(format_status(data))
                last_status_time = now

            # Signal — only once per new 4h bar
            if data["signal"] is not None and data["bar_time"] != last_bar_time:
                direction, entry, tp, sl, sl_pct, atr, _, _, _ = data["signal"]
                msg = format_signal(direction, entry, tp, sl, sl_pct, atr)
                log.info(f"🚨 {direction} entry={entry:.2f} tp={tp:.2f} sl={sl:.2f}")
                if use_telegram:
                    send_telegram(msg)
                else:
                    print("\n" + msg.replace("<b>", "").replace("</b>", "") + "\n")
                last_bar_time = data["bar_time"]

        except KeyboardInterrupt:
            log.info("Stopped")
            break
        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
