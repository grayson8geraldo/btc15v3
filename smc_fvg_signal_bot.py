#!/usr/bin/env python3
"""
SMC FVG + Discount/Premium Telegram Signal Bot
BTC/USDT 15m

Backtest results (2022-2026):
- PF 1.58 | WR 72% | DD -14.9%
- $200 → $36,100 over 4 years
- Out-of-sample PF 1.56 (vs in-sample 1.71)
- All 9 walk-forward halves profitable

Запуск:
    python3 smc_fvg_signal_bot.py

Перед запуском заполните TELEGRAM_TOKEN и CHAT_ID.
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

# Портфель
PORTFOLIO   = 200      # Депозит в $
RISK_PCT    = 2.0      # Риск на сделку %
MAX_SL_PCT  = 5.0      # Максимальная дистанция SL %

# Стратегия (оптимальные параметры из бэктеста)
RANGE_LOOKBACK   = 16    # 16 баров = 4h на 15m
ZONE_THRESHOLD   = 0.5   # 50% split
MIN_FVG_PCT      = 0.05  # Минимальный размер FVG в %

# Технические
CHECK_INTERVAL   = 60    # секунд
CANDLES_FETCH    = 100

# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("fvg_bot")


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram: ✅ sent")
        else:
            log.error(f"Telegram error: {resp.text}")
    except Exception as e:
        log.error(f"Telegram error: {e}")


def fetch_candles(symbol="BTCUSDT", interval="15m", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        candles = []
        for c in data:
            candles.append({
                "time": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                "open":  float(c[1]),
                "high":  float(c[2]),
                "low":   float(c[3]),
                "close": float(c[4]),
                "volume":float(c[5]),
            })
        return candles
    except Exception as e:
        log.error(f"Binance API error: {e}")
        return None


def analyze(candles):
    """Detect FVG signals in Discount/Premium zones."""
    if not candles or len(candles) < RANGE_LOOKBACK + 5:
        return None

    # Use LAST CLOSED candle as i (current bar = candles[-1] still open)
    # i = -2 (last closed), i-1 = -3, i-2 = -4
    n = len(candles)
    i = n - 2  # index of last closed candle

    # Dealing range from previous N closed bars (excluding the current one)
    range_bars = candles[i - RANGE_LOOKBACK : i]
    range_high = max(b["high"] for b in range_bars)
    range_low  = min(b["low"]  for b in range_bars)
    range_size = range_high - range_low
    if range_size <= 0:
        return None

    eq50         = (range_high + range_low) / 2
    discount_top = range_low + range_size * ZONE_THRESHOLD
    premium_bot  = range_high - range_size * ZONE_THRESHOLD

    last = candles[i]
    prev2 = candles[i - 2]  # i-2 for FVG
    prev1 = candles[i - 1]  # middle bar

    in_discount = last["close"] < discount_top
    in_premium  = last["close"] > premium_bot

    # FVG detection
    bull_fvg = last["low"] > prev2["high"]
    bear_fvg = last["high"] < prev2["low"]

    bull_fvg_size = ((last["low"] - prev2["high"]) / prev2["high"] * 100) if bull_fvg else 0
    bear_fvg_size = ((prev2["low"] - last["high"]) / prev2["low"] * 100) if bear_fvg else 0

    valid_bull = bull_fvg and bull_fvg_size >= MIN_FVG_PCT
    valid_bear = bear_fvg and bear_fvg_size >= MIN_FVG_PCT

    signal = None

    # LONG: bullish FVG in Discount
    if valid_bull and in_discount:
        entry = (last["low"] + prev2["high"]) / 2  # FVG midpoint
        sl    = min(prev2["low"], prev1["low"]) * 0.999
        tp    = eq50
        sl_pct = (entry - sl) / entry * 100
        if entry > sl and tp > entry and 0.1 < sl_pct < MAX_SL_PCT:
            signal = ("LONG", entry, tp, sl, sl_pct, bull_fvg_size)

    # SHORT: bearish FVG in Premium
    if valid_bear and in_premium and signal is None:
        entry = (last["high"] + prev2["low"]) / 2
        sl    = max(prev2["high"], prev1["high"]) * 1.001
        tp    = eq50
        sl_pct = (sl - entry) / entry * 100
        if entry < sl and tp < entry and 0.1 < sl_pct < MAX_SL_PCT:
            signal = ("SHORT", entry, tp, sl, sl_pct, bear_fvg_size)

    return {
        "signal": signal,
        "price": last["close"],
        "range_high": range_high,
        "range_low": range_low,
        "eq50": eq50,
        "in_discount": in_discount,
        "in_premium": in_premium,
        "candle_time": last["time"],
    }


def format_signal(direction, entry, tp, sl, sl_pct, fvg_size):
    """Format Telegram message with all trade params."""
    risk_amount = PORTFOLIO * (RISK_PCT / 100)
    pos_size = risk_amount / (sl_pct / 100) if sl_pct > 0 else 0
    leverage = pos_size / PORTFOLIO if PORTFOLIO > 0 else 0

    if direction == "LONG":
        rr = (tp - entry) / (entry - sl)
        emoji = "🟢"
        zone = "DISCOUNT"
    else:
        rr = (entry - tp) / (sl - entry)
        emoji = "🔴"
        zone = "PREMIUM"

    msg = (
        f"🚨 SMC FVG SIGNAL\n"
        f"{'━' * 28}\n"
        f"{emoji} <b>{direction} BTC/USDT</b>\n"
        f"Zone: <b>{zone}</b>\n"
        f"FVG size: {fvg_size:.3f}%\n"
        f"{'━' * 28}\n"
        f"📍 Entry: <b>${entry:,.2f}</b>\n"
        f"✅ Take Profit (50% line): <b>${tp:,.2f}</b>\n"
        f"❌ Stop Loss: <b>${sl:,.2f}</b>\n"
        f"📊 RR: <b>{rr:.2f}:1</b>\n"
        f"📏 SL distance: {sl_pct:.2f}%\n"
        f"{'━' * 28}\n"
        f"💰 Position: <b>${pos_size:,.2f}</b>\n"
        f"⚡ Leverage: <b>{leverage:.1f}x</b>\n"
        f"🎯 Risk: <b>${risk_amount:.2f}</b> ({RISK_PCT}%)\n"
        f"💼 Portfolio: ${PORTFOLIO}\n"
        f"{'━' * 28}\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return msg


def format_status(data):
    p = data["price"]
    zone = "DISCOUNT 🟢" if data["in_discount"] else "PREMIUM 🔴" if data["in_premium"] else "Equilibrium ⚪"
    msg = (
        f"📊 <b>SMC FVG Status</b>\n"
        f"Price: ${p:,.2f}\n"
        f"Zone: {zone}\n"
        f"Range High: ${data['range_high']:,.2f}\n"
        f"Range Low: ${data['range_low']:,.2f}\n"
        f"50% (TP target): ${data['eq50']:,.2f}\n"
        f"Range size: {(data['range_high']-data['range_low'])/p*100:.2f}%"
    )
    return msg


def main():
    log.info("=" * 50)
    log.info("SMC FVG + Discount/Premium Bot")
    log.info(f"Portfolio: ${PORTFOLIO} | Risk: {RISK_PCT}%")
    log.info(f"Range: {RANGE_LOOKBACK} bars | Min FVG: {MIN_FVG_PCT}%")
    log.info("=" * 50)

    use_telegram = TELEGRAM_TOKEN != "ВСТАВЬТЕ_ТОКЕН"
    if use_telegram:
        send_telegram(
            "🤖 <b>SMC FVG Bot запущен</b>\n"
            f"Portfolio: ${PORTFOLIO} | Risk: {RISK_PCT}%\n"
            f"Strategy: FVG + Discount/Premium\n"
            f"Range: {RANGE_LOOKBACK} bars (4h)\n"
            "Backtest: PF 1.58 | WR 72%\n"
            "Мониторю BTC/USDT 15m..."
        )

    last_signal_time = 0
    last_status_time = 0
    last_candle_time = None
    status_interval = 3600

    while True:
        try:
            candles = fetch_candles(limit=CANDLES_FETCH)
            if not candles:
                time.sleep(CHECK_INTERVAL)
                continue

            data = analyze(candles)
            if not data:
                time.sleep(CHECK_INTERVAL)
                continue

            now = time.time()

            # Status hourly
            if now - last_status_time > status_interval:
                log.info(f"BTC ${data['price']:,.2f} | "
                         f"Zone: {'DISC' if data['in_discount'] else 'PREM' if data['in_premium'] else 'EQ'} | "
                         f"50%: ${data['eq50']:,.2f}")
                if use_telegram:
                    send_telegram(format_status(data))
                last_status_time = now

            # Signal — only once per candle close
            if data["signal"] is not None and data["candle_time"] != last_candle_time:
                direction, entry, tp, sl, sl_pct, fvg_size = data["signal"]
                msg = format_signal(direction, entry, tp, sl, sl_pct, fvg_size)
                log.info(f"🚨 {direction} signal: entry={entry:.2f} tp={tp:.2f} sl={sl:.2f}")
                if use_telegram:
                    send_telegram(msg)
                else:
                    print("\n" + msg.replace("<b>", "").replace("</b>", "") + "\n")
                last_candle_time = data["candle_time"]
                last_signal_time = now

        except KeyboardInterrupt:
            log.info("Stopped")
            break
        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
