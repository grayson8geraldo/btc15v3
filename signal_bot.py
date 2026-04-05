#!/usr/bin/env python3
"""
Mean Reversion Signal Bot — BTC/USDT 15m
Мониторит цену BTC, отправляет торговые сигналы в Telegram.

Запуск: python3 signal_bot.py

Перед запуском заполните:
1. TELEGRAM_TOKEN — получите у @BotFather в Telegram
2. CHAT_ID — получите у @userinfobot в Telegram
"""

import time
import json
import logging
from datetime import datetime, timezone

import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════
# ─── НАСТРОЙКИ (ЗАПОЛНИТЕ ПЕРЕД ЗАПУСКОМ) ─────────────────────
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = "ВСТАВЬТЕ_ТОКЕН_ОТ_BOTFATHER"
CHAT_ID = "ВСТАВЬТЕ_ВАШ_CHAT_ID"

# Портфель
PORTFOLIO = 200       # Ваш депозит в $
RISK_PCT = 2.0        # Риск на сделку в %

# Стратегия (оптимальные параметры из бэктеста)
DEV_MA_LEN = 20       # Период EMA для средней
DEV_LOOKBACK = 20     # Период для стандартного отклонения
TIER3_MULT = 4.0      # Множитель Tier 3 (лучший PF 2.04)
SL_STDEV_MULT = 1.0   # SL = band - 1.0 * stdev
COOLDOWN_BARS = 2     # Минимум баров между сигналами

# Технические
CHECK_INTERVAL = 60   # Проверять каждые 60 секунд
CANDLES_TO_FETCH = 100  # Кол-во свечей для расчётов

# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("bot")


def send_telegram(text):
    """Отправить сообщение в Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram: сообщение отправлено")
        else:
            log.error(f"Telegram error: {resp.text}")
    except Exception as e:
        log.error(f"Telegram error: {e}")


def fetch_candles(symbol="BTCUSDT", interval="15m", limit=100):
    """Получить свечи с Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        candles = []
        for c in data:
            candles.append({
                "time": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return candles
    except Exception as e:
        log.error(f"Binance API error: {e}")
        return None


def calc_ema(values, period):
    """EMA расчёт."""
    ema = [values[0]]
    k = 2 / (period + 1)
    for i in range(1, len(values)):
        ema.append(values[i] * k + ema[-1] * (1 - k))
    return ema


def calc_stdev(values, period):
    """Rolling standard deviation."""
    result = [np.nan] * (period - 1)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        result.append(np.std(window, ddof=1))
    return result


def analyze(candles):
    """Анализ свечей и генерация сигналов."""
    if not candles or len(candles) < DEV_MA_LEN + 10:
        return None

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    # EMA (deviation mean)
    dev_ma = calc_ema(closes, DEV_MA_LEN)

    # Standard deviation
    stdev = calc_stdev(closes, DEV_LOOKBACK)

    i = len(candles) - 1  # Текущая (незакрытая) свеча
    j = len(candles) - 2  # Последняя закрытая свеча

    if np.isnan(stdev[j]) or stdev[j] == 0:
        return None

    # Bands на последней закрытой свече
    upper_band = dev_ma[j] + stdev[j] * TIER3_MULT
    lower_band = dev_ma[j] - stdev[j] * TIER3_MULT
    mean = dev_ma[j]

    # Текущие уровни (для алерта "approaching")
    upper_band_now = dev_ma[i] + stdev[j] * TIER3_MULT
    lower_band_now = dev_ma[i] - stdev[j] * TIER3_MULT
    mean_now = dev_ma[i]

    current_price = closes[i]
    current_low = lows[i]
    current_high = highs[i]

    signals = []

    # ── Проверка сигнала на ЗАКРЫТОЙ свече (j) ──────────────
    # LONG: low коснулся нижней банды, close выше банды, предыдущая свеча не касалась
    prev_lower = dev_ma[j - 1] + stdev[j - 1] * (-TIER3_MULT) if not np.isnan(stdev[j - 1]) else lower_band
    long_signal = (lows[j] <= lower_band and
                   closes[j] > lower_band and
                   lows[j - 1] > prev_lower)

    if long_signal:
        entry = lower_band
        tp = mean
        sl = entry - stdev[j] * SL_STDEV_MULT
        signals.append(("LONG", entry, tp, sl))

    # SHORT: high коснулся верхней банды, close ниже банды
    prev_upper = dev_ma[j - 1] + stdev[j - 1] * TIER3_MULT if not np.isnan(stdev[j - 1]) else upper_band
    short_signal = (highs[j] >= upper_band and
                    closes[j] < upper_band and
                    highs[j - 1] < prev_upper)

    if short_signal:
        entry = upper_band
        tp = mean
        sl = entry + stdev[j] * SL_STDEV_MULT
        signals.append(("SHORT", entry, tp, sl))

    # ── Проверка "APPROACHING" (цена подходит к банде) ──────
    dist_to_lower = (current_price - lower_band_now) / current_price * 100
    dist_to_upper = (upper_band_now - current_price) / current_price * 100

    approaching = None
    if 0 < dist_to_lower < 0.3:
        approaching = ("LONG", lower_band_now, mean_now,
                       lower_band_now - stdev[j] * SL_STDEV_MULT, dist_to_lower)
    elif 0 < dist_to_upper < 0.3:
        approaching = ("SHORT", upper_band_now, mean_now,
                       upper_band_now + stdev[j] * SL_STDEV_MULT, dist_to_upper)

    return {
        "signals": signals,
        "approaching": approaching,
        "price": current_price,
        "mean": mean_now,
        "upper": upper_band_now,
        "lower": lower_band_now,
        "stdev": stdev[j],
    }


def format_signal(direction, entry, tp, sl, is_approaching=False, dist=0):
    """Форматировать сообщение сигнала."""
    risk_amount = PORTFOLIO * (RISK_PCT / 100)

    if direction == "LONG":
        dist_sl = (entry - sl) / entry
        dist_tp = (tp - entry) / entry
        rr = dist_tp / dist_sl if dist_sl > 0 else 0
    else:
        dist_sl = (sl - entry) / entry
        dist_tp = (entry - tp) / entry
        rr = dist_tp / dist_sl if dist_sl > 0 else 0

    pos_size = risk_amount / dist_sl if dist_sl > 0 else 0
    leverage = pos_size / PORTFOLIO if PORTFOLIO > 0 else 0

    emoji = "🟢" if direction == "LONG" else "🔴"

    if is_approaching:
        header = f"⚠️ ПРИБЛИЖАЕТСЯ К ЗОНЕ ВХОДА"
        extra = f"\n📏 До банды: {dist:.2f}%"
    else:
        header = f"🚨 СИГНАЛ"
        extra = ""

    msg = (
        f"{header}\n"
        f"{'━' * 28}\n"
        f"{emoji} <b>{direction} BTC/USDT</b> (Tier 3)\n"
        f"{'━' * 28}\n"
        f"📍 Entry (лимит): <b>${entry:,.2f}</b>\n"
        f"✅ Take Profit: <b>${tp:,.2f}</b>\n"
        f"❌ Stop Loss: <b>${sl:,.2f}</b>\n"
        f"📊 RR: <b>{rr:.1f}:1</b>\n"
        f"{'━' * 28}\n"
        f"💰 Позиция: <b>${pos_size:,.2f}</b>\n"
        f"⚡ Плечо: <b>{leverage:.1f}x</b>\n"
        f"🎯 Риск: <b>${risk_amount:.2f}</b> ({RISK_PCT}%)\n"
        f"💼 Депозит: ${PORTFOLIO}"
        f"{extra}\n"
        f"{'━' * 28}\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return msg


def format_status(data):
    """Статус-сообщение."""
    p = data["price"]
    dist_up = (data["upper"] - p) / p * 100
    dist_dn = (p - data["lower"]) / p * 100
    msg = (
        f"📊 <b>BTC Status</b>\n"
        f"Price: ${p:,.2f}\n"
        f"Mean: ${data['mean']:,.2f}\n"
        f"Upper Band (4σ): ${data['upper']:,.2f} ({dist_up:.2f}% away)\n"
        f"Lower Band (4σ): ${data['lower']:,.2f} ({dist_dn:.2f}% away)\n"
        f"Stdev: ${data['stdev']:,.2f}"
    )
    return msg


def main():
    log.info("=" * 50)
    log.info("Mean Reversion Signal Bot запущен")
    log.info(f"Депозит: ${PORTFOLIO} | Риск: {RISK_PCT}%")
    log.info(f"Tier 3: {TIER3_MULT}σ | SL: {SL_STDEV_MULT}σ")
    log.info("=" * 50)

    if TELEGRAM_TOKEN == "ВСТАВЬТЕ_ТОКЕН_ОТ_BOTFATHER":
        log.warning("⚠️  TELEGRAM_TOKEN не настроен!")
        log.info("1. Откройте Telegram, найдите @BotFather")
        log.info("2. Отправьте /newbot, дайте имя")
        log.info("3. Скопируйте токен в TELEGRAM_TOKEN")
        log.info("4. Найдите @userinfobot, скопируйте ID в CHAT_ID")
        log.info("Бот будет работать в консольном режиме без Telegram\n")
        use_telegram = False
    else:
        use_telegram = True
        send_telegram("🤖 <b>Signal Bot запущен</b>\n"
                      f"Депозит: ${PORTFOLIO} | Риск: {RISK_PCT}%\n"
                      f"Tier 3 ({TIER3_MULT}σ) | SL {SL_STDEV_MULT}σ\n"
                      "Мониторю BTC/USDT 15m...")

    last_signal_time = 0
    last_approach_time = 0
    bars_since_signal = 999
    status_interval = 3600  # Статус каждый час
    last_status_time = 0

    while True:
        try:
            candles = fetch_candles(limit=CANDLES_TO_FETCH)
            if not candles:
                time.sleep(CHECK_INTERVAL)
                continue

            result = analyze(candles)
            if not result:
                time.sleep(CHECK_INTERVAL)
                continue

            now = time.time()

            # Статус каждый час
            if now - last_status_time > status_interval:
                status = format_status(result)
                log.info(f"BTC ${result['price']:,.2f} | "
                         f"Mean ${result['mean']:,.2f} | "
                         f"Band ↓${result['lower']:,.2f} ↑${result['upper']:,.2f}")
                if use_telegram:
                    send_telegram(status)
                last_status_time = now

            # Сигнал на закрытой свече
            for direction, entry, tp, sl in result["signals"]:
                if now - last_signal_time > 900:  # Не чаще 15 мин
                    msg = format_signal(direction, entry, tp, sl)
                    log.info(f"🚨 СИГНАЛ: {direction} entry={entry:.2f} tp={tp:.2f} sl={sl:.2f}")
                    if use_telegram:
                        send_telegram(msg)
                    else:
                        print("\n" + msg.replace("<b>", "").replace("</b>", "") + "\n")
                    last_signal_time = now

            # Approaching (цена подходит к банде)
            if result["approaching"] and now - last_approach_time > 600:
                d, entry, tp, sl, dist = result["approaching"]
                msg = format_signal(d, entry, tp, sl, is_approaching=True, dist=dist)
                log.info(f"⚠️ Approaching {d} zone, {dist:.2f}% away")
                if use_telegram:
                    send_telegram(msg)
                last_approach_time = now

        except KeyboardInterrupt:
            log.info("Бот остановлен")
            break
        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
