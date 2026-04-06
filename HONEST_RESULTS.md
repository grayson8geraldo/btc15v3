# 🚨 HONEST BACKTEST RESULTS — LOOK-AHEAD BIAS FOUND

## Critical Finding

All previously "profitable" strategies in this repository contain **look-ahead bias**.
After proper implementation (using only data available at the time of trade decision),
**NONE of the strategies are profitable**.

## What was wrong

### 1. Mean Reversion 4σ Strategy
- `dev_ma[i]` (EMA) and `stdev[i]` were computed INCLUDING `close[i]`
- At the moment price touches a band (during bar i), close[i] is not yet known
- The EMA therefore "knew" the future within the current bar
- This small edge made losing strategy appear profitable

**Fix:** Use `dev_ma = ...ewm(...).shift(1)` and `stdev = ...rolling(...).std().shift(1)`

**Result:** PF drops from 1.83 → **0.82** (unprofitable)

### 2. FVG + Discount/Premium Strategy
- Bullish FVG: `low[i] > high[i-2]` (gap up)
- Entry was computed as `(low[i] + high[i-2]) / 2` — midpoint of the gap
- This midpoint is BELOW low[i], meaning price NEVER touched that level on bar i
- The backtest "filled" at a price that wasn't available

**Fix:** Place pending limit order at midpoint, fill only on future bar retest

**Result:** PF drops from 1.58 → **0.76** (unprofitable)

### 3. Hybrid (4σ + Order Block)
- Inherits the same dev_ma/stdev look-ahead from #1
- Not separately tested but likely has the same issue

## Honest Backtest Results

| Strategy | Claimed PF | Honest PF | Status |
|---|---|---|---|
| Mean Reversion 4σ | 1.83 | **0.82** | UNPROFITABLE |
| FVG + Discount/Premium | 1.58 | **0.85** | UNPROFITABLE |
| Hybrid 4σ + OB | 3.05 | *not tested* | Likely UNPROFITABLE |

## DO NOT USE IN REAL TRADING

The following files produce false signals and should NOT be used:
- `signal_bot.py` (Mean Reversion Telegram bot)
- `smc_fvg_signal_bot.py` (FVG Telegram bot)
- `bounce_indicator_v2.pine` (Mean Reversion TradingView indicator)
- `smc_fvg_indicator.pine` (FVG TradingView indicator)

## Lesson Learned

ALWAYS use `.shift(1)` on indicator computations in backtests:
```python
dev_ma = df["Close"].ewm(span=20).mean().shift(1)  # CORRECT
# NOT: dev_ma = df["Close"].ewm(span=20).mean()    # WRONG - uses current bar
```

ALWAYS verify that entry prices were actually available (not below low or above high
of the current bar) before claiming a fill.

ALWAYS compare `shift(1)` vs no-shift results — if they differ significantly, you
have a look-ahead bug.
