# PROMPT — Add a "Tick Scalper" engine to TradePilot SUPER BOT (MT5, Python, Streamlit)

Copy everything below the line into your AI coding assistant, and attach `app.py` (TradePilot SUPER BOT) and `strategies.json`.

---

## Role
You are a senior Python quant developer. Extend my existing TradePilot SUPER BOT (`app.py`, attached). It's a rule-based MT5 bot with no AI in the trading loop. Add a new **Tick Scalper** engine and a new Streamlit tab, **"⚡ Tick Scalper"**. Keep the existing candle engine, settings, journal, backtester, locks (`MT5_LOCK`, `FILE_LOCK`), `_send_deal()`, `live_close_ticket()`, paper/live arming flow, and `BOT_MAGIC` style. Reuse the existing helpers and don't rewrite them. The new engine gets its own magic number (`TICK_MAGIC = 234600`) and its own state file and trades CSV (`bot_data/tick_trades.csv`).

## What the feature does
1. **Phase 1 – Calibration (5 minutes, configurable 1–15 min).** When I press **Start**, the engine streams ticks for each selected symbol. It places no trades during this phase. It measures the market and picks which tick strategies to switch on, with thresholds tuned to what it measured.
2. **Phase 2 – Continuous scalping.** It trades continuously, targeting 20+ short trades per minute in total across symbols and strategies. Each trade is held for seconds (default max hold 15 s). This continues until I press **Stop**, or until a risk limit halts it.
3. **Rolling re-calibration.** Every 5 minutes it re-runs the calibration on the last 5 minutes of ticks in the background. It can switch strategies off and on without stopping.

## Phase 1 — Calibration metrics (per symbol)
Compute these from `mt5.copy_ticks_from(symbol, start, count, mt5.COPY_TICKS_ALL)`, then keep a rolling `deque` of ticks fed by polling `symbol_info_tick` every 100–250 ms:
- `tick_rate`: ticks per second.
- `avg_spread`, `p90_spread`: in points.
- `move_5s`: median absolute mid-price change over 5-second windows.
- **`cost_ratio = (avg_spread + commission_in_points + expected_slippage) / move_5s`** ← the master filter.
- `efficiency_ratio` (Kaufman ER) over the 5 minutes: |net move| / sum(|tick moves|). ER > 0.35 means trending. ER < 0.15 means ranging or choppy.
- `trend_slope`: slope of a linear regression of mid-price over 5 minutes, normalised by `move_5s`.
- `range_hi`, `range_lo`: 5-minute high and low, plus the 10th and 90th percentile of mid-price.
- `tick_z_sd`: standard deviation of mid-price around its 30-second rolling mean.
- `uptick_ratio`: the share of up-ticks.

**Decision rules (show them in the UI as a "Calibration Report" card per symbol):**
- If `cost_ratio > 0.5`, mark the symbol **NOT TRADEABLE** at this time and don't trade it. Show the reason, e.g. "spread eats 80% of a typical 5-s move".
- If ER > 0.35, enable the trend/momentum strategies (S1, S4, S5, S6) in the direction of `trend_slope` only.
- If ER < 0.15 and the range is wide enough (`(range_hi - range_lo) > 8 × avg_spread`), enable the range/fade strategies (S2, S3).
- If the market is in between, enable S7 (compression breakout) only.
- Auto-set the thresholds from the measured distribution. Examples: z-entry is the 95th percentile of |z|, and the velocity threshold is the 90th percentile of 2-second velocity.

## Phase 2 — Tick strategies (each one is a pure function `(TickState) -> TickSignal | None`)
Build micro-bars from ticks: 1 s, 5 s and 10 s OHLC. Keep everything causal, so no future ticks are used.

| ID | Name | Entry rule | Exit rule |
|---|---|---|---|
| S1 | **Tick Momentum Burst** | N consecutive same-direction ticks (N auto-set, default 4), and price velocity over 2 s > threshold, and the direction agrees with the calibration trend | TP = 1.0–1.5 × `move_5s`, SL = 1.0 × `move_5s` + spread, time stop 10 s |
| S2 | **Micro Mean-Reversion (Tick Z-score)** | z = (mid − 30-s mean)/30-s SD. Sell at z > +2.5, buy at z < −2.5. Only when ER < 0.15 | Exit at z ≈ 0 (the mean), SL at z ±4, time stop 15 s |
| S3 | **Range Ping-Pong** | Buy when mid ≤ 10th percentile of the calibration range and the last tick is an up-tick. Sell when mid ≥ 90th percentile and the last tick is a down-tick | TP at range midpoint or 60% of the way across, SL just outside `range_lo`/`range_hi` + spread. Disable the strategy if the price closes a 10-s bar outside the range (range broken) |
| S4 | **5-Second Bar Breakout** | Close of the current 5-s bar > high of the previous 5-s bar (buy) or < its low (sell), in the 1-minute EMA(20) trend direction | SL = the other side of the previous 5-s bar, TP = 1.2R, time stop 3 bars |
| S5 | **Tick Order-Flow Imbalance** | Over a 3-s window, upticks/(upticks+downticks) > 0.70 → buy, < 0.30 → sell. Needs `tick_rate` above its calibration median | TP 1R, SL = 1 × `move_5s`, exit early if the imbalance flips |
| S6 | **Tick-VWAP Pullback (trend re-entry)** | In a calibrated trend, price pulls back to the 60-s tick-VWAP (volume = tick count), then prints 2 ticks back in the trend direction. Re-enter on every pullback | TP = previous micro swing high/low, SL = beyond the VWAP by 0.7 × `move_5s` |
| S7 | **Micro Squeeze Breakout** | 10-s tick range < 30% of the calibration average 10-s range, then the price breaks the squeeze high/low by ≥ 1 spread | TP = 2 × squeeze width, SL = opposite side of the squeeze |
| S8 | **Spread-Normalisation Fade** (optional, off by default) | After a spread spike (> p90) returns to normal, fade the impulse that happened during the spike | TP = half the impulse, time stop 5 s |

**Signal combination:** each strategy has its own cooldown (default 2 s) and fires independently. Only one open position per strategy per symbol. A new signal opposite to an open position from the same strategy closes that position and then reverses it (configurable).

**Throughput target:** to reach about 20 trades/minute, scan every strategy on every new tick (event-driven) for all enabled symbols. Show **live trades/minute** in the UI. Don't lower thresholds just to hit the trade-count target. If edge filters cut the rate below 20, show that clearly, e.g. "Running at 9 trades/min — filters are blocking low-edge signals".

## Execution requirements
- Use the existing `_send_deal()` (it already handles retries and filling-mode fallback). Every order **must carry a broker-side SL and TP**, because the Python loop can lag.
- Measure and log **latency per order**: tick time → order sent → fill time. Also log slippage in points.
- Respect MT5 limits. Handle retcode `10024` (too many requests) by backing off. Rate-limit globally with a configurable cap, default max 30 orders/minute.
- Require a **hedging account** for several positions per symbol. On a netting account, allow only 1 position per symbol and show a warning.
- Lot size: fixed micro-lot by default (0.01), or risk-based from SL distance with a hard cap.
- Time stop is enforced by the engine, which checks open tick positions every 100–250 ms.

## Risk controls (non-negotiable, enforced in the engine and not only the UI)
- **Session loss limit** (default 2% of equity) → halt and flatten.
- **Max consecutive losses** (default 8) → pause 2 min and re-calibrate.
- **Rolling expectancy kill-switch:** if the net average P/L per trade after costs over the last 50 trades is < 0, pause that strategy and log the reason.
- **Spread guard:** skip an entry if the current spread > 1.5 × calibration `avg_spread`.
- **News/rollover guard:** no trading 23:55–00:10 server time (spreads widen at rollover). Add a manual "news blackout" toggle.
- Max open tick positions in total (default 5).
- The **Stop** button stops new entries immediately. A second option is "Stop & Flatten".
- Startup always defaults to **PAPER**. Live needs the existing arming flow, plus typing `REAL` for real accounts.

## Paper mode (must be realistic, or the results are useless)
Simulate fills on real live ticks. Buy at the ask, sell at the bid. Add a configurable **commission per lot**, **latency** (default 150 ms: fill at the tick that arrives 150 ms after the signal), and **slippage**. Apply SL/TP against bid/ask correctly.

## Tick backtester (new section in the tab)
- Pull history with `mt5.copy_ticks_range` for a chosen symbol and date range (warn if the data exceeds ~2M ticks).
- Run the **same strategy functions** with the same cost/latency model.
- Report per strategy: trades, trades/min, win rate, average win/loss in points, **expectancy after costs**, profit factor, max drawdown, and the share of P/L eaten by spread + commission.
- Show a chart of the equity curve and a histogram of trade P/L.
- **Gate:** grey out the "Arm Live" button for a strategy until its paper or backtest record shows ≥ 1,000 trades with positive expectancy after costs.

## UI ("⚡ Tick Scalper" tab)
- Symbol multiselect, calibration length, strategy toggles (auto vs manual override), lot, cost settings.
- A big **START / STOP / STOP & FLATTEN**, plus a phase indicator: `CALIBRATING 3:12 left` → `SCALPING` → `HALTED (reason)`.
- Live panel that refreshes every 1 s:
  - trades/min
  - session P/L
  - P/L split into gross and cost
  - open positions with countdown to time stop
  - last 50 decisions with reasons, including skipped signals and why they were skipped
- Calibration report cards per symbol (cost_ratio colour-coded green < 0.3, amber 0.3–0.5, red > 0.5).

## Code quality
- Single file is fine (match `app.py`), but use clear sections: `TICK DATA`, `TICK STRATEGIES`, `CALIBRATION`, `TICK ENGINE`, `TICK BACKTEST`, `TICK UI`.
- Run the tick engine as a background thread like the existing `Engine`. It must survive cycle errors, never block the Streamlit thread, and persist state so open positions are re-adopted after a restart.
- Use type hints and docstrings. Unit-testable pure functions for the strategies and the calibration.
- Support offline mode (`SUPERBOT_OFFLINE=1`): load ticks from CSV for the backtest.
- Deliver the full updated `app.py`, the `requirements.txt` changes, and a short README section. The README should cover how to run it, what each strategy does, and why costs decide everything at this time-frame.

Not financial advice. The bot must never claim guaranteed profit anywhere in the UI.
