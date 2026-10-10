# TODO — Rule-based auto-executing forex bot (no AI)

Target: one Streamlit app (`app.py`) that scans several symbols, generates signals from
fixed, testable rules taken from `strategies.json`, sizes every trade from account risk,
sends and manages orders on MT5 unattended, and can prove (backtest + journal) whether
a strategy is worth running. `ai_trading.py` is out of scope and is not used.

Legend: `[x]` built in `app.py` · `[ ]` still to do · **P0** = must be true before any
real-money run.

---

## 1. What the audit of the existing bots found

These are the reasons the new bot is a rewrite of the engine rather than a fourth copy.
Files: `main.py` (manual), `tradepilot_auto.py` (Bollinger/RSI), `smc_ict_auto.py` (ICT/SMC).

### Money-losing or account-threatening (P0)

| # | Problem | Where | Fix in `app.py` |
|---|---------|-------|-----------------|
| A1 | Stop-outs are recorded as **0.00 profit**, so the daily loss kill-switch never sees a losing trade closed by the broker SL | `_manage_open_positions` (`profit_override=0.0`) in both auto bots | Realised P&L read from MT5 deal history (profit + commission + swap + fee) |
| A2 | Unlimited stacking: one signal opens positions until the broker refuses for margin, each sized at the full risk % — real risk per signal is N × risk % | `_open_positions_until_blocked` | One risk budget per trade, caps per symbol / per currency / total |
| A3 | Exit at +0.20 account currency with a stop many times further away: needs a very high win rate just to break even | `profit_take_amount` | R-multiple target, break-even, optional ATR trail. Quick-profit exit kept as an opt-in, off by default |
| A4 | A rejected close is ignored — the bot forgets the position while it is still open at the broker | `close_position()` result unused | Position stays tracked until the broker confirms it is gone |
| A5 | Bot state lives in `st.session_state`: refresh or close the tab and the thread keeps trading with no Stop button | all auto bots | Process-wide engine (`st.cache_resource`), state persisted to disk, positions re-adopted by magic number on start |
| A6 | Simulation never simulates a stop-loss — positions only ever close in profit, so simulated P&L is always positive | sim branch of `_manage_open_positions` | Paper broker fills at bid/ask and enforces SL/TP |
| A7 | Simulated flatten books raw price difference as money | `_close_and_record` (`realized_profit = diff`) | P&L via `order_calc_profit` |
| A8 | "Daily" limits reset on every Start, and are session counters, not per day | `BotState()` recreated on start | Day stats rebuilt from the journal by broker server date |
| A9 | Lot size rounds to nearest step and is forced **up** to the broker minimum even when that exceeds the risk budget; `round(lot, 2)` is wrong for other volume steps; sized on balance not equity | `calculate_lot_size` | Floor to step, hard risk cap for the minimum-lot case, `min(balance, equity)` |
| A10 | SL/TP not normalised to symbol digits or checked against the broker stops level; SELL stops ignore spread | `execute_market_order` | Normalised, stops-level floor, spread added to SELL stops |
| A11 | No requote / price-changed / filling-mode retry, no margin pre-check | `execute_market_order` | Retry loop, filling-mode fallback, `order_calc_margin` check |
| A12 | SMC bot re-enters every 2 s on the same stale signal (last sweep / last BOS can be hundreds of bars old) | `automation_loop`, `analyze` | One decision per closed bar, triggers must occur on the signal bar |
| A13 | Hitting max-trades in the Bollinger bot flattens every open position at market | `automation_loop` break → flatten | Limits halt new entries only |
| A14 | `smc_ict_auto.py` and `ai_trading.py` share magic 234300 | constants | New magic 234500 |

### Correctness and maintainability

- [x] Forming-candle detection relied on tick time; replaced by always reading from bar index 1.
- [x] 5-pip stop buffer is meaningless on gold/crypto; all buffers are ATR-based.
- [x] O(n²) loops in the SMC engine; replaced by vectorised, causal indicators.
- [x] No spread, session, weekend, cooldown or correlation filters.
- [x] No log file, no trade journal, no backtest — nothing showed whether a strategy has an edge.
- [x] `time.sleep(1); st.rerun()` whole-page loop replaced by a 2 s fragment.
- [ ] ~70 % of the code is duplicated across four files — retire the old bots once `app.py` has run on demo (keep `main.py` only if the manual dashboard is still wanted).
- [ ] No `requirements.txt`, README, tests in the repo, or git. `.env` holds API keys with no `.gitignore` — add one before `git init`.

---

## 2. Architecture (single file, layered)

```
MT5 terminal ──► data (closed bars, ticks, symbol specs)
                    │
                    ▼
              indicators (causal, vectorised)
                    │
                    ▼
     strategies ── Signal(direction, stop, target, reason) ── same code for live + backtest
                    │
        HTF filter ─┤─ confluence vote
                    ▼
               trade plan (stop floor/ceiling, R:R check)
                    │
               risk gates (day loss, streak, caps, spread, session, margin)
                    │
        ┌───────────┴───────────┐
   paper broker             MT5 broker
        └───────────┬───────────┘
            position manager (break-even, trail, time stop, opposite signal)
                    │
        journal (trades.csv) · state (json) · log (bot.log) ──► Streamlit UI
```

---

## 3. Build plan

### Phase 0 — Safety foundations (P0)
- [x] Paper mode is the default; live needs an explicit arm, and a typed `REAL` on a real-money account.
- [x] Every order carries a broker-side SL and TP.
- [x] Dedicated magic number; the bot never touches positions it does not own.
- [x] Engine survives browser refresh / tab close; one engine per process.
- [x] State and journal on disk (`bot_data/`); open positions adopted again after a restart.
- [x] No trade on the bar that was already closed when the bot started.
- [x] Stop (leave positions protected), Stop & flatten, and a panic "close all bot positions" that works with the engine stopped.
- [x] Transient errors never kill the loop; an unreadable position list is treated as "unknown", not "closed".
- [ ] Heartbeat alert (Telegram/email) when the loop stalls or MT5 disconnects.
- [ ] Run as a Windows service / scheduled task so a reboot restarts it.

### Phase 1 — Market data
- [x] Closed bars only, per symbol, per timeframe (M1–D1).
- [x] Higher-timeframe trend mapped without look-ahead.
- [x] Broker symbol suffix resolution (`EURUSD` → `EURUSDm`, `EURUSD.a`).
- [x] Symbol specs from the broker: digits, point, volume step/min/max, stops level, filling mode.
- [ ] Detect gaps / missing bars and skip evaluation on a broken series.
- [ ] Cache history locally for faster, larger backtests.

### Phase 2 — Strategy engine (from `strategies.json`)
- [x] Common `Signal` contract; one function per strategy; identical code path live and in backtest.
- [x] Eleven rule sets implemented (see coverage table, section 4).
- [x] Confluence vote: trade only when N strategies agree and none disagree on balance.
- [x] HTF trend filter for trend/reversal strategies; ADX range filter inside mean-reversion ones.
- [ ] #30 Role reversal (breakout → retest of the broken level, 3-part scale-in).
- [ ] #28 Swing day trading (contracting correction inside a trend).
- [ ] #5 Chart patterns (double top/bottom, triangles, head & shoulders) — needs a pattern detector.
- [ ] #27 RSI divergence system.
- [ ] #24 Fibonacci retracement confluence.
- [ ] Per-strategy parameters editable in the UI (currently constants in code).
- [ ] Per-symbol strategy allow-list driven by backtest results.

### Phase 3 — Risk management
- [x] Risk % of equity per trade, lot derived from stop distance with `order_calc_profit`.
- [x] Lot floored to volume step; minimum-lot trades allowed only under a hard risk cap.
- [x] Daily loss limit (realised + floating) — halts entries until the next server day.
- [x] Max trades per day, max consecutive losses, cooldown after a loss on a symbol.
- [x] Max open positions: total, per symbol, per currency (correlation guard, from #17/#18/#20).
- [x] Spread cap as a fraction of the stop distance.
- [x] Stop floor (ATR multiple and broker stops level) and stop ceiling.
- [x] Minimum reward:risk for strategies with a structural target.
- [x] Margin pre-check before sending.
- [ ] Weekly / max-drawdown-from-peak halt.
- [ ] Economic-calendar blackout (NFP, FOMC, CPI) — needs a calendar feed (#13, #26 "avoid news days").
- [ ] Swap-aware holding (no Wednesday triple-swap carry on scalps).

### Phase 4 — Execution
- [x] Market orders with requote / price-changed retry and filling-mode fallback.
- [x] Prices normalised to symbol digits; SELL stops include the spread.
- [x] Fill confirmed against the broker's position list; realised P&L taken from deal history.
- [x] Netting accounts limited to one position per symbol.
- [ ] Pending orders — limit/stop entries, expiry, OCO (#22). Needed for #24 and #32 proper.
- [ ] Partial close (scale out at 1R).
- [ ] Slippage statistics per symbol in the journal.

### Phase 5 — Position management
- [x] Break-even at a configurable R.
- [x] ATR trailing stop after a configurable R.
- [x] Time stop (max bars held).
- [x] Exit on the opposite signal from the same strategy (#25, #33).
- [x] Friday cut-off: no new entries and flatten after a chosen server hour.
- [x] Optional quick-profit exit in account currency (legacy behaviour, off by default).
- [ ] End-of-day flatten for pure day-trading profiles (#15).

### Phase 6 — Validation (do this before trusting any strategy)
- [x] Bar-by-bar backtester sharing the live strategy code: next-bar-open entry, spread cost, pessimistic SL-before-TP inside a bar, gap fills at the open.
- [x] Results in R: trades, win rate, profit factor, expectancy, max drawdown, per-strategy table, equity curve.
- [x] CSV backtest when MT5 is not available.
- [ ] Walk-forward / out-of-sample split and parameter-stability report.
- [ ] Commission and swap in the backtest cost model.
- [ ] Monte-Carlo reshuffle of trade order for drawdown confidence.
- [ ] Unit tests in the repo (indicators vs known values, no-look-ahead test, sizing, plan builder).

### Phase 7 — UI
- [x] Dashboard: engine status, halt reason, day P&L, open positions with live R, decision feed.
- [x] Scanner: current signals for every symbol with chart and recent signal markers.
- [x] Backtest tab, Journal tab (with CSV download), Strategy library tab (`strategies.json` coverage).
- [x] Settings persisted between sessions; locked while the engine runs.
- [ ] Per-strategy live performance vs backtest expectation (drift alarm).
- [ ] Password gate if the app is ever reachable beyond localhost.

### Phase 8 — Go-live checklist (in this order)
- [ ] Backtest each strategy on each intended symbol/timeframe; switch off anything with expectancy ≤ 0 R after spread.
- [ ] Paper mode for at least 2 weeks; compare journal against backtest expectancy.
- [ ] Demo account with live orders for at least 2 weeks; confirm fills, stops, restarts, weekend behaviour.
- [ ] Kill the app mid-trade on demo and confirm positions are re-adopted on restart.
- [ ] Real account at the smallest risk (≤ 0.25 %) with a daily loss limit; scale only after 50+ trades match expectation.

---

## 4. `strategies.json` coverage

| ID | Strategy | Status | How it is used |
|----|----------|--------|----------------|
| 1 | Trend trading | built | `trend_pullback` — EMA 20/50/200 stack, ADX ≥ 20, pullback and resume |
| 2 | Support & resistance | built | `sr_breakout` — close through the 20-bar high/low |
| 3 | Range trading | built | `bb_rsi_reversion` and `pivot_bounce`, both gated by ADX < 30 |
| 4 | Technical indicators | built | MA, Bollinger, RSI, Stochastic, MACD, ADX all computed; Momentum covered by MACD histogram |
| 5 | Chart patterns | todo | Needs a pattern detector (Phase 2) |
| 6 | Volume | built | Tick-volume above its 20-bar average confirms breakouts |
| 7 | Multiple time frames | built | HTF trend filter (EMA 50/200) on every trend/reversal strategy |
| 8 | Scalping | partial | M1/M5 supported with spread cap; no tick-level logic |
| 9 | Fading | built | `bb_rsi_reversion` — fade a band poke with RSI extreme |
| 10 | Daily pivots | built | `pivot_bounce` — rejection at S1/S2/R1/R2, target P |
| 11 | Momentum | built | MACD histogram in `trend_pullback`; `rsi2_momentum` |
| 12 | Technical analysis (umbrella) | built | The engine as a whole |
| 13 | Fundamental analysis | not automatable here | Needs macro data; only relevant as a news blackout (Phase 3) |
| 14 | Market sentiment | not automatable here | Needs positioning data (COT / broker sentiment feed) |
| 15 | Day trading | partial | Session-hours filter and Friday cut-off; end-of-day flatten is todo |
| 16 | Carry trade | out of scope | Multi-week swap strategy, not an intraday execution bot |
| 17 | Hedging | partial | Used as a risk rule: per-currency exposure cap |
| 18 | Portfolio / basket | partial | Multi-symbol scanning with total exposure caps |
| 19 | Buy and hold | out of scope | Not an automated-entry strategy |
| 20 | Spread / pair trading | todo | Needs a two-leg order model and cointegration test |
| 21 | Swing trading | built | Same rule sets on H1/H4 with the HTF filter |
| 22 | Order types | partial | Market + SL/TP; pending and OCO are todo (Phase 4) |
| 23 | Algorithmic trading | built | This bot |
| 24 | Momentum reversal | todo | Stochastic + S/R + Fibonacci on limit orders; COT part not automatable |
| 25 | MA crossover 20/60/100 | built | `ma_cross` — exits on the cross back |
| 26 | Heikin-Ashi | built* | `heikin_ashi` — two-candle colour flip, Stochastic confirmation |
| 27 | RSI systems | partial | 2-period RSI built; divergence system todo |
| 28 | Swing day trading | todo | Phase 2 |
| 29 | Candlestick patterns | built | `candle_patterns` — engulfing, hammer / shooting star at a 10-bar extreme |
| 30 | Role reversal | todo | Phase 2 |
| 31 | Bollinger squeeze | built* | `bb_squeeze` — full candle beyond the mid-band after a squeeze |
| 32 | Narrow range (NR7) | built* | `narrow_range` — close beyond the NR7 bar |
| 33 | 2-period RSI | built | `rsi2_momentum` — as written in the file: buy above 90, sell below 10, with the 100-MA trend |
| 34 | Binary options | out of scope | Not specified in the source and not a spot-FX product |

Also carried over from the project (not in `strategies.json`): the ICT/SMC engine from
`smc_ict_auto.py`, rebuilt as `smc` — liquidity-sweep rejection and first retrace into a
fair value gap created by a structure break.

\* Deviations from the JSON text, made so the rules work on any instrument:
- **#26** — the file says "long if Stochastic > 70, short if < 30". Straight after a run of
  opposite candles that almost never happens, so confirmation is %K above/below %D.
- **#31** — "band width approaching 0.0100" only fits EURUSD; a squeeze is band width in the
  lowest 20 % of the last 120 bars.
- **#32** — described on daily candles; applied to the chosen signal timeframe, entering on a
  close beyond the narrow bar (a stop order at the bar's edge needs pending orders).
- **#33** — the usual 2-period RSI system fades extremes; the file describes it as momentum.
  Implemented as written. The backtest tab will show which reading works on your symbols.

---

## 5. Definition of done

1. No path where a position exists at the broker without a stop, or without the bot tracking it.
2. Worst-case loss for a day is bounded by settings and survives a restart.
3. Every trade in the journal can be traced to a strategy, a bar, and a reason.
4. Every strategy that is switched on for live has a positive backtest expectancy after spread
   on that symbol/timeframe, and paper results within range of it.
