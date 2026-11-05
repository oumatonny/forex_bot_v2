Forex Trading Bot

## Gemini chart analysis

The chart analyzer uses Gemini. Create a local `.env` file (it is ignored by Git) with:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_VISION_MODEL=gemini-3.8-flash
```

Alternatively, enter the key in the chart analyzer field for the current session.

## Tick Scalper

The `⚡ Tick Scalper` tab calibrates selected MT5 symbols before enabling tick
strategies. It refuses symbols whose measured spread and configured execution
costs consume more than half of a typical five-second move. Start in Paper mode
and review the calibration report before considering any live use; this feature
does not guarantee profits and is not financial advice.

Calibration enables momentum strategies in trends, fade strategies in wide
ranges, and squeeze breakout in transitional conditions. Spread, commission,
slippage, and latency are material at this time scale, so a high trade count is
not a success criterion. Live trading must use the existing arming safeguards.
Tick Scalper’s Python exit manager has no time stop: it requests a close only
when estimated net profit is at least +0.05 account currency. Broker-side SL/TP
protection and an explicit emergency flatten can still close a position before
that target.
