"""
TradePilot AUTO - AI Market Analyst (OpenRouter powered)
==========================================================

Strategy signal logic (fully AI-driven, no fixed indicator rule set):
    On each evaluation cycle (while the bot is flat and not cooling down):
      - The last N closed candles on the chosen timeframe are pulled
        straight from MT5 (no ICT/SMC concepts, no FVG/OB/BOS/CHoCH -
        just raw OHLC price action).
      - Optionally, the last ~20 closed candles from a higher timeframe
        are attached as extra context.
      - That price action is handed to an LLM via OpenRouter
        (openrouter.ai - a single OpenAI-compatible endpoint that routes
        to whichever model/provider you configure), which is instructed
        to form its own independent view and return a strict JSON verdict: BUY / SELL /
        HOLD, a confidence score, a stop distance (as a multiple of ATR,
        so risk sizing stays deterministic even though direction/timing
        is AI-driven), a reward:risk ratio, and short reasons.
      - The bot only acts on BUY/SELL when confidence clears your
        configured minimum. HOLD, a low-confidence verdict, or any AI
        error (timeout, bad key, malformed response, etc.) all fail
        CLOSED to "no trade this cycle" - the app never invents a
        direction when the AI didn't clearly give one.
      - ATR (a generic volatility measure, not a strategy signal) is
        still computed locally purely so the stop-loss/target distance
        the AI proposes turns into sane, symbol-appropriate price
        levels rather than trusting the model to hallucinate exact
        decimals.

TRADE MANAGEMENT AND SIZING (unchanged from the previous version, strategy-agnostic):
  1. Exit is an ACTIVE check, not a wait on the broker SL/TP. Every poll
     cycle, every open position this bot holds is checked for unrealized
     profit, and is closed the moment that profit reaches
     `profit_take_amount` (account currency, default 0.20 - i.e. "any green
     is good enough"). A broker-side Stop Loss is still attached to every
     order as a crash/disconnect safety net (see CAVEATS) - it's not what
     this bot relies on to end a trade under normal operation. The
     broker-side "take profit" (AI's target) is likewise kept only as a
     distant backstop unless you opt in below; in practice the active
     profit-take exit will almost always fire first.
  2. On each AI-generated BUY/SELL signal that clears your confidence
     threshold, the bot opens positions repeatedly - same direction, same
     computed SL/target, same lot size - until it can't anymore:
        - Live mode: until the broker actually rejects an order (e.g. not
          enough free margin). That rejection IS the real, authoritative
          stopping point.
        - Simulation mode: a configurable `max_concurrent_positions` cap
          stands in for that condition (defaults to 20 if left at 0).
     A session-level `max_trades` cap and a hard internal per-cycle safety
     ceiling (200 orders/cycle) still apply underneath this as an
     engineering guard against a runaway loop.

READ THIS BEFORE ARMING LIVE - RISK OF THIS CONFIGURATION:
  Stacking same-direction positions on one signal does NOT diversify risk -
  every stacked position opens at nearly the same price, off the same
  signal, in the same direction. It multiplies exposure to that one AI
  verdict being wrong, and (live mode) can consume all free margin in one
  go. If the signal is wrong before the profit-take exit can trigger, every
  stacked position loses together and the per-trade Stop Loss is what
  stands between that and a margin call. On top of that, an LLM verdict is
  a probabilistic read of recent price action, not a verified quantitative
  forecast - it can be wrong, and it can be wrong confidently. Test in
  simulation mode and/or a demo account first; consider a non-zero
  `max_concurrent_positions` cap to bound worst-case exposure per signal.

CAVEATS:
  * The broker-side Stop Loss is what actually protects the account if this
    app crashes, the machine loses power, or the connection drops - do not
    remove it.
  * This file uses a background thread + shared state in st.session_state.
    That's tied to this one browser session/server process; closing the tab
    or a server restart means you lose your Stop button while positions
    could still be open. Always be able to check/close positions directly
    in the MT5 terminal as a fallback.
  * Every evaluation cycle spends one OpenRouter API call (which itself
    bills/rate-limits based on whichever underlying model you route to).
    A poll interval keeps the bot from hammering the API once per second
    while flat - see `analysis_poll_seconds` in the sidebar. Some models
    routed through OpenRouter are considerably slower than others; if you
    see read-timeout errors, raise the AI request timeout in the sidebar
    or switch to a faster model.
  * Not financial advice, no strategy is guaranteed to be profitable.
"""

import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import os
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional deps for the AI layer - the app still runs (with AI disabled and
# a clear sidebar message) if these aren't installed, rather than crashing.
try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up OPENROUTER_API_KEY / OPENROUTER_MODEL from a local .env file, if present
except ImportError:
    pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TradePilot AUTO - AI Market Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #00B4D8; }
    .signal-buy { background-color: #00B4D8; padding: 10px; border-radius: 8px; color: white; font-weight: bold; }
    .signal-sell { background-color: #FF6B6B; padding: 10px; border-radius: 8px; color: white; font-weight: bold; }
    .signal-neutral { background-color: #FFD93D; padding: 10px; border-radius: 8px; color: #333; font-weight: bold; }
    .status-running { background-color: #00B4D8; padding: 8px 14px; border-radius: 20px; color: white; font-weight: bold; display: inline-block; }
    .status-stopping { background-color: #FFD93D; padding: 8px 14px; border-radius: 20px; color: #333; font-weight: bold; display: inline-block; }
    .status-idle { background-color: #444; padding: 8px 14px; border-radius: 20px; color: white; font-weight: bold; display: inline-block; }
    .status-error { background-color: #FF6B6B; padding: 8px 14px; border-radius: 20px; color: white; font-weight: bold; display: inline-block; }
    .stButton > button { width: 100%; font-weight: bold; }
    div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

MT5_LOCK = threading.Lock()
BOT_MAGIC = 234300          # identifies this bot's own orders/positions
STACK_SAFETY_CEILING = 200  # hard per-cycle cap so a bug can't spin forever even if the broker somehow keeps accepting

TIMEFRAME_DURATIONS = {
    mt5.TIMEFRAME_M1: timedelta(minutes=1),
    mt5.TIMEFRAME_M5: timedelta(minutes=5),
    mt5.TIMEFRAME_M15: timedelta(minutes=15),
    mt5.TIMEFRAME_M30: timedelta(minutes=30),
    mt5.TIMEFRAME_H1: timedelta(hours=1),
    mt5.TIMEFRAME_H4: timedelta(hours=4),
}


# ============================================================================
# MT5 CONNECTION MANAGER
# ============================================================================
class MT5Connector:
    def __init__(self):
        self.connected = False
        self.account_info = None

    def connect(self):
        with MT5_LOCK:
            existing_account = mt5.account_info()
            if existing_account is not None:
                self.connected = True
                self.account_info = existing_account
                return True
            if not mt5.initialize():
                self.connected = False
                return False
            account = mt5.account_info()
            if account is None:
                mt5.shutdown()
                self.connected = False
                return False
            self.connected = True
            self.account_info = account
            return True

    def refresh_account_info(self):
        with MT5_LOCK:
            if self.connected:
                self.account_info = mt5.account_info()
        return self.account_info

    def disconnect(self):
        with MT5_LOCK:
            if self.connected:
                mt5.shutdown()
                self.connected = False

    def get_rates(self, symbol, timeframe=mt5.TIMEFRAME_M15, bars=300):
        if not self.connected and not self.connect():
            return None
        with MT5_LOCK:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        return df

    def get_server_time(self, symbol):
        with MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return datetime.utcfromtimestamp(tick.time)

    def get_symbol_info(self, symbol):
        if not self.connected:
            self.connect()
        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
            if info is not None and not info.visible:
                mt5.symbol_select(symbol, True)
                info = mt5.symbol_info(symbol)
        return info

    def get_current_price(self, symbol):
        if not self.connected:
            self.connect()
        with MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {'bid': tick.bid, 'ask': tick.ask}


def get_connector():
    if 'mt5_connector' not in st.session_state:
        st.session_state.mt5_connector = MT5Connector()
    return st.session_state.mt5_connector


def get_account_mode_label(account_info):
    if account_info is None:
        return "UNKNOWN", False
    mode = getattr(account_info, 'trade_mode', None)
    mapping = {
        getattr(mt5, 'ACCOUNT_TRADE_MODE_DEMO', 0): ("DEMO", False),
        getattr(mt5, 'ACCOUNT_TRADE_MODE_CONTEST', 1): ("CONTEST", False),
        getattr(mt5, 'ACCOUNT_TRADE_MODE_REAL', 2): ("REAL", True),
    }
    return mapping.get(mode, ("UNKNOWN", False))


def _resolve_filling_type(symbol_info):
    SYMBOL_FILLING_FOK = getattr(mt5, 'SYMBOL_FILLING_FOK', 1)
    SYMBOL_FILLING_IOC = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
    mode = getattr(symbol_info, 'filling_mode', 0) or 0
    if mode & SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if mode & SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def get_closed_bars(connector, symbol, timeframe, bars):
    """Fetch bars and drop the currently-forming (incomplete) candle so the
    AI only ever evaluates fully-closed candles."""
    df = connector.get_rates(symbol, timeframe, bars + 2)
    if df is None or len(df) < 3:
        return df
    server_time = connector.get_server_time(symbol)
    duration = TIMEFRAME_DURATIONS.get(timeframe, timedelta(minutes=15))
    if server_time is not None:
        last_bar_time = df['time'].iloc[-1].to_pydatetime()
        if server_time - last_bar_time < duration:
            df = df.iloc[:-1]
    return df.reset_index(drop=True)


# ============================================================================
# TRADE EXECUTION
# ============================================================================
def execute_market_order(connector, symbol, direction, lot_size, sl, tp,
                          deviation=20, magic=BOT_MAGIC, comment="TradePilot-AI"):
    if direction not in ('BUY', 'SELL'):
        return None, "Direction must be BUY or SELL."
    if not connector.connected and not connector.connect():
        return None, "Not connected to MT5."

    symbol_info = connector.get_symbol_info(symbol)
    if symbol_info is None:
        return None, f"Could not fetch symbol info for {symbol}."
    if not symbol_info.trade_mode or symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return None, f"Trading is disabled for {symbol}."

    with MT5_LOCK:
        tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, "Could not fetch current tick."

    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
    order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == 'BUY' else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lot_size),
        "type": order_type, "price": price, "sl": float(sl), "tp": float(tp),
        "deviation": deviation, "magic": magic, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": _resolve_filling_type(symbol_info),
    }
    with MT5_LOCK:
        result = mt5.order_send(request)
        last_err = mt5.last_error() if result is None else None

    if result is None:
        return None, f"order_send() returned None. last_error(): {last_err}"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return result, f"Order rejected - retcode {result.retcode}: {result.comment}"
    return result, f"Order filled - ticket #{result.order}, {lot_size} lots @ {result.price:.5f}"


def get_position_by_ticket(ticket):
    with MT5_LOCK:
        positions = mt5.positions_get(ticket=ticket)
    return positions[0] if positions else None


def get_bot_positions(symbol, magic=BOT_MAGIC):
    """All live MT5 positions on this symbol that belong to THIS bot
    (matched by magic number), so position management never touches a
    manually-placed or other-EA's position."""
    with MT5_LOCK:
        positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return []
    return [p for p in positions if p.magic == magic]


def get_open_positions(symbol=None):
    with MT5_LOCK:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None or len(positions) == 0:
        return pd.DataFrame()
    rows = []
    for p in positions:
        rows.append({
            'ticket': p.ticket, 'symbol': p.symbol,
            'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
            'volume': p.volume, 'open_price': p.price_open,
            'current_price': p.price_current, 'sl': p.sl, 'tp': p.tp,
            'profit': p.profit,
        })
    return pd.DataFrame(rows)


def close_position(ticket, deviation=20, comment="TradePilot-AI close"):
    with MT5_LOCK:
        positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None, f"No open position found with ticket #{ticket}."
    pos = positions[0]
    with MT5_LOCK:
        symbol_info = mt5.symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
    if symbol_info is None or tick is None:
        return None, "Could not fetch symbol info/tick for close."

    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
    price = tick.bid if is_buy else tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
        "type": close_type, "position": pos.ticket, "price": price,
        "deviation": deviation, "magic": BOT_MAGIC, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": _resolve_filling_type(symbol_info),
    }
    with MT5_LOCK:
        result = mt5.order_send(request)
    if result is None:
        return None, "order_send() returned None on close."
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return result, f"Close rejected - retcode {result.retcode}: {result.comment}"
    return result, f"Position #{ticket} closed @ {result.price:.5f}"


# ============================================================================
# AI MARKET ANALYST (OpenRouter - OpenAI-compatible chat completions API)
# ============================================================================
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


def get_openrouter_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'").strip()
    return key or None


def get_default_openrouter_model():
    """Reads OPENROUTER_MODEL from the environment/.env if set, so the
    sidebar pre-fills with whatever the user already configured."""
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)


def calculate_atr_series(df, period=14):
    """Generic volatility measure (Average True Range) - used only to turn
    the AI's proposed stop distance into an actual, symbol-appropriate
    price level. Not a trading signal on its own."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill()


def _format_candles(df, n):
    recent = df.tail(n)
    return "\n".join(
        f"{row.time:%Y-%m-%d %H:%M} O:{row.open:.5f} H:{row.high:.5f} L:{row.low:.5f} C:{row.close:.5f}"
        for row in recent.itertuples()
    )


def _build_analysis_messages(symbol, tf_label, df_ltf, current_price, atr_value,
                              htf_label=None, df_htf=None):
    window = df_ltf.tail(60)
    ltf_block = _format_candles(df_ltf, 60)
    first_close = window['close'].iloc[0]
    pct_change = ((current_price - first_close) / first_close * 100) if first_close else 0.0
    period_high = window['high'].max()
    period_low = window['low'].min()

    htf_block = ""
    if df_htf is not None and len(df_htf) > 0:
        htf_block = (
            f"\n\nHigher-timeframe context ({htf_label}, last {min(20, len(df_htf))} closed candles, "
            f"oldest to newest - for gauging the broader trend only, the timeframe above is primary):\n"
            f"{_format_candles(df_htf, 20)}"
        )

    system_prompt = (
        "You are an experienced, risk-averse discretionary forex/CFD trader. You are given raw recent "
        "price candles for one symbol and must form your own independent trading view purely from that "
        "price action - there is no indicator engine or pre-computed signal feeding you a suggestion. "
        "Most short windows of price action are noise and are not worth trading. Only return BUY or SELL "
        "when the recent structure genuinely supports a directional edge; return HOLD otherwise - HOLD is "
        "the correct answer most of the time. Respond with STRICT JSON ONLY, no markdown, no prose outside "
        "the JSON object, in exactly this shape: {\"verdict\": \"BUY\"|\"SELL\"|\"HOLD\", \"confidence\": "
        "<0-100 integer>, \"stop_atr_mult\": <number, how many ATR units away your stop should sit, "
        "typically 0.8 to 3.0>, \"reward_risk_ratio\": <number, target distance as a multiple of your stop "
        "distance, typically 1.0 to 3.0>, \"reasons\": [<up to 4 short strings, each under 20 words, "
        "explaining your read of the price action>]}"
    )

    user_prompt = f"""Symbol: {symbol}
Timeframe: {tf_label}
Current price: {current_price:.5f}
ATR(14): {atr_value:.5f}
Last-{len(window)}-candle range: {period_low:.5f} - {period_high:.5f} ({pct_change:+.2f}% net change over the window)

Recent closed candles, oldest to newest:
{ltf_block}{htf_block}

Based only on this price action, give your independent verdict on whether {symbol} should be bought, sold, or left alone right now on the {tf_label} chart."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openrouter_ai(messages, model, api_key, timeout=30, temperature=0.2, max_tokens=400,
                        retries=1, enable_reasoning=False):
    if not api_key:
        return None, "OPENROUTER_API_KEY not set in environment."
    if not model:
        return None, "No OpenRouter model configured."
    if not REQUESTS_AVAILABLE:
        return None, "The 'requests' package is not installed - run: pip install requests"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional app-attribution headers OpenRouter uses for its dashboard/leaderboards -
        # harmless to send, not required for the request to work.
        "HTTP-Referer": "https://tradepilot.local",
        "X-OpenRouter-Title": "TradePilot AUTO",
    }
    payload = {
        "model": model, "messages": messages,
        "temperature": temperature,
        "max_tokens": max(max_tokens, 800) if enable_reasoning else max_tokens,
        "stream": False,
    }
    if enable_reasoning:
        # Some models (e.g. x-ai/grok-4.3, deepseek reasoning variants) support an explicit
        # reasoning pass before the final answer - this just turns it on, no multi-turn
        # reasoning_details plumbing needed since each analysis cycle is a single fresh call.
        payload["reasoning"] = {"enabled": True}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = f"OpenRouter API request failed: {e}"
            continue  # transient - worth one retry before giving up
        except requests.exceptions.RequestException as e:
            return None, f"OpenRouter API request failed: {e}"

        if resp.status_code == 401:
            return None, (
                f"OpenRouter API returned HTTP 401 (unauthorized): {resp.text[:200]} - this is a "
                f"credentials problem, not a bug in this app. Check that OPENROUTER_API_KEY in your .env "
                f"is the exact, current key from openrouter.ai/keys (not expired/revoked/regenerated, no "
                f"stray quotes or trailing spaces), and that the app was restarted after editing .env."
            )
        if resp.status_code != 200:
            return None, f"OpenRouter API returned HTTP {resp.status_code}: {resp.text[:300]}"

        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            return None, f"Unexpected OpenRouter API response shape: {e}"

        return content, None

    return None, last_err or "OpenRouter API request failed after retry."


def _clamp(value, lo, hi, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return max(lo, min(hi, value))


def _parse_ai_analysis(raw_text):
    if not raw_text:
        return None
    text = raw_text.strip()
    text = re.sub(r'^```(?:json)?', '', text).strip()
    text = re.sub(r'```$', '', text).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    verdict = str(data.get('verdict', '')).upper().strip()
    if verdict not in ('BUY', 'SELL', 'HOLD'):
        verdict = 'HOLD'

    confidence = _clamp(data.get('confidence', 0), 0, 100, 0)
    stop_atr_mult = _clamp(data.get('stop_atr_mult', 1.5), 0.5, 4.0, 1.5)
    reward_risk_ratio = _clamp(data.get('reward_risk_ratio', 1.5), 0.5, 4.0, 1.5)

    reasons_raw = data.get('reasons', [])
    if isinstance(reasons_raw, str):
        reasons_raw = [reasons_raw]
    reasons = [str(r).strip()[:160] for r in reasons_raw if str(r).strip()][:5]

    return {
        'verdict': verdict, 'confidence': confidence,
        'stop_atr_mult': stop_atr_mult, 'reward_risk_ratio': reward_risk_ratio,
        'reasons': reasons,
    }


def analyze_market_ai(connector, symbol, timeframe, htf_timeframe, df_ltf, config):
    """The sole signal engine. Returns a dict shaped like the old rule-based
    analysis (direction/current_price/stop_loss/target/confidence/reasons)
    so the rest of the pipeline (execution, charting, UI) doesn't care that
    the source changed. Fails CLOSED: any AI error, unparseable response,
    HOLD verdict, or confidence below your threshold all resolve to
    direction='NEUTRAL' (no trade), never to a guessed direction."""
    atr_series = calculate_atr_series(df_ltf)
    atr_value = atr_series.iloc[-1]
    if not atr_value or atr_value != atr_value:  # NaN guard
        atr_value = (df_ltf['high'] - df_ltf['low']).mean()

    current_price = float(df_ltf['close'].iloc[-1])
    bar_time = df_ltf['time'].iloc[-1]

    df_htf = None
    if config.get('use_htf_context') and htf_timeframe is not None:
        df_htf = get_closed_bars(connector, symbol, htf_timeframe, 60)

    base = {
        'bar_time': bar_time, 'current_price': current_price, 'atr': atr_value,
        'stop_loss': None, 'target': None, 'ai_error': None,
        'ai_model': config.get('ai_model'),
    }

    messages = _build_analysis_messages(
        symbol, config.get('tf_label', str(timeframe)), df_ltf, current_price, atr_value,
        htf_label=config.get('htf_label'), df_htf=df_htf,
    )
    raw, err = call_openrouter_ai(
        messages, config['ai_model'], get_openrouter_api_key(), timeout=config.get('ai_timeout', 30),
        enable_reasoning=config.get('enable_reasoning', False),
    )
    if err:
        return {**base, 'direction': 'NEUTRAL', 'confidence': 0, 'ai_error': err,
                'reasons': [f"AI unavailable this cycle: {err}"]}

    parsed = _parse_ai_analysis(raw)
    if parsed is None:
        return {**base, 'direction': 'NEUTRAL', 'confidence': 0,
                'ai_error': "Could not parse AI response as JSON.",
                'reasons': [f"Could not parse AI response: {raw[:180]}"]}

    reasons = parsed['reasons'] if parsed['reasons'] else ['No reasoning returned by the model.']
    min_confidence = config.get('min_confidence', 65)

    if parsed['verdict'] == 'HOLD':
        return {**base, 'direction': 'NEUTRAL', 'confidence': parsed['confidence'], 'reasons': reasons}

    if parsed['confidence'] < min_confidence:
        reasons = reasons + [f"Confidence {parsed['confidence']:.0f}% below your {min_confidence}% threshold."]
        return {**base, 'direction': 'NEUTRAL', 'confidence': parsed['confidence'], 'reasons': reasons}

    direction = parsed['verdict']
    risk = atr_value * parsed['stop_atr_mult']
    if direction == 'BUY':
        stop_loss = current_price - risk
        target = current_price + risk * parsed['reward_risk_ratio']
    else:
        stop_loss = current_price + risk
        target = current_price - risk * parsed['reward_risk_ratio']

    return {**base, 'direction': direction, 'confidence': parsed['confidence'],
            'stop_loss': stop_loss, 'target': target, 'reasons': reasons}


# ============================================================================
# LOT SIZING & P&L ESTIMATION
# ============================================================================
def calculate_lot_size(symbol_info, account_balance, risk_percent, sl_distance):
    if symbol_info is None or sl_distance <= 0:
        return 0.01, 0.0
    risk_amount = account_balance * (risk_percent / 100)
    point = symbol_info.point or 0.0001
    tick_size = getattr(symbol_info, 'trade_tick_size', None) or point
    tick_value = getattr(symbol_info, 'trade_tick_value', None) or 1.0
    ticks_at_risk = sl_distance / point
    value_per_point = tick_value * (point / tick_size)
    if value_per_point <= 0:
        return 0.01, risk_amount
    raw_lot = risk_amount / (ticks_at_risk * value_per_point)
    step = symbol_info.volume_step or 0.01
    vol_min = symbol_info.volume_min or 0.01
    vol_max = symbol_info.volume_max or 100.0
    lot = round(raw_lot / step) * step
    lot = max(vol_min, min(lot, vol_max))
    return round(lot, 2), risk_amount


def estimate_open_profit(direction, entry_price, current_price, lot_size, symbol_info):
    """Approximate unrealized P&L in account currency for SIMULATION mode
    only (live mode reads the real profit straight from MT5 instead - this
    is an estimate that ignores spread/commission/swap)."""
    if symbol_info is None:
        return 0.0
    point = symbol_info.point or 0.0001
    tick_size = getattr(symbol_info, 'trade_tick_size', None) or point
    tick_value = getattr(symbol_info, 'trade_tick_value', None) or 1.0
    diff = (current_price - entry_price) if direction == 'BUY' else (entry_price - current_price)
    if tick_size <= 0:
        return 0.0
    ticks = diff / tick_size
    return ticks * tick_value * lot_size


# ============================================================================
# BOT STATE
# ============================================================================
@dataclass
class TradeRecord:
    ticket: int
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    profit: float
    exit_reason: str


class BotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = 'idle'          # idle, running, stopping, stopped, error
        self.mode = 'simulation'
        self.open_positions = []      # list of dicts: ticket, direction, entry_price, entry_time, lot_size
        self.last_analysis = None
        self.last_error = None
        self.last_cycle_note = None     # human-readable reason for what happened (or didn't) last cycle
        self.cycles_run = 0
        self.signals_seen = 0           # cycles where AI direction was BUY/SELL above confidence threshold
        self.signals_traded = 0         # cycles that actually resulted in opened positions
        self.trades = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.started_at = None
        self.last_poll_at = None

    def snapshot(self):
        with self.lock:
            return {
                'status': self.status, 'mode': self.mode,
                'open_positions': list(self.open_positions),
                'last_analysis': self.last_analysis, 'last_error': self.last_error,
                'last_cycle_note': self.last_cycle_note, 'cycles_run': self.cycles_run,
                'signals_seen': self.signals_seen, 'signals_traded': self.signals_traded,
                'trades': list(self.trades), 'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades, 'started_at': self.started_at,
                'last_poll_at': self.last_poll_at,
            }


def _close_and_record(bot_state, position, exit_price, reason, connector=None, live=True, profit_override=None):
    ticket = position['ticket']
    direction = position['direction']
    entry_price = position['entry_price']
    entry_time = position['entry_time']

    if profit_override is not None:
        realized_profit = profit_override
    elif live:
        pos = get_position_by_ticket(ticket)
        if pos is not None:
            realized_profit = pos.profit
            close_position(ticket)
        else:
            realized_profit = 0.0
    else:
        diff = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        realized_profit = diff

    record = TradeRecord(
        ticket=ticket or 0, direction=direction, entry_price=entry_price,
        exit_price=exit_price, entry_time=entry_time, exit_time=time.time(),
        profit=realized_profit or 0.0, exit_reason=reason,
    )
    with bot_state.lock:
        bot_state.trades.append(record)
        bot_state.daily_pnl += (realized_profit or 0.0)
        bot_state.open_positions = [p for p in bot_state.open_positions if p['ticket'] != ticket]


def flatten_all_positions(bot_state, connector, symbol, live, reason="closed on stop/flatten (user requested)"):
    with bot_state.lock:
        positions_snapshot = list(bot_state.open_positions)
    if not positions_snapshot:
        return
    for pos in positions_snapshot:
        price_info = connector.get_current_price(symbol)
        cp = price_info['bid'] if price_info else pos['entry_price']
        _close_and_record(bot_state, pos, cp, reason, connector=connector, live=live)


def _manage_open_positions(connector, symbol, config, bot_state, symbol_info):
    """Runs every poll cycle, regardless of whether a new signal fired.
    PRIMARY exit mechanism: any open position whose unrealized profit has
    reached `profit_take_amount` is closed immediately, rather than waiting
    for the broker SL/TP to end the trade. Also detects positions the
    broker's Stop Loss (safety net) already closed, so state stays accurate."""
    with bot_state.lock:
        positions_snapshot = list(bot_state.open_positions)
    if not positions_snapshot:
        return

    if config['live']:
        live_positions = {p.ticket: p for p in get_bot_positions(symbol, BOT_MAGIC)}
        for pos in positions_snapshot:
            mt5_pos = live_positions.get(pos['ticket'])
            if mt5_pos is None:
                price_info = connector.get_current_price(symbol)
                last_price = price_info['bid'] if price_info else pos['entry_price']
                _close_and_record(bot_state, pos, last_price,
                                   "closed by broker SL safety net (or manual close) - check MT5 history for exact P&L",
                                   connector=connector, live=False, profit_override=0.0)
                continue
            if mt5_pos.profit >= config['profit_take_amount']:
                close_position(pos['ticket'])
                _close_and_record(bot_state, pos, mt5_pos.price_current,
                                   f"closed on early profit target (>= {config['profit_take_amount']})",
                                   connector=connector, live=False, profit_override=mt5_pos.profit)
    else:
        price_info = connector.get_current_price(symbol)
        if price_info is None or symbol_info is None:
            return
        current_price = price_info['bid']
        for pos in positions_snapshot:
            profit = estimate_open_profit(pos['direction'], pos['entry_price'], current_price,
                                           pos['lot_size'], symbol_info)
            if profit >= config['profit_take_amount']:
                _close_and_record(bot_state, pos, current_price,
                                   f"closed on early profit target (>= {config['profit_take_amount']}) [simulated]",
                                   connector=connector, live=False, profit_override=profit)


def _open_positions_until_blocked(connector, symbol, config, bot_state, analysis, symbol_info):
    """On a fresh, confidence-cleared AI signal, keep opening positions -
    same direction, same computed SL/target, same lot size - until the
    broker itself rejects the next order (live mode), or a configurable cap
    is hit (simulation mode). Returns the number of positions actually
    opened."""
    account = connector.refresh_account_info()
    balance = account.balance if account else 10000.0
    sl_distance = abs(analysis['stop_loss'] - analysis['current_price'])
    lot_size, _ = calculate_lot_size(symbol_info, balance, config['risk_percent'], sl_distance)

    opened = 0
    while True:
        with bot_state.lock:
            current_open_count = len(bot_state.open_positions)
            daily_trades = bot_state.daily_trades

        if config['max_concurrent_positions'] > 0 and current_open_count >= config['max_concurrent_positions']:
            with bot_state.lock:
                bot_state.last_error = f"Stacking stopped: max_concurrent_positions ({config['max_concurrent_positions']}) reached."
            break
        if config['max_trades'] > 0 and daily_trades >= config['max_trades']:
            with bot_state.lock:
                bot_state.last_error = "Stacking stopped: max trades for this session reached."
            break
        if opened >= STACK_SAFETY_CEILING:
            with bot_state.lock:
                bot_state.last_error = f"Stacking stopped: internal per-cycle safety ceiling ({STACK_SAFETY_CEILING}) reached."
            break

        entry_price = analysis['current_price']
        ticket = None

        if config['live']:
            tp_price = analysis['target'] if config['use_tp_safety_net'] and analysis['target'] else 0.0
            result, msg = execute_market_order(
                connector, symbol, analysis['direction'], lot_size,
                analysis['stop_loss'], tp_price,
            )
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                with bot_state.lock:
                    bot_state.last_error = f"Stacking stopped by broker after {opened} position(s): {msg}"
                break
            ticket = result.order
            entry_price = result.price
        else:
            sim_cap = config['max_concurrent_positions'] if config['max_concurrent_positions'] > 0 else 20
            if current_open_count >= sim_cap:
                with bot_state.lock:
                    bot_state.last_error = (
                        f"Stacking stopped: simulated cap ({sim_cap}) reached - simulation mode has no "
                        f"real margin check, so this stands in for the broker rejection that would occur live."
                    )
                break
            ticket = (int(time.time() * 1000) + opened) % 10_000_000

        with bot_state.lock:
            bot_state.open_positions.append({
                'ticket': ticket, 'direction': analysis['direction'],
                'entry_price': entry_price, 'entry_time': time.time(), 'lot_size': lot_size,
            })
            bot_state.daily_trades += 1
        opened += 1
        time.sleep(0.05)  # brief pacing between stacked orders, not a strategy parameter

    return opened


def automation_loop(connector, symbol, timeframe, htf_timeframe, bars, config, stop_event, bot_state):
    """
    Runs continuously: manages open positions every tick (fast exit checks),
    and - once ALL of this bot's positions have closed - waits
    `reentry_delay_seconds`, then (no more often than `analysis_poll_seconds`,
    to respect OpenRouter/provider rate limits) asks the AI for a fresh verdict and
    opens a new batch if it clears the confidence threshold. Repeats for as
    long as conditions keep being met, until the person clicks Stop or a
    circuit breaker trips.
    """
    with bot_state.lock:
        bot_state.status = 'running'
        bot_state.mode = 'live' if config['live'] else 'simulation'
        if bot_state.started_at is None:
            bot_state.started_at = time.time()

    TICK = 1.0  # fast tick so exits are checked every second; AI calls are throttled separately below
    was_open = False
    last_batch_closed_time = None
    last_eval_time = None

    try:
        while not stop_event.is_set():
            symbol_info = connector.get_symbol_info(symbol)

            # 1. Manage existing positions every tick - active profit-take / SL-detection exit.
            _manage_open_positions(connector, symbol, config, bot_state, symbol_info)

            with bot_state.lock:
                currently_open = len(bot_state.open_positions) > 0
                bot_state.last_poll_at = time.time()
                bot_state.cycles_run += 1
                breached_loss = (config['daily_loss_limit'] > 0 and
                                  bot_state.daily_pnl <= -config['daily_loss_limit'])
                breached_trades = (config['max_trades'] > 0 and
                                    bot_state.daily_trades >= config['max_trades'])

            if breached_loss:
                with bot_state.lock:
                    bot_state.last_error = "Daily loss limit reached - automation stopped itself."
                break

            if was_open and not currently_open:
                # A batch just fully closed (by profit-take or SL) - start the re-entry cooldown.
                last_batch_closed_time = time.time()
                with bot_state.lock:
                    bot_state.last_cycle_note = (
                        f"Batch closed. Re-entering in {config['reentry_delay_seconds']}s if the AI still gives a signal."
                    )
            was_open = currently_open

            if breached_trades:
                with bot_state.lock:
                    bot_state.last_error = "Max trades for this session reached - automation stopped itself."
                if not currently_open:
                    break
                # Otherwise let the still-open batch keep being managed/closed above;
                # the loop will hit the break above on a later tick once it's flat.
            elif not currently_open:
                cooldown_ok = (
                    last_batch_closed_time is None or
                    (time.time() - last_batch_closed_time) >= config['reentry_delay_seconds']
                )
                poll_ok = (
                    last_eval_time is None or
                    (time.time() - last_eval_time) >= config['analysis_poll_seconds']
                )
                if cooldown_ok and poll_ok:
                    last_eval_time = time.time()
                    _evaluate_and_maybe_trade(connector, symbol, timeframe, htf_timeframe, bars,
                                               config, bot_state, symbol_info)
                else:
                    wait_for = 0.0
                    if not cooldown_ok:
                        wait_for = max(wait_for, config['reentry_delay_seconds'] - (time.time() - last_batch_closed_time))
                    if not poll_ok:
                        wait_for = max(wait_for, config['analysis_poll_seconds'] - (time.time() - last_eval_time))
                    with bot_state.lock:
                        bot_state.last_cycle_note = f"Waiting {wait_for:.0f}s before the next AI check."

            stop_event.wait(TICK)

        with bot_state.lock:
            bot_state.status = 'stopping'
        flatten_all_positions(bot_state, connector, symbol, config['live'], reason="closed on stop (user requested)")

        with bot_state.lock:
            bot_state.status = 'stopped'

    except Exception as e:  # noqa: BLE001
        with bot_state.lock:
            bot_state.status = 'error'
            bot_state.last_error = f"Automation crashed: {e}"


def _evaluate_and_maybe_trade(connector, symbol, timeframe, htf_timeframe, bars, config, bot_state, symbol_info):
    """Asks the AI for a fresh verdict on the latest closed candle and, if
    it's a BUY/SELL that clears the confidence threshold, opens a new
    batch. Only called when the bot currently has no open positions, the
    re-entry cooldown has elapsed, and the AI poll interval has elapsed -
    so this is what drives continuous re-entry after each batch closes."""
    df = get_closed_bars(connector, symbol, timeframe, bars)
    if df is None or len(df) < 20:
        with bot_state.lock:
            bot_state.last_error = f"No data for {symbol} this cycle."
        return

    analysis = analyze_market_ai(connector, symbol, timeframe, htf_timeframe, df, config)
    with bot_state.lock:
        bot_state.last_analysis = analysis
        bot_state.last_error = analysis['ai_error']

    if analysis['direction'] not in ('BUY', 'SELL'):
        with bot_state.lock:
            bot_state.last_cycle_note = analysis['reasons'][0] if analysis['reasons'] else "No trade this cycle."
        return

    with bot_state.lock:
        bot_state.signals_seen += 1

    opened = _open_positions_until_blocked(connector, symbol, config, bot_state, analysis, symbol_info)

    with bot_state.lock:
        if opened > 0:
            bot_state.signals_traded += 1
            bot_state.last_cycle_note = f"Opened {opened} {analysis['direction']} position(s)."
        elif bot_state.last_error is None:
            bot_state.last_cycle_note = "Signal fired but no positions were opened this attempt."


# ============================================================================
# CHARTING (manual "Analyze Now" preview only)
# ============================================================================
def plot_chart(df, analysis, symbol):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'],
                                  low=df['low'], close=df['close'], name='Price'))

    if analysis['direction'] in ('BUY', 'SELL') and analysis.get('stop_loss') and analysis.get('target'):
        fig.add_hline(y=analysis['stop_loss'], line_dash="dot", line_color="#FF6B6B",
                      annotation_text="SL (safety net)", annotation_position="right")
        fig.add_hline(y=analysis['target'], line_dash="dot", line_color="#00B4D8",
                      annotation_text="AI target (TP safety net)", annotation_position="right")

    fig.update_layout(title=f"{symbol} - AI Preview", template="plotly_dark", height=600,
                       showlegend=True, xaxis_rangeslider_visible=False)
    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    connector = get_connector()

    st.markdown('<div class="main-header">🤖 TradePilot AUTO - AI Market Analyst</div>',
                unsafe_allow_html=True)
    st.markdown("### Analyze → Arm → Start. Every signal is a live LLM (via OpenRouter) reading raw price action - no fixed indicator rules.")
    st.divider()

    if 'bot_state' not in st.session_state:
        st.session_state.bot_state = BotState()
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if 'preview' not in st.session_state:
        st.session_state.preview = None

    bot_state = st.session_state.bot_state
    ai_key_present = bool(get_openrouter_api_key())

    with st.sidebar:
        st.header("⚙️ Connection & Market")
        if connector.connect():
            account = connector.account_info
            mode_label, is_real = get_account_mode_label(account)
            badge = "🔴 REAL MONEY" if is_real else f"🟢 {mode_label}"
            st.success("Connected to MT5")
            st.info(f"#{account.login} | Balance: ${account.balance:.2f} | {badge}")
        else:
            st.error("Not connected to MT5 - make sure the terminal is running and logged in.")

        if not ai_key_present:
            st.error("OPENROUTER_API_KEY not found in environment/.env - the AI analyst can't run until it's set.")

        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD']
        symbol = st.selectbox("Symbol", symbols, index=0)

        timeframe_map = {
            "1 Minute": mt5.TIMEFRAME_M1, "5 Minutes": mt5.TIMEFRAME_M5,
            "15 Minutes": mt5.TIMEFRAME_M15, "30 Minutes": mt5.TIMEFRAME_M30, "1 Hour": mt5.TIMEFRAME_H1,
        }
        tf_choice = st.selectbox("Signal timeframe", list(timeframe_map.keys()), index=1)
        timeframe = timeframe_map[tf_choice]
        bars = st.slider("Bars to fetch", 50, 500, 150)

        st.divider()
        st.header("🤖 AI Analysis Settings")
        ai_model = st.text_input(
            "OpenRouter model", value=get_default_openrouter_model(),
            help="Any model slug OpenRouter supports, e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4-5, "
                 "or google/gemini-2.0-flash-001 - see openrouter.ai/models. Pre-filled from OPENROUTER_MODEL "
                 "in your .env if set. The base URL is the same for every model - only this string changes."
        )
        min_confidence = st.slider(
            "Minimum AI confidence to trade (%)", 50, 95, 65,
            help="A BUY/SELL verdict below this confidence is treated as no-trade for that cycle."
        )
        use_htf_context = st.checkbox(
            "Include higher-timeframe context", value=True,
            help="Attaches recent candles from a higher timeframe to the prompt so the AI can weigh the "
                 "broader trend. It's context for the model to reason about, not a separate rule."
        )
        htf_map = {"15 Minutes": mt5.TIMEFRAME_M15, "1 Hour": mt5.TIMEFRAME_H1, "4 Hours": mt5.TIMEFRAME_H4}
        htf_choice = st.selectbox("Higher timeframe for context", list(htf_map.keys()), index=1,
                                   disabled=not use_htf_context)
        htf_timeframe = htf_map[htf_choice]
        ai_timeout = st.number_input(
            "AI request timeout (seconds)", min_value=5, max_value=90, value=30, step=5,
            help="Some models routed through OpenRouter are slower than others. Raise this if you see "
                 "'Read timed out' errors."
        )
        enable_reasoning = st.checkbox(
            "Enable extended reasoning (if the model supports it)", value=False,
            help="Adds {'reasoning': {'enabled': true}} to the request. Improves quality on "
                 "reasoning-capable models (e.g. x-ai/grok-4.3, DeepSeek reasoning variants) at the cost "
                 "of extra latency and tokens - raise the timeout above if you turn this on."
        )
        analysis_poll_seconds = st.slider(
            "AI polling interval while flat (seconds)", 5, 60, 15,
            help="Caps how often the bot spends an OpenRouter API call while it has no open positions, so "
                 "it doesn't hammer the API (and its rate limit) once a second."
        )

        st.divider()
        st.header("💰 Exit & Position Sizing")
        profit_take_amount = st.number_input(
            "Close position as soon as profit reaches (account currency)", min_value=0.01, value=0.20, step=0.01,
            help="Checked every poll cycle for every open position this bot holds. This is what actually "
                 "closes trades under normal operation, not the broker TP."
        )
        use_tp_safety_net = st.checkbox("Also attach the AI's target as a broker-side TP safety net", value=False,
                                         help="Off by default - exits are driven purely by the live profit check above. "
                                              "SL is always attached regardless of this setting.")
        max_concurrent_positions = st.number_input(
            "Max concurrent positions per signal (0 = unlimited - stack until broker rejects)",
            min_value=0, value=0, step=1,
            help="On each confidence-cleared signal, the bot keeps opening same-direction positions until "
                 "this cap is hit. Live: 0 means it stacks until MT5 rejects an order (commonly insufficient "
                 "margin). Simulation: 0 falls back to a cap of 20 positions."
        )
        if max_concurrent_positions == 0:
            st.warning(
                "Unlimited stacking is enabled. Every AI signal that clears your confidence threshold will "
                "keep opening same-direction positions until your broker rejects the next one. All stacked "
                "positions share the same signal and a similar entry price - this multiplies exposure to "
                "that one verdict being wrong rather than diversifying it. Set a cap above if you want to "
                "bound worst-case exposure."
            )
        risk_percent = st.slider("Risk per trade (%, used for each stacked position's lot size)", 0.1, 2.0, 0.5, 0.1)

        st.divider()
        st.header("🤖 Automation Settings")
        live_enabled = st.checkbox(
            "Arm live order execution", value=False,
            help="Off = simulation mode: full logic runs and would-be trades are logged, but no real "
                 "order is sent. On = real orders on the connected account."
        )
        if live_enabled:
            st.warning("Armed - Start Automation will send real orders to MT5.")
        else:
            st.caption("Simulation mode - safe to test the logic before arming.")

        reentry_delay_seconds = st.slider(
            "Re-entry delay after a batch fully closes (seconds)", 0, 30, 2,
            help="Once every position from the last batch has closed, the bot waits this long before it's "
                 "even eligible to ask the AI for a fresh verdict (still subject to the polling interval above)."
        )

        default_loss_cap = round((connector.account_info.balance * 0.02), 2) if connector.account_info else 20.0
        daily_loss_limit = st.number_input(
            "Daily loss limit (account currency, 0 = disabled)", min_value=0.0,
            value=default_loss_cap, step=1.0,
            help="Kill-switch: automation stops itself if losses reach this. Defaults to ~2% of balance."
        )
        max_trades = st.number_input("Max trades this session (0 = unlimited)", min_value=0, value=50, step=5,
                                      help="Counts every stacked position individually, not just signals.")

    config = {
        'live': live_enabled, 'risk_percent': risk_percent, 'reentry_delay_seconds': reentry_delay_seconds,
        'daily_loss_limit': daily_loss_limit, 'max_trades': max_trades,
        'profit_take_amount': profit_take_amount, 'max_concurrent_positions': max_concurrent_positions,
        'use_tp_safety_net': use_tp_safety_net,
        'ai_model': ai_model, 'min_confidence': min_confidence, 'ai_timeout': ai_timeout,
        'analysis_poll_seconds': analysis_poll_seconds, 'enable_reasoning': enable_reasoning,
        'use_htf_context': use_htf_context, 'tf_label': tf_choice, 'htf_label': htf_choice,
    }

    running = bot_state.snapshot()['status'] in ('running', 'stopping')

    st.markdown("### 🔍 Preview")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📊 Analyze Now (preview only, places no trade, uses 1 AI call)",
                      use_container_width=True, disabled=running or not ai_key_present):
            with st.spinner(f"Asking the AI to analyze {symbol}..."):
                df = get_closed_bars(connector, symbol, timeframe, bars)
                if df is not None and len(df) >= 20:
                    analysis = analyze_market_ai(connector, symbol, timeframe, htf_timeframe, df, config)
                    st.session_state.preview = {'analysis': analysis, 'df': df}
                else:
                    st.error("Not enough closed-candle data available for this symbol/timeframe yet.")

    with col_b:
        if not running:
            start_clicked = st.button("🚀 Execute (Start Automation)", use_container_width=True,
                                       type="primary", disabled=not ai_key_present)
        else:
            st.button("🚀 Execute (Start Automation)", use_container_width=True, disabled=True)
            start_clicked = False

    if st.session_state.preview:
        analysis = st.session_state.preview['analysis']
        df = st.session_state.preview['df']
        if analysis.get('ai_error'):
            st.error(f"AI analysis unavailable: {analysis['ai_error']}")
        css_class = {'BUY': 'signal-buy', 'SELL': 'signal-sell'}.get(analysis['direction'], 'signal-neutral')
        icon = {'BUY': '📈', 'SELL': '📉'}.get(analysis['direction'], '⚖️')
        st.markdown(f'<div class="{css_class}">{icon} {analysis["direction"]}</div>', unsafe_allow_html=True)
        st.caption(f"AI confidence: **{analysis.get('confidence', 0):.0f}%** | model: {analysis.get('ai_model')}")
        for r in analysis['reasons']:
            st.markdown(f"- {r}")
        st.plotly_chart(plot_chart(df, analysis, symbol), use_container_width=True)

    if start_clicked:
        st.session_state.stop_event = threading.Event()
        st.session_state.bot_state = BotState()
        bot_state = st.session_state.bot_state

        with bot_state.lock:
            bot_state.status = 'running'
            bot_state.mode = 'live' if live_enabled else 'simulation'
            bot_state.started_at = time.time()

        thread = threading.Thread(
            target=automation_loop,
            args=(connector, symbol, timeframe, htf_timeframe, bars, config, st.session_state.stop_event, bot_state),
            daemon=True,
        )
        thread.start()
        st.rerun()

    st.divider()
    st.markdown("### 📡 Automation Status")

    snap = bot_state.snapshot()
    status_class = {
        'running': 'status-running', 'stopping': 'status-stopping',
        'stopped': 'status-idle', 'idle': 'status-idle', 'error': 'status-error',
    }.get(snap['status'], 'status-idle')
    st.markdown(f'<span class="{status_class}">{snap["status"].upper()}</span> &nbsp; '
                f'mode: **{snap["mode"]}**', unsafe_allow_html=True)

    if snap['last_error']:
        st.warning(snap['last_error'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily P&L", f"{snap['daily_pnl']:.2f}")
    col2.metric("Trades this session", snap['daily_trades'])
    col3.metric("Open positions", len(snap['open_positions']))
    col4.metric("Batches traded", f"{snap['signals_traded']}/{snap['signals_seen']}")

    if snap.get('last_cycle_note'):
        st.caption(f"🔄 {snap['last_cycle_note']}")

    a = snap['last_analysis']
    if a and a.get('current_price') is not None:
        st.caption(
            f"Last AI check ({a['bar_time']}): price {a['current_price']:.5f} | "
            f"verdict: **{a['direction']}** | confidence {a.get('confidence', 0):.0f}% | "
            f"model: {a.get('ai_model')}"
        )
    elif snap['status'] in ('running', 'stopping'):
        st.caption("Waiting on first cycle to complete (fetching data / asking the AI)...")

    if snap['open_positions']:
        st.markdown("#### 📌 This Bot's Open Positions")
        open_df = pd.DataFrame([{
            'ticket': p['ticket'], 'direction': p['direction'],
            'entry': round(p['entry_price'], 5), 'lot_size': p['lot_size'],
        } for p in snap['open_positions']])
        st.dataframe(open_df, use_container_width=True, hide_index=True)

    if snap['trades']:
        st.markdown("#### 📜 Trade Log (this session)")
        log_df = pd.DataFrame([{
            'ticket': t.ticket, 'direction': t.direction, 'entry': round(t.entry_price, 5),
            'exit': round(t.exit_price, 5), 'profit': round(t.profit, 5), 'reason': t.exit_reason,
        } for t in reversed(snap['trades'])])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    col_stop, col_force = st.columns(2)
    with col_stop:
        if st.button("⏹ Stop Automation", use_container_width=True,
                      disabled=snap['status'] not in ('running',)):
            st.session_state.stop_event.set()
            st.rerun()
    with col_force:
        if st.button("🛑 Force Flatten Now", use_container_width=True,
                      disabled=snap['status'] not in ('running', 'stopping')):
            st.session_state.stop_event.set()
            flatten_all_positions(bot_state, connector, symbol, snap['mode'] == 'live',
                                   reason="force-flattened by user")
            st.rerun()

    st.divider()
    st.markdown("### 📂 Open MT5 Positions (all, not just this bot's)")
    positions_df = get_open_positions()
    if positions_df.empty:
        st.caption("No open positions.")
    else:
        st.dataframe(positions_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("⚠️ For educational purposes. Not financial advice. AI verdicts are probabilistic, not "
               "guaranteed forecasts. Test in simulation/demo before arming live orders.")

    if snap['status'] in ('running', 'stopping'):
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()