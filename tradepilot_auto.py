"""
TradePilot AUTO - 15m Bollinger Band Mean Reversion Bot
==========================================================

Strategy signal logic (per your pseudocode, implemented as-is):

    On each newly closed 15m candle:
      - Compute Bollinger Bands (20, 2.0) and RSI(14) for the closed candle.
      - SHORT: previous candle's high poked above the previous upper band,
        current candle's close is back inside the current upper band, and
        current RSI >= 70 (overbought). SL = highest high of the last 3
        candles + 5 pips. Structural target = the middle band (the mean).
      - LONG: previous candle's low poked below the previous lower band,
        current candle's close is back inside the current lower band, and
        current RSI <= 30 (oversold). SL = lowest low of the last 3 candles
        - 5 pips. Structural target = the middle band (the mean).
      - Only one setup is checked per cycle in short-circuit order (short
        first, then long), matching the pseudocode's sequential IF/RETURN
        structure - exactly one signal (or none) can fire per candle.

TRADE MANAGEMENT AND SIZING - CHANGED PER YOUR LAST REQUEST:
  1. Exit no longer waits on the broker Stop Loss to be the thing that ends
     a trade. Every poll cycle, every open position this bot holds is
     checked for unrealized profit, and is closed the moment that profit
     reaches `profit_take_amount` (account currency, default 0.20 - i.e.
     a very small "any green is good enough" exit). A broker-side Stop
     Loss is still attached to every order as a crash/disconnect safety
     net (see CAVEATS) - it's just no longer the thing this bot is relying
     on to end a winning or losing trade under normal operation. The
     broker-side "take profit" (middle band) is likewise kept only as a
     distant backstop; in practice the active +profit_take_amount exit
     will almost always fire first.
  2. HasOpenPosition() -> RETURN has been removed. On each newly closed
     candle that produces a BUY/SELL signal, the bot now opens positions
     repeatedly - same direction, same computed SL/target, same lot size -
     until it can't anymore:
        - Live mode: until the broker actually rejects an order (e.g.
          "not enough money" / margin). That broker rejection IS the "no
          other position can be placed" condition you asked for - it's
          the real, authoritative stopping point, not a number this app
          invents.
        - Simulation mode: there is no real broker margin to hit, so a
          configurable `max_concurrent_positions` cap stands in for that
          condition (defaults to 20 if left at 0/unlimited, since an
          uncapped simulation loop has no natural stopping point at all).
     A session-level `max_trades` cap (existing feature) and a hard
     internal per-cycle safety ceiling (200 orders/cycle) still apply
     underneath this, purely as an engineering guard against a runaway
     loop - they are not meant to be the normal stopping condition.

READ THIS BEFORE ARMING LIVE - RISK OF THIS CONFIGURATION:
  Stacking same-direction positions on one signal, instead of using the
  signal to open exactly one position, does NOT diversify risk - every
  stacked position is opened at nearly the same price, in the same
  direction, off the same candle. It multiplies exposure to that single
  signal being wrong, and (live mode) can consume all of the account's
  free margin in one candle close, which is the mechanism that stops the
  stacking loop. If that signal is wrong before the +profit_take_amount
  exit can trigger, every stacked position loses together and the
  per-trade Stop Loss is what's left standing between that and a margin
  call. This is a materially higher-risk configuration than the single-
  position version. Test in simulation mode and/or a demo account first,
  and consider setting `max_concurrent_positions` to a non-zero cap rather
  than leaving stacking uncapped, if you want to bound worst-case exposure
  per signal.

CAVEATS (unchanged):
  * A real broker-side Stop Loss is attached to every order and is what
    actually protects the account if this app crashes, the machine loses
    power, or the connection drops - do not remove it, even though normal
    exits now happen earlier via the active profit-take check.
  * This file uses a background thread + shared state in st.session_state
    to run the loop. That's tied to this one browser session/server
    process; closing the tab or a server restart means you lose your Stop
    button while positions could still be open. Always be able to
    check/close positions directly in the MT5 terminal as a fallback.
  * Not financial advice, no strategy is guaranteed to be profitable.

FIX LOG:
  * Startup race condition fixed: bot_state.status is set to 'running'
    synchronously in the MAIN thread, before the background automation
    thread is started, so the immediate st.rerun() after thread.start()
    always sees 'running' and the auto-refresh loop engages right away.
  * Strategy engine: Bollinger Band(20, 2.0) + RSI(14) mean reversion,
    replacing the earlier EMA/RSI pullback logic.
  * Exit management: active "close as soon as unrealized profit >=
    profit_take_amount" check every poll cycle, instead of waiting on the
    broker SL/TP to end a trade (SL is retained as a crash-protection
    safety net only - see above).
  * Position sizing per signal: stacks positions until the broker rejects
    further orders (live) or a configurable cap is hit (simulation),
    instead of opening exactly one position per signal. See the risk note
    above before arming live.
"""

import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TradePilot AUTO - Bollinger Mean Reversion Bot",
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
BOT_MAGIC = 234200          # identifies this bot's own orders/positions
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

    def get_rates(self, symbol, timeframe=mt5.TIMEFRAME_M15, bars=500):
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


def get_pip_value(symbol_info):
    """Standard convention: 5- and 3-digit quoted symbols use 1 pip = 10 points;
    2- and 4-digit symbols use 1 pip = 1 point."""
    if symbol_info is None:
        return 0.0001
    point = symbol_info.point or 0.0001
    digits = getattr(symbol_info, 'digits', 5)
    return point * 10 if digits in (3, 5) else point


def get_closed_bars(connector, symbol, timeframe, bars):
    """Fetch bars and drop the currently-forming (incomplete) candle so the
    strategy only ever evaluates fully-closed candles."""
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
                          deviation=20, magic=BOT_MAGIC, comment="TradePilot-BB-RSI"):
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


def close_position(ticket, deviation=20, comment="TradePilot-BB-RSI close"):
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
# STRATEGY ENGINE: 15m Bollinger Band(20, 2.0) + RSI(14) mean reversion
# ============================================================================
class BollingerRSIMeanReversionPredictor:
    """Implements the pseudocode exactly:
        - SHORT: prev candle's high poked above prev upper band, current
          close is back inside the current upper band, current RSI >=
          overbought threshold. SL = highest high of last 3 candles + pip
          buffer. Structural target = current middle band.
        - LONG: prev candle's low poked below prev lower band, current
          close is back inside the current lower band, current RSI <=
          oversold threshold. SL = lowest low of last 3 candles - pip
          buffer. Structural target = current middle band.
      Checked short-circuit, short first then long, so at most one signal
      fires per candle - matching the pseudocode's sequential IF/RETURN.
    """

    def __init__(self, df, pip_value, cfg):
        self.df = df.copy().reset_index(drop=True)
        self.pip = pip_value
        self.cfg = cfg  # bb_period, bb_stddev, rsi_period, rsi_overbought, rsi_oversold, sl_buffer_pips

    def _bollinger(self, period, stddev_mult):
        middle = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std(ddof=0)
        upper = middle + stddev_mult * std
        lower = middle - stddev_mult * std
        return upper, middle, lower

    def _rsi(self, period):
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def analyze(self):
        cfg = self.cfg
        min_bars = max(cfg['bb_period'], cfg['rsi_period']) + 5
        if len(self.df) < min_bars:
            return {
                'direction': 'NEUTRAL', 'bar_time': None, 'current_price': None,
                'upper_band': None, 'middle_band': None, 'lower_band': None,
                'rsi_current': None, 'stop_loss': None, 'target': None,
                'reasons': [f"Need at least {min_bars} closed bars, have {len(self.df)}."],
            }

        upper_series, middle_series, lower_series = self._bollinger(cfg['bb_period'], cfg['bb_stddev'])
        rsi_series = self._rsi(cfg['rsi_period'])

        current_price = self.df['close'].iloc[-1]
        bar_time = self.df['time'].iloc[-1]
        upper_band = upper_series.iloc[-1]
        middle_band = middle_series.iloc[-1]
        lower_band = lower_series.iloc[-1]
        prev_upper_band = upper_series.iloc[-2]
        prev_lower_band = lower_series.iloc[-2]
        rsi_current = rsi_series.iloc[-1]

        prev_high = self.df['high'].iloc[-2]
        prev_low = self.df['low'].iloc[-2]

        reasons = []
        direction = 'NEUTRAL'
        stop_loss = None
        target = None

        # ---- SHORT setup: pierced upper band, closed back inside, RSI overbought ----
        price_poked_above = prev_high > prev_upper_band
        closed_back_inside_upper = current_price < upper_band
        reasons.append(
            f"Prev high {prev_high:.5f} {'>' if price_poked_above else '<='} prev upper band "
            f"{prev_upper_band:.5f} ({'poked above' if price_poked_above else 'no poke'})"
        )

        if price_poked_above and closed_back_inside_upper and rsi_current >= cfg['rsi_overbought']:
            reasons.append(f"Closed back inside upper band ({current_price:.5f} < {upper_band:.5f})")
            reasons.append(f"RSI {rsi_current:.1f} >= overbought threshold {cfg['rsi_overbought']}")
            direction = 'SELL'
            highest_high = self.df['high'].iloc[-3:].max()
            stop_loss = highest_high + cfg['sl_buffer_pips'] * self.pip
            target = middle_band
        else:
            if price_poked_above and not closed_back_inside_upper:
                reasons.append("Price poked above but has not closed back inside the upper band yet")
            elif price_poked_above and rsi_current < cfg['rsi_overbought']:
                reasons.append(f"RSI {rsi_current:.1f} below overbought threshold {cfg['rsi_overbought']}")

            # ---- LONG setup: pierced lower band, closed back inside, RSI oversold ----
            price_poked_below = prev_low < prev_lower_band
            closed_back_inside_lower = current_price > lower_band
            reasons.append(
                f"Prev low {prev_low:.5f} {'<' if price_poked_below else '>='} prev lower band "
                f"{prev_lower_band:.5f} ({'poked below' if price_poked_below else 'no poke'})"
            )

            if price_poked_below and closed_back_inside_lower and rsi_current <= cfg['rsi_oversold']:
                reasons.append(f"Closed back inside lower band ({current_price:.5f} > {lower_band:.5f})")
                reasons.append(f"RSI {rsi_current:.1f} <= oversold threshold {cfg['rsi_oversold']}")
                direction = 'BUY'
                lowest_low = self.df['low'].iloc[-3:].min()
                stop_loss = lowest_low - cfg['sl_buffer_pips'] * self.pip
                target = middle_band
            elif price_poked_below and not closed_back_inside_lower:
                reasons.append("Price poked below but has not closed back inside the lower band yet")
            elif price_poked_below and rsi_current > cfg['rsi_oversold']:
                reasons.append(f"RSI {rsi_current:.1f} above oversold threshold {cfg['rsi_oversold']}")

        return {
            'direction': direction, 'bar_time': bar_time, 'current_price': current_price,
            'upper_band': upper_band, 'middle_band': middle_band, 'lower_band': lower_band,
            'rsi_current': rsi_current, 'stop_loss': stop_loss, 'target': target,
            'reasons': reasons,
        }


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
        self.last_evaluated_bar_time = None
        self.last_error = None
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
                'trades': list(self.trades), 'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades, 'started_at': self.started_at,
                'last_poll_at': self.last_poll_at,
            }


def _close_and_record(bot_state, position, exit_price, reason, connector=None, live=True, profit_override=None):
    """Removes `position` from bot_state.open_positions and logs a
    TradeRecord. If live and profit_override is None, attempts to close it
    on the broker first and uses the broker's realized profit."""
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
    """Closes every position this bot currently has open, live or simulated."""
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
    Per your request, this is now the PRIMARY exit mechanism: any open
    position whose unrealized profit has reached `profit_take_amount` is
    closed immediately, rather than waiting for the broker SL/TP to end
    the trade. Also detects positions the broker's Stop Loss (safety net)
    already closed, so the log/state stay accurate."""
    with bot_state.lock:
        positions_snapshot = list(bot_state.open_positions)
    if not positions_snapshot:
        return

    if config['live']:
        live_positions = {p.ticket: p for p in get_bot_positions(symbol, BOT_MAGIC)}
        for pos in positions_snapshot:
            mt5_pos = live_positions.get(pos['ticket'])
            if mt5_pos is None:
                # No longer open on the broker - the SL safety net (or a
                # manual close) already ended it. Exact P&L should be
                # confirmed in MT5 history; we don't have it here.
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
    """Per your request: on a fresh signal, keep opening positions - same
    direction, same computed SL/target, same lot size - until the broker
    itself rejects the next order (live mode), or a configurable cap is
    hit (simulation mode, where there's no real margin to run out of).
    Returns the number of positions actually opened this call."""
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
            result, msg = execute_market_order(
                connector, symbol, analysis['direction'], lot_size,
                analysis['stop_loss'], analysis['target'],
            )
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                # This IS the "no other position can be placed" condition -
                # the broker rejected it (commonly: not enough free margin).
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


def automation_loop(connector, symbol, timeframe, bars, config, stop_event, bot_state):
    # status/mode/started_at are set by the caller (main thread) right
    # before this thread is started - see the `start_clicked` block in
    # main(). Set here too, defensively.
    with bot_state.lock:
        bot_state.status = 'running'
        bot_state.mode = 'live' if config['live'] else 'simulation'
        if bot_state.started_at is None:
            bot_state.started_at = time.time()

    try:
        while not stop_event.is_set():
            _run_one_cycle(connector, symbol, timeframe, bars, config, bot_state)

            with bot_state.lock:
                bot_state.last_poll_at = time.time()
                breached_loss = (config['daily_loss_limit'] > 0 and
                                  bot_state.daily_pnl <= -config['daily_loss_limit'])
                breached_trades = (config['max_trades'] > 0 and
                                    bot_state.daily_trades >= config['max_trades'])
            if breached_loss:
                with bot_state.lock:
                    bot_state.last_error = "Daily loss limit reached - automation stopped itself."
                break
            if breached_trades:
                with bot_state.lock:
                    bot_state.last_error = "Max trades for this session reached - automation stopped itself."
                break

            stop_event.wait(config['poll_interval'])

        # ---- winding down: stop opening new trades, close everything now ----
        with bot_state.lock:
            bot_state.status = 'stopping'
        flatten_all_positions(bot_state, connector, symbol, config['live'], reason="closed on stop (user requested)")

        with bot_state.lock:
            bot_state.status = 'stopped'

    except Exception as e:  # noqa: BLE001
        with bot_state.lock:
            bot_state.status = 'error'
            bot_state.last_error = f"Automation crashed: {e}"


def _run_one_cycle(connector, symbol, timeframe, bars, config, bot_state):
    symbol_info = connector.get_symbol_info(symbol)

    # 1. Manage existing positions FIRST, every cycle - this is the active
    #    early-profit-take exit, independent of whether a new signal fires.
    _manage_open_positions(connector, symbol, config, bot_state, symbol_info)

    # 2. Evaluate the strategy on the latest closed candle.
    df = get_closed_bars(connector, symbol, timeframe, bars)
    if df is None or len(df) < 20:
        with bot_state.lock:
            bot_state.last_error = f"No data for {symbol} this cycle."
        return

    pip = get_pip_value(symbol_info)
    strategy_cfg = {
        'bb_period': config['bb_period'], 'bb_stddev': config['bb_stddev'],
        'rsi_period': config['rsi_period'], 'rsi_overbought': config['rsi_overbought'],
        'rsi_oversold': config['rsi_oversold'], 'sl_buffer_pips': config['sl_buffer_pips'],
    }
    analysis = BollingerRSIMeanReversionPredictor(df, pip, strategy_cfg).analyze()
    with bot_state.lock:
        bot_state.last_analysis = analysis
        if analysis['bar_time'] is not None:
            bot_state.last_error = None  # clear stale "no data" warning now that a cycle succeeded

    if analysis['bar_time'] is None:
        return

    with bot_state.lock:
        already_evaluated = (bot_state.last_evaluated_bar_time == analysis['bar_time'])
        bot_state.last_evaluated_bar_time = analysis['bar_time']

    if already_evaluated:
        return  # this candle already produced a decision - wait for the next one

    if analysis['direction'] not in ('BUY', 'SELL'):
        return  # no signal this candle - by design, nothing is opened

    # 3. Signal fired: stack positions until the broker (or the sim cap) blocks it.
    _open_positions_until_blocked(connector, symbol, config, bot_state, analysis, symbol_info)


# ============================================================================
# CHARTING (manual "Analyze Now" preview only)
# ============================================================================
def plot_chart(df, analysis, symbol, cfg):
    predictor = BollingerRSIMeanReversionPredictor(df, 0.0001, cfg)
    upper_series, middle_series, lower_series = predictor._bollinger(cfg['bb_period'], cfg['bb_stddev'])
    rsi = predictor._rsi(cfg['rsi_period'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'],
                                  low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=upper_series, name="Upper Band",
                              line=dict(color="#FF6B6B", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=middle_series, name="Middle Band (SMA)",
                              line=dict(color="#FFD93D", width=1.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=lower_series, name="Lower Band",
                              line=dict(color="#00FF88", width=1.2)), row=1, col=1)

    if analysis['direction'] in ('BUY', 'SELL'):
        fig.add_hline(y=analysis['stop_loss'], line_dash="dot", line_color="#FF6B6B", row=1, col=1,
                      annotation_text="SL (safety net)", annotation_position="right")
        fig.add_hline(y=analysis['target'], line_dash="dot", line_color="#00B4D8", row=1, col=1,
                      annotation_text="Structural target (mid band)", annotation_position="right")

    fig.add_trace(go.Scatter(x=df['time'], y=rsi, name="RSI 14", line=dict(color="#C77DFF")), row=2, col=1)
    fig.add_hline(y=cfg['rsi_overbought'], line_dash="dash", line_color="#FF6B6B", row=2, col=1)
    fig.add_hline(y=cfg['rsi_oversold'], line_dash="dash", line_color="#00FF88", row=2, col=1)

    fig.update_layout(title=f"{symbol} - Bollinger Mean Reversion Preview", template="plotly_dark", height=650,
                       showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    connector = get_connector()

    st.markdown('<div class="main-header">🤖 TradePilot AUTO - Bollinger Mean Reversion Bot</div>',
                unsafe_allow_html=True)
    st.markdown("### Analyze → Arm → Start. Stacks positions per signal, exits each on first sign of profit.")
    st.divider()

    if 'bot_state' not in st.session_state:
        st.session_state.bot_state = BotState()
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if 'preview' not in st.session_state:
        st.session_state.preview = None

    bot_state = st.session_state.bot_state

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

        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD']
        symbol = st.selectbox("Symbol", symbols, index=0)

        timeframe_map = {
            "5 Minutes": mt5.TIMEFRAME_M5, "15 Minutes": mt5.TIMEFRAME_M15,
            "30 Minutes": mt5.TIMEFRAME_M30, "1 Hour": mt5.TIMEFRAME_H1,
        }
        tf_choice = st.selectbox("Signal timeframe", list(timeframe_map.keys()), index=1)
        timeframe = timeframe_map[tf_choice]

        st.divider()
        st.header("📐 Strategy Parameters")
        bb_period = st.number_input("Bollinger Band period", min_value=5, value=20, step=1)
        bb_stddev = st.number_input("Bollinger Band std dev", min_value=0.5, value=2.0, step=0.1)
        bars = st.slider("Bars to fetch", 100, 1000, 300)
        rsi_period = st.number_input("RSI period", min_value=2, value=14, step=1)
        rsi_overbought = st.number_input("RSI overbought threshold", min_value=50, max_value=99, value=70, step=1)
        rsi_oversold = st.number_input("RSI oversold threshold", min_value=1, max_value=50, value=30, step=1)
        sl_buffer_pips = st.number_input("Stop-loss buffer beyond swing (pips)", min_value=0.0, value=5.0, step=0.5,
                                          help="This SL is now a crash/disconnect safety net only - normal exits "
                                               "happen earlier via the profit-take setting below.")

        st.divider()
        st.header("💰 Exit & Position Sizing")
        profit_take_amount = st.number_input(
            "Close position as soon as profit reaches (account currency)", min_value=0.01, value=0.20, step=0.01,
            help="Checked every poll cycle for every open position this bot holds. Replaces waiting on the "
                 "broker Stop Loss/Take Profit to end a trade - the SL is still sent to the broker as a "
                 "crash-protection safety net, but under normal operation this is what actually closes trades."
        )
        max_concurrent_positions = st.number_input(
            "Max concurrent positions per signal (0 = unlimited - stack until broker rejects)",
            min_value=0, value=0, step=1,
            help="On each signal, the bot keeps opening same-direction positions until this cap is hit. "
                 "Live mode: 0 means it stacks until MT5 itself rejects an order (commonly insufficient "
                 "margin) - that rejection is what stops it. Simulation mode has no real margin check, so "
                 "0 falls back to a cap of 20 positions."
        )
        if max_concurrent_positions == 0:
            st.warning(
                "Unlimited stacking is enabled. On every signal the bot will keep opening same-direction "
                "positions until your broker rejects the next one (commonly: insufficient free margin). "
                "All stacked positions share the same signal and a similar entry price, so this multiplies "
                "exposure to that signal being wrong rather than diversifying it. Set a cap above if you "
                "want to bound worst-case exposure per signal."
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

        poll_interval = st.slider("Check interval (seconds)", 10, 120, 30,
                                   help="How often open positions are checked against the profit-take "
                                        "threshold, and how often a newly-closed candle is checked for a signal.")

        default_loss_cap = round((connector.account_info.balance * 0.02), 2) if connector.account_info else 20.0
        daily_loss_limit = st.number_input(
            "Daily loss limit (account currency, 0 = disabled)", min_value=0.0,
            value=default_loss_cap, step=1.0,
            help="Kill-switch: automation stops itself if losses reach this. Defaults to ~2% of balance."
        )
        max_trades = st.number_input("Max trades this session (0 = unlimited)", min_value=0, value=50, step=5,
                                      help="Counts every stacked position individually, not just signals.")

    strategy_cfg = {
        'bb_period': bb_period, 'bb_stddev': bb_stddev, 'rsi_period': rsi_period,
        'rsi_overbought': rsi_overbought, 'rsi_oversold': rsi_oversold,
        'sl_buffer_pips': sl_buffer_pips,
    }

    running = bot_state.snapshot()['status'] in ('running', 'stopping')

    st.markdown("### 🔍 Preview")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📊 Analyze Now (preview only, places no trade)", use_container_width=True, disabled=running):
            with st.spinner(f"Analyzing {symbol}..."):
                df = get_closed_bars(connector, symbol, timeframe, bars)
                if df is not None and len(df) >= 20:
                    symbol_info = connector.get_symbol_info(symbol)
                    pip = get_pip_value(symbol_info)
                    analysis = BollingerRSIMeanReversionPredictor(df, pip, strategy_cfg).analyze()
                    st.session_state.preview = {'analysis': analysis, 'df': df, 'cfg': strategy_cfg}
                else:
                    st.error("Not enough closed-candle data available for this symbol/timeframe yet.")

    with col_b:
        if not running:
            start_clicked = st.button("🚀 Execute (Start Automation)", use_container_width=True, type="primary")
        else:
            st.button("🚀 Execute (Start Automation)", use_container_width=True, disabled=True)
            start_clicked = False

    if st.session_state.preview:
        analysis = st.session_state.preview['analysis']
        df = st.session_state.preview['df']
        css_class = {'BUY': 'signal-buy', 'SELL': 'signal-sell'}.get(analysis['direction'], 'signal-neutral')
        icon = {'BUY': '📈', 'SELL': '📉'}.get(analysis['direction'], '⚖️')
        if analysis['current_price'] is not None:
            st.markdown(f'<div class="{css_class}">{icon} {analysis["direction"]}</div>', unsafe_allow_html=True)
            for r in analysis['reasons']:
                st.markdown(f"- {r}")
            st.plotly_chart(plot_chart(df, analysis, symbol, st.session_state.preview['cfg']),
                             use_container_width=True)
        else:
            st.warning(analysis['reasons'][0])

    if start_clicked:
        st.session_state.stop_event = threading.Event()
        st.session_state.bot_state = BotState()
        bot_state = st.session_state.bot_state
        config = {
            'live': live_enabled, 'risk_percent': risk_percent, 'poll_interval': poll_interval,
            'daily_loss_limit': daily_loss_limit, 'max_trades': max_trades,
            'profit_take_amount': profit_take_amount, 'max_concurrent_positions': max_concurrent_positions,
            **strategy_cfg,
        }

        # Status is set synchronously in the MAIN thread, before the
        # background thread starts, so the immediate st.rerun() below
        # always sees 'running' (see FIX LOG at top of file).
        with bot_state.lock:
            bot_state.status = 'running'
            bot_state.mode = 'live' if live_enabled else 'simulation'
            bot_state.started_at = time.time()

        thread = threading.Thread(
            target=automation_loop,
            args=(connector, symbol, timeframe, bars, config, st.session_state.stop_event, bot_state),
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

    col1, col2, col3 = st.columns(3)
    col1.metric("Daily P&L", f"{snap['daily_pnl']:.2f}")
    col2.metric("Trades this session", snap['daily_trades'])
    col3.metric("Open positions", len(snap['open_positions']))

    a = snap['last_analysis']
    if a and a.get('current_price') is not None:
        st.caption(
            f"Last closed candle ({a['bar_time']}): price {a['current_price']:.5f} | "
            f"upper {a['upper_band']:.5f} | mid {a['middle_band']:.5f} | lower {a['lower_band']:.5f} | "
            f"RSI {a['rsi_current']:.1f} | signal: **{a['direction']}**"
        )
    elif snap['status'] in ('running', 'stopping'):
        st.caption("Waiting on first cycle to complete (fetching data / evaluating strategy)...")

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
    st.caption("⚠️ For educational purposes. Not financial advice. Test in simulation/demo before "
               "arming live orders.")

    if snap['status'] in ('running', 'stopping'):
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()