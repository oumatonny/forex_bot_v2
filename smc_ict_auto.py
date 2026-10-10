"""
TradePilot AUTO - ICT/SMC Stacking Bot
==========================================================

Strategy signal logic (ICT/SMC, same engine as the manual-execution version):
    On each newly closed candle on the chosen timeframe:
      - Real swing highs/lows (fractals), Fair Value Gaps, Order Blocks,
        liquidity sweeps, and Break of Structure / Change of Character are
        computed.
      - FVGs/OBs are filtered to unmitigated only (price hasn't traded back
        through them yet) and the nearest one to current price is used.
      - A weighted score (FVG, OB, sweep, BOS/CHoCH) determines BUY / SELL /
        NEUTRAL.
      - Optional Higher-Timeframe bias filter: the signal is only acted on
        if it agrees with the HTF's own structural trend ("trade with the
        trend").
      - SL is anchored to the nearest relevant swing point (ATR fallback);
        the structural mid-point target (analogous to a "return to
        equilibrium" level) is kept only as a broker-side TP safety net -
        see trade management below.

TRADE MANAGEMENT AND SIZING (unchanged from your last version, strategy-agnostic):
  1. Exit is an ACTIVE check, not a wait on the broker SL/TP. Every poll
     cycle, every open position this bot holds is checked for unrealized
     profit, and is closed the moment that profit reaches
     `profit_take_amount` (account currency, default 0.20 - i.e. "any green
     is good enough"). A broker-side Stop Loss is still attached to every
     order as a crash/disconnect safety net (see CAVEATS) - it's not what
     this bot relies on to end a trade under normal operation. The
     broker-side "take profit" (structural mid-point) is likewise kept only
     as a distant backstop; in practice the active profit-take exit will
     almost always fire first.
  2. On each newly closed candle that produces a BUY/SELL signal aligned
     with the HTF bias, the bot opens positions repeatedly - same
     direction, same computed SL/target, same lot size - until it can't
     anymore:
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
  candle, in the same direction. It multiplies exposure to that one signal
  being wrong, and (live mode) can consume all free margin in one candle
  close. If the signal is wrong before the profit-take exit can trigger,
  every stacked position loses together and the per-trade Stop Loss is what
  stands between that and a margin call. Test in simulation mode and/or a
  demo account first; consider a non-zero `max_concurrent_positions` cap to
  bound worst-case exposure per signal.

CAVEATS:
  * The broker-side Stop Loss is what actually protects the account if this
    app crashes, the machine loses power, or the connection drops - do not
    remove it, even though normal exits now happen earlier via the active
    profit-take check.
  * This file uses a background thread + shared state in st.session_state.
    That's tied to this one browser session/server process; closing the tab
    or a server restart means you lose your Stop button while positions
    could still be open. Always be able to check/close positions directly
    in the MT5 terminal as a fallback.
  * Not financial advice, no strategy is guaranteed to be profitable.
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TradePilot AUTO - ICT/SMC Stacking Bot",
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
                          deviation=20, magic=BOT_MAGIC, comment="TradePilot-ICT-SMC"):
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


def close_position(ticket, deviation=20, comment="TradePilot-ICT-SMC close"):
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
# STRATEGY ENGINE: ICT/SMC (FVG, Order Blocks, Sweeps, BOS/CHoCH, HTF bias)
# ============================================================================
class SMCPredictor:
    def __init__(self, df, symbol):
        self.df = df.copy().reset_index(drop=True)
        self.symbol = symbol
        self.atr = None

    def _calculate_atr(self, period=14):
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean().bfill()

    def _find_swing_points(self, left=2, right=2):
        highs = self.df['high'].values
        lows = self.df['low'].values
        n = len(self.df)
        swing_highs, swing_lows = [], []
        for i in range(left, n - right):
            window_h = highs[i - left:i + right + 1]
            if highs[i] == window_h.max() and np.argmax(window_h) == left:
                swing_highs.append({'index': i, 'price': float(highs[i])})
            window_l = lows[i - left:i + right + 1]
            if lows[i] == window_l.min() and np.argmin(window_l) == left:
                swing_lows.append({'index': i, 'price': float(lows[i])})
        return swing_highs, swing_lows

    def _find_fair_value_gaps(self):
        fvg_list = []
        for i in range(2, len(self.df)):
            atr_val = self.atr.iloc[i] if not np.isnan(self.atr.iloc[i]) else self.atr.mean()
            if not atr_val or np.isnan(atr_val):
                continue
            if self.df['low'].iloc[i] > self.df['high'].iloc[i - 2]:
                gap = self.df['low'].iloc[i] - self.df['high'].iloc[i - 2]
                fvg_list.append({'type': 'bullish', 'top': self.df['low'].iloc[i],
                                  'bottom': self.df['high'].iloc[i - 2], 'index': i,
                                  'strength': 'strong' if gap > atr_val * 0.5 else 'moderate'})
            elif self.df['high'].iloc[i] < self.df['low'].iloc[i - 2]:
                gap = self.df['low'].iloc[i - 2] - self.df['high'].iloc[i]
                fvg_list.append({'type': 'bearish', 'top': self.df['low'].iloc[i - 2],
                                  'bottom': self.df['high'].iloc[i], 'index': i,
                                  'strength': 'strong' if gap > atr_val * 0.5 else 'moderate'})
        return fvg_list

    def _find_order_blocks(self):
        ob_list = []
        for i in range(3, len(self.df)):
            atr_val = self.atr.iloc[i] if not np.isnan(self.atr.iloc[i]) else self.atr.mean()
            if not atr_val or np.isnan(atr_val):
                continue
            move_up = self.df['close'].iloc[i] - self.df['high'].iloc[i - 1]
            move_down = self.df['low'].iloc[i - 1] - self.df['close'].iloc[i]
            if move_up > atr_val * 0.5:
                ob_list.append({'type': 'bullish', 'high': self.df['high'].iloc[i - 1],
                                  'low': self.df['low'].iloc[i - 1], 'index': i - 1,
                                  'strength': 'major' if move_up > atr_val else 'minor'})
            elif move_down > atr_val * 0.5:
                ob_list.append({'type': 'bearish', 'high': self.df['high'].iloc[i - 1],
                                  'low': self.df['low'].iloc[i - 1], 'index': i - 1,
                                  'strength': 'major' if move_down > atr_val else 'minor'})
        return ob_list

    def _is_fvg_mitigated(self, fvg):
        for j in range(fvg['index'] + 1, len(self.df)):
            if self.df['low'].iloc[j] <= fvg['top'] and self.df['high'].iloc[j] >= fvg['bottom']:
                return True
        return False

    def _is_ob_mitigated(self, ob):
        for j in range(ob['index'] + 1, len(self.df)):
            if self.df['low'].iloc[j] <= ob['high'] and self.df['high'].iloc[j] >= ob['low']:
                return True
        return False

    @staticmethod
    def _fvg_near_edge(fvg):
        return fvg['top'] if fvg['type'] == 'bullish' else fvg['bottom']

    @staticmethod
    def _ob_near_edge(ob):
        return ob['high'] if ob['type'] == 'bullish' else ob['low']

    def _find_liquidity_sweeps(self, swing_highs, swing_lows):
        sweeps = []
        for i in range(len(self.df)):
            highs_before = [s for s in swing_highs if s['index'] < i]
            lows_before = [s for s in swing_lows if s['index'] < i]
            if highs_before:
                recent_high = highs_before[-1]['price']
                if self.df['high'].iloc[i] > recent_high and self.df['close'].iloc[i] < recent_high:
                    sweeps.append({'type': 'sell_sweep', 'level': recent_high, 'index': i})
            if lows_before:
                recent_low = lows_before[-1]['price']
                if self.df['low'].iloc[i] < recent_low and self.df['close'].iloc[i] > recent_low:
                    sweeps.append({'type': 'buy_sweep', 'level': recent_low, 'index': i})
        return sweeps

    def _identify_market_structure(self, swing_highs, swing_lows):
        events = []
        trend = None
        for i in range(len(self.df)):
            highs_before = [s for s in swing_highs if s['index'] < i]
            lows_before = [s for s in swing_lows if s['index'] < i]
            if not highs_before or not lows_before:
                continue
            close = self.df['close'].iloc[i]
            recent_high = highs_before[-1]['price']
            recent_low = lows_before[-1]['price']
            if close > recent_high:
                event_type = 'bullish_choch' if trend == 'bearish' else 'bullish_bos'
                events.append({'type': event_type, 'index': i, 'level': recent_high})
                trend = 'bullish'
            elif close < recent_low:
                event_type = 'bearish_choch' if trend == 'bullish' else 'bearish_bos'
                events.append({'type': event_type, 'index': i, 'level': recent_low})
                trend = 'bearish'
        return events, trend

    def analyze(self, htf_bias=None):
        self.atr = self._calculate_atr()
        atr_value = self.atr.iloc[-1]
        if not atr_value or np.isnan(atr_value):
            atr_value = (self.df['high'] - self.df['low']).mean()

        swing_highs, swing_lows = self._find_swing_points()
        fvg = self._find_fair_value_gaps()
        ob = self._find_order_blocks()
        sweeps = self._find_liquidity_sweeps(swing_highs, swing_lows)
        structure_events, ltf_trend = self._identify_market_structure(swing_highs, swing_lows)

        current_price = self.df['close'].iloc[-1]
        bar_time = self.df['time'].iloc[-1]
        unmitigated_fvg = [f for f in fvg if not self._is_fvg_mitigated(f)]
        unmitigated_ob = [o for o in ob if not self._is_ob_mitigated(o)]

        score = 0
        reasons = []

        if unmitigated_fvg:
            nearest_fvg = min(unmitigated_fvg, key=lambda f: abs(current_price - self._fvg_near_edge(f)))
            if abs(current_price - self._fvg_near_edge(nearest_fvg)) < atr_value * 0.5:
                if nearest_fvg['type'] == 'bullish':
                    score += 3
                    reasons.append(f"Price near unmitigated bullish FVG (${nearest_fvg['top']:.5f})")
                else:
                    score -= 3
                    reasons.append(f"Price near unmitigated bearish FVG (${nearest_fvg['bottom']:.5f})")

        if unmitigated_ob:
            nearest_ob = min(unmitigated_ob, key=lambda o: abs(current_price - self._ob_near_edge(o)))
            if abs(current_price - self._ob_near_edge(nearest_ob)) < atr_value * 0.35:
                if nearest_ob['type'] == 'bullish':
                    score += 2
                    reasons.append(f"Near unmitigated bullish OB at ${nearest_ob['high']:.5f}")
                else:
                    score -= 2
                    reasons.append(f"Near unmitigated bearish OB at ${nearest_ob['low']:.5f}")

        if sweeps:
            last_sweep = sweeps[-1]
            if last_sweep['type'] == 'buy_sweep':
                score += 2
                reasons.append(f"Buy-side sweep of ${last_sweep['level']:.5f}")
            else:
                score -= 2
                reasons.append(f"Sell-side sweep of ${last_sweep['level']:.5f}")

        if structure_events:
            last_event = structure_events[-1]
            is_choch = 'choch' in last_event['type']
            weight = 2 if is_choch else 1
            if 'bullish' in last_event['type']:
                score += weight
                reasons.append(f"Bullish {'CHoCH' if is_choch else 'BOS'} at ${last_event['level']:.5f}")
            else:
                score -= weight
                reasons.append(f"Bearish {'CHoCH' if is_choch else 'BOS'} at ${last_event['level']:.5f}")

        if score >= 3:
            direction = 'BUY'
            confidence = min(80, 50 + (score * 5))
        elif score <= -3:
            direction = 'SELL'
            confidence = min(80, 50 + (abs(score) * 5))
        else:
            direction = 'NEUTRAL'
            confidence = 50

        # --- diagnostics: capture the raw score-only signal BEFORE any HTF
        # adjustment, so it's possible to tell whether SELLs are being
        # generated and then filtered by HTF bias, or never generated at
        # all by the score. Purely additive - does not change direction,
        # confidence, or any downstream trading decision.
        raw_direction = direction
        raw_score = score

        aligned_with_htf = True
        if htf_bias in ('bullish', 'bearish') and direction in ('BUY', 'SELL'):
            aligned_with_htf = (htf_bias == 'bullish' and direction == 'BUY') or \
                                (htf_bias == 'bearish' and direction == 'SELL')
            if aligned_with_htf:
                confidence = min(92, confidence + 10)
                reasons.append(f"✅ Aligned with HTF {htf_bias} bias")
            else:
                confidence = max(25, confidence - 25)
                reasons.append(f"⚠️ Against HTF {htf_bias} bias")

        highs_all = [s['price'] for s in swing_highs]
        lows_all = [s['price'] for s in swing_lows]
        buffer = atr_value * 0.15
        stop_loss = None
        target = None

        if direction == 'BUY':
            structural_sl = lows_all[-1] if lows_all else current_price - atr_value * 0.5
            stop_loss = min(structural_sl - buffer, current_price - atr_value * 0.3)
            risk = current_price - stop_loss
            target = current_price + risk * 1.0
        elif direction == 'SELL':
            structural_sl = highs_all[-1] if highs_all else current_price + atr_value * 0.5
            stop_loss = max(structural_sl + buffer, current_price + atr_value * 0.3)
            risk = stop_loss - current_price
            target = current_price - risk * 1.0

        return {
            'direction': direction, 'bar_time': bar_time, 'current_price': current_price,
            'stop_loss': stop_loss, 'target': target, 'confidence': confidence,
            'reasons': reasons if reasons else ['No clear signals detected'],
            'unmitigated_fvg': unmitigated_fvg, 'unmitigated_ob': unmitigated_ob,
            'ltf_trend': ltf_trend, 'htf_bias': htf_bias, 'aligned_with_htf': aligned_with_htf,
            'atr': atr_value,
            'raw_direction': raw_direction, 'raw_score': raw_score,
        }


def get_htf_bias(connector, symbol, htf_timeframe, bars=150):
    df_htf = connector.get_rates(symbol, htf_timeframe, bars)
    if df_htf is None or len(df_htf) < 20:
        return None
    p = SMCPredictor(df_htf, symbol)
    p.atr = p._calculate_atr()
    swing_highs, swing_lows = p._find_swing_points()
    _, trend = p._identify_market_structure(swing_highs, swing_lows)
    return trend


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
        self.signals_seen = 0           # cycles where direction was BUY/SELL (before HTF filter)
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
    """On a fresh, HTF-aligned signal, keep opening positions - same
    direction, same computed SL/target, same lot size - until the broker
    itself rejects the next order (live mode), or a configurable cap is hit
    (simulation mode). Returns the number of positions actually opened."""
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
    `reentry_delay_seconds` and then re-evaluates the strategy and opens a
    fresh batch, repeating for as long as the signal/HTF conditions keep
    being met, until the person clicks Stop or a circuit breaker trips.
    This is no longer gated to "once per newly-closed candle" - it re-enters
    as soon as the cooldown after a closed batch elapses, using whatever the
    latest closed-candle analysis says at that moment.
    """
    with bot_state.lock:
        bot_state.status = 'running'
        bot_state.mode = 'live' if config['live'] else 'simulation'
        if bot_state.started_at is None:
            bot_state.started_at = time.time()

    TICK = 1.0  # fast tick so exits and re-entries are checked every second
    was_open = False
    last_batch_closed_time = None

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
                        f"Batch closed. Re-entering in {config['reentry_delay_seconds']}s if conditions still met."
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
                ready_to_reenter = (
                    last_batch_closed_time is None or
                    (time.time() - last_batch_closed_time) >= config['reentry_delay_seconds']
                )
                if ready_to_reenter:
                    _evaluate_and_maybe_trade(connector, symbol, timeframe, htf_timeframe, bars,
                                               config, bot_state, symbol_info)
                else:
                    remaining = config['reentry_delay_seconds'] - (time.time() - last_batch_closed_time)
                    with bot_state.lock:
                        bot_state.last_cycle_note = f"Cooling down before re-entry ({remaining:.1f}s left)."

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
    """Evaluates ICT/SMC on the latest closed candle and, if it's a
    BUY/SELL signal (HTF-aligned if that filter is on), opens a new batch.
    Only called when the bot currently has no open positions and the
    re-entry cooldown has elapsed - so this is what drives continuous
    re-entry after each batch closes, not a once-per-candle check."""
    df = get_closed_bars(connector, symbol, timeframe, bars)
    if df is None or len(df) < 20:
        with bot_state.lock:
            bot_state.last_error = f"No data for {symbol} this cycle."
        return

    htf_bias = None
    if config['use_htf_bias']:
        htf_bias = get_htf_bias(connector, symbol, htf_timeframe)

    analysis = SMCPredictor(df, symbol).analyze(htf_bias=htf_bias)
    with bot_state.lock:
        bot_state.last_analysis = analysis
        if analysis['bar_time'] is not None:
            bot_state.last_error = None

    if analysis['direction'] not in ('BUY', 'SELL'):
        with bot_state.lock:
            bot_state.last_cycle_note = "No trade - neutral signal on the latest closed candle."
        return

    if config['use_htf_bias'] and not analysis['aligned_with_htf']:
        with bot_state.lock:
            bot_state.last_cycle_note = f"No trade - {analysis['direction']} is against HTF {analysis['htf_bias']} bias."
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

    for fvg in analysis.get('unmitigated_fvg', [])[-5:]:
        color = "rgba(0, 255, 0, 0.15)" if fvg['type'] == 'bullish' else "rgba(255, 0, 0, 0.15)"
        fig.add_hrect(y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, line_width=0)

    for ob in analysis.get('unmitigated_ob', [])[-3:]:
        if ob['type'] == 'bullish':
            fig.add_hrect(y0=ob['low'], y1=ob['high'], fillcolor="rgba(0, 200, 255, 0.2)",
                          line_width=1, line_color="rgba(0, 200, 255, 0.5)")
        else:
            fig.add_hrect(y0=ob['low'], y1=ob['high'], fillcolor="rgba(255, 100, 100, 0.2)",
                          line_width=1, line_color="rgba(255, 100, 100, 0.5)")

    if analysis['direction'] in ('BUY', 'SELL'):
        fig.add_hline(y=analysis['stop_loss'], line_dash="dot", line_color="#FF6B6B",
                      annotation_text="SL (safety net)", annotation_position="right")
        if analysis['target']:
            fig.add_hline(y=analysis['target'], line_dash="dot", line_color="#00B4D8",
                          annotation_text="Structural target (TP safety net)", annotation_position="right")

    fig.update_layout(title=f"{symbol} - ICT/SMC Preview", template="plotly_dark", height=600,
                       showlegend=True, xaxis_rangeslider_visible=False)
    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    connector = get_connector()

    st.markdown('<div class="main-header">🤖 TradePilot AUTO - ICT/SMC Stacking Bot</div>',
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
            "1 Minute": mt5.TIMEFRAME_M1, "5 Minutes": mt5.TIMEFRAME_M5,
            "15 Minutes": mt5.TIMEFRAME_M15, "30 Minutes": mt5.TIMEFRAME_M30, "1 Hour": mt5.TIMEFRAME_H1,
        }
        tf_choice = st.selectbox("Signal timeframe", list(timeframe_map.keys()), index=1)
        timeframe = timeframe_map[tf_choice]
        bars = st.slider("Bars to fetch", 50, 500, 150)

        st.divider()
        st.header("📐 ICT/SMC Parameters")
        use_htf_bias = st.checkbox("Require HTF trend alignment", value=True,
                                    help="Only acts on a signal if it agrees with the higher timeframe's own structural trend - the 'trade with the trend' filter.")
        htf_map = {"15 Minutes": mt5.TIMEFRAME_M15, "1 Hour": mt5.TIMEFRAME_H1, "4 Hours": mt5.TIMEFRAME_H4}
        htf_choice = st.selectbox("HTF for bias", list(htf_map.keys()), index=1, disabled=not use_htf_bias)
        htf_timeframe = htf_map[htf_choice]

        st.divider()
        st.header("💰 Exit & Position Sizing")
        profit_take_amount = st.number_input(
            "Close position as soon as profit reaches (account currency)", min_value=0.01, value=0.20, step=0.01,
            help="Checked every poll cycle for every open position this bot holds. This is what actually "
                 "closes trades under normal operation, not the broker TP."
        )
        use_tp_safety_net = st.checkbox("Also attach the structural target as a broker-side TP safety net", value=False,
                                         help="Off by default - exits are driven purely by the live profit check above. "
                                              "SL is always attached regardless of this setting.")
        max_concurrent_positions = st.number_input(
            "Max concurrent positions per signal (0 = unlimited - stack until broker rejects)",
            min_value=0, value=0, step=1,
            help="On each aligned signal, the bot keeps opening same-direction positions until this cap is "
                 "hit. Live: 0 means it stacks until MT5 rejects an order (commonly insufficient margin). "
                 "Simulation: 0 falls back to a cap of 20 positions."
        )
        if max_concurrent_positions == 0:
            st.warning(
                "Unlimited stacking is enabled. Every aligned signal will keep opening same-direction "
                "positions until your broker rejects the next one. All stacked positions share the same "
                "signal and a similar entry price - this multiplies exposure to that signal being wrong "
                "rather than diversifying it. Set a cap above if you want to bound worst-case exposure."
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
            help="Once every position from the last batch has closed, the bot waits this long, then "
                 "re-checks the strategy and opens a fresh batch if conditions are still met - repeating "
                 "continuously until you stop it or a circuit breaker trips."
        )

        default_loss_cap = round((connector.account_info.balance * 0.02), 2) if connector.account_info else 20.0
        daily_loss_limit = st.number_input(
            "Daily loss limit (account currency, 0 = disabled)", min_value=0.0,
            value=default_loss_cap, step=1.0,
            help="Kill-switch: automation stops itself if losses reach this. Defaults to ~2% of balance."
        )
        max_trades = st.number_input("Max trades this session (0 = unlimited)", min_value=0, value=50, step=5,
                                      help="Counts every stacked position individually, not just signals.")

    running = bot_state.snapshot()['status'] in ('running', 'stopping')

    st.markdown("### 🔍 Preview")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📊 Analyze Now (preview only, places no trade)", use_container_width=True, disabled=running):
            with st.spinner(f"Analyzing {symbol}..."):
                df = get_closed_bars(connector, symbol, timeframe, bars)
                htf_bias = get_htf_bias(connector, symbol, htf_timeframe) if use_htf_bias else None
                if df is not None and len(df) >= 20:
                    analysis = SMCPredictor(df, symbol).analyze(htf_bias=htf_bias)
                    st.session_state.preview = {'analysis': analysis, 'df': df}
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
        st.markdown(f'<div class="{css_class}">{icon} {analysis["direction"]}</div>', unsafe_allow_html=True)
        if analysis.get('htf_bias'):
            st.caption(f"HTF bias: **{analysis['htf_bias'].upper()}** | LTF structure trend: **{(analysis.get('ltf_trend') or 'n/a').upper()}**")
        # Diagnostic line: raw score-based signal vs the final HTF-filtered signal,
        # so you can tell apart "score never produced a SELL" from "SELL was
        # produced but blocked by the HTF alignment filter".
        st.caption(
            f"Raw signal (score only, pre-HTF): **{analysis.get('raw_direction', 'n/a')}** "
            f"(score {analysis.get('raw_score', 'n/a')}) → Final signal (post-HTF): **{analysis['direction']}**"
        )
        for r in analysis['reasons']:
            st.markdown(f"- {r}")
        st.plotly_chart(plot_chart(df, analysis, symbol), use_container_width=True)

    if start_clicked:
        st.session_state.stop_event = threading.Event()
        st.session_state.bot_state = BotState()
        bot_state = st.session_state.bot_state
        config = {
            'live': live_enabled, 'risk_percent': risk_percent, 'reentry_delay_seconds': reentry_delay_seconds,
            'daily_loss_limit': daily_loss_limit, 'max_trades': max_trades,
            'profit_take_amount': profit_take_amount, 'max_concurrent_positions': max_concurrent_positions,
            'use_tp_safety_net': use_tp_safety_net, 'use_htf_bias': use_htf_bias,
        }

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
            f"Last closed candle ({a['bar_time']}): price {a['current_price']:.5f} | "
            f"signal: **{a['direction']}** (raw: {a.get('raw_direction', 'n/a')}, score {a.get('raw_score', 'n/a')}) | "
            f"confidence {a.get('confidence', '-')}% | "
            f"HTF bias: {(a.get('htf_bias') or 'off')}"
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