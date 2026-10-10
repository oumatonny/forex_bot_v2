"""
TradePilot - MT5 ICT/SMC Prediction Dashboard (fixed)

Fixes applied vs the original version:
  1. FVG / OB "near edge" now depends on zone type (bullish vs bearish) instead of
     always checking the same edge -> bearish signals were being triggered/missed
     on the wrong price level before.
  2. Order block detection thresholds are ATR-relative instead of a flat 0.2%,
     so they scale sensibly across EURUSD / XAUUSD / BTCUSD / etc.
  3. "Market structure" now uses real swing highs/lows (fractals) and tracks
     trend state to distinguish genuine Break of Structure (BOS) from
     Change of Character (CHoCH). The old version just compared adjacent
     candles, which fired on almost every bar in a trend.
  4. FVGs and Order Blocks are checked for mitigation (has price already
     traded back through the zone?) before being used - mitigated zones are
     excluded, and the NEAREST unmitigated zone to current price is used
     instead of always grabbing the most recently formed one.
  5. Added an optional Higher-Timeframe bias filter: pulls a higher timeframe,
     computes its structural trend, and boosts/penalizes confidence depending
     on whether the LTF signal agrees with HTF direction.
  6. SL is now anchored to the nearest relevant swing point (structure-based)
     with an ATR fallback, instead of a flat ATR multiple disconnected from
     price structure.
  7. Lot sizing now uses real symbol_info (point, tick value, tick size,
     volume step/min/max) instead of a hardcoded point_value = 0.01, which
     was wrong for most non-5-digit-FX instruments (gold, crypto, JPY pairs).
  8. MT5 connector is cached in st.session_state instead of being
     re-instantiated (and re-initialized) on every Streamlit rerun.
  9. Auto-refresh actually refreshes now via st.rerun() at the end of the run
     loop (documented caveat: for production use, prefer the
     streamlit-autorefresh component instead of a blocking sleep).
"""

import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TradePilot - MT5 ICT/SMC Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #00B4D8; }
    .signal-buy { background-color: #00B4D8; padding: 10px; border-radius: 8px; color: white; font-weight: bold; }
    .signal-sell { background-color: #FF6B6B; padding: 10px; border-radius: 8px; color: white; font-weight: bold; }
    .signal-neutral { background-color: #FFD93D; padding: 10px; border-radius: 8px; color: #333; font-weight: bold; }
    .metric-card { background-color: #1E1E2E; padding: 15px; border-radius: 10px; border-left: 4px solid #00B4D8; }
    .prediction-box { border: 2px solid #00B4D8; padding: 20px; border-radius: 12px; background-color: #0E1117; }
    .stButton > button { width: 100%; background-color: #00B4D8; color: white; font-weight: bold; }
    .stButton > button:hover { background-color: #0096C7; }
    div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MT5 CONNECTION MANAGER
# ============================================================================
class MT5Connector:
    """Handles connection to MetaTrader 5 terminal"""

    def __init__(self):
        self.connected = False
        self.account_info = None

    def connect(self):
        """Initialize MT5 connection (idempotent - safe to call repeatedly)."""
        # If we already have a live terminal + account session, don't re-init.
        existing_account = mt5.account_info()
        if existing_account is not None:
            self.connected = True
            self.account_info = existing_account
            return True

        if not mt5.initialize():
            st.error("❌ MT5 initialization failed. Make sure MT5 is running.")
            self.connected = False
            return False

        account = mt5.account_info()
        if account is None:
            st.error("❌ Please login to MT5 first.")
            mt5.shutdown()
            self.connected = False
            return False

        self.connected = True
        self.account_info = account
        return True

    def disconnect(self):
        """Close MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def get_rates(self, symbol, timeframe=mt5.TIMEFRAME_M1, bars=100):
        """Fetch OHLCV data for a symbol"""
        if not self.connected:
            if not self.connect():
                return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            st.warning(f"No data for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        return df

    def get_symbol_info(self, symbol):
        """Get symbol details including point value, digits, tick value, etc."""
        if not self.connected:
            self.connect()
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        return info

    def get_current_price(self, symbol):
        """Get current bid/ask price"""
        if not self.connected:
            self.connect()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {'bid': tick.bid, 'ask': tick.ask}


def get_connector():
    """Reuse a single connector across reruns instead of re-instantiating it."""
    if 'mt5_connector' not in st.session_state:
        st.session_state.mt5_connector = MT5Connector()
    return st.session_state.mt5_connector


# ============================================================================
# TRADE EXECUTION
# ============================================================================
def get_account_mode_label(account_info):
    """Returns ('DEMO'|'REAL'|'CONTEST'|'UNKNOWN', is_real) for a clear UI badge."""
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
    """
    Picking the wrong ORDER_FILLING_* mode is one of the most common causes of
    order_send() silently failing (retcode 10030 'Unsupported filling mode').
    symbol_info.filling_mode is a bitmask describing what the broker/symbol
    supports. The Python MetaTrader5 package does NOT expose the
    SYMBOL_FILLING_FOK / SYMBOL_FILLING_IOC bitmask constants (only the
    ORDER_FILLING_* ones used when actually placing an order), so we use the
    documented raw bit values here: bit 0 (1) = FOK supported, bit 1 (2) = IOC
    supported. getattr(..., default) is used defensively in case a future
    package version does add these names.
    """
    SYMBOL_FILLING_FOK = getattr(mt5, 'SYMBOL_FILLING_FOK', 1)
    SYMBOL_FILLING_IOC = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)

    mode = getattr(symbol_info, 'filling_mode', 0) or 0
    if mode & SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if mode & SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def execute_market_order(connector, symbol, direction, lot_size, sl, tp,
                          deviation=20, magic=234000, comment="TradePilot ICT/SMC"):
    """
    Sends a real market order to MT5. Returns (result, message).
    `result` is the raw OrderSendResult (or None on hard failure) so the caller
    can inspect retcode/ticket/etc; `message` is a human-readable summary.
    """
    if direction not in ('BUY', 'SELL'):
        return None, "Direction must be BUY or SELL - refusing to send a NEUTRAL order."

    if not connector.connected and not connector.connect():
        return None, "Not connected to MT5."

    symbol_info = connector.get_symbol_info(symbol)
    if symbol_info is None:
        return None, f"Could not fetch symbol info for {symbol}."

    if not symbol_info.trade_mode or symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return None, f"Trading is disabled for {symbol} on this account/symbol."

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, "Could not fetch current tick."

    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
    order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == 'BUY' else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _resolve_filling_type(symbol_info),
    }

    result = mt5.order_send(request)

    if result is None:
        return None, f"order_send() returned None. last_error(): {mt5.last_error()}"

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return result, f"❌ Order rejected - retcode {result.retcode}: {result.comment}"

    return result, f"✅ Order filled - ticket #{result.order}, {lot_size} lots @ {result.price:.5f}"


def get_open_positions(connector, symbol=None):
    """Returns open positions as a DataFrame (optionally filtered to one symbol)."""
    if not connector.connected and not connector.connect():
        return pd.DataFrame()

    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None or len(positions) == 0:
        return pd.DataFrame()

    rows = []
    for p in positions:
        rows.append({
            'ticket': p.ticket,
            'symbol': p.symbol,
            'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
            'volume': p.volume,
            'open_price': p.price_open,
            'current_price': p.price_current,
            'sl': p.sl,
            'tp': p.tp,
            'profit': p.profit,
        })
    return pd.DataFrame(rows)


def close_position(connector, ticket, deviation=20, comment="TradePilot close"):
    """Closes a single open position by ticket, fully, at market."""
    if not connector.connected and not connector.connect():
        return None, "Not connected to MT5."

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None, f"No open position found with ticket #{ticket}."

    pos = positions[0]
    symbol_info = connector.get_symbol_info(pos.symbol)
    if symbol_info is None:
        return None, f"Could not fetch symbol info for {pos.symbol}."

    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return None, "Could not fetch current tick."

    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
    price = tick.bid if is_buy else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": pos.ticket,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _resolve_filling_type(symbol_info),
    }

    result = mt5.order_send(request)
    if result is None:
        return None, f"order_send() returned None. last_error(): {mt5.last_error()}"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return result, f"❌ Close rejected - retcode {result.retcode}: {result.comment}"
    return result, f"✅ Position #{ticket} closed."


# ============================================================================
# ICT/SMC ANALYSIS ENGINE
# ============================================================================
class SMCPredictor:
    """Analyzes price data using ICT/SMC principles and generates predictions"""

    def __init__(self, df, symbol):
        self.df = df.copy().reset_index(drop=True)
        self.symbol = symbol
        self.atr = None

    # ------------------------------------------------------------------
    # Core indicators
    # ------------------------------------------------------------------
    def _calculate_atr(self, period=14):
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        # Backfill early NaNs so we always have a usable value even on short lookbacks
        atr = atr.bfill()
        return atr

    def _find_swing_points(self, left=2, right=2):
        """
        Real swing highs/lows (fractals): a bar is a swing high if its high is the
        max within a window of `left` bars before and `right` bars after it (and
        similarly for swing lows). This replaces the old adjacent-candle-only logic
        that fired on nearly every bar in a trend.
        """
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

    # ------------------------------------------------------------------
    # Zone detection
    # ------------------------------------------------------------------
    def _find_fair_value_gaps(self):
        """Identify Fair Value Gaps (FVG), ATR-relative strength."""
        fvg_list = []
        for i in range(2, len(self.df)):
            atr_val = self.atr.iloc[i] if not np.isnan(self.atr.iloc[i]) else self.atr.mean()
            if not atr_val or np.isnan(atr_val):
                continue

            if self.df['low'].iloc[i] > self.df['high'].iloc[i - 2]:
                gap = self.df['low'].iloc[i] - self.df['high'].iloc[i - 2]
                fvg_list.append({
                    'type': 'bullish',
                    'top': self.df['low'].iloc[i],
                    'bottom': self.df['high'].iloc[i - 2],
                    'index': i,
                    'strength': 'strong' if gap > atr_val * 0.5 else 'moderate'
                })
            elif self.df['high'].iloc[i] < self.df['low'].iloc[i - 2]:
                gap = self.df['low'].iloc[i - 2] - self.df['high'].iloc[i]
                fvg_list.append({
                    'type': 'bearish',
                    'top': self.df['low'].iloc[i - 2],
                    'bottom': self.df['high'].iloc[i],
                    'index': i,
                    'strength': 'strong' if gap > atr_val * 0.5 else 'moderate'
                })
        return fvg_list

    def _find_order_blocks(self):
        """Identify Order Blocks (OB) using ATR-relative displacement (not a flat %)."""
        ob_list = []
        for i in range(3, len(self.df)):
            atr_val = self.atr.iloc[i] if not np.isnan(self.atr.iloc[i]) else self.atr.mean()
            if not atr_val or np.isnan(atr_val):
                continue

            move_up = self.df['close'].iloc[i] - self.df['high'].iloc[i - 1]
            move_down = self.df['low'].iloc[i - 1] - self.df['close'].iloc[i]

            if move_up > atr_val * 0.5:
                ob_list.append({
                    'type': 'bullish',
                    'high': self.df['high'].iloc[i - 1],
                    'low': self.df['low'].iloc[i - 1],
                    'index': i - 1,
                    'strength': 'major' if move_up > atr_val else 'minor'
                })
            elif move_down > atr_val * 0.5:
                ob_list.append({
                    'type': 'bearish',
                    'high': self.df['high'].iloc[i - 1],
                    'low': self.df['low'].iloc[i - 1],
                    'index': i - 1,
                    'strength': 'major' if move_down > atr_val else 'minor'
                })
        return ob_list

    def _is_fvg_mitigated(self, fvg):
        """A gap is mitigated once a later candle's range trades back into it."""
        for j in range(fvg['index'] + 1, len(self.df)):
            if self.df['low'].iloc[j] <= fvg['top'] and self.df['high'].iloc[j] >= fvg['bottom']:
                return True
        return False

    def _is_ob_mitigated(self, ob):
        """An order block is mitigated once price trades back into its range."""
        for j in range(ob['index'] + 1, len(self.df)):
            if self.df['low'].iloc[j] <= ob['high'] and self.df['high'].iloc[j] >= ob['low']:
                return True
        return False

    @staticmethod
    def _fvg_near_edge(fvg):
        """The edge of the gap price will touch FIRST, depending on approach direction."""
        return fvg['top'] if fvg['type'] == 'bullish' else fvg['bottom']

    @staticmethod
    def _ob_near_edge(ob):
        return ob['high'] if ob['type'] == 'bullish' else ob['low']

    # ------------------------------------------------------------------
    # Liquidity & structure
    # ------------------------------------------------------------------
    def _find_liquidity_sweeps(self, swing_highs, swing_lows):
        """
        A sweep = price wicks beyond the most recent confirmed swing high/low but
        closes back inside it (stop hunt). Anchored to real swing points instead
        of a naive rolling max/min, which flagged sweeps constantly in normal chop.
        """
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
        """
        Real BOS/CHoCH: track the prevailing trend. A close beyond the last
        confirmed swing high, while trend is neutral/bullish, is a BOS (trend
        continuation). The same break while trend is bearish is a CHoCH (trend
        reversal signal). Symmetric logic for downside breaks.
        """
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

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
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
        last_high = self.df['high'].iloc[-5:].max()
        last_low = self.df['low'].iloc[-5:].min()

        # Only consider zones price hasn't already traded through
        unmitigated_fvg = [f for f in fvg if not self._is_fvg_mitigated(f)]
        unmitigated_ob = [o for o in ob if not self._is_ob_mitigated(o)]

        score = 0
        reasons = []

        # --- Fair Value Gap (nearest unmitigated, correct near-edge) ---
        nearest_fvg = None
        if unmitigated_fvg:
            nearest_fvg = min(unmitigated_fvg, key=lambda f: abs(current_price - self._fvg_near_edge(f)))
            dist = abs(current_price - self._fvg_near_edge(nearest_fvg))
            if dist < atr_value * 0.5:
                if nearest_fvg['type'] == 'bullish':
                    score += 3
                    reasons.append(f"Price near unmitigated bullish FVG (${nearest_fvg['top']:.5f})")
                else:
                    score -= 3
                    reasons.append(f"Price near unmitigated bearish FVG (${nearest_fvg['bottom']:.5f})")

        # --- Order Block (nearest unmitigated, correct near-edge) ---
        nearest_ob = None
        if unmitigated_ob:
            nearest_ob = min(unmitigated_ob, key=lambda o: abs(current_price - self._ob_near_edge(o)))
            dist = abs(current_price - self._ob_near_edge(nearest_ob))
            if dist < atr_value * 0.35:
                if nearest_ob['type'] == 'bullish':
                    score += 2
                    reasons.append(f"Near unmitigated bullish OB at ${nearest_ob['high']:.5f}")
                else:
                    score -= 2
                    reasons.append(f"Near unmitigated bearish OB at ${nearest_ob['low']:.5f}")

        # --- Liquidity sweep (most recent) ---
        if sweeps:
            last_sweep = sweeps[-1]
            if last_sweep['type'] == 'buy_sweep':
                score += 2
                reasons.append(f"Buy-side sweep of ${last_sweep['level']:.5f}")
            else:
                score -= 2
                reasons.append(f"Sell-side sweep of ${last_sweep['level']:.5f}")

        # --- Structure (BOS/CHoCH, weighted - CHoCH matters more, it's a reversal signal) ---
        if structure_events:
            last_event = structure_events[-1]
            is_choch = 'choch' in last_event['type']
            weight = 2 if is_choch else 1
            if 'bullish' in last_event['type']:
                score += weight
                label = "CHoCH (reversal)" if is_choch else "BOS (continuation)"
                reasons.append(f"Bullish {label} confirmed at ${last_event['level']:.5f}")
            else:
                score -= weight
                label = "CHoCH (reversal)" if is_choch else "BOS (continuation)"
                reasons.append(f"Bearish {label} confirmed at ${last_event['level']:.5f}")

        # --- Determine direction & base confidence ---
        if score >= 3:
            direction = 'BUY'
            confidence = min(80, 50 + (score * 5))
        elif score <= -3:
            direction = 'SELL'
            confidence = min(80, 50 + (abs(score) * 5))
        else:
            direction = 'NEUTRAL'
            confidence = 50

        # --- HTF bias gate: reward alignment, penalize counter-trend signals ---
        if htf_bias in ('bullish', 'bearish') and direction in ('BUY', 'SELL'):
            aligned = (htf_bias == 'bullish' and direction == 'BUY') or \
                      (htf_bias == 'bearish' and direction == 'SELL')
            if aligned:
                confidence = min(92, confidence + 10)
                reasons.append(f"✅ Aligned with HTF {htf_bias} bias")
            else:
                confidence = max(25, confidence - 25)
                reasons.append(f"⚠️ Against HTF {htf_bias} bias - confidence reduced")

        # --- SL/TP anchored to structure where possible, ATR fallback otherwise ---
        highs_all = [s['price'] for s in swing_highs]
        lows_all = [s['price'] for s in swing_lows]
        buffer = atr_value * 0.15

        if direction == 'BUY':
            structural_sl = lows_all[-1] if lows_all else current_price - atr_value * 0.5
            sl = min(structural_sl - buffer, current_price - atr_value * 0.3)
            risk = current_price - sl
            prediction_price = current_price + atr_value * 0.3
            tp1 = current_price + risk * 1.0
            tp2 = current_price + risk * 2.0
        elif direction == 'SELL':
            structural_sl = highs_all[-1] if highs_all else current_price + atr_value * 0.5
            sl = max(structural_sl + buffer, current_price + atr_value * 0.3)
            risk = sl - current_price
            prediction_price = current_price - atr_value * 0.3
            tp1 = current_price - risk * 1.0
            tp2 = current_price - risk * 2.0
        else:
            prediction_price = current_price
            sl = current_price - atr_value * 0.3
            tp1 = current_price + atr_value * 0.3
            tp2 = current_price + atr_value * 0.6

        return {
            'direction': direction,
            'score': score,
            'confidence': confidence,
            'current_price': current_price,
            'prediction_5min': prediction_price,
            'stop_loss': sl,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'reasons': reasons if reasons else ['No clear signals detected'],
            'fvg': fvg,
            'unmitigated_fvg': unmitigated_fvg,
            'ob': ob,
            'unmitigated_ob': unmitigated_ob,
            'sweeps': sweeps,
            'structure': structure_events,
            'ltf_trend': ltf_trend,
            'htf_bias': htf_bias,
            'last_high': last_high,
            'last_low': last_low,
            'atr': atr_value,
        }

    def get_prediction_text(self, analysis):
        direction = analysis['direction']
        confidence = analysis['confidence']
        if direction == 'BUY':
            return f"📈 **BUY** Signal - Confidence: {confidence}%"
        elif direction == 'SELL':
            return f"📉 **SELL** Signal - Confidence: {confidence}%"
        else:
            return f"⚖️ **NEUTRAL** - Confidence: {confidence}%"


def get_htf_bias(connector, symbol, htf_timeframe=mt5.TIMEFRAME_H4, bars=150):
    """
    Pull a higher timeframe and derive a directional bias from its own
    structure (BOS/CHoCH trend state). Returns 'bullish', 'bearish', or None.
    """
    df_htf = connector.get_rates(symbol, htf_timeframe, bars)
    if df_htf is None or len(df_htf) < 20:
        return None
    htf_predictor = SMCPredictor(df_htf, symbol)
    htf_predictor.atr = htf_predictor._calculate_atr()
    swing_highs, swing_lows = htf_predictor._find_swing_points()
    _, trend = htf_predictor._identify_market_structure(swing_highs, swing_lows)
    return trend


# ============================================================================
# LOT SIZING (fixed - uses real symbol_info instead of a hardcoded 0.01 point value)
# ============================================================================
def calculate_lot_size(symbol_info, account_balance, risk_percent, sl_distance):
    """
    Correct risk-based position sizing:
        ticks_at_risk   = sl_distance / point
        value_per_point = tick_value * (point / tick_size)
        lot_size         = risk_amount / (ticks_at_risk * value_per_point)
    Rounded to the instrument's volume step and clamped to [volume_min, volume_max].
    """
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


# ============================================================================
# CHARTING FUNCTIONS
# ============================================================================
def plot_candlestick(df, analysis, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price'
        ),
        row=1, col=1
    )

    # Only plot unmitigated zones - mitigated ones are no longer actionable
    for fvg in analysis.get('unmitigated_fvg', [])[-5:]:
        color = "rgba(0, 255, 0, 0.15)" if fvg['type'] == 'bullish' else "rgba(255, 0, 0, 0.15)"
        fig.add_hrect(y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, line_width=0, row=1, col=1)

    for ob in analysis.get('unmitigated_ob', [])[-3:]:
        if ob['type'] == 'bullish':
            fig.add_hrect(y0=ob['low'], y1=ob['high'], fillcolor="rgba(0, 200, 255, 0.2)",
                          line_width=1, line_color="rgba(0, 200, 255, 0.5)", row=1, col=1)
        else:
            fig.add_hrect(y0=ob['low'], y1=ob['high'], fillcolor="rgba(255, 100, 100, 0.2)",
                          line_width=1, line_color="rgba(255, 100, 100, 0.5)", row=1, col=1)

    if analysis:
        price = analysis['current_price']
        pred = analysis['prediction_5min']
        sl = analysis['stop_loss']
        tp1 = analysis['take_profit_1']

        fig.add_hline(y=price, line_dash="dash", line_color="white", row=1, col=1)

        if analysis['direction'] == 'BUY':
            fig.add_hline(y=pred, line_dash="dot", line_color="#00B4D8", row=1, col=1,
                         annotation_text="Target", annotation_position="right")
            fig.add_hline(y=sl, line_dash="dot", line_color="#FF6B6B", row=1, col=1,
                         annotation_text="SL", annotation_position="right")
            fig.add_hline(y=tp1, line_dash="dot", line_color="#00FF88", row=1, col=1,
                         annotation_text="TP1", annotation_position="right")
        elif analysis['direction'] == 'SELL':
            fig.add_hline(y=pred, line_dash="dot", line_color="#FF6B6B", row=1, col=1,
                         annotation_text="Target", annotation_position="right")
            fig.add_hline(y=sl, line_dash="dot", line_color="#00B4D8", row=1, col=1,
                         annotation_text="SL", annotation_position="right")
            fig.add_hline(y=tp1, line_dash="dot", line_color="#FF4444", row=1, col=1,
                         annotation_text="TP1", annotation_position="right")

    colors = ['#00B4D8' if c >= o else '#FF6B6B' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['tick_volume'], name='Volume', marker_color=colors), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} - ICT/SMC Analysis",
        template="plotly_dark",
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    mt5_connector = get_connector()

    st.markdown('<div class="main-header">📊 TradePilot - ICT/SMC Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Connect to MT5 → Analyze → Get Structure-Based Predictions")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Settings")

        if mt5_connector.connect():
            st.success("✅ Connected to MT5")
            account = mt5_connector.account_info
            mode_label, is_real = get_account_mode_label(account)
            badge = "🔴 REAL MONEY" if is_real else f"🟢 {mode_label}"
            st.info(f"Account: {account.login} | Balance: ${account.balance:.2f} | {badge}")
        else:
            st.error("❌ Not connected to MT5")
            st.info("Make sure MT5 is running and logged in")

        st.divider()
        st.subheader("⚡ Trade Execution")
        enable_trading = st.checkbox(
            "Arm live order execution",
            value=False,
            help="Off by default. When unchecked, the Execute Trade button stays disabled "
                 "even if a signal is showing - this is a deliberate safety gate."
        )
        if enable_trading:
            st.warning("Armed - clicking Execute will send a REAL order to MT5.")

        st.divider()
        st.subheader("📊 Symbol")

        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD']
        selected_symbol = st.selectbox("Choose Symbol", symbols, index=0)

        timeframe_map = {
            "1 Minute": mt5.TIMEFRAME_M1,
            "5 Minutes": mt5.TIMEFRAME_M5,
            "15 Minutes": mt5.TIMEFRAME_M15,
            "1 Hour": mt5.TIMEFRAME_H1,
            "4 Hours": mt5.TIMEFRAME_H4,
            "Daily": mt5.TIMEFRAME_D1
        }
        selected_tf = st.selectbox("Timeframe", list(timeframe_map.keys()), index=2)
        bars_count = st.slider("Bars to Analyze", 50, 300, 100)

        use_htf_bias = st.checkbox("Use Higher-Timeframe bias filter", value=True)
        htf_choice = st.selectbox(
            "HTF for bias", ["1 Hour", "4 Hours", "Daily"], index=1,
            disabled=not use_htf_bias
        )

        st.divider()
        analyze_clicked = st.button("🔍 Analyze Now", use_container_width=True)
        auto_refresh = st.checkbox("Auto-refresh every 30s")

        st.divider()
        st.caption("⚠️ Disclaimer: For educational purposes only. Not financial advice.")

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    if analyze_clicked:
        with st.spinner(f"Analyzing {selected_symbol}..."):
            timeframe = timeframe_map[selected_tf]
            df = mt5_connector.get_rates(selected_symbol, timeframe, bars_count)

            if df is not None and len(df) > 20:
                st.session_state.df = df

                htf_bias = None
                if use_htf_bias:
                    htf_bias = get_htf_bias(mt5_connector, selected_symbol, timeframe_map[htf_choice])

                predictor = SMCPredictor(df, selected_symbol)
                analysis = predictor.analyze(htf_bias=htf_bias)
                st.session_state.analysis_result = analysis
                st.session_state.last_update = datetime.now()

                tick = mt5_connector.get_current_price(selected_symbol)
                if tick:
                    st.session_state.current_bid = tick['bid']
                    st.session_state.current_ask = tick['ask']
            else:
                st.error("No data available for this symbol")

    if st.session_state.analysis_result is not None:
        analysis = st.session_state.analysis_result
        df = st.session_state.df
        symbol = selected_symbol

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            direction = analysis['direction']
            css_class = {'BUY': 'signal-buy', 'SELL': 'signal-sell'}.get(direction, 'signal-neutral')
            icon = {'BUY': '📈', 'SELL': '📉'}.get(direction, '⚖️')
            st.markdown(f'<div class="{css_class}">{icon} {direction}</div>', unsafe_allow_html=True)

        with col2:
            st.metric("Confidence", f"{analysis['confidence']}%")
        with col3:
            st.metric("Current Price", f"{analysis['current_price']:.5f}")
        with col4:
            st.metric("Target", f"{analysis['prediction_5min']:.5f}")
        with col5:
            change = analysis['prediction_5min'] - analysis['current_price']
            change_pct = (change / analysis['current_price']) * 100
            st.metric("Expected Change", f"{change:.5f}", f"{change_pct:.3f}%")

        if analysis.get('htf_bias'):
            st.caption(f"🧭 HTF bias: **{analysis['htf_bias'].upper()}** | LTF structure trend: **{(analysis.get('ltf_trend') or 'n/a').upper()}**")

        st.divider()
        st.markdown("### 📋 Prediction Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Stop Loss", f"{analysis['stop_loss']:.5f}")
            st.caption(f"Risk: {abs(analysis['stop_loss'] - analysis['current_price']):.5f} pts (structure-based)")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("TP1 (1:1)", f"{analysis['take_profit_1']:.5f}")
            st.caption(f"Reward: {abs(analysis['take_profit_1'] - analysis['current_price']):.5f} pts")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("TP2 (2:1)", f"{analysis['take_profit_2']:.5f}")
            st.caption(f"Reward: {abs(analysis['take_profit_2'] - analysis['current_price']):.5f} pts")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🔍 Signal Analysis")
        for reason in analysis['reasons']:
            st.markdown(f"✅ {reason}")

        score = analysis['score']
        st.markdown(f"**Score:** {score}/8 (before HTF adjustment)")
        st.progress((max(-8, min(8, score)) + 8) / 16)

        st.divider()
        st.markdown("### 📈 Interactive Chart")
        fig = plot_candlestick(df, analysis, symbol)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 📝 Trade Plan Summary")

        col1, col2 = st.columns(2)
        with col1:
            sl_distance = abs(analysis['stop_loss'] - analysis['current_price'])
            tp_distance = abs(analysis['take_profit_1'] - analysis['current_price'])
            rr = tp_distance / sl_distance if sl_distance > 0 else 0
            st.code(f"""
📊 {symbol} - {analysis['direction']} SIGNAL
Confidence: {analysis['confidence']}%

📌 ENTRY: {analysis['current_price']:.5f}
🔴 SL:     {analysis['stop_loss']:.5f}
🟢 TP1:    {analysis['take_profit_1']:.5f}
🟢 TP2:    {analysis['take_profit_2']:.5f}

Risk: {sl_distance:.5f} pts
Reward: {tp_distance:.5f} pts
R:R: 1:{rr:.2f}
            """, language="text")

        with col2:
            st.markdown("#### Risk Management")
            account_balance = mt5_connector.account_info.balance if mt5_connector.account_info else 10000.0
            risk_percent = st.slider("Risk per trade (%)", 0.5, 2.0, 1.0, 0.1)

            symbol_info = mt5_connector.get_symbol_info(symbol)
            sl_distance = abs(analysis['stop_loss'] - analysis['current_price'])
            lot_size, risk_amount = calculate_lot_size(symbol_info, account_balance, risk_percent, sl_distance)

            st.metric("Suggested Lot Size", f"{lot_size:.2f}")
            st.metric("Risk Amount", f"${risk_amount:.2f}")
            if symbol_info is None:
                st.warning("⚠️ Could not fetch symbol_info - lot size defaulted to 0.01. Check the symbol is enabled in Market Watch.")
            st.caption(f"Based on {risk_percent}% risk of ${account_balance:.2f}, using this symbol's real tick value/step.")

        # ===== EXECUTE TRADE =====
        st.divider()
        st.markdown("### ⚡ Execute Trade")

        if analysis['direction'] == 'NEUTRAL':
            st.info("Signal is NEUTRAL - nothing to execute.")
        elif not enable_trading:
            st.button("🚀 Execute Trade", use_container_width=True, disabled=True)
            st.caption("Check 'Arm live order execution' in the sidebar to enable this.")
        else:
            mode_label, is_real = get_account_mode_label(mt5_connector.account_info)
            if is_real:
                st.error("⚠️ This is a REAL money account. Double-check everything below before confirming.")

            st.markdown(f"""
**Preview** — {analysis['direction']} {lot_size} lots of {symbol}
SL: `{analysis['stop_loss']:.5f}`  TP: `{analysis['take_profit_1']:.5f}`
Estimated risk: **${risk_amount:.2f}** ({risk_percent}% of ${account_balance:.2f})
            """)

            confirm = st.checkbox(
                f"I've reviewed this {analysis['direction']} order and want to submit it to my {mode_label} account",
                key="confirm_trade"
            )

            if st.button("🚀 Execute Trade", use_container_width=True, disabled=not confirm):
                with st.spinner("Sending order to MT5..."):
                    result, message = execute_market_order(
                        mt5_connector, symbol, analysis['direction'],
                        lot_size, analysis['stop_loss'], analysis['take_profit_1']
                    )
                if result is not None and getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                    st.success(message)
                else:
                    st.error(message)

        # ===== OPEN POSITIONS =====
        st.divider()
        st.markdown("### 📂 Open Positions")
        positions_df = get_open_positions(mt5_connector)
        if positions_df.empty:
            st.caption("No open positions.")
        else:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            close_ticket = st.selectbox(
                "Select a position to close",
                positions_df['ticket'].tolist(),
                format_func=lambda t: f"#{t} - " + positions_df.loc[positions_df['ticket'] == t, 'symbol'].iloc[0]
            )
            if st.button("❌ Close Selected Position", use_container_width=True):
                with st.spinner(f"Closing #{close_ticket}..."):
                    result, message = close_position(mt5_connector, close_ticket)
                if result is not None and getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.divider()
        st.caption(f"🔄 Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        st.info("👈 Click 'Analyze Now' to get your prediction")
        st.markdown("""
        ### How it works:
        1. ✅ Ensure MT5 is running and logged in
        2. 📊 Select a symbol, timeframe, and (optionally) an HTF bias filter
        3. 🔍 Click Analyze Now
        4. 📈 Get a structure-anchored entry/SL/TP plan

        **ICT/SMC Analysis includes:**
        - Fair Value Gaps (FVG) — nearest unmitigated only
        - Order Blocks (OB) — nearest unmitigated only
        - Liquidity Sweeps — anchored to real swing points
        - Break of Structure (BOS) & Change of Character (CHoCH)
        - Higher-timeframe bias gate
        """)

    # --- Auto-refresh: reruns the whole script every 30s while the box is checked.
    # For production use, prefer the `streamlit-autorefresh` component instead of
    # a blocking sleep, since this approach ties up the Streamlit run thread.
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()