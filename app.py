"""
TradePilot SUPER BOT - rule-based, multi-strategy, auto-executing MT5 bot (no AI)
=================================================================================

Run:   streamlit run app.py

What it does
    * Scans several symbols on every newly closed candle.
    * Generates signals from fixed rules taken from strategies.json (trend
      pullback, MA crossover, Bollinger/RSI fade, Bollinger squeeze, daily
      pivots, Heikin-Ashi, 2-period RSI, candlestick patterns, narrow range,
      support/resistance breakout) plus the project's ICT/SMC engine.
    * Sizes every trade from account risk and the stop distance, attaches a
      broker-side SL and TP, and manages the position (break-even, ATR trail,
      time stop, opposite-signal exit).
    * Enforces account-level limits: daily loss, trades per day, losing
      streak, open-position caps, spread, session hours.
    * Backtests the very same strategy code so you can see what each rule set
      has actually done on your symbol/timeframe before switching it on.

Modes
    * Paper (default): real MT5 prices, simulated fills, SL/TP enforced, no
      order is ever sent.
    * Live: real orders on whichever account the MT5 terminal is logged in
      to. Needs an explicit arm; a real-money account needs a typed REAL.

Things to know before arming live
    * The engine is one background thread per Streamlit *process*. It keeps
      running if you close the browser tab. It stops if the Streamlit process
      or the machine stops - open positions are then protected only by their
      broker-side SL/TP, and are picked up again on the next start.
    * Position checks run every few seconds, not every tick. Break-even and
      trailing moves can lag fast markets; the broker-side SL is what bounds
      the loss.
    * A backtest on bar data is an approximation (no commission/swap, spread
      taken from bar data, intrabar order unknown). Treat a good backtest as
      permission to paper-trade, not as proof.
    * Not financial advice. No rule set here is guaranteed to be profitable.

Set SUPERBOT_OFFLINE=1 to run without MetaTrader5 (CSV backtests only).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    if os.environ.get("SUPERBOT_OFFLINE"):
        raise ImportError("offline mode")
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# ============================================================================
# CONSTANTS & FILES
# ============================================================================
BOT_MAGIC = 234500          # this bot's own orders/positions (distinct from the older bots)
WARMUP = 210                # bars needed before the slowest indicator (EMA 200) is usable
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
DEFAULT_SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'XAUUSD', 'BTCUSD']

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "bot_data"
TRADES_CSV = DATA_DIR / "trades.csv"
SETTINGS_JSON = DATA_DIR / "settings.json"
LOG_FILE = DATA_DIR / "bot.log"
STRATEGIES_JSON = BASE_DIR / "strategies.json"

TRADE_FIELDS = ['mode', 'ticket', 'symbol', 'strategy', 'direction', 'lot', 'entry', 'exit',
                'sl_initial', 'tp', 'open_time', 'close_time', 'pnl', 'r_multiple', 'reason']

MT5_LOCK = threading.RLock()
FILE_LOCK = threading.Lock()


def _get_logger():
    logger = logging.getLogger("superbot")
    if not logger.handlers:
        DATA_DIR.mkdir(exist_ok=True)
        handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = _get_logger()


# ============================================================================
# SETTINGS
# ============================================================================
@dataclass
class Settings:
    symbols: list = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    timeframe: str = "M15"
    htf: str = "H4"
    bars: int = 600
    strategies: list = field(default_factory=lambda: ['trend_pullback', 'bb_rsi_reversion', 'smc'])
    min_votes: int = 1
    use_htf_filter: bool = True
    mode: str = "paper"              # paper | live
    # risk
    risk_pct: float = 0.5
    max_risk_pct: float = 1.0        # hard cap when the broker minimum lot forces more than risk_pct
    max_positions: int = 3
    max_per_symbol: int = 1
    max_per_currency: int = 2
    daily_loss_pct: float = 3.0
    max_daily_trades: int = 10
    max_consec_losses: int = 3
    cooldown_bars: int = 3
    # trade plan
    rr: float = 1.5
    min_rr: float = 1.0
    use_strategy_target: bool = True
    min_sl_atr: float = 0.5
    max_sl_atr: float = 4.0
    max_spread_frac: float = 0.15    # spread may cost at most this fraction of the stop distance
    # management
    be_at_r: float = 1.0             # 0 = off
    trail_atr: float = 0.0           # 0 = off
    trail_start_r: float = 1.0
    max_bars_held: int = 0           # 0 = off
    quick_profit: float = 0.0        # account currency, 0 = off (legacy "any green" exit)
    exit_on_opposite: bool = True
    # filters
    session_start: int = 0           # broker server hours
    session_end: int = 24
    friday_close_hour: int = -1      # -1 = off
    # engine
    poll_seconds: float = 2.0
    deviation: int = 20

    @classmethod
    def load(cls):
        try:
            raw = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in raw.items() if k in known})
        s.strategies = [k for k in s.strategies if k in STRATEGIES] or cls().strategies
        s.mode = "paper"             # never come back up pre-armed for live
        return s

    def save(self):
        save_json(SETTINGS_JSON, asdict(self))


def save_json(path, obj):
    DATA_DIR.mkdir(exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with FILE_LOCK:
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)


def append_trade(row):
    DATA_DIR.mkdir(exist_ok=True)
    with FILE_LOCK:
        is_new = not TRADES_CSV.exists()
        with open(TRADES_CSV, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=TRADE_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in TRADE_FIELDS})


def load_trades():
    if not TRADES_CSV.exists():
        return pd.DataFrame(columns=TRADE_FIELDS)
    with FILE_LOCK:
        try:
            return pd.read_csv(TRADES_CSV)
        except (OSError, ValueError):
            return pd.DataFrame(columns=TRADE_FIELDS)


# ============================================================================
# INDICATORS (all causal: the value at bar i uses bars <= i only)
# ============================================================================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rma(series, n):
    """Wilder's smoothing."""
    return series.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n):
    delta = close.diff()
    gain = rma(delta.clip(lower=0), n)
    loss = rma(-delta.clip(upper=0), n)
    out = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out = out.where(loss != 0, 100.0)
    return out.where((gain != 0) | (loss != 0), 50.0)


def true_range(df):
    prev_close = df['close'].shift()
    return pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(),
                      (df['low'] - prev_close).abs()], axis=1).max(axis=1)


def add_indicators(df):
    d = df.reset_index(drop=True).copy()
    o, h, l, c = d['open'], d['high'], d['low'], d['close']
    if 'tick_volume' not in d:
        d['tick_volume'] = 1.0

    d['ema20'], d['ema50'], d['ema200'] = ema(c, 20), ema(c, 50), ema(c, 200)
    d['sma20'], d['sma60'], d['sma100'] = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(100).mean()
    d['rsi14'], d['rsi2'] = rsi(c, 14), rsi(c, 2)

    tr = true_range(d)
    d['atr'] = rma(tr, 14)

    std = c.rolling(20).std(ddof=0)
    d['bb_mid'] = d['sma20']
    d['bb_up'], d['bb_lo'] = d['bb_mid'] + 2 * std, d['bb_mid'] - 2 * std
    d['bb_pct'] = ((d['bb_up'] - d['bb_lo']) / d['bb_mid']).rolling(120).rank(pct=True)

    macd = ema(c, 12) - ema(c, 26)
    d['macd_hist'] = macd - ema(macd, 9)

    up_move, down_move = h.diff(), -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_safe = d['atr'].replace(0, np.nan)
    plus_di, minus_di = 100 * rma(plus_dm, 14) / atr_safe, 100 * rma(minus_dm, 14) / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d['adx'] = rma(dx.fillna(0), 14)

    lowest, highest = l.rolling(14).min(), h.rolling(14).max()
    raw_k = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    d['stoch_k'] = raw_k.rolling(3).mean()
    d['stoch_d'] = d['stoch_k'].rolling(7).mean()

    ha_close = ((o + h + l + c) / 4).to_numpy()
    ha_open = np.empty(len(d))
    if len(d):
        ha_open[0] = (o.iloc[0] + c.iloc[0]) / 2
        for k in range(1, len(d)):
            ha_open[k] = (ha_open[k - 1] + ha_close[k - 1]) / 2
    d['ha_green'] = ha_close > ha_open

    d['don_hi'], d['don_lo'] = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    d['hi10'], d['lo10'] = h.rolling(10).max(), l.rolling(10).min()
    d['vol_ma'] = d['tick_volume'].rolling(20).mean()
    rng = h - l
    d['nr7'] = rng <= rng.rolling(7).min()

    # Fractal swings (2 bars either side). A swing at bar j is only known at
    # j+2, hence the shift - swing_hi[i] is the last swing high confirmed by bar i.
    d['swing_hi'] = h.where(h == h.rolling(5, center=True).max()).shift(2).ffill()
    d['swing_lo'] = l.where(l == l.rolling(5, center=True).min()).shift(2).ffill()

    # Classic floor pivots from the previous day's high/low/close.
    day = d['time'].dt.floor('D')
    daily = d.groupby(day).agg(ph=('high', 'max'), pl=('low', 'min'), pc=('close', 'last')).shift(1)
    piv = (daily['ph'] + daily['pl'] + daily['pc']) / 3
    d['piv'] = day.map(piv)
    d['r1'], d['s1'] = day.map(2 * piv - daily['pl']), day.map(2 * piv - daily['ph'])
    d['r2'], d['s2'] = day.map(piv + daily['ph'] - daily['pl']), day.map(piv - (daily['ph'] - daily['pl']))
    return d


def to_arrays(d):
    return {col: d[col].to_numpy() for col in d.columns}


def htf_trend_series(htf_df):
    """+1 / -1 / 0 per HTF bar: price and EMA 50 on the same side of EMA 200."""
    c = htf_df['close']
    e50, e200 = ema(c, 50), ema(c, 200)
    return np.where((c > e50) & (e50 > e200), 1, np.where((c < e50) & (e50 < e200), -1, 0))


def map_htf_to_ltf(ltf_times, ltf_tf, htf_df, htf_tf):
    """HTF trend as it was known at each LTF bar's close (no look-ahead)."""
    if htf_df is None or len(htf_df) < 60:
        return np.zeros(len(ltf_times), dtype=int)
    left = pd.DataFrame({'t': (ltf_times + pd.Timedelta(seconds=TF_SECONDS[ltf_tf])).astype('datetime64[ns]')})
    right = pd.DataFrame({
        't': (htf_df['time'] + pd.Timedelta(seconds=TF_SECONDS[htf_tf])).astype('datetime64[ns]'),
        'trend': htf_trend_series(htf_df),
    })
    merged = pd.merge_asof(left, right, on='t', direction='backward')
    return merged['trend'].fillna(0).astype(int).to_numpy()


# ============================================================================
# STRATEGIES
# Each takes (A, i): A = dict of numpy arrays from add_indicators, i = index of
# the last CLOSED bar. Returns a Signal or None. Stops are price levels on the
# bid chart; the execution layer applies spread, floors and normalisation.
# ============================================================================
@dataclass
class Signal:
    direction: int               # +1 buy, -1 sell
    sl: float
    tp: float | None             # structural target, or None to use the R multiple
    reason: str


def s_trend_pullback(A, i):
    c, h, l, e20, atr = A['close'], A['high'], A['low'], A['ema20'], A['atr'][i]
    if A['adx'][i] < 20:
        return None
    hist_rising = A['macd_hist'][i] > A['macd_hist'][i - 1]
    if e20[i] > A['ema50'][i] > A['ema200'][i]:
        touched = l[i] <= e20[i] or l[i - 1] <= e20[i - 1]
        if touched and c[i] > e20[i] and c[i] > h[i - 1] and hist_rising:
            return Signal(1, l[i - 4:i + 1].min() - 0.3 * atr, None, "Uptrend pullback to EMA20, resumed")
    if e20[i] < A['ema50'][i] < A['ema200'][i]:
        touched = h[i] >= e20[i] or h[i - 1] >= e20[i - 1]
        if touched and c[i] < e20[i] and c[i] < l[i - 1] and not hist_rising:
            return Signal(-1, h[i - 4:i + 1].max() + 0.3 * atr, None, "Downtrend pullback to EMA20, resumed")
    return None


def s_ma_cross(A, i):
    fast, slow, c, atr = A['sma20'], A['sma60'], A['close'][i], A['atr'][i]
    if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1] and c > A['sma100'][i]:
        return Signal(1, c - 2 * atr, None, "SMA20 crossed above SMA60, price above SMA100")
    if fast[i] < slow[i] and fast[i - 1] >= slow[i - 1] and c < A['sma100'][i]:
        return Signal(-1, c + 2 * atr, None, "SMA20 crossed below SMA60, price below SMA100")
    return None


def s_bb_rsi_reversion(A, i):
    c, h, l, atr = A['close'][i], A['high'], A['low'], A['atr'][i]
    if A['adx'][i] >= 30:
        return None
    r = A['rsi14']
    if h[i - 1] > A['bb_up'][i - 1] and c < A['bb_up'][i] and max(r[i], r[i - 1]) >= 70:
        return Signal(-1, h[i - 2:i + 1].max() + 0.3 * atr, A['bb_mid'][i],
                      "Poked upper band, closed back inside, RSI overbought")
    if l[i - 1] < A['bb_lo'][i - 1] and c > A['bb_lo'][i] and min(r[i], r[i - 1]) <= 30:
        return Signal(1, l[i - 2:i + 1].min() - 0.3 * atr, A['bb_mid'][i],
                      "Poked lower band, closed back inside, RSI oversold")
    return None


def s_bb_squeeze(A, i):
    recent = A['bb_pct'][i - 6:i]
    if not np.isfinite(recent).any() or np.nanmin(recent) > 0.2:
        return None
    o, c, h, l, mid, atr = A['open'][i], A['close'][i], A['high'], A['low'], A['bb_mid'], A['atr'][i]
    vol_ok = A['tick_volume'][i] > A['vol_ma'][i]
    if l[i] > mid[i] and c > o and not l[i - 1] > mid[i - 1] and vol_ok:
        return Signal(1, l[i - 1] - 0.1 * atr, None, "Squeeze released: full candle above the mid-band")
    if h[i] < mid[i] and c < o and not h[i - 1] < mid[i - 1] and vol_ok:
        return Signal(-1, h[i - 1] + 0.1 * atr, None, "Squeeze released: full candle below the mid-band")
    return None


def s_pivot_bounce(A, i):
    o, c, h, l, atr, piv = A['open'][i], A['close'], A['high'][i], A['low'][i], A['atr'][i], A['piv'][i]
    if A['adx'][i] >= 30 or not np.isfinite(piv):
        return None
    for name in ('s1', 's2'):
        lvl = A[name][i]
        if l <= lvl < c[i] and c[i] > o and c[i - 1] > lvl and piv > c[i]:
            return Signal(1, l - 0.3 * atr, piv, f"Rejected daily {name.upper()}, target pivot")
    for name in ('r1', 'r2'):
        lvl = A[name][i]
        if h >= lvl > c[i] and c[i] < o and c[i - 1] < lvl and piv < c[i]:
            return Signal(-1, h + 0.3 * atr, piv, f"Rejected daily {name.upper()}, target pivot")
    return None


def s_heikin_ashi(A, i):
    g, k, d, atr = A['ha_green'], A['stoch_k'][i], A['stoch_d'][i], A['atr'][i]
    if g[i] and g[i - 1] and not g[i - 2] and not g[i - 3] and k > d:
        return Signal(1, A['low'][i - 2:i + 1].min() - 0.3 * atr, None, "Two green Heikin-Ashi after red, %K above %D")
    if not g[i] and not g[i - 1] and g[i - 2] and g[i - 3] and k < d:
        return Signal(-1, A['high'][i - 2:i + 1].max() + 0.3 * atr, None, "Two red Heikin-Ashi after green, %K below %D")
    return None


def s_rsi2_momentum(A, i):
    r, c, atr = A['rsi2'], A['close'][i], A['atr'][i]
    if r[i] > 90 and r[i - 1] <= 90 and c > A['sma100'][i]:
        return Signal(1, c - 1.5 * atr, None, "RSI(2) pushed above 90 with price above SMA100")
    if r[i] < 10 and r[i - 1] >= 10 and c < A['sma100'][i]:
        return Signal(-1, c + 1.5 * atr, None, "RSI(2) dropped below 10 with price below SMA100")
    return None


def s_candle_patterns(A, i):
    o, c, h, l, atr = A['open'], A['close'], A['high'], A['low'], A['atr'][i]
    rng = h[i] - l[i]
    body_top, body_bot = max(o[i], c[i]), min(o[i], c[i])
    at_low = min(l[i], l[i - 1]) <= A['lo10'][i]
    at_high = max(h[i], h[i - 1]) >= A['hi10'][i]
    if at_low:
        engulf = (c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1]
                  and c[i] - o[i] > o[i - 1] - c[i - 1])
        hammer = rng >= 0.8 * atr and body_bot - l[i] >= 0.6 * rng and l[i] <= A['lo10'][i]
        if engulf or hammer:
            return Signal(1, min(l[i], l[i - 1]) - 0.2 * atr, None,
                          "Bullish engulfing at a 10-bar low" if engulf else "Hammer at a 10-bar low")
    if at_high:
        engulf = (c[i - 1] > o[i - 1] and c[i] < o[i] and c[i] <= o[i - 1] and o[i] >= c[i - 1]
                  and o[i] - c[i] > c[i - 1] - o[i - 1])
        star = rng >= 0.8 * atr and h[i] - body_top >= 0.6 * rng and h[i] >= A['hi10'][i]
        if engulf or star:
            return Signal(-1, max(h[i], h[i - 1]) + 0.2 * atr, None,
                          "Bearish engulfing at a 10-bar high" if engulf else "Shooting star at a 10-bar high")
    return None


def s_narrow_range(A, i):
    if not A['nr7'][i - 1]:
        return None
    c, atr = A['close'][i], A['atr'][i]
    if c > A['high'][i - 1]:
        return Signal(1, A['low'][i - 1] - 0.1 * atr, None, "Closed above the narrowest bar of 7")
    if c < A['low'][i - 1]:
        return Signal(-1, A['high'][i - 1] + 0.1 * atr, None, "Closed below the narrowest bar of 7")
    return None


def s_sr_breakout(A, i):
    c, atr = A['close'], A['atr'][i]
    if A['tick_volume'][i] <= A['vol_ma'][i]:
        return None
    if c[i] > A['don_hi'][i] and c[i - 1] <= A['don_hi'][i - 1]:
        return Signal(1, A['don_hi'][i] - atr, None, "Closed above 20-bar resistance on rising volume")
    if c[i] < A['don_lo'][i] and c[i - 1] >= A['don_lo'][i - 1]:
        return Signal(-1, A['don_lo'][i] + atr, None, "Closed below 20-bar support on rising volume")
    return None


def s_smc(A, i):
    o, c, h, l, atr = A['open'], A['close'], A['high'], A['low'], A['atr'][i]
    sw_hi, sw_lo = A['swing_hi'], A['swing_lo']

    # 1. Liquidity sweep: wick through the last confirmed swing, close back inside.
    if l[i] < sw_lo[i - 1] < c[i] and c[i] > o[i]:
        return Signal(1, l[i] - 0.2 * atr, None, "Swept sell-side liquidity and closed back above the swing low")
    if h[i] > sw_hi[i - 1] > c[i] and c[i] < o[i]:
        return Signal(-1, h[i] + 0.2 * atr, None, "Swept buy-side liquidity and closed back below the swing high")

    # 2. First retrace into a fair value gap left by a structure break.
    for k in range(i - 1, max(i - 20, 4), -1):
        if l[k] > h[k - 2]:                                   # bullish FVG between bar k-2 and k
            top, bottom = l[k], h[k - 2]
            broke = c[k - 1] > sw_hi[k - 2] or c[k] > sw_hi[k - 1]
            if (top - bottom >= 0.25 * atr and broke and l[k + 1:i].min(initial=np.inf) > top
                    and l[i] <= top and c[i] > bottom and c[i] > o[i]):
                return Signal(1, min(bottom, l[k - 1]) - 0.2 * atr, None, "Retraced into a bullish FVG after a break of structure")
        if h[k] < l[k - 2]:                                   # bearish FVG
            top, bottom = l[k - 2], h[k]
            broke = c[k - 1] < sw_lo[k - 2] or c[k] < sw_lo[k - 1]
            if (top - bottom >= 0.25 * atr and broke and h[k + 1:i].max(initial=-np.inf) < bottom
                    and h[i] >= bottom and c[i] < top and c[i] < o[i]):
                return Signal(-1, max(top, h[k - 1]) + 0.2 * atr, None, "Retraced into a bearish FVG after a break of structure")
    return None


@dataclass
class StrategyDef:
    name: str
    kind: str        # 'trend' = must agree with the HTF trend, 'reversion' = range play, exempt
    fn: object
    json_ids: tuple
    rule: str


STRATEGIES = {
    'trend_pullback': StrategyDef("Trend pullback (EMA 20/50/200 + ADX + MACD)", 'trend', s_trend_pullback, (1, 11, 21),
                                  "EMAs stacked, ADX >= 20, bar touches EMA20 then closes beyond the prior bar with MACD histogram turning."),
    'ma_cross': StrategyDef("MA crossover 20/60/100", 'trend', s_ma_cross, (25,),
                            "SMA20 crosses SMA60 on the right side of SMA100. Exits on the cross back."),
    'bb_rsi_reversion': StrategyDef("Bollinger + RSI fade", 'reversion', s_bb_rsi_reversion, (3, 9),
                                    "Band poke, close back inside, RSI 70/30, ADX < 30. Target: mid-band."),
    'bb_squeeze': StrategyDef("Bollinger squeeze breakout", 'trend', s_bb_squeeze, (31, 6),
                              "Band width in its lowest 20%, then a full candle beyond the mid-band on above-average volume."),
    'pivot_bounce': StrategyDef("Daily pivot bounce", 'reversion', s_pivot_bounce, (10,),
                                "Rejection candle at S1/S2 or R1/R2, ADX < 30. Target: daily pivot."),
    'heikin_ashi': StrategyDef("Heikin-Ashi reversal", 'trend', s_heikin_ashi, (26,),
                               "Two same-colour Heikin-Ashi candles after two opposite, Stochastic %K on the right side of %D."),
    'rsi2_momentum': StrategyDef("2-period RSI", 'trend', s_rsi2_momentum, (33, 27),
                                 "RSI(2) crosses 90 (buy) / 10 (sell) with the SMA100 trend. Exits on the opposite signal."),
    'candle_patterns': StrategyDef("Candlestick patterns", 'trend', s_candle_patterns, (29,),
                                   "Engulfing, hammer or shooting star at a 10-bar extreme."),
    'narrow_range': StrategyDef("Narrow range (NR7) breakout", 'trend', s_narrow_range, (32,),
                                "Close beyond the narrowest bar of the last seven."),
    'sr_breakout': StrategyDef("Support/resistance breakout", 'trend', s_sr_breakout, (2, 6),
                               "Close through the 20-bar high/low on above-average volume."),
    'smc': StrategyDef("ICT/SMC sweep + FVG", 'trend', s_smc, (),
                       "Liquidity sweep rejection, or first retrace into a fair value gap left by a structure break."),
}

# strategies.json id -> (status, how the bot uses it). Ids covered by a rule set
# above are filled in from STRATEGIES; these are the rest.
JSON_NOTES = {
    4: ("built", "Indicators computed: MA, Bollinger, RSI, Stochastic, MACD, ADX"),
    5: ("todo", "Needs a chart-pattern detector"),
    7: ("built", "Higher-timeframe trend filter on every trend/reversal strategy"),
    8: ("partial", "M1/M5 supported with a spread cap; no tick-level logic"),
    12: ("built", "The engine as a whole"),
    13: ("not automatable", "Needs macro data; relevant only as a news blackout"),
    14: ("not automatable", "Needs positioning / sentiment data"),
    15: ("partial", "Session-hours filter and Friday cut-off"),
    16: ("out of scope", "Multi-week swap strategy"),
    17: ("partial", "Used as a risk rule: per-currency exposure cap"),
    18: ("partial", "Multi-symbol scanning with total exposure caps"),
    19: ("out of scope", "Not an automated-entry strategy"),
    20: ("todo", "Needs a two-leg order model"),
    22: ("partial", "Market orders with SL/TP; pending and OCO not yet"),
    23: ("built", "This bot"),
    24: ("todo", "Needs pending limit orders and Fibonacci levels"),
    28: ("todo", "Correction-in-trend detector"),
    30: ("todo", "Breakout-retest logic"),
    34: ("out of scope", "Not specified in the source; not spot FX"),
}


def evaluate_signals(A, i, keys, htf_trend, use_htf):
    """Returns (signals, filtered): signals = [(key, Signal)], filtered = notes on signals the HTF filter dropped."""
    signals, filtered = [], []
    if i < 25:
        return signals, filtered
    for key in keys:
        sig = STRATEGIES[key].fn(A, i)
        if sig is None or not np.isfinite(sig.sl):
            continue
        if sig.tp is not None and not np.isfinite(sig.tp):
            sig.tp = None
        if use_htf and STRATEGIES[key].kind == 'trend' and htf_trend != sig.direction:
            filtered.append(f"{key} {'BUY' if sig.direction == 1 else 'SELL'} dropped: HTF trend is "
                            f"{ {1: 'up', -1: 'down', 0: 'flat'}[int(htf_trend)] }")
            continue
        signals.append((key, sig))
    return signals, filtered


def combine_signals(signals, min_votes):
    """Lead signal when one side out-votes the other by at least min_votes, else None."""
    buys = [x for x in signals if x[1].direction == 1]
    sells = [x for x in signals if x[1].direction == -1]
    if len(buys) - len(sells) >= min_votes:
        return buys[0][0], buys[0][1], len(buys)
    if len(sells) - len(buys) >= min_votes:
        return sells[0][0], sells[0][1], len(sells)
    return None


@dataclass
class Plan:
    entry: float
    sl: float
    tp: float
    risk: float      # price distance entry -> stop


def build_plan(direction, entry, sl, tp, atr, s, min_stop_dist=0.0):
    """Turns a signal into entry/SL/TP, or (None, reason) when it isn't worth taking."""
    if not (np.isfinite(atr) and atr > 0):
        return None, "ATR not available"
    risk = max((entry - sl) * direction, s.min_sl_atr * atr, min_stop_dist)
    if risk > s.max_sl_atr * atr:
        return None, f"stop {risk / atr:.1f} ATR away, wider than the {s.max_sl_atr} ATR limit"
    if tp is not None and s.use_strategy_target:
        reward = (tp - entry) * direction
        if reward < s.min_rr * risk:
            return None, f"target is only {reward / risk:.2f}R, below the {s.min_rr}R minimum"
    else:
        tp = entry + direction * s.rr * risk
    return Plan(entry, entry - direction * risk, tp, risk), ""


# ============================================================================
# BACKTEST (same strategy functions as live)
# ============================================================================
def simulate(A, keys, s, htf, spread):
    """One position at a time. Signal on closed bar i -> entry at the open of
    bar i+1. Bars are bid prices: buys enter at ask and exit at bid, sells the
    reverse. If SL and TP both sit inside one bar, SL is assumed first."""
    o, h, l, c, atr, times = A['open'], A['high'], A['low'], A['close'], A['atr'], A['time']
    trades, pos, cool_until = [], None, -1

    def close(i, price, reason):
        nonlocal pos, cool_until
        r = (price - pos['entry']) * pos['dir'] / pos['risk']
        trades.append({'strategy': pos['strategy'], 'direction': 'BUY' if pos['dir'] == 1 else 'SELL',
                       'entry_time': times[pos['i']], 'exit_time': times[i], 'entry': pos['entry'],
                       'exit': price, 'r': r, 'bars': i - pos['i'] + 1, 'reason': reason})
        if r < 0:
            cool_until = i + s.cooldown_bars
        pos = None

    for i in range(min(WARMUP, len(c) // 3), len(c) - 1):
        sp = spread[i]
        if pos is not None:
            d = pos['dir']
            if d == 1:
                if o[i] <= pos['sl']:
                    close(i, o[i], 'SL (gap)')
                elif l[i] <= pos['sl']:
                    close(i, pos['sl'], 'SL')
                elif h[i] >= pos['tp']:
                    close(i, pos['tp'], 'TP')
            else:
                if o[i] + sp >= pos['sl']:
                    close(i, o[i] + sp, 'SL (gap)')
                elif h[i] + sp >= pos['sl']:
                    close(i, pos['sl'], 'SL')
                elif l[i] + sp <= pos['tp']:
                    close(i, pos['tp'], 'TP')
        if pos is not None:
            d = pos['dir']
            px = c[i] if d == 1 else c[i] + sp
            r_now = (px - pos['entry']) * d / pos['risk']
            if s.max_bars_held > 0 and i - pos['i'] + 1 >= s.max_bars_held:
                close(i, px, 'time stop')
            else:
                if s.be_at_r > 0 and not pos['be'] and r_now >= s.be_at_r:
                    pos['be'] = True
                    pos['sl'] = max(pos['sl'] * d, (pos['entry'] + d * 0.05 * pos['risk']) * d) * d
                if s.trail_atr > 0 and r_now >= s.trail_start_r and atr[i] > 0:
                    pos['sl'] = max(pos['sl'] * d, (px - d * s.trail_atr * atr[i]) * d) * d

        signals, _ = evaluate_signals(A, i, keys, htf[i], s.use_htf_filter)
        if pos is not None and s.exit_on_opposite and any(
                k == pos['strategy'] and sg.direction == -pos['dir'] for k, sg in signals):
            close(i, c[i] if pos['dir'] == 1 else c[i] + sp, 'opposite signal')

        if pos is None and i >= cool_until:
            lead = combine_signals(signals, s.min_votes)
            if lead:
                key, sig, _ = lead
                nsp = spread[i + 1]
                entry = o[i + 1] + (nsp if sig.direction == 1 else 0.0)
                stop = sig.sl + (nsp if sig.direction == -1 else 0.0)
                plan, _ = build_plan(sig.direction, entry, stop, sig.tp, atr[i], s)
                if plan and nsp <= s.max_spread_frac * plan.risk:
                    pos = {'dir': sig.direction, 'entry': plan.entry, 'sl': plan.sl, 'tp': plan.tp,
                           'risk': plan.risk, 'strategy': key, 'i': i + 1, 'be': False}
    return pd.DataFrame(trades, columns=['strategy', 'direction', 'entry_time', 'exit_time',
                                         'entry', 'exit', 'r', 'bars', 'reason'])


def summarize(r_values):
    r = np.asarray(r_values, dtype=float)
    if len(r) == 0:
        return {'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'expectancy': 0.0, 'total': 0.0, 'max_dd': 0.0}
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    equity = np.cumsum(r)
    return {
        'trades': len(r), 'win_rate': float((r > 0).mean() * 100),
        'profit_factor': float(wins / losses) if losses > 0 else float('inf'),
        'expectancy': float(r.mean()), 'total': float(r.sum()),
        'max_dd': float((np.maximum.accumulate(np.maximum(equity, 0)) - equity).max()),
    }


def run_backtest(df, s, htf_df, spread):
    """spread: price-unit array aligned with df (or a scalar). Returns (portfolio trades, per-strategy table, indicator frame)."""
    d = add_indicators(df)
    A = to_arrays(d)
    htf = map_htf_to_ltf(d['time'], s.timeframe, htf_df, s.htf) if s.use_htf_filter else np.zeros(len(d), dtype=int)
    spread = np.broadcast_to(np.asarray(spread, dtype=float), (len(d),))
    portfolio = simulate(A, s.strategies, s, htf, spread)
    rows = []
    for key in s.strategies:
        solo = simulate(A, [key], s, htf, spread)
        rows.append({'strategy': STRATEGIES[key].name, **summarize(solo['r'])})
    return portfolio, pd.DataFrame(rows), d


# ============================================================================
# MT5 ACCESS (every call goes through one lock - the package is not thread-safe)
# ============================================================================
def mt(name, *args, **kwargs):
    with MT5_LOCK:
        return getattr(mt5, name)(*args, **kwargs)


def mt5_connect():
    if not MT5_AVAILABLE:
        return False
    with MT5_LOCK:
        if mt5.account_info() is not None:
            return True
        if not mt5.initialize():
            return False
        return mt5.account_info() is not None


def account_mode(account):
    """('DEMO'|'REAL'|'CONTEST'|'UNKNOWN', is_real). Unknown is treated as real."""
    mode = getattr(account, 'trade_mode', None)
    if mode == getattr(mt5, 'ACCOUNT_TRADE_MODE_DEMO', 0):
        return "DEMO", False
    if mode == getattr(mt5, 'ACCOUNT_TRADE_MODE_CONTEST', 1):
        return "CONTEST", False
    return ("REAL" if mode == getattr(mt5, 'ACCOUNT_TRADE_MODE_REAL', 2) else "UNKNOWN"), True


def resolve_symbol(name):
    """Finds the broker's actual symbol for a plain name (EURUSD -> EURUSDm, EURUSD.a, ...)."""
    info = mt('symbol_info', name)
    if info is None:
        matches = [x.name for x in (mt('symbols_get', group=f"*{name}*") or []) if x.name.upper().startswith(name.upper())]
        if not matches:
            return None
        name = min(matches, key=len)
        info = mt('symbol_info', name)
    if info is not None and not info.visible:
        mt('symbol_select', name, True)
    return name if info is not None else None


def fetch_bars(symbol, tf, count):
    """Closed bars only: position 0 is the candle still forming, so start at 1."""
    rates = mt('copy_rates_from_pos', symbol, getattr(mt5, f"TIMEFRAME_{tf}"), 1, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def last_closed_bar_time(symbol, tf):
    rates = mt('copy_rates_from_pos', symbol, getattr(mt5, f"TIMEFRAME_{tf}"), 1, 1)
    return int(rates[0]['time']) if rates is not None and len(rates) else None


def calc_pnl(symbol, info, direction, lot, entry, exit_price):
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    value = mt('order_calc_profit', order_type, symbol, float(lot), float(entry), float(exit_price))
    if value is not None:
        return float(value)
    tick_size = getattr(info, 'trade_tick_size', 0) or info.point
    return (exit_price - entry) * direction / tick_size * (getattr(info, 'trade_tick_value', 0) or 1.0) * lot


def round_lot_down(raw, info):
    step = info.volume_step or 0.01
    lot = math.floor(raw / step + 1e-9) * step
    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(min(lot, info.volume_max or lot), decimals)


def _filling_modes(info):
    mode = getattr(info, 'filling_mode', 0) or 0
    order = []
    if mode & 1:
        order.append(mt5.ORDER_FILLING_FOK)
    if mode & 2:
        order.append(mt5.ORDER_FILLING_IOC)
    for m in (mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK):
        if m not in order:
            order.append(m)
    return order


def _send_deal(symbol, order_type, volume, deviation, comment, sl=0.0, tp=0.0, position=None):
    """Market deal with retries on requote/price change and filling-mode fallback. Returns (result|None, message)."""
    info = mt('symbol_info', symbol)
    if info is None:
        return None, f"no symbol info for {symbol}"
    fillings = _filling_modes(info)
    retry_codes = {getattr(mt5, n, -1) for n in ('TRADE_RETCODE_REQUOTE', 'TRADE_RETCODE_PRICE_CHANGED', 'TRADE_RETCODE_PRICE_OFF')}
    message = "no attempt made"
    for attempt in range(4):
        tick = mt('symbol_info_tick', symbol)
        if tick is None:
            return None, "no tick"
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume), "type": order_type,
            "price": tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid,
            "deviation": int(deviation), "magic": BOT_MAGIC, "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": fillings[0],
        }
        if sl:
            request["sl"] = round(float(sl), info.digits)
        if tp:
            request["tp"] = round(float(tp), info.digits)
        if position is not None:
            request["position"] = int(position)
        with MT5_LOCK:
            result = mt5.order_send(request)
            last_error = mt5.last_error() if result is None else None
        if result is None:
            return None, f"order_send returned None: {last_error}"
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result, "filled"
        message = f"retcode {result.retcode}: {result.comment}"
        if result.retcode == getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030) and len(fillings) > 1:
            fillings.pop(0)
        elif result.retcode in retry_codes:
            time.sleep(0.3)
        else:
            break
    return None, message


def live_close_ticket(ticket, deviation=20, comment="SuperBot close"):
    """(ok, message). ok is True when the position is closed or already gone."""
    positions = mt('positions_get', ticket=int(ticket))
    if positions is None:
        return False, "positions_get failed"
    if not positions:
        return True, "already closed"
    p = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    result, message = _send_deal(p.symbol, close_type, p.volume, deviation, comment, position=p.ticket)
    return result is not None, message


def close_all_bot_positions():
    """Panic button for when the engine is not running. Returns a list of messages."""
    positions = mt('positions_get')
    if positions is None:
        return ["Could not read positions from MT5."]
    out = []
    for p in positions:
        if p.magic == BOT_MAGIC:
            ok, message = live_close_ticket(p.ticket)
            out.append(f"#{p.ticket} {p.symbol}: {'closed' if ok else 'FAILED - ' + message}")
    return out or ["No positions with this bot's magic number."]


# ============================================================================
# ENGINE
# ============================================================================
@dataclass
class Position:
    ticket: int
    symbol: str
    direction: int
    lot: float
    entry: float
    sl: float
    tp: float
    sl_initial: float
    init_risk: float          # price distance entry -> initial stop
    risk_money: float
    strategy: str
    open_time: str            # broker server time, ISO
    open_epoch: float
    be_done: bool = False
    pnl: float = 0.0
    price: float = 0.0
    pending_reason: str = ""  # set once the bot itself has closed it and is waiting for deal history
    close_attempts: int = 0
    retry_after: float = 0.0


class Engine:
    def __init__(self, settings):
        self.s = settings
        self.live = settings.mode == 'live'
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.thread = None
        self.status = 'idle'              # idle, running, stopped, error
        self.positions = {}
        self.decisions = deque(maxlen=300)
        self.symbol_state = {}
        self.symbols = []                 # broker symbol names
        self.last_bar = {}
        self.cooldown_until = {}
        self.closed_today = []
        self.opened_today = 0
        self.day = None
        self.day_start_equity = None
        self.day_halt = None
        self.friday_flat_done = None
        self.paper_seq = 1
        self.flatten_on_stop = False
        self.flatten_req = False
        self.last_cycle = None
        self.last_error = None
        self.server_time = None
        self.state_file = DATA_DIR / f"state_{settings.mode}.json"

    # ---- lifecycle ---------------------------------------------------------
    def start(self):
        self.status = 'running'
        self.thread = threading.Thread(target=self._run, name="superbot-engine", daemon=True)
        self.thread.start()

    def stop(self, flatten=False):
        self.flatten_on_stop = flatten
        self.stop_evt.set()

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def _run(self):
        try:
            self._startup()
            while not self.stop_evt.is_set():
                try:
                    self._cycle()
                except Exception as exc:  # noqa: BLE001 - the loop must outlive transient errors so positions stay managed
                    log.exception("cycle failed")
                    self.last_error = f"Cycle error (still running): {exc}"
                self.stop_evt.wait(self.s.poll_seconds)
            if self.flatten_on_stop:
                self._flatten_all("closed on stop")
            self.status = 'stopped'
            self._note('ENGINE', '', 'STOPPED', 'flattened' if self.flatten_on_stop else 'open positions keep their broker SL/TP')
        except Exception as exc:  # noqa: BLE001
            log.exception("engine stopped on error")
            self.status = 'error'
            self.last_error = f"Engine stopped: {exc}"
        finally:
            self._save_state()

    def _startup(self):
        if not mt5_connect():
            raise RuntimeError("MT5 is not connected - start the terminal and log in.")
        for name in self.s.symbols:
            resolved = resolve_symbol(name)
            if resolved is None:
                self._note(name, '', 'SKIPPED', 'symbol not offered by this broker')
            else:
                self.symbols.append(resolved)
        if not self.symbols:
            raise RuntimeError("None of the selected symbols exist on this broker.")
        if self.live:
            account = mt('account_info')
            if getattr(account, 'margin_mode', 2) != getattr(mt5, 'ACCOUNT_MARGIN_MODE_RETAIL_HEDGING', 2):
                self.s.max_per_symbol = 1
                self._note('ENGINE', '', 'INFO', 'netting account: one position per symbol')
        self._load_state()
        for sym in self.symbols:
            # The bar that is already closed at start-up is analysed for display
            # but never traded - it may have closed most of a candle ago.
            self.last_bar[sym] = last_closed_bar_time(sym, self.s.timeframe)
            self._evaluate(sym, trade=False)
        self._note('ENGINE', '', 'STARTED', f"{self.s.mode} | {', '.join(self.symbols)} | {self.s.timeframe} | "
                                            f"{len(self.s.strategies)} strategies")

    # ---- persistence -------------------------------------------------------
    def _load_state(self):
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        known = {f.name for f in fields(Position)}
        for item in raw.get('positions', []):
            pos = Position(**{k: v for k, v in item.items() if k in known})
            self.positions[pos.ticket] = pos
        self.day = raw.get('day')
        self.day_start_equity = raw.get('day_start_equity')
        self.day_halt = raw.get('day_halt')
        self.opened_today = raw.get('opened_today', 0)
        self.cooldown_until = raw.get('cooldown_until', {})
        self.paper_seq = raw.get('paper_seq', 1)
        if self.day:
            self._load_closed_today()
        if self.positions:
            self._note('ENGINE', '', 'INFO', f"resumed {len(self.positions)} open position(s) from the last run")

    def _save_state(self):
        with self.lock:
            save_json(self.state_file, {
                'positions': [asdict(p) for p in self.positions.values()],
                'day': self.day, 'day_start_equity': self.day_start_equity, 'day_halt': self.day_halt,
                'opened_today': self.opened_today, 'cooldown_until': self.cooldown_until,
                'paper_seq': self.paper_seq,
            })

    def _load_closed_today(self):
        trades = load_trades()
        if trades.empty:
            self.closed_today = []
            return
        today = trades[(trades['mode'] == self.s.mode) & (trades['close_time'].astype(str).str[:10] == self.day)]
        self.closed_today = [float(x) for x in today['pnl']]

    def _note(self, symbol, strategy, action, detail):
        stamp = (self.server_time or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.decisions.appendleft({'time': stamp, 'symbol': symbol, 'strategy': strategy,
                                       'action': action, 'detail': detail})
        log.info("%s | %s | %s | %s", symbol, strategy, action, detail)

    # ---- main cycle --------------------------------------------------------
    def _cycle(self):
        if not mt5_connect():
            self.last_error = "MT5 disconnected - retrying. Open positions are protected by their broker-side SL/TP."
            return
        self._update_clock()
        if self.flatten_req:
            self.flatten_req = False
            self._flatten_all("flattened by user")
        self._manage()
        self._check_halts()
        self._friday_cutoff()
        for sym in self.symbols:
            bar_time = last_closed_bar_time(sym, self.s.timeframe)
            if bar_time is not None and bar_time != self.last_bar.get(sym):
                self.last_bar[sym] = bar_time
                self._evaluate(sym, trade=True)
        self.last_cycle = time.time()
        if self.last_error and self.last_error.startswith("MT5 disconnected"):
            self.last_error = None

    def _update_clock(self):
        stamps = [t.time for t in (mt('symbol_info_tick', sym) for sym in self.symbols) if t is not None]
        if stamps:
            # MT5 tick times are broker server wall-clock encoded as a UTC epoch.
            self.server_time = datetime.fromtimestamp(max(stamps), tz=timezone.utc).replace(tzinfo=None)
        elif self.server_time is None:
            self.server_time = datetime.now(timezone.utc).replace(tzinfo=None)
        today = self.server_time.strftime("%Y-%m-%d")
        if today != self.day:
            account = mt('account_info')
            self.day = today
            self.day_start_equity = float(account.balance) if account else None
            self.day_halt = None
            self.opened_today = 0
            self._load_closed_today()
            self._save_state()

    def _check_halts(self):
        if self.day_halt:
            return
        with self.lock:
            floating = sum(p.pnl for p in self.positions.values())
        realised = sum(self.closed_today)
        reason = None
        if self.day_start_equity and self.s.daily_loss_pct > 0:
            limit = self.day_start_equity * self.s.daily_loss_pct / 100
            if realised + floating <= -limit:
                reason = f"daily loss limit hit ({realised + floating:.2f} vs -{limit:.2f})"
        streak = 0
        for pnl in reversed(self.closed_today):
            if pnl >= 0:
                break
            streak += 1
        if reason is None and 0 < self.s.max_consec_losses <= streak:
            reason = f"{streak} losses in a row"
        if reason is None and 0 < self.s.max_daily_trades <= self.opened_today:
            reason = f"{self.opened_today} trades opened today (limit)"
        if reason:
            self.day_halt = reason
            self._note('ENGINE', '', 'HALTED', f"{reason} - no new entries until the next server day")
            self._save_state()

    def _friday_cutoff(self):
        if self.s.friday_close_hour < 0 or not self._past_friday_cutoff() or self.friday_flat_done == self.day:
            return
        self.friday_flat_done = self.day
        if self.positions:
            self._flatten_all("Friday cut-off")

    def _past_friday_cutoff(self):
        return (self.s.friday_close_hour >= 0 and self.server_time is not None
                and self.server_time.weekday() == 4 and self.server_time.hour >= self.s.friday_close_hour)

    # ---- analysis ----------------------------------------------------------
    def _evaluate(self, sym, trade):
        df = fetch_bars(sym, self.s.timeframe, self.s.bars)
        if df is None or len(df) < 60:
            self.symbol_state[sym] = {'note': 'not enough bar data'}
            return
        A = to_arrays(add_indicators(df))
        i = len(df) - 1
        htf_trend = 0
        if self.s.use_htf_filter:
            htf_df = fetch_bars(sym, self.s.htf, 300)
            if htf_df is not None and len(htf_df) >= 60:
                htf_trend = int(htf_trend_series(htf_df)[-1])
        signals, filtered = evaluate_signals(A, i, self.s.strategies, htf_trend, self.s.use_htf_filter)
        atr = float(A['atr'][i])
        self.symbol_state[sym] = {
            'bar': str(df['time'].iloc[-1]), 'close': float(A['close'][i]), 'atr': atr, 'htf_trend': htf_trend,
            'signals': [f"{k}: {'BUY' if sg.direction == 1 else 'SELL'}" for k, sg in signals], 'filtered': filtered,
        }
        if not trade:
            return

        if self.s.exit_on_opposite:
            with self.lock:
                mine = [p for p in self.positions.values() if p.symbol == sym and not p.pending_reason]
            for pos in mine:
                if any(k == pos.strategy and sg.direction == -pos.direction for k, sg in signals):
                    self._close(pos, "opposite signal")

        lead = combine_signals(signals, self.s.min_votes)
        if lead is None:
            if signals:
                self._note(sym, ', '.join(k for k, _ in signals), 'SKIPPED',
                           f"signals do not out-vote each other by {self.s.min_votes}")
            for item in filtered:
                self._note(sym, item.split(' ')[0], 'SKIPPED', item)
            return
        key, sig, votes = lead
        self._enter(sym, key, sig, votes, atr)

    # ---- entries -----------------------------------------------------------
    def _entry_block(self, sym, info):
        if self.day_halt:
            return f"halted: {self.day_halt}"
        hour = self.server_time.hour if self.server_time else 0
        start, end = self.s.session_start, self.s.session_end
        in_session = start <= hour < end if start <= end else (hour >= start or hour < end)
        if not in_session:
            return f"outside session hours ({start:02d}-{end:02d} server time)"
        if self._past_friday_cutoff():
            return "past the Friday cut-off"
        if info.trade_mode != getattr(mt5, 'SYMBOL_TRADE_MODE_FULL', 4):
            return "symbol is not fully tradable right now"
        if time.time() < self.cooldown_until.get(sym, 0):
            return "cooling down after a loss on this symbol"
        with self.lock:
            open_positions = list(self.positions.values())
        if len(open_positions) >= self.s.max_positions:
            return f"max open positions ({self.s.max_positions})"
        if sum(p.symbol == sym for p in open_positions) >= self.s.max_per_symbol:
            return "already in a position on this symbol"
        currencies = {info.currency_base, info.currency_profit}
        for cur in currencies:
            exposed = 0
            for p in open_positions:
                p_info = mt('symbol_info', p.symbol)
                if p_info is not None and cur in (p_info.currency_base, p_info.currency_profit):
                    exposed += 1
            if exposed >= self.s.max_per_currency:
                return f"already {exposed} open position(s) involving {cur}"
        return None

    def _enter(self, sym, key, sig, votes, atr):
        side = 'BUY' if sig.direction == 1 else 'SELL'
        info, tick, account = mt('symbol_info', sym), mt('symbol_info_tick', sym), mt('account_info')
        if info is None or tick is None or account is None:
            return self._note(sym, key, 'BLOCKED', f"{side}: no symbol/tick/account data")
        block = self._entry_block(sym, info)
        if block:
            return self._note(sym, key, 'BLOCKED', f"{side}: {block}")

        spread = tick.ask - tick.bid
        entry = tick.ask if sig.direction == 1 else tick.bid
        stop = sig.sl + (spread if sig.direction == -1 else 0.0)   # sell stops trigger on the ask
        min_stop = (getattr(info, 'trade_stops_level', 0) or 0) * info.point + spread
        plan, why = build_plan(sig.direction, entry, stop, sig.tp, atr, self.s, min_stop)
        if plan is None:
            return self._note(sym, key, 'SKIPPED', f"{side}: {why}")
        if spread > self.s.max_spread_frac * plan.risk:
            return self._note(sym, key, 'BLOCKED', f"{side}: spread is {spread / plan.risk:.0%} of the stop distance")

        equity = min(account.balance, account.equity)
        budget = equity * self.s.risk_pct / 100
        loss_per_lot = -calc_pnl(sym, info, sig.direction, 1.0, plan.entry, plan.sl)
        if loss_per_lot <= 0:
            return self._note(sym, key, 'BLOCKED', f"{side}: could not price the risk per lot")
        lot = round_lot_down(budget / loss_per_lot, info)
        if lot < info.volume_min:
            min_risk = info.volume_min * loss_per_lot
            if min_risk > equity * self.s.max_risk_pct / 100:
                return self._note(sym, key, 'BLOCKED',
                                  f"{side}: minimum lot would risk {min_risk:.2f} ({min_risk / equity:.2%} of equity), "
                                  f"above the {self.s.max_risk_pct}% cap")
            lot = info.volume_min
        risk_money = lot * loss_per_lot
        order_type = mt5.ORDER_TYPE_BUY if sig.direction == 1 else mt5.ORDER_TYPE_SELL

        if self.live:
            margin = mt('order_calc_margin', order_type, sym, float(lot), float(plan.entry))
            if margin is not None and margin > account.margin_free:
                return self._note(sym, key, 'BLOCKED', f"{side}: needs {margin:.2f} margin, {account.margin_free:.2f} free")
            result, message = _send_deal(sym, order_type, lot, self.s.deviation, f"SB {key}", plan.sl, plan.tp)
            if result is None:
                return self._note(sym, key, 'REJECTED', f"{side} {lot}: {message}")
            ticket, fill = int(result.order), float(result.price or plan.entry)
            found = mt('positions_get', ticket=ticket)
            if not found:   # netting accounts: the position ticket differs from the order ticket
                found = [p for p in (mt('positions_get', symbol=sym) or []) if p.magic == BOT_MAGIC]
            if found:
                ticket, fill = int(found[0].ticket), float(found[0].price_open)
        else:
            ticket, fill = self.paper_seq, plan.entry
            self.paper_seq += 1

        pos = Position(ticket=ticket, symbol=sym, direction=sig.direction, lot=lot, entry=fill, sl=plan.sl,
                       tp=plan.tp, sl_initial=plan.sl, init_risk=abs(fill - plan.sl) or plan.risk,
                       risk_money=risk_money, strategy=key,
                       open_time=self.server_time.isoformat(sep=' ', timespec='seconds'),
                       open_epoch=time.time(), price=fill)
        with self.lock:
            self.positions[ticket] = pos
            self.opened_today += 1
        self._save_state()
        self._note(sym, key, 'OPENED', f"{side} {lot} @ {fill:.{info.digits}f} | SL {plan.sl:.{info.digits}f} | "
                                       f"TP {plan.tp:.{info.digits}f} | risk {risk_money:.2f} | votes {votes} | {sig.reason}")

    # ---- position management ----------------------------------------------
    def _manage(self):
        broker = {}
        if self.live:
            raw = mt('positions_get')
            if raw is None:
                return      # unknown is not the same as closed - try again next cycle
            broker = {p.ticket: p for p in raw if p.magic == BOT_MAGIC}
            for ticket, bp in broker.items():
                if ticket not in self.positions:
                    self._adopt(bp)
        with self.lock:
            tracked = list(self.positions.values())

        for pos in tracked:
            d = pos.direction
            if self.live:
                bp = broker.get(pos.ticket)
                if bp is None:
                    self._finalize_live(pos)
                    continue
                pos.sl, pos.tp, pos.lot = bp.sl, bp.tp, bp.volume
                pos.price, pos.pnl = bp.price_current, bp.profit + getattr(bp, 'swap', 0.0)
            else:
                tick, info = mt('symbol_info_tick', pos.symbol), mt('symbol_info', pos.symbol)
                if tick is None or info is None:
                    continue
                pos.price = tick.bid if d == 1 else tick.ask
                if (pos.price - pos.sl) * d <= 0:
                    self._close_paper(pos, pos.price, 'SL', info)   # a gap fills at the worse price
                    continue
                if (pos.price - pos.tp) * d >= 0:
                    self._close_paper(pos, pos.tp, 'TP', info)
                    continue
                pos.pnl = calc_pnl(pos.symbol, info, d, pos.lot, pos.entry, pos.price)

            if pos.pending_reason or time.time() < pos.retry_after:
                continue
            if self.s.quick_profit > 0 and pos.pnl >= self.s.quick_profit:
                self._close(pos, f"quick profit >= {self.s.quick_profit}")
                continue
            if self.s.max_bars_held > 0 and time.time() - pos.open_epoch >= self.s.max_bars_held * TF_SECONDS[self.s.timeframe]:
                self._close(pos, "time stop")
                continue

            r_now = (pos.price - pos.entry) * d / pos.init_risk if pos.init_risk else 0.0
            new_sl, to_be = pos.sl, False
            if self.s.be_at_r > 0 and not pos.be_done and r_now >= self.s.be_at_r:
                candidate = pos.entry + d * 0.05 * pos.init_risk
                if (candidate - new_sl) * d > 0:
                    new_sl = candidate
                to_be = True
            atr = self.symbol_state.get(pos.symbol, {}).get('atr')
            if self.s.trail_atr > 0 and atr and r_now >= self.s.trail_start_r:
                candidate = pos.price - d * self.s.trail_atr * atr
                if (candidate - new_sl) * d > 0.1 * atr:     # don't pester the broker with tiny moves
                    new_sl = candidate
            if (new_sl - pos.sl) * d > 0:
                self._modify_sl(pos, new_sl, "break-even" if to_be else "trail")
            elif to_be:
                pos.be_done = True

    def _adopt(self, bp):
        direction = 1 if bp.type == mt5.POSITION_TYPE_BUY else -1
        atr = self.symbol_state.get(bp.symbol, {}).get('atr') or 0.0
        risk = abs(bp.price_open - bp.sl) if bp.sl else atr
        pos = Position(ticket=int(bp.ticket), symbol=bp.symbol, direction=direction, lot=bp.volume,
                       entry=bp.price_open, sl=bp.sl, tp=bp.tp, sl_initial=bp.sl, init_risk=risk, risk_money=0.0,
                       strategy='adopted', open_time=str(datetime.fromtimestamp(bp.time, tz=timezone.utc).replace(tzinfo=None)),
                       open_epoch=time.time(), price=bp.price_current)
        with self.lock:
            self.positions[pos.ticket] = pos
        self._save_state()
        self._note(bp.symbol, 'adopted', 'INFO', f"picked up untracked bot position #{bp.ticket}"
                                                 + ("" if bp.sl else " - IT HAS NO STOP LOSS"))

    def _modify_sl(self, pos, new_sl, why):
        info = mt('symbol_info', pos.symbol)
        if info is None:
            return
        new_sl = round(new_sl, info.digits)
        if self.live:
            if abs(pos.price - new_sl) < (getattr(info, 'trade_stops_level', 0) or 0) * info.point:
                return
            request = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol, "position": int(pos.ticket),
                       "sl": new_sl, "tp": float(pos.tp)}
            result = mt('order_send', request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                pos.retry_after = time.time() + 30
                return self._note(pos.symbol, pos.strategy, 'WARN',
                                  f"#{pos.ticket} stop move rejected: {getattr(result, 'comment', 'no result')}")
        pos.sl = new_sl
        if why == "break-even":
            pos.be_done = True
        self._save_state()
        self._note(pos.symbol, pos.strategy, 'MODIFIED', f"#{pos.ticket} stop -> {new_sl:.{info.digits}f} ({why})")

    def _close(self, pos, reason):
        if self.live:
            ok, message = live_close_ticket(pos.ticket, self.s.deviation)
            if not ok:
                pos.retry_after = time.time() + 5     # still open at the broker: keep tracking, try again shortly
                return self._note(pos.symbol, pos.strategy, 'WARN', f"#{pos.ticket} close failed ({reason}): {message}")
            pos.pending_reason = reason
            self._finalize_live(pos)
        else:
            tick, info = mt('symbol_info_tick', pos.symbol), mt('symbol_info', pos.symbol)
            if tick is None or info is None:
                return
            self._close_paper(pos, tick.bid if pos.direction == 1 else tick.ask, reason, info)

    def _close_paper(self, pos, price, reason, info):
        self._record_close(pos, price, calc_pnl(pos.symbol, info, pos.direction, pos.lot, pos.entry, price), reason)

    def _finalize_live(self, pos):
        """Position is gone from the broker: book it with the real P&L from deal history."""
        deals = mt('history_deals_get', position=int(pos.ticket))
        exits = [x for x in (deals or []) if x.entry != getattr(mt5, 'DEAL_ENTRY_IN', 0)]
        if not exits:
            pos.close_attempts += 1
            if pos.close_attempts < 15:
                return      # history can lag the position list by a moment
            return self._record_close(pos, pos.price, pos.pnl, (pos.pending_reason or "closed at broker") + " (P&L unverified)")
        pnl = sum(x.profit + x.commission + x.swap + getattr(x, 'fee', 0.0) for x in deals)
        reasons = {getattr(mt5, 'DEAL_REASON_SL', 4): 'SL', getattr(mt5, 'DEAL_REASON_TP', 5): 'TP',
                   getattr(mt5, 'DEAL_REASON_SO', 6): 'STOP-OUT'}
        reason = pos.pending_reason or reasons.get(exits[-1].reason, 'closed outside the bot')
        self._record_close(pos, exits[-1].price, pnl, reason)

    def _record_close(self, pos, exit_price, pnl, reason):
        r_multiple = pnl / pos.risk_money if pos.risk_money else 0.0
        append_trade({
            'mode': self.s.mode, 'ticket': pos.ticket, 'symbol': pos.symbol, 'strategy': pos.strategy,
            'direction': 'BUY' if pos.direction == 1 else 'SELL', 'lot': pos.lot, 'entry': pos.entry,
            'exit': exit_price, 'sl_initial': pos.sl_initial, 'tp': pos.tp, 'open_time': pos.open_time,
            'close_time': (self.server_time or datetime.now()).isoformat(sep=' ', timespec='seconds'),
            'pnl': round(pnl, 2), 'r_multiple': round(r_multiple, 2), 'reason': reason,
        })
        with self.lock:
            self.positions.pop(pos.ticket, None)
            self.closed_today.append(float(pnl))
            if pnl < 0:
                self.cooldown_until[pos.symbol] = time.time() + self.s.cooldown_bars * TF_SECONDS[self.s.timeframe]
        self._save_state()
        self._note(pos.symbol, pos.strategy, 'CLOSED', f"#{pos.ticket} {reason} | P&L {pnl:.2f} ({r_multiple:+.2f}R)")

    def _flatten_all(self, reason):
        with self.lock:
            tracked = list(self.positions.values())
        for pos in tracked:
            if not pos.pending_reason:
                self._close(pos, reason)

    # ---- UI ----------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            positions = [{
                'ticket': p.ticket, 'symbol': p.symbol, 'side': 'BUY' if p.direction == 1 else 'SELL',
                'lot': p.lot, 'entry': p.entry, 'price': p.price, 'sl': p.sl, 'tp': p.tp, 'pnl': round(p.pnl, 2),
                'R': round((p.price - p.entry) * p.direction / p.init_risk, 2) if p.init_risk else 0.0,
                'strategy': p.strategy, 'opened': p.open_time,
            } for p in self.positions.values()]
            return {
                'status': self.status if self.is_alive() or self.status != 'running' else 'error',
                'mode': self.s.mode, 'halt': self.day_halt, 'positions': positions,
                'decisions': list(self.decisions), 'symbol_state': dict(self.symbol_state),
                'realised': sum(self.closed_today), 'closed_count': len(self.closed_today),
                'wins': sum(x > 0 for x in self.closed_today), 'opened_today': self.opened_today,
                'floating': sum(p.pnl for p in self.positions.values()), 'day': self.day,
                'day_start_equity': self.day_start_equity, 'last_cycle': self.last_cycle,
                'last_error': self.last_error, 'server_time': self.server_time,
            }


@st.cache_resource
def get_runtime():
    """One engine per Streamlit process, shared by every browser session and surviving page refreshes."""
    return {'engine': None}


# ============================================================================
# CHARTS
# ============================================================================
def chart_colors():
    try:
        dark = st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit or bare run
        dark = True
    return {
        'blue': '#3987e5' if dark else '#2a78d6', 'orange': '#d95926' if dark else '#eb6834',
        'aqua': '#199e70' if dark else '#1baf7a', 'muted': '#898781',
        'up': '#0ca30c', 'down': '#e66767' if dark else '#d03b3b',
        'grid': '#2c2c2a' if dark else '#e1e0d9',
    }


def price_chart(d, markers, title):
    """Candles with EMAs and Bollinger bands; RSI in its own panel (never a second y-axis)."""
    col = chart_colors()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.76, 0.24])
    fig.add_trace(go.Candlestick(x=d['time'], open=d['open'], high=d['high'], low=d['low'], close=d['close'],
                                 name='Price', increasing_line_color=col['up'], decreasing_line_color=col['down'],
                                 increasing_line_width=1, decreasing_line_width=1), row=1, col=1)
    for name, column, colour in (('EMA 20', 'ema20', col['blue']), ('EMA 50', 'ema50', col['orange']),
                                 ('EMA 200', 'ema200', col['aqua'])):
        fig.add_trace(go.Scatter(x=d['time'], y=d[column], name=name, line=dict(color=colour, width=2)), row=1, col=1)
    for name, column in (('Bollinger upper', 'bb_up'), ('Bollinger lower', 'bb_lo')):
        fig.add_trace(go.Scatter(x=d['time'], y=d[column], name=name, showlegend=column == 'bb_up',
                                 legendgroup='bb', line=dict(color=col['muted'], width=1, dash='dot')), row=1, col=1)
    if len(markers):
        for direction, label, symbol, colour in ((1, 'Buy signal', 'triangle-up', col['up']),
                                                 (-1, 'Sell signal', 'triangle-down', col['down'])):
            m = markers[markers['direction'] == direction]
            if len(m):
                fig.add_trace(go.Scatter(x=m['time'], y=m['price'], mode='markers', name=label, text=m['label'],
                                         hovertemplate='%{text}<extra></extra>',
                                         marker=dict(symbol=symbol, size=11, color=colour,
                                                     line=dict(width=1, color='white'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['time'], y=d['rsi14'], name='RSI 14', showlegend=False,
                             line=dict(color=col['blue'], width=2)), row=2, col=1)
    for level in (30, 70):
        fig.add_hline(y=level, line_dash='dot', line_color=col['muted'], line_width=1, row=2, col=1)
    fig.update_layout(title=title, height=620, xaxis_rangeslider_visible=False, hovermode='x unified',
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation='h', y=1.02, x=0))
    fig.update_xaxes(gridcolor=col['grid'])
    fig.update_yaxes(gridcolor=col['grid'])
    fig.update_yaxes(title_text='RSI 14', range=[0, 100], row=2, col=1)
    return fig


def equity_chart(x, y, y_title, title):
    col = chart_colors()
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', name=y_title, line=dict(color=col['blue'], width=2),
                               hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'))
    fig.add_hline(y=0, line_color=col['muted'], line_width=1)
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    fig.update_xaxes(gridcolor=col['grid'])
    fig.update_yaxes(gridcolor=col['grid'], title_text=y_title)
    return fig


def recent_signal_markers(d, keys, htf, use_htf, lookback=200):
    A = to_arrays(d)
    rows = []
    for i in range(max(25, len(d) - lookback), len(d)):
        for key, sig in evaluate_signals(A, i, keys, htf[i], use_htf)[0]:
            rows.append({'time': d['time'].iloc[i], 'direction': sig.direction,
                         'price': A['low'][i] if sig.direction == 1 else A['high'][i],
                         'label': f"{STRATEGIES[key].name}: {sig.reason}"})
    return pd.DataFrame(rows, columns=['time', 'direction', 'price', 'label'])


# ============================================================================
# UI
# ============================================================================
def sidebar(running, connected, account):
    """Builds Settings from the sidebar. Everything is locked while the engine runs."""
    saved = st.session_state.setdefault('settings', Settings.load())
    lock = running
    with st.sidebar:
        st.header("Account")
        if not MT5_AVAILABLE:
            st.warning("MetaTrader5 package not loaded - CSV backtests only.")
        elif connected:
            label, is_real = account_mode(account)
            st.success(f"#{account.login} · {label}{' · REAL MONEY' if is_real else ''}")
            st.caption(f"Balance {account.balance:,.2f} {account.currency} · Equity {account.equity:,.2f}")
        else:
            st.error("MT5 not connected. Start the terminal and log in, then refresh.")

        st.header("Market")
        options = sorted(set(DEFAULT_SYMBOLS) | set(saved.symbols))
        symbols = st.multiselect("Symbols", options, default=saved.symbols, disabled=lock)
        extra = st.text_input("Extra symbols (comma separated)", "", disabled=lock,
                              help="Plain names are fine - broker suffixes like EURUSDm are resolved automatically.")
        symbols = symbols + [x.strip().upper() for x in extra.split(',') if x.strip()]
        tfs = list(TF_SECONDS)
        timeframe = st.selectbox("Signal timeframe", tfs, index=tfs.index(saved.timeframe), disabled=lock)
        use_htf = st.checkbox("Trade with the higher-timeframe trend", saved.use_htf_filter, disabled=lock,
                              help="Trend and reversal strategies only fire in the direction of the higher "
                                   "timeframe (EMA 50/200). Range strategies are exempt.")
        htf = st.selectbox("Higher timeframe", tfs, index=tfs.index(saved.htf), disabled=lock or not use_htf)

        st.header("Strategies")
        strategies = st.multiselect("Active rule sets", list(STRATEGIES), default=saved.strategies, disabled=lock,
                                    format_func=lambda k: STRATEGIES[k].name)
        min_votes = st.slider("Strategies that must agree", 1, 4, saved.min_votes, disabled=lock,
                              help="1 = any single strategy can open a trade. Higher = confluence only.")

        with st.expander("Risk limits", expanded=True):
            risk_pct = st.slider("Risk per trade (% of equity)", 0.1, 2.0, saved.risk_pct, 0.05, disabled=lock)
            max_risk_pct = st.slider("Hard cap if the minimum lot risks more (%)", 0.1, 5.0, saved.max_risk_pct, 0.1,
                                     disabled=lock, help="Small accounts: when the risk-sized lot is below the broker "
                                                         "minimum, the minimum lot is used only if it stays under this.")
            daily_loss_pct = st.slider("Daily loss limit (% of day-start balance)", 0.5, 10.0, saved.daily_loss_pct, 0.5, disabled=lock)
            max_daily_trades = st.number_input("Max trades per day", 1, 200, saved.max_daily_trades, disabled=lock)
            max_consec_losses = st.number_input("Stop for the day after N losses in a row (0 = off)", 0, 20, saved.max_consec_losses, disabled=lock)
            max_positions = st.number_input("Max open positions", 1, 20, saved.max_positions, disabled=lock)
            max_per_symbol = st.number_input("Max per symbol", 1, 5, saved.max_per_symbol, disabled=lock)
            max_per_currency = st.number_input("Max sharing a currency", 1, 10, saved.max_per_currency, disabled=lock)
            cooldown_bars = st.number_input("Cooldown after a loss (bars)", 0, 50, saved.cooldown_bars, disabled=lock)

        with st.expander("Stops, targets and exits"):
            rr = st.slider("Target (R multiple of the stop)", 0.5, 5.0, saved.rr, 0.1, disabled=lock)
            use_target = st.checkbox("Use the strategy's own target when it has one", saved.use_strategy_target, disabled=lock,
                                     help="Mid-band for the Bollinger fade, daily pivot for the pivot bounce.")
            min_rr = st.slider("Minimum R for a strategy target", 0.3, 3.0, saved.min_rr, 0.1, disabled=lock)
            min_sl_atr = st.slider("Minimum stop distance (ATR)", 0.2, 2.0, saved.min_sl_atr, 0.1, disabled=lock)
            max_sl_atr = st.slider("Maximum stop distance (ATR)", 1.0, 10.0, saved.max_sl_atr, 0.5, disabled=lock)
            be_at_r = st.slider("Move stop to break-even at (R, 0 = off)", 0.0, 3.0, saved.be_at_r, 0.1, disabled=lock)
            trail_atr = st.slider("ATR trailing stop (multiple, 0 = off)", 0.0, 5.0, saved.trail_atr, 0.5, disabled=lock)
            trail_start_r = st.slider("Start trailing at (R)", 0.0, 3.0, saved.trail_start_r, 0.1, disabled=lock)
            max_bars_held = st.number_input("Time stop (bars, 0 = off)", 0, 500, saved.max_bars_held, disabled=lock)
            exit_on_opposite = st.checkbox("Close on the opposite signal from the same strategy", saved.exit_on_opposite, disabled=lock)
            quick_profit = st.number_input("Quick-profit exit (account currency, 0 = off)", 0.0, 10000.0, saved.quick_profit, 0.1, disabled=lock,
                                           help="The older bots' 'close at any small profit' rule. Off by default: taking "
                                                "cents while risking a full stop needs a very high win rate to break even.")

        with st.expander("Filters"):
            max_spread_frac = st.slider("Max spread (fraction of stop distance)", 0.02, 0.5, saved.max_spread_frac, 0.01, disabled=lock)
            session_start, session_end = st.slider("Trading hours (broker server time)", 0, 24,
                                                   (saved.session_start, saved.session_end), disabled=lock)
            friday = st.checkbox("Flatten and stop entering on Friday evening", saved.friday_close_hour >= 0, disabled=lock)
            friday_hour = st.slider("Friday cut-off hour (server)", 12, 23, max(saved.friday_close_hour, 20), disabled=lock or not friday)

    s = Settings(
        symbols=symbols, timeframe=timeframe, htf=htf, bars=saved.bars, strategies=strategies, min_votes=min_votes,
        use_htf_filter=use_htf, mode=saved.mode, risk_pct=risk_pct, max_risk_pct=max(max_risk_pct, risk_pct),
        max_positions=int(max_positions), max_per_symbol=int(max_per_symbol), max_per_currency=int(max_per_currency),
        daily_loss_pct=daily_loss_pct, max_daily_trades=int(max_daily_trades), max_consec_losses=int(max_consec_losses),
        cooldown_bars=int(cooldown_bars), rr=rr, min_rr=min_rr, use_strategy_target=use_target, min_sl_atr=min_sl_atr,
        max_sl_atr=max_sl_atr, max_spread_frac=max_spread_frac, be_at_r=be_at_r, trail_atr=trail_atr,
        trail_start_r=trail_start_r, max_bars_held=int(max_bars_held), quick_profit=quick_profit,
        exit_on_opposite=exit_on_opposite, session_start=session_start, session_end=session_end,
        friday_close_hour=friday_hour if friday else -1, poll_seconds=saved.poll_seconds, deviation=saved.deviation,
    )
    return s


def controls(runtime, s, connected, account):
    engine = runtime['engine']
    running = engine is not None and engine.is_alive()

    if not running:
        left, right = st.columns([2, 3])
        with left:
            mode_label = st.radio("Mode", ["Paper - simulated fills, no orders sent", "Live - real orders on the connected account"],
                                  disabled=not connected)
        live = mode_label.startswith("Live")
        armed = True
        with right:
            if live and connected:
                label, is_real = account_mode(account)
                if is_real:
                    st.error(f"Account #{account.login} is a {label} account. Orders will use real money.")
                    armed = st.text_input("Type REAL to arm") == "REAL"
                else:
                    armed = st.checkbox(f"Arm live orders on {label} account #{account.login}")
            elif not live:
                st.info("Paper mode uses live MT5 prices and enforces stops and targets, but never sends an order.")
        problems = []
        if not connected:
            problems.append("MT5 is not connected")
        if not s.symbols:
            problems.append("pick at least one symbol")
        if not s.strategies:
            problems.append("pick at least one strategy")
        if live and not armed:
            problems.append("arm live trading")
        if st.button("Start bot", type="primary", disabled=bool(problems), width="stretch"):
            s.mode = 'live' if live else 'paper'
            s.save()
            st.session_state['settings'] = s
            runtime['engine'] = Engine(s)
            runtime['engine'].start()
            st.rerun()
        if problems:
            st.caption("To start: " + "; ".join(problems) + ".")
        if connected and st.button("Close every position this bot owns at the broker", width="stretch"):
            for line in close_all_bot_positions():
                st.write(line)
    else:
        c1, c2, c3 = st.columns(3)
        if c1.button("Stop (leave positions with their SL/TP)", width="stretch"):
            engine.stop(flatten=False)
            engine.thread.join(timeout=15)
            st.rerun()
        if c2.button("Stop and close all bot positions", width="stretch"):
            engine.stop(flatten=True)
            engine.thread.join(timeout=30)
            st.rerun()
        if c3.button("Close all bot positions, keep running", width="stretch"):
            engine.flatten_req = True
            st.toast("Flatten requested - it runs on the next cycle.")


@st.fragment(run_every=2)
def live_panel(runtime):
    engine = runtime['engine']
    if engine is None:
        st.info("The bot is not running. Choose settings in the sidebar, then start it in paper mode first.")
        return
    snap = engine.snapshot()
    status = snap['status'].upper()
    stale = snap['last_cycle'] and time.time() - snap['last_cycle'] > 30 and status == 'RUNNING'
    line = f"**{status}** · {snap['mode']} mode"
    if snap['server_time']:
        line += f" · server time {snap['server_time']:%Y-%m-%d %H:%M:%S}"
    st.markdown(line)
    if status == 'ERROR':
        st.error(snap['last_error'] or "The engine thread has died. Check bot_data/bot.log, then start it again.")
    elif snap['last_error']:
        st.warning(snap['last_error'])
    if stale:
        st.warning("No completed cycle for over 30 seconds.")
    if snap['halt']:
        st.error(f"New entries halted for today: {snap['halt']}")

    m = st.columns(5)
    m[0].metric("Realised today", f"{snap['realised']:.2f}")
    m[1].metric("Floating", f"{snap['floating']:.2f}")
    m[2].metric("Open positions", len(snap['positions']))
    m[3].metric("Opened today", snap['opened_today'])
    m[4].metric("Closed today (wins)", f"{snap['closed_count']} ({snap['wins']})")

    st.subheader("Open positions")
    if snap['positions']:
        st.dataframe(pd.DataFrame(snap['positions']), hide_index=True, width="stretch")
    else:
        st.caption("None.")

    st.subheader("Symbols")
    rows = [{'symbol': sym, 'last closed bar': v.get('bar', ''), 'close': v.get('close'),
             'HTF trend': {1: 'up', -1: 'down', 0: 'flat'}.get(v.get('htf_trend'), ''),
             'signals on that bar': ', '.join(v.get('signals', [])) or v.get('note', '-'),
             'dropped by HTF filter': len(v.get('filtered', []))} for sym, v in snap['symbol_state'].items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Decision feed")
    if snap['decisions']:
        st.dataframe(pd.DataFrame(snap['decisions']), hide_index=True, width="stretch", height=320)
    else:
        st.caption("Nothing yet - decisions appear when a candle closes.")


def tab_scanner(s, connected):
    st.caption("What every active strategy says about the last closed candle, right now. Places no trade.")
    if not connected:
        st.info("Connect MT5 to scan.")
        return
    if st.button("Scan now", width="stretch"):
        results = {}
        for name in s.symbols:
            sym = resolve_symbol(name)
            df = fetch_bars(sym, s.timeframe, s.bars) if sym else None
            if df is None or len(df) < 60:
                continue
            d = add_indicators(df)
            htf_df = fetch_bars(sym, s.htf, 300) if s.use_htf_filter else None
            htf = map_htf_to_ltf(d['time'], s.timeframe, htf_df, s.htf) if s.use_htf_filter else np.zeros(len(d), dtype=int)
            signals, filtered = evaluate_signals(to_arrays(d), len(d) - 1, s.strategies, htf[-1], s.use_htf_filter)
            results[sym] = {'d': d, 'htf': htf, 'signals': signals, 'filtered': filtered}
        st.session_state['scan'] = results
    scan = st.session_state.get('scan')
    if not scan:
        return
    table = [{'symbol': sym, 'last closed bar': str(r['d']['time'].iloc[-1]), 'close': r['d']['close'].iloc[-1],
              'HTF trend': {1: 'up', -1: 'down', 0: 'flat'}[int(r['htf'][-1])],
              'signals': '; '.join(f"{'BUY' if sg.direction == 1 else 'SELL'} - {STRATEGIES[k].name}: {sg.reason}"
                                   for k, sg in r['signals']) or '-',
              'dropped by HTF filter': '; '.join(r['filtered']) or '-'} for sym, r in scan.items()]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    sym = st.selectbox("Chart", list(scan))
    r = scan[sym]
    view = r['d'].tail(250)
    markers = recent_signal_markers(r['d'], s.strategies, r['htf'], s.use_htf_filter)
    st.plotly_chart(price_chart(view, markers[markers['time'] >= view['time'].iloc[0]],
                                f"{sym} {s.timeframe} - signals from the active strategies"), width="stretch")


def tab_backtest(s, connected):
    st.caption("Runs the same strategy code as the live engine over history. Results are in R (multiples of the "
               "amount risked per trade), after spread. No commission or swap; if a stop and a target fall in "
               "the same bar the stop is assumed to be hit first.")
    df = htf_df = None
    spread = 0.0
    label = ""
    if connected:
        c1, c2 = st.columns(2)
        name = c1.selectbox("Symbol", s.symbols or DEFAULT_SYMBOLS)
        count = c2.select_slider("Bars of history", [1000, 2000, 5000, 10000, 20000], 5000)
        source_ready = True
    else:
        source_ready = False
        st.info("MT5 is not connected, so upload bar data instead: a CSV with time, open, high, low, close "
                "(and optionally tick_volume). The higher-timeframe filter is skipped for CSV data.")
        upload = st.file_uploader("Bar data (CSV)", type="csv")
        spread = st.number_input("Spread in price units (e.g. 0.00012)", 0.0, 100.0, 0.0, format="%.5f")
        if upload is not None:
            df = pd.read_csv(upload)
            df.columns = [c.strip().lower() for c in df.columns]
            missing = {'time', 'open', 'high', 'low', 'close'} - set(df.columns)
            if missing:
                st.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
                df = None
            else:
                df['time'] = pd.to_datetime(df['time'])
                label = upload.name
                source_ready = True
    if not s.strategies:
        st.warning("Pick at least one strategy in the sidebar.")
        return
    if st.button("Run backtest", type="primary", disabled=not source_ready, width="stretch"):
        with st.spinner("Backtesting..."):
            run_settings = s
            if connected:
                sym = resolve_symbol(name)
                df = fetch_bars(sym, s.timeframe, count) if sym else None
                if df is None or len(df) < 300:
                    st.error("Not enough history for this symbol/timeframe.")
                    return
                info = mt('symbol_info', sym)
                spread = df['spread'].to_numpy() * info.point
                span = int(len(df) * TF_SECONDS[s.timeframe] / TF_SECONDS[s.htf]) + 300
                htf_df = fetch_bars(sym, s.htf, min(span, 20000)) if s.use_htf_filter else None
                label = f"{sym} {s.timeframe}"
            else:
                run_settings = Settings(**{**asdict(s), 'use_htf_filter': False})
            trades, table, _ = run_backtest(df, run_settings, htf_df, spread)
            st.session_state['bt'] = {'trades': trades, 'table': table, 'label': label,
                                      'span': f"{df['time'].iloc[0]:%Y-%m-%d} to {df['time'].iloc[-1]:%Y-%m-%d}"}
    bt = st.session_state.get('bt')
    if not bt:
        return
    trades, stats = bt['trades'], summarize(bt['trades']['r'])
    st.subheader(f"{bt['label']} · {bt['span']}")
    m = st.columns(6)
    m[0].metric("Trades", stats['trades'])
    m[1].metric("Win rate", f"{stats['win_rate']:.1f}%")
    m[2].metric("Profit factor", "n/a" if not np.isfinite(stats['profit_factor']) else f"{stats['profit_factor']:.2f}")
    m[3].metric("Expectancy", f"{stats['expectancy']:+.2f} R")
    m[4].metric("Total", f"{stats['total']:+.1f} R")
    m[5].metric("Max drawdown", f"{stats['max_dd']:.1f} R")
    if stats['trades'] < 30:
        st.warning("Fewer than 30 trades - too few to conclude anything. Use more history.")
    if stats['trades']:
        st.plotly_chart(equity_chart(trades['exit_time'], trades['r'].cumsum(), "Cumulative R",
                                     "Equity curve, all active strategies together (one position at a time)"), width="stretch")
    st.markdown("**Each strategy on its own**")
    st.dataframe(bt['table'], hide_index=True, width="stretch", column_config={
        'win_rate': st.column_config.NumberColumn("win rate %", format="%.1f"),
        'profit_factor': st.column_config.NumberColumn("profit factor", format="%.2f"),
        'expectancy': st.column_config.NumberColumn("expectancy (R)", format="%+.2f"),
        'total': st.column_config.NumberColumn("total (R)", format="%+.1f"),
        'max_dd': st.column_config.NumberColumn("max drawdown (R)", format="%.1f"),
    })
    with st.expander("Trades"):
        st.dataframe(trades, hide_index=True, width="stretch")


def tab_journal():
    trades = load_trades()
    if trades.empty:
        st.info("No closed trades yet. Every trade the bot closes is written to bot_data/trades.csv.")
        return
    mode = st.radio("Show", ["paper", "live"], horizontal=True)
    t = trades[trades['mode'] == mode].copy()
    if t.empty:
        st.caption(f"No {mode} trades yet.")
        return
    stats = summarize(t['r_multiple'])
    m = st.columns(5)
    m[0].metric("Trades", len(t))
    m[1].metric("Net P&L", f"{t['pnl'].sum():.2f}")
    m[2].metric("Win rate", f"{(t['pnl'] > 0).mean() * 100:.1f}%")
    m[3].metric("Expectancy", f"{stats['expectancy']:+.2f} R")
    m[4].metric("Max drawdown", f"{stats['max_dd']:.1f} R")
    st.plotly_chart(equity_chart(pd.to_datetime(t['close_time']), t['pnl'].cumsum(), "Cumulative P&L",
                                 f"Closed {mode} trades"), width="stretch")
    by = t.groupby('strategy').agg(trades=('pnl', 'size'), net_pnl=('pnl', 'sum'),
                                   win_rate=('pnl', lambda x: (x > 0).mean() * 100),
                                   avg_r=('r_multiple', 'mean')).reset_index()
    st.markdown("**By strategy**")
    st.dataframe(by, hide_index=True, width="stretch")
    st.dataframe(t.iloc[::-1], hide_index=True, width="stretch")
    st.download_button("Download journal (CSV)", trades.to_csv(index=False), "trades.csv", "text/csv")


def tab_library():
    st.caption("Every entry in strategies.json and what this bot does with it.")
    try:
        library = json.loads(STRATEGIES_JSON.read_text(encoding="utf-8"))['strategies']
    except (OSError, ValueError, KeyError):
        st.error("strategies.json not found or unreadable next to app.py.")
        return
    built = {}
    for key, definition in STRATEGIES.items():
        for json_id in definition.json_ids:
            built.setdefault(json_id, []).append(definition.name)
    rows = []
    for item in library:
        if item['id'] in built:
            status, how = "built", "Rule set: " + "; ".join(built[item['id']])
        else:
            status, how = JSON_NOTES.get(item['id'], ("todo", ""))
        rows.append({'id': item['id'], 'strategy': item['name'], 'category': item['category'],
                     'status': status, 'in this bot': how})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=560)
    st.markdown("**Rules of the built strategies**")
    st.dataframe(pd.DataFrame([{'rule set': d.name, 'follows HTF trend': d.kind == 'trend', 'rule': d.rule}
                               for d in STRATEGIES.values()]), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="TradePilot Super Bot", page_icon="🤖", layout="wide")
    runtime = get_runtime()
    engine = runtime['engine']
    running = engine is not None and engine.is_alive()
    connected = mt5_connect()
    account = mt('account_info') if connected else None
    connected = connected and account is not None

    st.title("TradePilot Super Bot")
    st.caption("Rule-based, multi-strategy auto-execution on MetaTrader 5. No AI. Not financial advice - "
               "backtest, then paper, then demo, before any real money.")

    s = sidebar(running, connected, account)
    tabs = st.tabs(["Dashboard", "Scanner", "Backtest", "Journal", "Strategy library"])
    with tabs[0]:
        controls(runtime, s, connected, account)
        st.divider()
        live_panel(runtime)
    with tabs[1]:
        tab_scanner(s, connected)
    with tabs[2]:
        tab_backtest(s, connected)
    with tabs[3]:
        tab_journal()
    with tabs[4]:
        tab_library()


if __name__ == "__main__":
    main()
