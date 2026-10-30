"""
TradePilot Strategy Advisor (app2.py)
=====================================

Reads the live MT5 charts, runs every rule set from app.py (built from
strategies.json) on each newly closed candle, and turns the best current setup
into a trade plan: direction, entry, stop loss, take profit and the reasons.

* One plan at a time: after a plan is given the advisor waits the chosen
  interval (10 minutes by default) before it gives another.
* Every strategy is ranked by a backtest on the same symbol/timeframe; a setup
  only becomes a plan when its strategy has a positive expectancy there.
* Each plan is then followed on live prices until it hits TP or SL, so the
  history shows how the advice actually played out.
* Advice only - this app never sends an order. app.py is the auto-trader.

Run:  streamlit run app2.py   (can run next to app.py: streamlit run app2.py --server.port 8502)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import app` works however this is launched

import numpy as np
import pandas as pd
import streamlit as st

import app as core

PLANS_JSON = core.DATA_DIR / "advisor_plans.json"
ADVISOR_SETTINGS_JSON = core.DATA_DIR / "advisor_settings.json"
ADVISOR_LOG = core.DATA_DIR / "advisor.log"
STATS_REFRESH_SECONDS = 3600      # re-rank the strategies on each symbol once an hour
PLAN_MAX_AGE_HOURS = 48           # an unresolved plan is marked expired after this


log = core.attach_file_log("advisor", ADVISOR_LOG, 1_000_000)


# ============================================================================
# SETTINGS
# ============================================================================
@dataclass
class AdvisorSettings:
    symbols: list = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'])
    timeframe: str = "M15"
    htf: str = "H4"
    strategies: list = field(default_factory=lambda: list(core.STRATEGIES))
    use_htf_filter: bool = True
    min_votes: int = 1
    interval_minutes: int = 10        # wait this long after a plan before giving the next
    history_bars: int = 3000          # bars used to rank each strategy
    min_trades: int = 15              # backtest trades a strategy needs before it is trusted
    min_expectancy: float = 0.0       # R per trade the strategy must have shown on this symbol
    rr: float = 1.5
    min_rr: float = 1.0
    poll_seconds: float = 15.0

    @classmethod
    def load(cls):
        try:
            raw = json.loads(ADVISOR_SETTINGS_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = set(cls.__dataclass_fields__)
        s = cls(**{k: v for k, v in raw.items() if k in known})
        s.strategies = [k for k in s.strategies if k in core.STRATEGIES] or cls().strategies
        return s

    def save(self):
        core.save_json(ADVISOR_SETTINGS_JSON, asdict(self))

    def core_settings(self):
        """The app.py Settings the shared strategy / plan / backtest code expects."""
        return core.Settings(symbols=self.symbols, timeframe=self.timeframe, htf=self.htf, strategies=self.strategies,
                             min_votes=1, use_htf_filter=self.use_htf_filter, rr=self.rr, min_rr=self.min_rr)


# ============================================================================
# ADVISOR ENGINE
# ============================================================================
class Advisor:
    def __init__(self, settings):
        self.s = settings
        self.cs = settings.core_settings()
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.thread = None
        self.status = 'idle'
        self.last_error = None
        self.last_scan = None
        self.symbols = []
        self.last_bar = {}
        self.symbol_state = {}            # sym -> what the chart shows on the last closed bar
        self.stats = {}                   # sym -> {'time': epoch, 'by_key': {key: summarize()}}
        self.candidates = {}              # sym -> [candidate dicts] for its latest bar
        self.issued_keys = set()          # (sym, bar, strategy) already turned into a plan
        self.next_allowed = 0.0
        self.plans = self._load_plans()
        if self.plans:
            last = max(p['issued_epoch'] for p in self.plans)
            self.next_allowed = last + self.s.interval_minutes * 60

    # ---- lifecycle ---------------------------------------------------------
    def start(self):
        self.status = 'running'
        self.thread = threading.Thread(target=self._run, name="advisor", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def _run(self):
        try:
            if not core.mt5_connect():
                raise RuntimeError("MT5 is not connected - start the terminal and log in.")
            for name in self.s.symbols:
                resolved = core.resolve_symbol(name)
                if resolved:
                    self.symbols.append(resolved)
            if not self.symbols:
                raise RuntimeError("None of the selected symbols exist on this broker.")
            log.info("STARTED %s %s", self.s.timeframe, ", ".join(self.symbols))
            while not self.stop_evt.is_set():
                try:
                    self._cycle()
                except Exception as exc:  # noqa: BLE001 - keep advising through transient errors
                    log.exception("cycle failed")
                    self.last_error = f"Cycle error (still running): {exc}"
                self.stop_evt.wait(self.s.poll_seconds)
            self.status = 'stopped'
            log.info("STOPPED")
        except Exception as exc:  # noqa: BLE001
            log.exception("advisor stopped on error")
            self.status = 'error'
            self.last_error = str(exc)

    # ---- persistence -------------------------------------------------------
    @staticmethod
    def _load_plans():
        try:
            return json.loads(PLANS_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _save_plans(self):
        with self.lock:
            core.save_json(PLANS_JSON, self.plans[-500:])

    # ---- cycle -------------------------------------------------------------
    def _cycle(self):
        if not core.mt5_connect():
            self.last_error = "MT5 disconnected - retrying."
            return
        for sym in self.symbols:
            bar_time = core.last_closed_bar_time(sym, self.s.timeframe)
            if bar_time is not None and bar_time != self.last_bar.get(sym):
                self.last_bar[sym] = bar_time
                self._read_chart(sym, bar_time)
        self._track_plans()
        if time.time() >= self.next_allowed:
            self._maybe_issue()
        self.last_scan = time.time()
        if self.last_error and self.last_error.startswith("MT5 disconnected"):
            self.last_error = None

    def _rank_strategies(self, sym, A, htf, spread):
        """Backtest each strategy alone on this symbol so its live signals can be trusted or ignored."""
        by_key = {}
        for key in self.s.strategies:
            trades = core.simulate(A, [key], self.cs, htf, spread)
            by_key[key] = core.summarize(trades['r'])
        self.stats[sym] = {'time': time.time(), 'by_key': by_key}

    def _read_chart(self, sym, bar_time):
        df = core.fetch_bars(sym, self.s.timeframe, self.s.history_bars)
        info = core.mt('symbol_info', sym)
        if df is None or len(df) < 300 or info is None:
            self.symbol_state[sym] = {'note': 'not enough bar data'}
            self.candidates[sym] = []
            return
        d = core.add_indicators(df)
        A = core.to_arrays(d)
        htf_df = core.fetch_bars(sym, self.s.htf, 300 + len(df) * core.TF_SECONDS[self.s.timeframe]
                                 // core.TF_SECONDS[self.s.htf]) if self.s.use_htf_filter else None
        htf = (core.map_htf_to_ltf(d['time'], self.s.timeframe, htf_df, self.s.htf) if self.s.use_htf_filter
               else np.zeros(len(d), dtype=int))
        spread = df['spread'].to_numpy() * info.point
        if time.time() - self.stats.get(sym, {}).get('time', 0) > STATS_REFRESH_SECONDS:
            self._rank_strategies(sym, A, htf, spread)
        by_key = self.stats[sym]['by_key']

        i = len(d) - 1
        signals, filtered = core.evaluate_signals(A, i, self.s.strategies, int(htf[i]), self.s.use_htf_filter)
        atr = float(A['atr'][i])
        candidates, notes = [], []
        for key, sig in signals:
            same = [k for k, sg in signals if sg.direction == sig.direction]
            against = [k for k, sg in signals if sg.direction != sig.direction]
            side = 'BUY' if sig.direction == 1 else 'SELL'
            st_ = by_key.get(key, core.summarize([]))
            if len(same) - len(against) < self.s.min_votes:
                notes.append(f"{key} {side}: not enough agreement ({len(same)} for, {len(against)} against)")
            elif st_['trades'] < self.s.min_trades:
                notes.append(f"{key} {side}: only {st_['trades']} backtest trades on {sym} - not trusted yet")
            elif st_['expectancy'] <= self.s.min_expectancy:
                notes.append(f"{key} {side}: backtest expectancy {st_['expectancy']:+.2f}R on {sym} - not worth it")
            else:
                candidates.append({'key': key, 'sig': sig, 'atr': atr, 'bar': bar_time, 'votes': len(same),
                                   'agree': same, 'stats': st_})
        self.candidates[sym] = candidates
        self.symbol_state[sym] = {
            'bar': str(d['time'].iloc[-1]), 'close': float(A['close'][i]), 'atr': atr, 'htf_trend': int(htf[i]),
            'digits': info.digits,
            'signals': [f"{k}: {'BUY' if sg.direction == 1 else 'SELL'}" for k, sg in signals],
            'filtered': filtered, 'notes': notes, 'ready': [c['key'] for c in candidates],
            'chart': d.tail(200).reset_index(drop=True),
        }

    def _maybe_issue(self):
        pool = []
        for sym, cands in self.candidates.items():
            for c in cands:
                if c['bar'] == self.last_bar.get(sym) and (sym, c['bar'], c['key']) not in self.issued_keys:
                    pool.append((sym, c))
        # Most agreement first, then the strategy with the best track record on that symbol.
        pool.sort(key=lambda x: (x[1]['votes'], x[1]['stats']['expectancy'] * min(1.0, x[1]['stats']['trades'] / 50)),
                  reverse=True)
        for sym, c in pool:
            self.issued_keys.add((sym, c['bar'], c['key']))
            plan = self._plan(sym, c)
            if plan is not None:
                with self.lock:
                    self.plans.append(plan)
                self.next_allowed = plan['issued_epoch'] + self.s.interval_minutes * 60
                self._save_plans()
                log.info("PLAN %s %s %s entry %s SL %s TP %s (%s)", plan['symbol'], plan['direction'],
                         plan['strategy'], plan['entry'], plan['sl'], plan['tp'], plan['reason'])
                return

    def _plan(self, sym, c):
        sig, info, tick = c['sig'], core.mt('symbol_info', sym), core.mt('symbol_info_tick', sym)
        if info is None or tick is None:
            return None
        spread = tick.ask - tick.bid
        entry = tick.ask if sig.direction == 1 else tick.bid
        stop = sig.sl + (spread if sig.direction == -1 else 0.0)
        min_stop = (getattr(info, 'trade_stops_level', 0) or 0) * info.point + spread
        plan, why = core.build_plan(sig.direction, entry, stop, sig.tp, c['atr'], self.cs, min_stop)
        if plan is None:
            self.symbol_state.setdefault(sym, {}).setdefault('notes', []).append(f"{c['key']}: {why}")
            return None
        definition, st_, dg = core.STRATEGIES[c['key']], c['stats'], info.digits
        now = datetime.now()
        risk, reward = float(plan.risk), abs(float(plan.tp) - float(plan.entry))
        if dg in (3, 5):    # forex quotes: show pips; metals/indices/crypto: show the price distance
            pip = info.point * 10
            risk_txt, reward_txt = f"{risk / pip:.1f} pips", f"{reward / pip:.1f} pips"
        else:
            risk_txt, reward_txt = f"{risk:.{dg}f}", f"{reward:.{dg}f}"
        return {
            'id': int(now.timestamp() * 1000), 'issued': now.strftime("%Y-%m-%d %H:%M:%S"),
            'issued_epoch': now.timestamp(), 'valid_until': (now + timedelta(minutes=self.s.interval_minutes)).strftime("%H:%M"),
            'symbol': sym, 'timeframe': self.s.timeframe, 'direction': 'BUY' if sig.direction == 1 else 'SELL',
            'strategy': definition.name, 'strategy_key': c['key'], 'rule': definition.rule, 'reason': sig.reason,
            'agreeing': [core.STRATEGIES[k].name for k in c['agree']], 'votes': c['votes'],
            'htf_trend': {1: 'up', -1: 'down', 0: 'flat'}.get(self.symbol_state.get(sym, {}).get('htf_trend'), '?'),
            'entry': round(float(plan.entry), dg), 'sl': round(float(plan.sl), dg), 'tp': round(float(plan.tp), dg),
            'rr': round(reward / risk, 2), 'risk_txt': risk_txt, 'reward_txt': reward_txt,
            'digits': dg, 'bt_trades': int(st_['trades']), 'bt_win_rate': round(float(st_['win_rate']), 1),
            'bt_expectancy': round(float(st_['expectancy']), 2),
            'bt_profit_factor': round(float(st_['profit_factor']), 2) if np.isfinite(st_['profit_factor']) else None,
            'status': 'open', 'result_r': None, 'closed': None,
        }

    def _track_plans(self):
        """Follows each open plan on live prices (as if entered at market when issued) until TP, SL or expiry."""
        changed = False
        with self.lock:
            open_plans = [p for p in self.plans if p['status'] == 'open']
        for p in open_plans:
            if time.time() - p['issued_epoch'] > PLAN_MAX_AGE_HOURS * 3600:
                p.update(status='expired', closed=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                changed = True
                continue
            tick = core.mt('symbol_info_tick', p['symbol'])
            if tick is None:
                continue
            d = 1 if p['direction'] == 'BUY' else -1
            price = tick.bid if d == 1 else tick.ask
            risk = abs(p['entry'] - p['sl']) or 1e-12
            if (price - p['sl']) * d <= 0:
                p.update(status='SL hit', result_r=-1.0)
            elif (price - p['tp']) * d >= 0:
                p.update(status='TP hit', result_r=round((p['tp'] - p['entry']) * d / risk, 2))
            else:
                p['live_r'] = round((price - p['entry']) * d / risk, 2)
                continue
            p['closed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            log.info("RESULT %s %s %s %s", p['symbol'], p['direction'], p['strategy'], p['status'])
        if changed:
            self._save_plans()

    # ---- UI ----------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return {
                'status': self.status if self.is_alive() or self.status != 'running' else 'error',
                'last_error': self.last_error, 'last_scan': self.last_scan, 'next_allowed': self.next_allowed,
                'plans': [dict(p) for p in self.plans], 'symbol_state': dict(self.symbol_state),
                'stats': {sym: v['by_key'] for sym, v in self.stats.items()},
            }


@st.cache_resource
def get_runtime():
    """One advisor per Streamlit process: keeps its 10-minute clock across page refreshes."""
    return {'advisor': None}


# ============================================================================
# UI
# ============================================================================
def sidebar(running):
    saved = st.session_state.setdefault('advisor_settings', AdvisorSettings.load())
    lock = running
    with st.sidebar:
        st.header("Charts to read")
        options = sorted(set(core.DEFAULT_SYMBOLS) | set(saved.symbols))
        symbols = st.multiselect("Symbols", options, default=saved.symbols, disabled=lock)
        extra = st.text_input("Extra symbols (comma separated)", "", disabled=lock)
        symbols = symbols + [x.strip().upper() for x in extra.split(',') if x.strip()]
        tfs = list(core.TF_SECONDS)
        timeframe = st.selectbox("Chart timeframe", tfs, index=tfs.index(saved.timeframe), disabled=lock)
        use_htf = st.checkbox("Only with the higher-timeframe trend", saved.use_htf_filter, disabled=lock)
        htf = st.selectbox("Higher timeframe", tfs, index=tfs.index(saved.htf), disabled=lock or not use_htf)

        st.header("Strategies")
        strategies = st.multiselect("Rule sets to use", list(core.STRATEGIES), default=saved.strategies, disabled=lock,
                                    format_func=lambda k: core.STRATEGIES[k].name)
        min_votes = st.slider("Strategies that must agree", 1, 4, saved.min_votes, disabled=lock)

        st.header("Pacing and quality")
        interval = st.number_input("Minutes to wait after each strategy", 1, 240, saved.interval_minutes, disabled=lock)
        min_trades = st.number_input("Backtest trades needed to trust a strategy", 0, 200, saved.min_trades, disabled=lock)
        min_exp = st.slider("Minimum backtest expectancy (R per trade)", -0.5, 1.0, saved.min_expectancy, 0.05,
                            disabled=lock, help="A setup is only given when its strategy made money on this "
                                                "symbol/timeframe in the backtest. 0 = break-even or better.")
        history_bars = st.select_slider("Bars used for the backtest", [1000, 2000, 3000, 5000, 10000],
                                        saved.history_bars, disabled=lock)
        rr = st.slider("Target (R multiple of the stop)", 0.5, 5.0, saved.rr, 0.1, disabled=lock)
    return AdvisorSettings(symbols=symbols, timeframe=timeframe, htf=htf, strategies=strategies, use_htf_filter=use_htf,
                           min_votes=min_votes, interval_minutes=int(interval), history_bars=int(history_bars),
                           min_trades=int(min_trades), min_expectancy=float(min_exp), rr=rr, min_rr=saved.min_rr,
                           poll_seconds=saved.poll_seconds)


def controls(runtime, s, connected):
    advisor = runtime['advisor']
    running = advisor is not None and advisor.is_alive()
    if running:
        if st.button("Stop advisor", width="stretch"):
            advisor.stop()
            advisor.thread.join(timeout=20)
            st.rerun()
        return
    problems = [p for p, bad in (("MT5 is not connected", not connected), ("pick at least one symbol", not s.symbols),
                                 ("pick at least one strategy", not s.strategies)) if bad]
    if st.button("Start advisor", type="primary", disabled=bool(problems), width="stretch"):
        s.save()
        st.session_state['advisor_settings'] = s
        runtime['advisor'] = Advisor(s)
        runtime['advisor'].start()
        st.rerun()
    if problems:
        st.caption("To start: " + "; ".join(problems) + ".")


def plan_card(p):
    side_icon = "🟢" if p['direction'] == 'BUY' else "🔴"
    dg = p.get('digits', 5)
    with st.container(border=True):
        st.markdown(f"### {side_icon} {p['direction']} {p['symbol']} ({p['timeframe']}) · {p['strategy']}")
        st.caption(f"Given {p['issued']} · next strategy not before {p['valid_until']} · status: **{p['status']}**"
                   + (f" ({p['live_r']:+.2f}R now)" if p['status'] == 'open' and p.get('live_r') is not None else ""))
        m = st.columns(4)
        m[0].metric("Entry (market)", f"{p['entry']:.{dg}f}")
        m[1].metric("Stop loss", f"{p['sl']:.{dg}f}", f"-{p['risk_txt']}", delta_color="off")
        m[2].metric("Take profit", f"{p['tp']:.{dg}f}", f"+{p['reward_txt']}", delta_color="off")
        m[3].metric("Reward : risk", f"1 : {p['rr']:.2f}")
        st.markdown(f"**Why now:** {p['reason']}")
        st.markdown(f"**Rule:** {p['rule']}")
        st.markdown(f"**Agreeing strategies ({p['votes']}):** {', '.join(p['agreeing'])} · HTF trend: {p['htf_trend']}")
        pf = p['bt_profit_factor']
        st.markdown(f"**Track record on {p['symbol']} {p['timeframe']} (backtest):** {p['bt_trades']} trades, "
                    f"{p['bt_win_rate']}% wins, expectancy {p['bt_expectancy']:+.2f}R"
                    + (f", profit factor {pf:.2f}" if pf is not None else ""))


@st.fragment(run_every=2)
def live_panel(runtime):
    advisor = runtime['advisor']
    if advisor is None:
        st.info("Choose the charts and strategies in the sidebar, then start the advisor.")
        return
    snap = advisor.snapshot()
    status = snap['status'].upper()
    wait = snap['next_allowed'] - time.time()
    line = f"**{status}**"
    if snap['last_scan']:
        line += f" · last chart read {datetime.fromtimestamp(snap['last_scan']):%H:%M:%S}"
    st.markdown(line)
    if snap['last_error']:
        (st.error if status == 'ERROR' else st.warning)(snap['last_error'])
    if status == 'RUNNING':
        if wait > 0:
            st.info(f"⏳ Strategy given - next one in **{int(wait // 60):02d}:{int(wait % 60):02d}**")
        else:
            st.info("🔎 Reading the charts - a strategy is given as soon as a qualifying setup appears on a closed candle.")

    plans = snap['plans']
    seen = st.session_state.get('seen_plan')
    if plans and seen is not None and plans[-1]['id'] != seen:
        st.toast(f"New strategy: {plans[-1]['direction']} {plans[-1]['symbol']}", icon="📈")
    st.session_state['seen_plan'] = plans[-1]['id'] if plans else 0

    st.subheader("Latest strategy")
    if plans:
        plan_card(plans[-1])
    else:
        st.caption("None yet.")

    st.subheader("What the charts show now")
    rows = [{'symbol': sym, 'last closed bar': v.get('bar', ''), 'close': v.get('close'),
             'HTF trend': {1: 'up', -1: 'down', 0: 'flat'}.get(v.get('htf_trend'), ''),
             'signals': ', '.join(v.get('signals', [])) or v.get('note', '-'),
             'qualifying': ', '.join(v.get('ready', [])) or '-',
             'not taken': '; '.join(v.get('notes', []) + v.get('filtered', [])) or '-'}
            for sym, v in snap['symbol_state'].items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Strategy history")
    if plans:
        h = pd.DataFrame(plans[::-1])
        h['result_r'] = pd.to_numeric(h['result_r'], errors='coerce')
        done = h[h['result_r'].notna()]
        m = st.columns(4)
        m[0].metric("Strategies given", len(h))
        m[1].metric("Resolved", len(done))
        m[2].metric("TP hit rate", f"{(done['result_r'] > 0).mean() * 100:.0f}%" if len(done) else "-")
        m[3].metric("Net result", f"{done['result_r'].sum():+.2f} R" if len(done) else "-")
        st.dataframe(h[['issued', 'symbol', 'direction', 'strategy', 'entry', 'sl', 'tp', 'rr', 'status', 'result_r',
                        'reason']], hide_index=True, width="stretch")


@st.fragment(run_every=30)
def chart_panel(runtime):
    advisor = runtime['advisor']
    if advisor is None:
        return
    snap = advisor.snapshot()
    states = {k: v for k, v in snap['symbol_state'].items() if 'chart' in v}
    if not states:
        return
    plans = snap['plans']
    default = plans[-1]['symbol'] if plans and plans[-1]['symbol'] in states else next(iter(states))
    sym = st.selectbox("Chart", list(states), index=list(states).index(default))
    d = states[sym]['chart']
    fig = core.price_chart(d, pd.DataFrame(columns=['time', 'price', 'direction', 'label']),
                           f"{sym} {advisor.s.timeframe}")
    plan = next((p for p in reversed(plans) if p['symbol'] == sym and p['status'] == 'open'), None)
    if plan:
        col = core.chart_colors()
        for level, name, colour in ((plan['entry'], 'Entry', col['blue']), (plan['sl'], 'SL', col['down']),
                                    (plan['tp'], 'TP', col['up'])):
            fig.add_hline(y=level, line_color=colour, line_dash='dash', line_width=1.5, row=1, col=1,
                          annotation_text=f"{name} {level}", annotation_position="right")
    st.plotly_chart(fig, width="stretch")


def tab_ranking(runtime):
    advisor = runtime['advisor']
    if advisor is None:
        st.caption("Start the advisor to rank the strategies.")
        return
    stats = advisor.snapshot()['stats']
    if not stats:
        st.caption("Ranking runs on the first chart read for each symbol.")
        return
    rows = [{'symbol': sym, 'strategy': core.STRATEGIES[k].name, **v} for sym, by_key in stats.items()
            for k, v in by_key.items()]
    table = pd.DataFrame(rows).sort_values(['symbol', 'expectancy'], ascending=[True, False])
    st.caption("Backtest of each strategy alone on each symbol (in R, after spread). Only strategies above the "
               "sidebar's minimum trades and expectancy are used for live strategies. Re-ranked every hour.")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        'win_rate': st.column_config.NumberColumn("win rate %", format="%.1f"),
        'profit_factor': st.column_config.NumberColumn("profit factor", format="%.2f"),
        'expectancy': st.column_config.NumberColumn("expectancy (R)", format="%+.2f"),
        'total': st.column_config.NumberColumn("total (R)", format="%+.1f"),
        'max_dd': st.column_config.NumberColumn("max drawdown (R)", format="%.1f"),
    })


def main():
    st.set_page_config(page_title="TradePilot Strategy Advisor", page_icon="🧭", layout="wide")
    runtime = get_runtime()
    advisor = runtime['advisor']
    running = advisor is not None and advisor.is_alive()
    connected = core.mt5_connect() and core.mt('account_info') is not None

    st.title("TradePilot Strategy Advisor")
    st.caption("Reads the live MT5 charts with the strategies from strategies.json and gives one trade strategy at a "
               "time, then waits before giving the next. Advice only - no orders are sent. Not financial advice.")
    s = sidebar(running)
    controls(runtime, s, connected)
    tabs = st.tabs(["Strategies", "Chart", "Strategy ranking"])
    with tabs[0]:
        live_panel(runtime)
    with tabs[1]:
        chart_panel(runtime)
    with tabs[2]:
        tab_ranking(runtime)


if __name__ == "__main__":
    main()
