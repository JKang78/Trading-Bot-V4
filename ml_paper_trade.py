"""
LIVE PAPER-TRADING RUNNER for the longer-horizon ML V3 strategy.

Why a separate runner (not the main bot's DRY_RUN)?
---------------------------------------------------
The main bot runs once per cron cycle and reads REAL open positions from Kraken.
Its DRY_RUN mode just skips real orders - it does NOT remember simulated
positions between runs, so it cannot actually track paper P&L over days.

This runner keeps its own state file (ml_paper_state.json) so it can hold a
"paper" position for ~3 days across many independent cron runs, then close it
and record the result - exactly like the backtest, but on live data going
forward. It NEVER places a real order. It only simulates and notifies.

How to use
----------
Run it on a schedule (e.g. hourly). Each run it will:
  1. Close any paper positions whose ~2-day hold is up (record P&L).
  2. Open new paper positions when the cost-aware ML model is highly confident.
  3. Save state and send a Telegram summary of what happened.

    ./venv/bin/python ml_paper_trade.py
"""

import os
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from kraken_bot_v4_advanced import Config, Telegram
from backtest import get_history
from ml_strategy import (
    KrakenCostModel,
    MLSwingStrategy,
    btc_regime_state,
    compute_btc_regime_frame,
)


# ─────────────────────────── Settings (env-overridable) ───────────────────────────
STATE_FILE = os.getenv('ML_PAPER_STATE_FILE', 'ml_paper_state.json')
SYMBOLS = [s.strip() for s in os.getenv(
    'ML_PAPER_SYMBOLS', 'XRP-USD,ADA-USD,SOL-USD,BTC-USD,ETH-USD').split(',') if s.strip()]
PERIOD = os.getenv('ML_PAPER_PERIOD', '720d')     # history for training
INTERVAL = os.getenv('ML_PAPER_INTERVAL', '1h')
HORIZON = int(os.getenv('ML_PAPER_HORIZON', '72'))
BUY_THR = float(os.getenv('ML_PAPER_BUY_THR', '0.70'))
SELL_THR = float(os.getenv('ML_PAPER_SELL_THR', '0.35'))
EXIT_THR = float(os.getenv('ML_PAPER_EXIT_THR', '0.40'))
USE_FNG_FEATURES = os.getenv('ML_PAPER_FNG_FEATURES', 'true').lower() == 'true'
USE_FNG_FILTER = os.getenv('ML_PAPER_FNG_FILTER', 'true').lower() == 'true'
LEVERAGE = float(os.getenv('ML_PAPER_LEVERAGE', '2'))
FEE_PER_SIDE = float(os.getenv('ML_PAPER_FEE', '0.001'))  # maker fee per side
START_EQUITY = float(os.getenv('ML_PAPER_START_EQUITY', '1000'))
POSITION_FRACTION = float(os.getenv('ML_PAPER_POSITION_FRACTION', '0.2'))  # margin per trade
MAX_OPEN = int(os.getenv('ML_PAPER_MAX_OPEN', '3'))
MAKER_ENTRY_FEE = float(os.getenv('ML_PAPER_MAKER_ENTRY_FEE', '0.0023'))
TAKER_ENTRY_FEE = float(os.getenv('ML_PAPER_TAKER_ENTRY_FEE', '0.0040'))
TAKER_EXIT_FEE = float(os.getenv('ML_PAPER_TAKER_EXIT_FEE', '0.0040'))
MARGIN_OPEN_FEE = float(os.getenv('ML_PAPER_MARGIN_OPEN_FEE', '0.0004'))
ROLLOVER_FEE_4H = float(os.getenv('ML_PAPER_ROLLOVER_FEE_4H', '0.0004'))
SPREAD_BUFFER = float(os.getenv('ML_PAPER_SPREAD_BUFFER', '0.0005'))
SLIPPAGE_BUFFER = float(os.getenv('ML_PAPER_SLIPPAGE_BUFFER', '0.0010'))
MINIMUM_EDGE = float(os.getenv('ML_PAPER_MINIMUM_EDGE', '0.0075'))
EV_COST_MULTIPLIER = float(os.getenv('ML_PAPER_EV_COST_MULTIPLIER', '1.5'))
USE_BTC_FEATURES = os.getenv('ML_PAPER_BTC_FEATURES', 'false').lower() == 'true'
USE_BTC_REGIME_FILTER = os.getenv('ML_PAPER_BTC_REGIME_FILTER', 'false').lower() == 'true'
USE_RELATIVE_STRENGTH_FILTER = os.getenv('ML_PAPER_RELATIVE_STRENGTH_FILTER', 'false').lower() == 'true'
USE_EXPECTED_VALUE_FILTER = os.getenv('ML_PAPER_EXPECTED_VALUE_FILTER', 'false').lower() == 'true'


def confidence_size_multiplier(probability: float, threshold: float) -> float:
    if probability < threshold + 0.03:
        return 0.50
    if probability < threshold + 0.07:
        return 0.75
    return 1.00


def load_state() -> dict:
    """Read the saved paper account, or start a fresh one."""
    path = Path(STATE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {'equity': START_EQUITY, 'start_equity': START_EQUITY,
            'open': {}, 'closed': []}


def save_state(state: dict) -> None:
    """Write the paper account back to disk."""
    state['updated_at'] = pd.Timestamp.utcnow().isoformat()
    Path(STATE_FILE).write_text(json.dumps(state, indent=2, default=str))


def net_pnl_pct(direction: str, entry: float, exit_price: float,
                bars_held: int, cost_model: KrakenCostModel) -> float:
    """Profit % on the margin, after fees (includes leverage on both sides)."""
    if direction == 'long':
        gross = (exit_price - entry) / entry * 100 * LEVERAGE
    else:
        gross = (entry - exit_price) / entry * 100 * LEVERAGE
    total_cost = cost_model.estimated_total_cost(bars_held, 'maker') * 100 * LEVERAGE
    return gross - total_cost


def main() -> None:
    config = Config()
    telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    cost_model = KrakenCostModel(
        maker_entry_fee=MAKER_ENTRY_FEE,
        taker_entry_fee=TAKER_ENTRY_FEE,
        taker_exit_fee=TAKER_EXIT_FEE,
        margin_open_fee=MARGIN_OPEN_FEE,
        margin_rollover_fee_4h=ROLLOVER_FEE_4H,
        spread_buffer=SPREAD_BUFFER,
        slippage_buffer=SLIPPAGE_BUFFER,
        minimum_edge=MINIMUM_EDGE,
    )
    strategy = MLSwingStrategy(
        horizon=HORIZON, buy_thr=BUY_THR, sell_thr=SELL_THR, exit_thr=EXIT_THR,
        use_fng_features=USE_FNG_FEATURES, use_fng_filter=USE_FNG_FILTER,
        long_only=(SELL_THR <= 0), cost_model=cost_model,
        use_btc_features=USE_BTC_FEATURES,
        use_btc_regime_filter=USE_BTC_REGIME_FILTER,
        use_relative_strength_filter=USE_RELATIVE_STRENGTH_FILTER,
        use_expected_value_filter=USE_EXPECTED_VALUE_FILTER,
        ev_cost_multiplier=EV_COST_MULTIPLIER,
    )

    state = load_state()
    actions = []  # human-readable lines describing what happened this run

    # Download fresh data once per coin.
    data_by_symbol = {}
    for symbol in SYMBOLS:
        try:
            df = get_history(symbol, PERIOD, INTERVAL)
            if not df.empty:
                data_by_symbol[symbol] = df
        except Exception as e:
            print(f"  ⚠️ {symbol}: data error {e}")
    btc_data = None
    btc_regimes = None
    if USE_BTC_FEATURES or USE_BTC_REGIME_FILTER or USE_RELATIVE_STRENGTH_FILTER:
        try:
            btc_data = get_history('BTC-USD', PERIOD, INTERVAL)
            btc_regimes = compute_btc_regime_frame(btc_data)
        except Exception as e:
            print(f"  ⚠️ BTC regime data error {e}")
            btc_data = None
            btc_regimes = None

    # ── 1) Close paper positions when hold is up OR model says bail ──
    for symbol in list(state['open'].keys()):
        pos = state['open'][symbol]
        df = data_by_symbol.get(symbol)
        if df is None:
            continue
        now = df.index[-1]
        exit_due = pd.Timestamp(pos['exit_due'])
        time_due = now >= exit_due
        early_exit = False
        exit_prob = pos.get('prob_up', 0.5)
        if not time_due and pos['direction'] == 'long' and EXIT_THR > 0:
            early_exit, exit_prob = strategy.should_exit_early(df, btc_data)
        btc_exit = False
        if USE_BTC_REGIME_FILTER and btc_regimes is not None:
            btc_exit = btc_regime_state(btc_regimes, now).block_new_entries
            if btc_exit:
                early_exit = True
        if not time_due and not early_exit:
            continue

        current_price = float(df['Close'].iloc[-1])
        bars_held = max(1, int((now - pd.Timestamp(pos['entry_time'])) / pd.Timedelta(hours=1)))
        pnl_pct = net_pnl_pct(pos['direction'], pos['entry_price'], current_price, bars_held, cost_model)
        pnl_usd = pos['margin_usd'] * pnl_pct / 100
        state['equity'] += pnl_usd

        closed = {**pos, 'symbol': symbol, 'exit_price': current_price,
                  'exit_time': str(now), 'pnl_pct': round(pnl_pct, 3),
                  'pnl_usd': round(pnl_usd, 2),
                  'exit_reason': 'btc_regime' if btc_exit else 'model_exit' if early_exit else 'time'}
        state['closed'].append(closed)
        del state['open'][symbol]
        tag = f" ({closed['exit_reason']})" if early_exit else ""
        actions.append(
            f"CLOSE {symbol} {pos['direction']} @ {current_price:.4f} "
            f"-> {pnl_pct:+.2f}% ({pnl_usd:+.2f}$){tag}")

    # ── 2) Rank new paper positions by EV/risk ──
    latest_ts = max(df.index[-1] for df in data_by_symbol.values()) if data_by_symbol else pd.Timestamp.utcnow()
    btc_state = btc_regime_state(btc_regimes, latest_ts) if btc_regimes is not None else None
    allowed_max_open = min(MAX_OPEN, btc_state.max_positions) if USE_BTC_REGIME_FILTER and btc_state else MAX_OPEN
    open_slots = max(0, allowed_max_open - len(state['open']))
    if USE_BTC_REGIME_FILTER and btc_state and btc_state.block_new_entries:
        actions.append(f"skip all entries: BTC regime weak ({','.join(btc_state.reasons) or 'weak'})")

    candidates = []
    for symbol in SYMBOLS:
        if symbol in state['open']:
            continue  # already holding this coin
        df = data_by_symbol.get(symbol)
        if df is None:
            continue

        sig = strategy.get_signal(df, btc_data)
        if sig.signal not in ('BUY', 'SELL'):
            if sig.blocked_reason:
                actions.append(
                    f"skip {symbol}: {sig.blocked_reason} "
                    f"(p={sig.prob_up:.2f}, ev={sig.expected_value:.2%}, "
                    f"thr={sig.dynamic_threshold:.2f}, rs7={sig.relative_strength_7d:.2%})")
            continue

        candidates.append((symbol, df, sig))

    candidates.sort(key=lambda item: item[2].score, reverse=True)

    for symbol, df, sig in candidates[:open_slots]:
        now = df.index[-1]
        entry_price = float(df['Close'].iloc[-1])
        conf_mult = confidence_size_multiplier(sig.prob_up, sig.dynamic_threshold)
        margin_usd = round(state['equity'] * POSITION_FRACTION * conf_mult * sig.regime_size_multiplier, 2)
        if margin_usd <= 0:
            continue
        exit_due = (now + timedelta(hours=HORIZON))

        state['open'][symbol] = {
            'direction': 'long' if sig.signal == 'BUY' else 'short',
            'entry_price': entry_price,
            'entry_time': str(now),
            'exit_due': str(exit_due),
            'prob_up': round(sig.prob_up, 3),
            'expected_value': round(sig.expected_value, 5),
            'dynamic_threshold': round(sig.dynamic_threshold, 3),
            'estimated_cost': round(sig.estimated_cost, 5),
            'btc_regime': sig.btc_regime,
            'relative_strength_7d': round(sig.relative_strength_7d, 5),
            'margin_usd': margin_usd,
            'leverage': LEVERAGE,
            'model_version': 'v3_cost_aware_ev_btc_rs',
        }
        actions.append(
            f"OPEN {symbol} {'LONG' if sig.signal == 'BUY' else 'SHORT'} "
            f"@ {entry_price:.4f} (p={sig.prob_up:.2f}, thr={sig.dynamic_threshold:.2f}, "
            f"ev={sig.expected_value:.2%}, score={sig.score:.2f}, margin ${margin_usd})")

    save_state(state)

    # ── 3) Report ──
    total_return = (state['equity'] / state['start_equity'] - 1) * 100
    n_closed = len(state['closed'])
    wins = [t for t in state['closed'] if t['pnl_pct'] > 0]
    win_rate = (len(wins) / n_closed * 100) if n_closed else 0.0

    # Include the config in the header so parallel paper traders (e.g. the
    # h72 candidate) are distinguishable in Telegram.
    header = (f"🧪 <b>ML PAPER TRADING</b> (no real money) "
              f"[h={HORIZON}, thr={BUY_THR:g}{'' if SELL_THR > 0 else ', long-only'}]")
    body = (
        f"\nEquity: ${state['equity']:.2f}  ({total_return:+.1f}%)"
        f"\nOpen positions: {len(state['open'])}"
        f"\nClosed trades: {n_closed}  |  Win rate: {win_rate:.1f}%"
    )
    if actions:
        body += "\n\n<b>This run:</b>\n" + "\n".join(f"• {a}" for a in actions)
    else:
        body += "\n\n(no new actions this run)"

    print(header.replace('<b>', '').replace('</b>', ''))
    print(body.replace('<b>', '').replace('</b>', ''))

    # Only ping Telegram when something actually happened, to avoid spam.
    if actions:
        telegram.send(header + body)


if __name__ == "__main__":
    main()
