"""
LIVE PAPER-TRADING RUNNER for the longer-horizon ML strategy.

Why a separate runner (not the main bot's DRY_RUN)?
---------------------------------------------------
The main bot runs once per cron cycle and reads REAL open positions from Kraken.
Its DRY_RUN mode just skips real orders - it does NOT remember simulated
positions between runs, so it cannot actually track paper P&L over days.

This runner keeps its own state file (ml_paper_state.json) so it can hold a
"paper" position for ~2 days across many independent cron runs, then close it
and record the result - exactly like the backtest, but on live data going
forward. It NEVER places a real order. It only simulates and notifies.

How to use
----------
Run it on a schedule (e.g. hourly). Each run it will:
  1. Close any paper positions whose ~2-day hold is up (record P&L).
  2. Open new paper positions when the ML model is highly confident.
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
from ml_strategy import MLSwingStrategy


# ─────────────────────────── Settings (env-overridable) ───────────────────────────
STATE_FILE = os.getenv('ML_PAPER_STATE_FILE', 'ml_paper_state.json')
SYMBOLS = [s.strip() for s in os.getenv(
    'ML_PAPER_SYMBOLS', 'XRP-USD,ADA-USD,SOL-USD,BTC-USD,ETH-USD').split(',') if s.strip()]
PERIOD = os.getenv('ML_PAPER_PERIOD', '720d')     # history for training
INTERVAL = os.getenv('ML_PAPER_INTERVAL', '1h')
HORIZON = int(os.getenv('ML_PAPER_HORIZON', '72'))
BUY_THR = float(os.getenv('ML_PAPER_BUY_THR', '0.68'))
SELL_THR = float(os.getenv('ML_PAPER_SELL_THR', '0.35'))
EXIT_THR = float(os.getenv('ML_PAPER_EXIT_THR', '0.40'))
USE_FNG_FEATURES = os.getenv('ML_PAPER_FNG_FEATURES', 'true').lower() == 'true'
USE_FNG_FILTER = os.getenv('ML_PAPER_FNG_FILTER', 'true').lower() == 'true'
LEVERAGE = float(os.getenv('ML_PAPER_LEVERAGE', '2'))
FEE_PER_SIDE = float(os.getenv('ML_PAPER_FEE', '0.001'))  # maker fee per side
START_EQUITY = float(os.getenv('ML_PAPER_START_EQUITY', '1000'))
POSITION_FRACTION = float(os.getenv('ML_PAPER_POSITION_FRACTION', '0.2'))  # margin per trade
MAX_OPEN = int(os.getenv('ML_PAPER_MAX_OPEN', '3'))


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


def net_pnl_pct(direction: str, entry: float, exit_price: float) -> float:
    """Profit % on the margin, after fees (includes leverage on both sides)."""
    if direction == 'long':
        gross = (exit_price - entry) / entry * 100 * LEVERAGE
    else:
        gross = (entry - exit_price) / entry * 100 * LEVERAGE
    fee_cost = FEE_PER_SIDE * 100 * LEVERAGE * 2
    return gross - fee_cost


def main() -> None:
    config = Config()
    telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    strategy = MLSwingStrategy(
        horizon=HORIZON, buy_thr=BUY_THR, sell_thr=SELL_THR, exit_thr=EXIT_THR,
        use_fng_features=USE_FNG_FEATURES, use_fng_filter=USE_FNG_FILTER,
        long_only=(SELL_THR <= 0),
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
            early_exit, exit_prob = strategy.should_exit_early(df)
        if not time_due and not early_exit:
            continue

        current_price = float(df['Close'].iloc[-1])
        pnl_pct = net_pnl_pct(pos['direction'], pos['entry_price'], current_price)
        pnl_usd = pos['margin_usd'] * pnl_pct / 100
        state['equity'] += pnl_usd

        closed = {**pos, 'symbol': symbol, 'exit_price': current_price,
                  'exit_time': str(now), 'pnl_pct': round(pnl_pct, 3),
                  'pnl_usd': round(pnl_usd, 2),
                  'exit_reason': 'model_exit' if early_exit else 'time'}
        state['closed'].append(closed)
        del state['open'][symbol]
        tag = f" ({closed['exit_reason']})" if early_exit else ""
        actions.append(
            f"CLOSE {symbol} {pos['direction']} @ {current_price:.4f} "
            f"-> {pnl_pct:+.2f}% ({pnl_usd:+.2f}$){tag}")

    # ── 2) Open new paper positions when the model is confident ──
    for symbol in SYMBOLS:
        if symbol in state['open']:
            continue  # already holding this coin
        if len(state['open']) >= MAX_OPEN:
            break
        df = data_by_symbol.get(symbol)
        if df is None:
            continue

        sig = strategy.get_signal(df)
        if sig.signal not in ('BUY', 'SELL'):
            continue

        now = df.index[-1]
        entry_price = float(df['Close'].iloc[-1])
        margin_usd = round(state['equity'] * POSITION_FRACTION, 2)
        exit_due = (now + timedelta(hours=HORIZON))

        state['open'][symbol] = {
            'direction': 'long' if sig.signal == 'BUY' else 'short',
            'entry_price': entry_price,
            'entry_time': str(now),
            'exit_due': str(exit_due),
            'prob_up': round(sig.prob_up, 3),
            'margin_usd': margin_usd,
            'leverage': LEVERAGE,
        }
        actions.append(
            f"OPEN {symbol} {'LONG' if sig.signal == 'BUY' else 'SHORT'} "
            f"@ {entry_price:.4f} (p_up={sig.prob_up:.2f}, margin ${margin_usd})")

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
