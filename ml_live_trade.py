"""
REAL-MONEY LIVE TRADER for the longer-horizon ML strategy.

⚠️ THIS PLACES REAL ORDERS ON KRAKEN. ⚠️

It trades the validated longer-horizon ML strategy (see ml_strategy.py) with a
few conservative, user-chosen settings:
- Coins: XRP, ADA, SOL (where the backtested edge was strongest).
- Each trade uses 20% of usable margin, at 2x leverage.
- Hold ~2 days (48 x 1h bars), then close (time-based exit).
- At most one position per coin.

Safety design
-------------
- Entries and exits are MARKET orders (reliable fills; a time-boxed strategy
  must be able to get in and out on schedule).
- State (which positions we opened and when to close them) is saved to
  ml_live_state.json so it survives between independent cron runs.
- Set ML_LIVE_DRY_RUN=true to run the full logic WITHOUT placing real orders
  (read-only account calls only) - useful for testing.

This is separate from the old swing bot, which should be disabled so the two do
not fight over the same Kraken account.
"""

import os
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from kraken_bot_v4_advanced import Config, KrakenClient, Telegram
from backtest import get_history
from ml_strategy import MLSwingStrategy


# ─────────────────────────── Settings (env-overridable) ───────────────────────────
STATE_FILE = os.getenv('ML_LIVE_STATE_FILE', 'ml_live_state.json')
SYMBOLS = [s.strip() for s in os.getenv('ML_LIVE_SYMBOLS', 'XRP-USD,ADA-USD,SOL-USD').split(',') if s.strip()]
PERIOD = os.getenv('ML_LIVE_PERIOD', '720d')
INTERVAL = os.getenv('ML_LIVE_INTERVAL', '1h')
HORIZON = int(os.getenv('ML_LIVE_HORIZON', '48'))
BUY_THR = float(os.getenv('ML_LIVE_BUY_THR', '0.65'))
SELL_THR = float(os.getenv('ML_LIVE_SELL_THR', '0.35'))
LEVERAGE = int(os.getenv('ML_LIVE_LEVERAGE', '2'))
POSITION_FRACTION = float(os.getenv('ML_LIVE_POSITION_FRACTION', '0.20'))
MAX_OPEN = int(os.getenv('ML_LIVE_MAX_OPEN', '3'))
MARGIN_SAFETY_FACTOR = float(os.getenv('ML_LIVE_MARGIN_SAFETY', '1.5'))
DRY_RUN = os.getenv('ML_LIVE_DRY_RUN', 'false').lower() == 'true'


def load_state() -> dict:
    """Read our record of open ML positions, or start empty."""
    path = Path(STATE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {'open': {}, 'closed': []}


def save_state(state: dict) -> None:
    state['updated_at'] = pd.Timestamp.utcnow().isoformat()
    Path(STATE_FILE).write_text(json.dumps(state, indent=2, default=str))


def pair_map(config: Config) -> dict:
    """Map yfinance symbol -> its Kraken pair + minimum order volume."""
    return {p.yf_symbol: p for p in config.TRADING_PAIRS}


def main() -> None:
    config = Config()
    kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, config.KRAKEN_API_URL)
    telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    strategy = MLSwingStrategy(horizon=HORIZON, buy_thr=BUY_THR, sell_thr=SELL_THR)
    pairs = pair_map(config)

    mode = "🧪 DRY-RUN (no real orders)" if DRY_RUN else "💰 REAL MONEY"
    print(f"ML LIVE TRADER | {mode} | coins={SYMBOLS} | {POSITION_FRACTION:.0%}/trade @ {LEVERAGE}x")

    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("❌ Missing Kraken credentials - aborting.")
        return

    state = load_state()
    actions = []

    # Fresh data per coin (used for both signals and current price).
    data_by_symbol = {}
    for symbol in SYMBOLS:
        try:
            df = get_history(symbol, PERIOD, INTERVAL)
            if not df.empty:
                data_by_symbol[symbol] = df
        except Exception as e:
            print(f"  ⚠️ {symbol}: data error {e}")

    # Read live account state (safe, read-only).
    try:
        available_margin = kraken.get_available_margin()
    except Exception as e:
        print(f"❌ Could not read margin: {e}")
        return
    usable_margin = available_margin / MARGIN_SAFETY_FACTOR

    # ── 1) Close positions whose ~2-day hold is complete ──
    for symbol in list(state['open'].keys()):
        pos = state['open'][symbol]
        df = data_by_symbol.get(symbol)
        if df is None:
            continue
        now = df.index[-1]
        if now < pd.Timestamp(pos['exit_due']):
            continue

        current_price = float(df['Close'].iloc[-1])
        kp = pairs.get(symbol)
        try:
            if not DRY_RUN:
                kraken.close_position(kp.kraken_pair, pos['direction'], pos['volume'], LEVERAGE)
            # Informational realized P&L (price move x leverage).
            if pos['direction'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100 * LEVERAGE
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100 * LEVERAGE
            state['closed'].append({**pos, 'symbol': symbol, 'exit_price': current_price,
                                    'exit_time': str(now), 'pnl_pct': round(pnl_pct, 3)})
            del state['open'][symbol]
            actions.append(f"CLOSE {symbol} {pos['direction']} @ {current_price:.4f} -> {pnl_pct:+.2f}%")
        except Exception as e:
            actions.append(f"⚠️ close {symbol} failed: {e}")

    # ── 2) Open new positions on high-confidence signals ──
    for symbol in SYMBOLS:
        if symbol in state['open']:
            continue
        if len(state['open']) >= MAX_OPEN:
            break
        df = data_by_symbol.get(symbol)
        kp = pairs.get(symbol)
        if df is None or kp is None:
            continue

        sig = strategy.get_signal(df)
        if sig.signal not in ('BUY', 'SELL'):
            continue

        current_price = float(df['Close'].iloc[-1])
        margin_usd = usable_margin * POSITION_FRACTION
        volume = (margin_usd * LEVERAGE) / current_price

        if margin_usd <= 0 or volume < kp.min_volume:
            actions.append(f"skip {symbol}: volume {volume:.8f} < min {kp.min_volume} (margin ${margin_usd:.2f})")
            continue

        order_type = 'buy' if sig.signal == 'BUY' else 'sell'
        direction = 'long' if sig.signal == 'BUY' else 'short'
        now = df.index[-1]
        try:
            if not DRY_RUN:
                kraken.place_order(pair=kp.kraken_pair, order_type=order_type,
                                   volume=volume, leverage=LEVERAGE, reduce_only=False)
            state['open'][symbol] = {
                'direction': direction,
                'entry_price': current_price,
                'entry_time': str(now),
                'exit_due': str(now + timedelta(hours=HORIZON)),
                'volume': round(volume, 8),
                'prob_up': round(sig.prob_up, 3),
                'leverage': LEVERAGE,
            }
            actions.append(f"OPEN {symbol} {direction.upper()} @ {current_price:.4f} "
                           f"vol={volume:.6f} (p_up={sig.prob_up:.2f}, margin ${margin_usd:.2f})")
        except Exception as e:
            actions.append(f"⚠️ open {symbol} failed: {e}")

    save_state(state)

    # ── 3) Report ──
    n_closed = len(state['closed'])
    wins = [t for t in state['closed'] if t.get('pnl_pct', 0) > 0]
    win_rate = (len(wins) / n_closed * 100) if n_closed else 0.0

    header = f"{'🧪 ML DRY-RUN' if DRY_RUN else '💰 ML LIVE'} TRADER"
    body = (f"\nUsable margin: ${usable_margin:.2f}"
            f"\nOpen positions: {len(state['open'])}/{MAX_OPEN}"
            f"\nClosed trades: {n_closed} | Win rate: {win_rate:.1f}%")
    if actions:
        body += "\n\n<b>This run:</b>\n" + "\n".join(f"• {a}" for a in actions)
    else:
        body += "\n\n(no new actions this run)"

    print(body.replace('<b>', '').replace('</b>', ''))
    if actions:
        telegram.send(f"<b>{header}</b>{body}")


if __name__ == "__main__":
    main()
