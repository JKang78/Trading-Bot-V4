"""
REAL-MONEY LIVE TRADER for the longer-horizon ML strategy.

⚠️ THIS PLACES REAL ORDERS ON KRAKEN. ⚠️

It trades the validated longer-horizon ML strategy (see ml_strategy.py) with a
few conservative, user-chosen settings:
- Coins: XRP, ADA, SOL, LINK, DOGE (each passed walk-forward validation with
  positive expectancy in both the early and holdout periods).
- Each trade uses 20% of usable margin at 2x leverage, so all five coins
  together can use the full usable margin (balance / 1.5) but never more.
- Hold ~2 days (48 x 1h bars), then close (time-based exit).
- At most one position per coin.

Safety design
-------------
- Entries try a post-only LIMIT order first (cheaper maker fee) and fall back
  to MARKET if it doesn't fill within ~90s. Exits are always MARKET orders
  (a time-boxed strategy must be able to get out on schedule).
- Long-only by default: the backtested short side barely covered its costs.
- State (which positions we opened and when to close them) is saved to
  ml_live_state.json so it survives between independent cron runs.
- Set ML_LIVE_DRY_RUN=true to run the full logic WITHOUT placing real orders
  (read-only account calls only) - useful for testing.

This is separate from the old swing bot, which should be disabled so the two do
not fight over the same Kraken account.
"""

import os
import json
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from kraken_bot_v4_advanced import Config, KrakenClient, Telegram
from backtest import get_history
from ml_strategy import MLSwingStrategy


# ─────────────────────────── Settings (env-overridable) ───────────────────────────
STATE_FILE = os.getenv('ML_LIVE_STATE_FILE', 'ml_live_state.json')
SYMBOLS = [s.strip() for s in os.getenv(
    'ML_LIVE_SYMBOLS', 'XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD').split(',') if s.strip()]
PERIOD = os.getenv('ML_LIVE_PERIOD', '720d')
INTERVAL = os.getenv('ML_LIVE_INTERVAL', '1h')
HORIZON = int(os.getenv('ML_LIVE_HORIZON', '48'))
BUY_THR = float(os.getenv('ML_LIVE_BUY_THR', '0.65'))
SELL_THR = float(os.getenv('ML_LIVE_SELL_THR', '0.35'))
LEVERAGE = int(os.getenv('ML_LIVE_LEVERAGE', '2'))
POSITION_FRACTION = float(os.getenv('ML_LIVE_POSITION_FRACTION', '0.20'))
MAX_OPEN = int(os.getenv('ML_LIVE_MAX_OPEN', '5'))
MARGIN_SAFETY_FACTOR = float(os.getenv('ML_LIVE_MARGIN_SAFETY', '1.5'))
DRY_RUN = os.getenv('ML_LIVE_DRY_RUN', 'false').lower() == 'true'

# Long-only: backtests show longs earn ~+3.0%/trade vs ~+0.9% for shorts, and
# shorts pay extra margin costs. Set ML_LIVE_LONG_ONLY=false to re-enable shorts.
LONG_ONLY = os.getenv('ML_LIVE_LONG_ONLY', 'true').lower() == 'true'

# Maker-first entries: try a post-only LIMIT order (cheaper maker fee ~0.16%)
# and only fall back to a MARKET order (taker fee ~0.26%) if it doesn't fill
# within MAKER_WAIT_SEC. Entries aren't time-critical for a ~2-day hold.
USE_MAKER_ENTRY = os.getenv('ML_LIVE_MAKER_ENTRY', 'true').lower() == 'true'
MAKER_WAIT_SEC = int(os.getenv('ML_LIVE_MAKER_WAIT_SEC', '90'))


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


def enter_position(kraken: KrakenClient, kraken_pair: str, order_type: str,
                   volume: float, leverage: int, fallback_price: float) -> tuple:
    """
    Open a position, trying the cheap way first.

    1. Place a post-only LIMIT order at the best bid (buy) / best ask (sell),
       which pays the lower maker fee if it fills.
    2. Wait up to MAKER_WAIT_SEC, checking every few seconds.
    3. If it hasn't fully filled, cancel it and MARKET-order the remainder,
       so we always end up with the full position this run.

    Returns (average_fill_price, how) where how is 'maker', 'taker', or 'mixed'.
    """
    if not USE_MAKER_ENTRY:
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=volume, leverage=leverage, reduce_only=False)
        return fallback_price, 'taker'

    # Rest the order on our side of the spread so it can't cross (= maker).
    try:
        bid, ask = kraken.get_bid_ask(kraken_pair)
        decimals = kraken.get_pair_decimals(kraken_pair)
        limit_price = round(bid if order_type == 'buy' else ask, decimals)
        result = kraken.place_order(pair=kraken_pair, order_type=order_type,
                                    volume=volume, leverage=leverage, reduce_only=False,
                                    ordertype='limit', price=limit_price, post_only=True)
        txid = result.get('txid', [None])[0]
    except Exception as e:
        # Post-only rejected (e.g. price would cross) -> just take the market.
        print(f"   maker entry failed ({e}); falling back to market")
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=volume, leverage=leverage, reduce_only=False)
        return fallback_price, 'taker'

    if txid is None:
        return limit_price, 'maker'  # order accepted but no txid returned; assume resting fill

    # Poll until filled or out of patience.
    deadline = time.time() + MAKER_WAIT_SEC
    while time.time() < deadline:
        time.sleep(5)
        try:
            info = kraken.query_order(txid)
        except Exception:
            continue
        if info.get('status') == 'closed':
            return float(info.get('price', limit_price) or limit_price), 'maker'

    # Not (fully) filled in time: cancel and market-order whatever is missing.
    filled_vol = 0.0
    try:
        kraken.cancel_order(txid)
        info = kraken.query_order(txid)
        filled_vol = float(info.get('vol_exec', 0) or 0)
    except Exception as e:
        print(f"   ⚠️ cancel/query after maker wait failed: {e}")

    remaining = volume - filled_vol
    if remaining > 0:
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=remaining, leverage=leverage, reduce_only=False)
    how = 'mixed' if filled_vol > 0 else 'taker'
    return fallback_price, how


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
        if LONG_ONLY and sig.signal == 'SELL':
            actions.append(f"skip {symbol}: SELL signal ignored (long-only mode, p_up={sig.prob_up:.2f})")
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
            fill_how = 'dry-run'
            entry_price = current_price
            if not DRY_RUN:
                entry_price, fill_how = enter_position(
                    kraken, kp.kraken_pair, order_type, volume, LEVERAGE, current_price)
            state['open'][symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': str(now),
                'exit_due': str(now + timedelta(hours=HORIZON)),
                'volume': round(volume, 8),
                'prob_up': round(sig.prob_up, 3),
                'leverage': LEVERAGE,
                'entry_fill': fill_how,
            }
            actions.append(f"OPEN {symbol} {direction.upper()} @ {entry_price:.4f} "
                           f"vol={volume:.6f} ({fill_how} fill, p_up={sig.prob_up:.2f}, "
                           f"margin ${margin_usd:.2f})")
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
