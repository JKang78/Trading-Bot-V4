"""
REAL-MONEY LIVE TRADER for the longer-horizon ML strategy.

⚠️ THIS PLACES REAL ORDERS ON KRAKEN. ⚠️

It trades the validated longer-horizon ML V3 strategy (see ml_strategy.py) with a
few conservative, user-chosen settings:
- Coins: XRP, ADA, SOL, LINK, DOGE (each passed walk-forward validation with
  positive expectancy in both the early and holdout periods).
- Each trade uses 20% of usable margin at 2x leverage, so all five coins
  together can use the full usable margin (balance / 1.5) but never more.
- Hold ~3 days (72 x 1h bars), then close (time-based exit).
- At most one position per coin.

Safety design
-------------
- Entries try a post-only LIMIT order first. The bot only falls back to MARKET
  if the expected value still survives taker fees and slippage. Exits are
  always MARKET orders (a time-boxed strategy must be able to get out).
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
from ml_strategy import (
    KrakenCostModel,
    MLSwingStrategy,
    btc_regime_state,
    compute_btc_regime_frame,
    expected_value,
)


# ─────────────────────────── Settings (env-overridable) ───────────────────────────
STATE_FILE = os.getenv('ML_LIVE_STATE_FILE', 'ml_live_state.json')
SYMBOLS = [s.strip() for s in os.getenv(
    'ML_LIVE_SYMBOLS', 'XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD').split(',') if s.strip()]
PERIOD = os.getenv('ML_LIVE_PERIOD', '720d')
INTERVAL = os.getenv('ML_LIVE_INTERVAL', '1h')
HORIZON = int(os.getenv('ML_LIVE_HORIZON', '72'))
BUY_THR = float(os.getenv('ML_LIVE_BUY_THR', '0.70'))
SELL_THR = float(os.getenv('ML_LIVE_SELL_THR', '0.35'))
EXIT_THR = float(os.getenv('ML_LIVE_EXIT_THR', '0.40'))
USE_FNG_FEATURES = os.getenv('ML_LIVE_FNG_FEATURES', 'true').lower() == 'true'
USE_FNG_FILTER = os.getenv('ML_LIVE_FNG_FILTER', 'true').lower() == 'true'
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

MAKER_ENTRY_FEE = float(os.getenv('ML_LIVE_MAKER_ENTRY_FEE', '0.0023'))
TAKER_ENTRY_FEE = float(os.getenv('ML_LIVE_TAKER_ENTRY_FEE', '0.0040'))
TAKER_EXIT_FEE = float(os.getenv('ML_LIVE_TAKER_EXIT_FEE', '0.0040'))
MARGIN_OPEN_FEE = float(os.getenv('ML_LIVE_MARGIN_OPEN_FEE', '0.0004'))
ROLLOVER_FEE_4H = float(os.getenv('ML_LIVE_ROLLOVER_FEE_4H', '0.0004'))
SPREAD_BUFFER = float(os.getenv('ML_LIVE_SPREAD_BUFFER', '0.0005'))
SLIPPAGE_BUFFER = float(os.getenv('ML_LIVE_SLIPPAGE_BUFFER', '0.0010'))
MINIMUM_EDGE = float(os.getenv('ML_LIVE_MINIMUM_EDGE', '0.0075'))
EV_COST_MULTIPLIER = float(os.getenv('ML_LIVE_EV_COST_MULTIPLIER', '1.5'))
USE_BTC_FEATURES = os.getenv('ML_LIVE_BTC_FEATURES', 'false').lower() == 'true'
USE_BTC_REGIME_FILTER = os.getenv('ML_LIVE_BTC_REGIME_FILTER', 'false').lower() == 'true'
USE_RELATIVE_STRENGTH_FILTER = os.getenv('ML_LIVE_RELATIVE_STRENGTH_FILTER', 'false').lower() == 'true'
USE_EXPECTED_VALUE_FILTER = os.getenv('ML_LIVE_EXPECTED_VALUE_FILTER', 'false').lower() == 'true'


def confidence_size_multiplier(probability: float, threshold: float) -> float:
    if probability < threshold + 0.03:
        return 0.50
    if probability < threshold + 0.07:
        return 0.75
    return 1.00


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
                   volume: float, leverage: int, fallback_price: float,
                   allow_market_fallback: bool) -> tuple:
    """
    Open a position, trying the cheap way first.

    1. Place a post-only LIMIT order at the best bid (buy) / best ask (sell),
       which pays the lower maker fee if it fills.
    2. Wait up to MAKER_WAIT_SEC, checking every few seconds.
    3. If it hasn't fully filled, cancel it and MARKET-order the remainder,
       so we always end up with the full position this run.

    Returns (average_fill_price, how, filled_volume).
    """
    if not USE_MAKER_ENTRY:
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=volume, leverage=leverage, reduce_only=False)
        return fallback_price, 'taker', volume

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
        print(f"   maker entry failed ({e})")
        if not allow_market_fallback:
            return None, 'maker_failed_skip_taker', 0.0
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=volume, leverage=leverage, reduce_only=False)
        return fallback_price, 'taker', volume

    if txid is None:
        return limit_price, 'maker_unknown', volume

    # Poll until filled or out of patience.
    deadline = time.time() + MAKER_WAIT_SEC
    while time.time() < deadline:
        time.sleep(5)
        try:
            info = kraken.query_order(txid)
        except Exception:
            continue
        if info.get('status') == 'closed':
            return float(info.get('price', limit_price) or limit_price), 'maker', volume

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
        if not allow_market_fallback:
            if filled_vol > 0:
                return limit_price, 'partial_maker_skip_taker', filled_vol
            return None, 'skipped_unfilled_maker', 0.0
        kraken.place_order(pair=kraken_pair, order_type=order_type,
                           volume=remaining, leverage=leverage, reduce_only=False)
    how = 'mixed' if filled_vol > 0 else 'taker'
    return fallback_price, how, volume


def main() -> None:
    config = Config()
    kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, config.KRAKEN_API_URL)
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
        long_only=LONG_ONLY, cost_model=cost_model,
        use_btc_features=USE_BTC_FEATURES,
        use_btc_regime_filter=USE_BTC_REGIME_FILTER,
        use_relative_strength_filter=USE_RELATIVE_STRENGTH_FILTER,
        use_expected_value_filter=USE_EXPECTED_VALUE_FILTER,
        ev_cost_multiplier=EV_COST_MULTIPLIER,
    )
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
    btc_data = None
    btc_regimes = None
    if USE_BTC_FEATURES or USE_BTC_REGIME_FILTER or USE_RELATIVE_STRENGTH_FILTER:
        try:
            btc_data = get_history('BTC-USD', PERIOD, INTERVAL)
            btc_regimes = compute_btc_regime_frame(btc_data)
        except Exception as e:
            print(f"❌ Could not read BTC regime data: {e}")
            return

    # Read live account state (safe, read-only).
    try:
        available_margin = kraken.get_available_margin()
    except Exception as e:
        print(f"❌ Could not read margin: {e}")
        return
    usable_margin = available_margin / MARGIN_SAFETY_FACTOR

    # ── 1) Close positions: time limit OR model says bail early ──
    for symbol in list(state['open'].keys()):
        pos = state['open'][symbol]
        df = data_by_symbol.get(symbol)
        if df is None:
            continue
        now = df.index[-1]
        time_due = now >= pd.Timestamp(pos['exit_due'])
        early_exit = False
        exit_prob = pos.get('prob_up', 0.5)
        if not time_due and pos['direction'] == 'long' and EXIT_THR > 0:
            early_exit, exit_prob = strategy.should_exit_early(df, btc_data)
        btc_exit = False
        if USE_BTC_REGIME_FILTER and btc_regimes is not None:
            btc_state = btc_regime_state(btc_regimes, now)
            btc_exit = btc_state.block_new_entries
        if btc_exit:
            early_exit = True
        if not time_due and not early_exit:
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
            reason = 'btc_regime' if btc_exit else 'model_exit' if early_exit else 'time'
            state['closed'].append({**pos, 'symbol': symbol, 'exit_price': current_price,
                                    'exit_time': str(now), 'pnl_pct': round(pnl_pct, 3),
                                    'exit_reason': reason, 'exit_prob_up': round(exit_prob, 3)})
            del state['open'][symbol]
            tag = f" ({reason}, p_up={exit_prob:.2f})" if early_exit else ""
            actions.append(f"CLOSE {symbol} {pos['direction']} @ {current_price:.4f} -> {pnl_pct:+.2f}%{tag}")
        except Exception as e:
            actions.append(f"⚠️ close {symbol} failed: {e}")

    # ── 2) Rank eligible high-margin signals, then open top opportunities ──
    latest_ts = max(df.index[-1] for df in data_by_symbol.values()) if data_by_symbol else pd.Timestamp.utcnow()
    btc_state = btc_regime_state(btc_regimes, latest_ts) if btc_regimes is not None else None
    allowed_max_open = min(MAX_OPEN, btc_state.max_positions) if USE_BTC_REGIME_FILTER and btc_state else MAX_OPEN
    open_slots = max(0, allowed_max_open - len(state['open']))
    if USE_BTC_REGIME_FILTER and btc_state and btc_state.block_new_entries:
        actions.append(f"skip all entries: BTC regime weak ({','.join(btc_state.reasons) or 'weak'})")

    candidates = []
    for symbol in SYMBOLS:
        if symbol in state['open']:
            continue
        df = data_by_symbol.get(symbol)
        kp = pairs.get(symbol)
        if df is None or kp is None:
            continue

        sig = strategy.get_signal(df, btc_data)
        if sig.signal not in ('BUY', 'SELL'):
            if sig.blocked_reason == 'fng_fear_bucket':
                actions.append(f"skip {symbol}: F&G fear bucket 25-40 (p_up={sig.prob_up:.2f})")
            elif sig.blocked_reason:
                actions.append(
                    f"skip {symbol}: {sig.blocked_reason} "
                    f"(p={sig.prob_up:.2f}, ev={sig.expected_value:.3%}, "
                    f"thr={sig.dynamic_threshold:.2f}, rs7={sig.relative_strength_7d:.2%})")
            continue
        if LONG_ONLY and sig.signal == 'SELL':
            actions.append(f"skip {symbol}: SELL signal ignored (long-only mode, p_up={sig.prob_up:.2f})")
            continue

        candidates.append((symbol, kp, df, sig))

    candidates.sort(key=lambda item: item[3].score, reverse=True)

    for symbol, kp, df, sig in candidates[:open_slots]:
        current_price = float(df['Close'].iloc[-1])
        conf_mult = confidence_size_multiplier(sig.prob_up, sig.dynamic_threshold)
        margin_usd = usable_margin * POSITION_FRACTION * conf_mult * sig.regime_size_multiplier
        volume = (margin_usd * LEVERAGE) / current_price

        if margin_usd <= 0 or volume < kp.min_volume:
            actions.append(f"skip {symbol}: volume {volume:.8f} < min {kp.min_volume} (margin ${margin_usd:.2f})")
            continue

        order_type = 'buy' if sig.signal == 'BUY' else 'sell'
        direction = 'long' if sig.signal == 'BUY' else 'short'
        now = df.index[-1]
        taker_cost = cost_model.estimated_total_cost(HORIZON, 'taker')
        taker_ev = expected_value(sig.prob_up, sig.avg_win, sig.avg_loss, taker_cost)
        allow_market_fallback = taker_ev > 0 and taker_ev > EV_COST_MULTIPLIER * taker_cost
        try:
            fill_how = 'dry-run'
            entry_price = current_price
            filled_volume = volume
            if not DRY_RUN:
                entry_price, fill_how, filled_volume = enter_position(
                    kraken, kp.kraken_pair, order_type, volume, LEVERAGE,
                    current_price, allow_market_fallback)
            if entry_price is None or filled_volume <= 0:
                actions.append(
                    f"skip {symbol}: maker not filled and taker EV too low "
                    f"(maker_ev={sig.expected_value:.3%}, taker_ev={taker_ev:.3%})")
                continue
            state['open'][symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': str(now),
                'exit_due': str(now + timedelta(hours=HORIZON)),
                'volume': round(filled_volume, 8),
                'prob_up': round(sig.prob_up, 3),
                'expected_value': round(sig.expected_value, 5),
                'dynamic_threshold': round(sig.dynamic_threshold, 3),
                'estimated_cost': round(sig.estimated_cost, 5),
                'btc_regime': sig.btc_regime,
                'relative_strength_7d': round(sig.relative_strength_7d, 5),
                'margin_usd': round(margin_usd, 2),
                'leverage': LEVERAGE,
                'entry_fill': fill_how,
                'model_version': 'v3_cost_aware_ev_btc_rs',
            }
            actions.append(f"OPEN {symbol} {direction.upper()} @ {entry_price:.4f} "
                           f"vol={filled_volume:.6f} ({fill_how}, p={sig.prob_up:.2f}, "
                           f"thr={sig.dynamic_threshold:.2f}, ev={sig.expected_value:.2%}, "
                           f"score={sig.score:.2f}, margin ${margin_usd:.2f})")
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
