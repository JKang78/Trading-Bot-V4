"""
EXPECTANCY TUNING (research only - does not touch the live bot)

Goal: find settings with higher expectancy per trade WITHOUT fooling ourselves.

The overfitting trap
--------------------
If we try 20 settings on our full history and pick the best, that "best" is
partly luck and will disappoint in live trading. Defense used here:

1. Every config is run through the SAME walk-forward backtest with the honest
   fee model (trading fees + Kraken margin open/rollover costs).
2. Each config's trades are split at a fixed calendar date (2025-10-01) into
   an EARLY half and a LATE (holdout) half. We only trust a config if it is
   positive and decent in BOTH halves - not just overall.
3. We also report trades/year, because 10 great trades a year can still earn
   less money than 60 good ones.

It also tests a feature idea: giving the altcoin models BTC's own momentum
as extra inputs (alts often follow BTC with a lag).
"""

import numpy as np
import pandas as pd

import ml_strategy_backtest as msb
from backtest import get_history
from research_edge import build_features

SYMBOLS = ['XRP-USD', 'ADA-USD', 'SOL-USD']
SPLIT_DATE = pd.Timestamp('2025-10-01')

# Honest cost model: maker entry (0.16%) + taker exit (0.26%) averaged per side,
# plus the default margin open + rollover fees inside the backtest.
FEE_PER_SIDE = 0.0021
LEVERAGE = 2.0

# Long-only (validated earlier): sell threshold 0 disables shorts.
GRID = [
    # (horizon, buy_thr)
    (48, 0.60), (48, 0.65), (48, 0.68), (48, 0.70),
    (72, 0.60), (72, 0.65), (72, 0.68), (72, 0.70),
    (96, 0.60), (96, 0.65), (96, 0.68), (96, 0.70),
]


def run_config(data_by_symbol: dict, horizon: int, buy_thr: float) -> list:
    """Run the walk-forward backtest for one config, return all pooled trades."""
    trades = []
    for symbol, data in data_by_symbol.items():
        r = msb.backtest_symbol(
            symbol, data, horizon=horizon, buy_thr=buy_thr, sell_thr=0.0,
            fee_rate=FEE_PER_SIDE, leverage=LEVERAGE, model_name='logistic',
            train_min=4000, retrain_every=720, atr_stop_mult=0.0, atr_period=14,
        )
        trades.extend(r.get('trades', []))
    return trades


def describe(trades: list) -> dict:
    """Expectancy, win rate and profit factor for a list of trades."""
    if not trades:
        return {'n': 0, 'exp': float('nan'), 'win': float('nan'), 'pf': float('nan')}
    p = np.array([t['net_pnl_pct'] for t in trades])
    losses = p[p <= 0]
    pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
    return {'n': len(p), 'exp': p.mean(), 'win': (p > 0).mean() * 100, 'pf': pf}


def report(label: str, trades: list) -> None:
    """Print one config's overall + early/late split stats on one line."""
    early = [t for t in trades if pd.Timestamp(t['entry_time']) < SPLIT_DATE]
    late = [t for t in trades if pd.Timestamp(t['entry_time']) >= SPLIT_DATE]
    o, e, l = describe(trades), describe(early), describe(late)
    both_positive = (e['n'] > 0 and l['n'] > 0 and e['exp'] > 0 and l['exp'] > 0)
    flag = 'ROBUST' if both_positive else '  -   '
    print(f"  {label:28s} n={o['n']:4d} exp={o['exp']:+6.2f}% pf={o['pf']:5.2f} | "
          f"early n={e['n']:3d} exp={e['exp']:+6.2f}% | "
          f"late n={l['n']:3d} exp={l['exp']:+6.2f}% | {flag}")


def add_btc_features(alt_features: pd.DataFrame, btc_data: pd.DataFrame) -> pd.DataFrame:
    """Append BTC momentum/trend columns to an altcoin's feature table."""
    btc_close = btc_data['Close']
    extra = pd.DataFrame(index=btc_data.index)
    extra['btc_ret_24'] = btc_close.pct_change(24)
    extra['btc_ret_72'] = btc_close.pct_change(72)
    ema50 = btc_close.ewm(span=50, adjust=False).mean()
    extra['btc_dist_ema50'] = (btc_close - ema50) / ema50
    # Align BTC rows to the altcoin's timestamps (forward-fill small gaps).
    extra = extra.reindex(alt_features.index, method='ffill')
    return alt_features.join(extra)


def main() -> None:
    print("Downloading data once per coin (+BTC for the feature test)...")
    data_by_symbol = {s: get_history(s, '720d', '1h') for s in SYMBOLS}
    btc = get_history('BTC-USD', '720d', '1h')

    print(f"\nAll configs long-only, {LEVERAGE:.0f}x, fee {FEE_PER_SIDE*100:.2f}%/side "
          f"+ margin costs. Holdout split at {SPLIT_DATE.date()}.\n")

    print("── Grid: horizon x buy threshold ──")
    for horizon, buy_thr in GRID:
        trades = run_config(data_by_symbol, horizon, buy_thr)
        report(f"h={horizon} thr={buy_thr:.2f}", trades)

    # ── BTC-features experiment: patch the feature builder the backtest uses,
    # rerun the most promising configs, then restore it. Research-only. ──
    print("\n── Same grid rows + BTC features for the alt models ──")
    original_build_features = msb.build_features
    msb.build_features = lambda data: add_btc_features(build_features(data), btc)
    try:
        for horizon, buy_thr in [(48, 0.65), (72, 0.65), (48, 0.70), (72, 0.70)]:
            trades = run_config(data_by_symbol, horizon, buy_thr)
            report(f"h={horizon} thr={buy_thr:.2f} +BTC", trades)
    finally:
        msb.build_features = original_build_features


if __name__ == "__main__":
    main()
