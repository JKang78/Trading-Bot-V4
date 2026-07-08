"""
SENTIMENT (FEAR & GREED) EXPERIMENT (research only - does not touch the live bot)

Question: does market sentiment improve the validated ML strategy?

The Fear & Greed index (alternative.me) is the one sentiment source with full
daily HISTORY, so unlike news headlines it can be backtested honestly.

Two tests, same walk-forward harness and honest fee model as tune_expectancy.py:

1. FEATURE TEST - give the model three extra inputs (yesterday's F&G level,
   its 7-day change, its distance from its 30-day average) and see if
   expectancy improves in BOTH the early and holdout halves.
2. DIAGNOSTIC - bucket the baseline strategy's trades by the F&G level at
   entry. If expectancy differs sharply and CONSISTENTLY across buckets, a
   sentiment entry filter might be worth building. If not, it isn't.

Look-ahead safety: the index for day D is published at the start of day D; we
still shift it by one full day, so a trade at any hour of day D only ever sees
the value from day D-1.
"""

import numpy as np
import pandas as pd
import requests

import ml_strategy_backtest as msb
from backtest import get_history
from research_edge import build_features

SYMBOLS = ['XRP-USD', 'ADA-USD', 'SOL-USD', 'LINK-USD', 'DOGE-USD']
SPLIT_DATE = pd.Timestamp('2025-10-01')
FEE_PER_SIDE = 0.0021
LEVERAGE = 2.0
HORIZON = 48
BUY_THR = 0.65


def fetch_fear_greed() -> pd.Series:
    """Download the full daily Fear & Greed history (0=extreme fear, 100=greed)."""
    r = requests.get('https://api.alternative.me/fng/',
                     params={'limit': 0, 'format': 'json'}, timeout=30).json()
    rows = [(pd.Timestamp(int(d['timestamp']), unit='s'), float(d['value']))
            for d in r['data']]
    s = pd.Series(dict(rows)).sort_index()
    s.index.name = 'date'
    return s


def make_fng_features(fng: pd.Series) -> pd.DataFrame:
    """Daily F&G -> feature columns, shifted 1 day so there is no look-ahead."""
    df = pd.DataFrame(index=fng.index)
    df['fng'] = fng / 100.0                       # level, scaled 0..1
    df['fng_chg_7'] = fng.diff(7) / 100.0         # week-over-week swing
    df['fng_vs_ma30'] = (fng - fng.rolling(30).mean()) / 100.0
    return df.shift(1)


def run_config(data_by_symbol: dict) -> list:
    """Walk-forward backtest (long-only, honest fees) pooled over all coins."""
    trades = []
    for symbol, data in data_by_symbol.items():
        r = msb.backtest_symbol(
            symbol, data, horizon=HORIZON, buy_thr=BUY_THR, sell_thr=0.0,
            fee_rate=FEE_PER_SIDE, leverage=LEVERAGE, model_name='logistic',
            train_min=4000, retrain_every=720, atr_stop_mult=0.0, atr_period=14,
        )
        trades.extend(r.get('trades', []))
    return trades


def describe(trades: list) -> dict:
    if not trades:
        return {'n': 0, 'exp': float('nan'), 'pf': float('nan')}
    p = np.array([t['net_pnl_pct'] for t in trades])
    losses = p[p <= 0]
    pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
    return {'n': len(p), 'exp': p.mean(), 'pf': pf}


def report(label: str, trades: list) -> None:
    early = [t for t in trades if pd.Timestamp(t['entry_time']) < SPLIT_DATE]
    late = [t for t in trades if pd.Timestamp(t['entry_time']) >= SPLIT_DATE]
    o, e, l = describe(trades), describe(early), describe(late)
    ok = e['n'] > 0 and l['n'] > 0 and e['exp'] > 0 and l['exp'] > 0
    print(f"  {label:24s} n={o['n']:4d} exp={o['exp']:+6.2f}% pf={o['pf']:5.2f} | "
          f"early exp={e['exp']:+6.2f}% | late exp={l['exp']:+6.2f}% | "
          f"{'ROBUST' if ok else '  -   '}")


def main() -> None:
    print("Downloading Fear & Greed history and price data...")
    fng = fetch_fear_greed()
    fng_feats = make_fng_features(fng)
    print(f"  F&G: {len(fng)} days ({fng.index.min().date()} .. {fng.index.max().date()})")
    data_by_symbol = {s: get_history(s, '720d', '1h') for s in SYMBOLS}

    print(f"\nAll runs long-only, h={HORIZON}, thr={BUY_THR}, {LEVERAGE:.0f}x, "
          f"honest fees, holdout split {SPLIT_DATE.date()}.\n")

    # ── 1) Baseline vs +F&G features ──
    print("── Feature test ──")
    baseline_trades = run_config(data_by_symbol)
    report("baseline (price only)", baseline_trades)

    original = msb.build_features
    msb.build_features = lambda data: build_features(data).join(
        fng_feats.reindex(data.index, method='ffill'))
    try:
        fng_trades = run_config(data_by_symbol)
    finally:
        msb.build_features = original
    report("+ Fear&Greed features", fng_trades)

    # ── 2) Diagnostic: baseline expectancy by F&G level at entry ──
    print("\n── Baseline trades bucketed by F&G at entry (shifted 1 day) ──")
    fng_shifted = fng.shift(1)
    rows = []
    for t in baseline_trades:
        day = pd.Timestamp(t['entry_time']).normalize()
        val = fng_shifted.get(day, np.nan)
        if not np.isnan(val):
            rows.append({'fng': val, 'pnl': t['net_pnl_pct'],
                         'entry_time': t['entry_time']})
    d = pd.DataFrame(rows)
    d['bucket'] = pd.cut(d['fng'], [0, 25, 40, 55, 100],
                         labels=['extreme fear (<25)', 'fear (25-40)',
                                 'neutral (40-55)', 'greed (>55)'])
    for name, g in d.groupby('bucket', observed=True):
        e = g[g['entry_time'] < SPLIT_DATE]['pnl']
        l = g[g['entry_time'] >= SPLIT_DATE]['pnl']
        print(f"  {name:20s} n={len(g):4d} exp={g['pnl'].mean():+6.2f}% | "
              f"early n={len(e):3d} exp={e.mean() if len(e) else float('nan'):+6.2f}% | "
              f"late n={len(l):3d} exp={l.mean() if len(l) else float('nan'):+6.2f}%")


if __name__ == "__main__":
    main()
