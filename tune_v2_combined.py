"""
COMBINED V2 TUNING (research only)

Tests the best individual improvements together:
- Higher entry threshold (0.68 vs 0.65)
- Fear & Greed features
- Adaptive model exit (bail if P(up) drops during hold)
- Optional F&G entry filter (skip the 25-40 fear bucket that loses money)

Same honesty rules: walk-forward, honest fees, early/late holdout at 2025-10-01.
"""

import numpy as np
import pandas as pd
import requests

from backtest import get_history
from research_edge import build_features, build_labels, make_model

SYMBOLS = ['XRP-USD', 'ADA-USD', 'SOL-USD', 'LINK-USD', 'DOGE-USD']
SPLIT_DATE = pd.Timestamp('2025-10-01')
FEE_PER_SIDE = 0.0021
LEVERAGE = 2.0
MARGIN_OPEN_FEE = 0.0002
ROLLOVER_FEE = 0.0002
TRAIN_MIN = 4000
RETRAIN_EVERY = 720


def fetch_fear_greed() -> pd.Series:
    """Download full daily Fear & Greed history."""
    r = requests.get('https://api.alternative.me/fng/',
                     params={'limit': 0, 'format': 'json'}, timeout=30).json()
    rows = [(pd.Timestamp(int(d['timestamp']), unit='s'), float(d['value']))
            for d in r['data']]
    return pd.Series(dict(rows)).sort_index()


def make_fng_features(fng: pd.Series) -> pd.DataFrame:
    """Daily F&G features, shifted 1 day to avoid look-ahead."""
    df = pd.DataFrame(index=fng.index)
    df['fng'] = fng / 100.0
    df['fng_chg_7'] = fng.diff(7) / 100.0
    df['fng_vs_ma30'] = (fng - fng.rolling(30).mean()) / 100.0
    return df.shift(1)


def build_enhanced_features(data: pd.DataFrame, fng_feats: pd.DataFrame) -> pd.DataFrame:
    """Price features + Fear & Greed, aligned to hourly bars."""
    return build_features(data).join(fng_feats.reindex(data.index, method='ffill'))


def backtest(data: pd.DataFrame, horizon: int, buy_thr: float, exit_thr: float,
             fng_feats: pd.DataFrame, fng_filter: bool) -> list:
    """Walk-forward long-only backtest with optional enhancements."""
    feats = build_enhanced_features(data, fng_feats)
    labels, _ = build_labels(data, horizon)
    X = feats.values
    y = labels.values
    close = data['Close'].values
    n = len(data)
    valid_feat = ~np.isnan(X).any(axis=1)
    fee_cost_pct = FEE_PER_SIDE * 100 * LEVERAGE * 2

    fng_shifted = fetch_fear_greed().shift(1) if fng_filter else None

    model = None
    last_train_pos = -10**9
    trades = []
    i = TRAIN_MIN

    while i < n - 1:
        if not valid_feat[i]:
            i += 1
            continue

        if model is None or (i - last_train_pos) >= RETRAIN_EVERY:
            train_upto = i - horizon
            if train_upto > TRAIN_MIN // 2:
                mask = np.zeros(n, dtype=bool)
                mask[:train_upto + 1] = True
                mask &= valid_feat & ~np.isnan(y)
                if mask.sum() >= 300 and len(np.unique(y[mask])) > 1:
                    model = make_model('logistic')
                    model.fit(X[mask], y[mask])
                    last_train_pos = i

        if model is None:
            i += 1
            continue

        prob_up = float(model.predict_proba(X[i:i + 1])[:, 1][0])
        if prob_up <= buy_thr:
            i += 1
            continue

        # Skip the fear bucket (25-40) that historically loses money.
        if fng_filter and fng_shifted is not None:
            day = pd.Timestamp(data.index[i]).normalize()
            fng_val = fng_shifted.get(day, np.nan)
            if not np.isnan(fng_val) and 25 <= fng_val < 40:
                i += 1
                continue

        entry_price = close[i]
        exit_idx = min(i + horizon, n - 1)
        exit_reason = 'time'

        if exit_thr > 0:
            for j in range(i + 1, exit_idx):
                if not valid_feat[j]:
                    continue
                p = float(model.predict_proba(X[j:j + 1])[:, 1][0])
                if p < exit_thr:
                    exit_idx = j
                    exit_reason = 'model_exit'
                    break

        exit_price = close[exit_idx]
        bars_held = exit_idx - i
        gross_pct = (exit_price - entry_price) / entry_price * 100 * LEVERAGE
        margin_cost_pct = (MARGIN_OPEN_FEE + ROLLOVER_FEE * (bars_held // 4)) * 100 * LEVERAGE
        net_pct = gross_pct - fee_cost_pct - margin_cost_pct

        trades.append({
            'entry_time': data.index[i],
            'net_pnl_pct': net_pct,
            'prob_up': prob_up,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
        })
        i = exit_idx + 1

    return trades


def describe(trades: list) -> dict:
    if not trades:
        return {'n': 0, 'exp': float('nan'), 'win': float('nan'), 'pf': float('nan')}
    p = np.array([t['net_pnl_pct'] for t in trades])
    losses = p[p <= 0]
    pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
    return {'n': len(p), 'exp': p.mean(), 'win': (p > 0).mean() * 100, 'pf': pf}


def report(label: str, trades: list) -> None:
    early = [t for t in trades if pd.Timestamp(t['entry_time']) < SPLIT_DATE]
    late = [t for t in trades if pd.Timestamp(t['entry_time']) >= SPLIT_DATE]
    o, e, l = describe(trades), describe(early), describe(late)
    ok = e['n'] > 0 and l['n'] > 0 and e['exp'] > 0 and l['exp'] > 0
    total = sum(t['net_pnl_pct'] for t in trades)
    print(f"  {label:38s} n={o['n']:4d} exp={o['exp']:+6.2f}% pf={o['pf']:5.2f} "
          f"total={total:+7.1f}% | early={e['exp']:+6.2f}% late={l['exp']:+6.2f}% | "
          f"{'ROBUST' if ok else '  -   '}")


def main() -> None:
    print("Downloading Fear & Greed + price data...")
    fng = fetch_fear_greed()
    fng_feats = make_fng_features(fng)
    data_by_symbol = {s: get_history(s, '720d', '1h') for s in SYMBOLS}

    configs = [
        ('BASELINE h=48 thr=0.65', 48, 0.65, 0.0, False),
        ('V2 h=48 thr=0.68', 48, 0.68, 0.0, False),
        ('V2 h=48 thr=0.68 +exit0.40', 48, 0.68, 0.40, False),
        ('V2 h=48 thr=0.68 +exit0.40 +F&G filter', 48, 0.68, 0.40, True),
        ('V2 h=72 thr=0.68 +exit0.40', 72, 0.68, 0.40, False),
        ('V2 h=72 thr=0.68 +exit0.40 +F&G filter', 72, 0.68, 0.40, True),
    ]

    print(f"\nLong-only, {LEVERAGE:.0f}x, honest fees, holdout {SPLIT_DATE.date()}\n")
    for label, horizon, buy_thr, exit_thr, fng_filter in configs:
        trades = []
        for data in data_by_symbol.values():
            trades.extend(backtest(data, horizon, buy_thr, exit_thr, fng_feats, fng_filter))
        report(label, trades)


if __name__ == '__main__':
    main()
