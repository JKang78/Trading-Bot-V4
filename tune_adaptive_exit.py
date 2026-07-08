"""
ADAPTIVE EXIT EXPERIMENT (research only - does not touch the live bot)

Idea: keep the entry rules exactly as validated (long when P(up) > 0.65), but
during the 48-bar hold, re-ask the model every bar. If its probability that
price rises drops below an exit threshold, close early instead of waiting for
the timer. Cutting trades the model has given up on should trim losers (and
save rollover fees) without reducing HOW OFTEN we trade.

Same honesty rules as tune_expectancy.py: walk-forward, full fee model, and an
early/late holdout split at 2025-10-01. exit_thr=0 reproduces the pure
time-based exit as a sanity baseline.
"""

import numpy as np
import pandas as pd

from backtest import get_history
from research_edge import build_features, build_labels, make_model

SYMBOLS = ['XRP-USD', 'ADA-USD', 'SOL-USD']
SPLIT_DATE = pd.Timestamp('2025-10-01')

FEE_PER_SIDE = 0.0021
LEVERAGE = 2.0
MARGIN_OPEN_FEE = 0.0002
ROLLOVER_FEE = 0.0002
HORIZON = 48
BUY_THR = 0.65
TRAIN_MIN = 4000
RETRAIN_EVERY = 720


def backtest_adaptive(data: pd.DataFrame, exit_thr: float) -> list:
    """Walk-forward, long-only, time-boxed hold with optional early model exit."""
    feats = build_features(data)
    labels, _ = build_labels(data, HORIZON)
    X = feats.values
    y = labels.values
    close = data['Close'].values
    n = len(data)
    valid_feat = ~np.isnan(X).any(axis=1)

    fee_cost_pct = FEE_PER_SIDE * 100 * LEVERAGE * 2

    model = None
    last_train_pos = -10**9
    trades = []
    i = TRAIN_MIN

    while i < n - 1:
        if not valid_feat[i]:
            i += 1
            continue

        if model is None or (i - last_train_pos) >= RETRAIN_EVERY:
            train_upto = i - HORIZON
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
        if prob_up <= BUY_THR:
            i += 1
            continue

        entry_price = close[i]
        exit_idx = min(i + HORIZON, n - 1)
        exit_reason = 'time'

        # During the hold, ask the model again each bar. Below exit_thr = bail.
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


def main() -> None:
    print("Downloading data...")
    data_by_symbol = {s: get_history(s, '720d', '1h') for s in SYMBOLS}

    print(f"\nEntry fixed at thr={BUY_THR}, h={HORIZON}, long-only, honest fees. "
          f"Varying the early-exit threshold.\n")
    for exit_thr in (0.0, 0.35, 0.40, 0.45, 0.50):
        trades = []
        for s in SYMBOLS:
            trades.extend(backtest_adaptive(data_by_symbol[s], exit_thr))
        early = [t for t in trades if pd.Timestamp(t['entry_time']) < SPLIT_DATE]
        late = [t for t in trades if pd.Timestamp(t['entry_time']) >= SPLIT_DATE]
        o, e, l = describe(trades), describe(early), describe(late)
        n_early_exits = sum(1 for t in trades if t['exit_reason'] == 'model_exit')
        total = sum(t['net_pnl_pct'] for t in trades)
        label = 'time-only (baseline)' if exit_thr == 0 else f'exit if p<{exit_thr:.2f}'
        flag = 'ROBUST' if (e['n'] and l['n'] and e['exp'] > 0 and l['exp'] > 0) else '  -   '
        print(f"  {label:22s} n={o['n']:4d} exp={o['exp']:+6.2f}% pf={o['pf']:5.2f} "
              f"total={total:+7.1f}% early_exits={n_early_exits:3d} | "
              f"early exp={e['exp']:+6.2f}% | late exp={l['exp']:+6.2f}% | {flag}")


if __name__ == "__main__":
    main()
