"""
LONGER-HORIZON ML STRATEGY - HONEST WALK-FORWARD BACKTEST (Step 2 of "build a real edge")

What research_edge.py told us
-----------------------------
There is no edge at the 1h timeframe the current bot trades, but a small, real
edge appears when predicting direction ~48 hours ahead (especially on XRP/ADA/SOL).

What this script does
---------------------
Turns that finding into an actual tradeable strategy and tests it *properly*:

1. Walk-forward: the model is retrained only on PAST data, then makes decisions
   on strictly later data. It is retrained periodically (like you would in real
   life), never using the future.
2. Non-overlapping trades: only one position at a time. Enter when the model is
   confident, hold for a fixed number of bars (a few days), then exit and look
   for the next trade. This is realistic and avoids the overlapping-window
   inflation that can flatter research numbers.
3. Realistic costs: maker fee on both sides, scaled by leverage. Optional ATR
   stop-loss to cut losing trades early.

It does NOT touch the live bot. It tells us whether this strategy is worth
promoting to live trading.
"""

import argparse
import numpy as np
import pandas as pd

# Reuse the same data + feature code so research and backtest stay consistent.
from backtest import get_history, compute_atr
from research_edge import build_features, build_labels, make_model


def backtest_symbol(
    symbol: str,
    data: pd.DataFrame,
    horizon: int,
    buy_thr: float,
    sell_thr: float,
    fee_rate: float,
    leverage: float,
    model_name: str,
    train_min: int,
    retrain_every: int,
    atr_stop_mult: float,
    atr_period: int,
    starting_equity: float = 1000.0,
) -> dict:
    """Run the walk-forward backtest for one coin and return its stats."""
    feats = build_features(data)
    labels, _fwd = build_labels(data, horizon)

    feature_cols = list(feats.columns)
    X_all = feats.values
    y_all = labels.values
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    atr_all = compute_atr(data, atr_period).values if atr_stop_mult > 0 else None

    n = len(data)
    # A row is usable as a feature only if none of its features are NaN.
    valid_feat = ~np.isnan(X_all).any(axis=1)

    fee_cost_pct = fee_rate * 100 * leverage * 2  # round-trip fee, % of margin

    model = None
    last_train_pos = -10**9

    trades = []
    equity = starting_equity
    equity_curve = [equity]

    # Start once we have enough history to train the first model.
    i = train_min
    oos_start_price = close[i] if i < n else close[-1]

    while i < n - 1:
        if not valid_feat[i]:
            i += 1
            continue

        # ---- Retrain periodically, using ONLY data whose label is known ----
        # A sample at position j only has a valid 48h label once we are at least
        # `horizon` bars past it, so we train on rows with index <= i - horizon.
        if model is None or (i - last_train_pos) >= retrain_every:
            train_upto = i - horizon
            if train_upto > train_min // 2:
                mask = np.zeros(n, dtype=bool)
                mask[:train_upto + 1] = True
                mask &= valid_feat & ~np.isnan(y_all)
                if mask.sum() >= 300 and len(np.unique(y_all[mask])) > 1:
                    model = make_model(model_name)
                    model.fit(X_all[mask], y_all[mask])
                    last_train_pos = i

        if model is None:
            i += 1
            continue

        # ---- Decide whether to enter ----
        prob_up = float(model.predict_proba(X_all[i:i + 1])[:, 1][0])
        if prob_up > buy_thr:
            direction = 'long'
        elif prob_up < sell_thr:
            direction = 'short'
        else:
            i += 1
            equity_curve.append(equity)
            continue

        entry_price = close[i]
        entry_atr = atr_all[i] if atr_all is not None else None

        # Precompute stop level (optional).
        if entry_atr is not None and not np.isnan(entry_atr) and entry_atr > 0:
            if direction == 'long':
                stop_price = entry_price - atr_stop_mult * entry_atr
            else:
                stop_price = entry_price + atr_stop_mult * entry_atr
        else:
            stop_price = None

        # ---- Hold for up to `horizon` bars; exit on stop or at the time limit ----
        exit_idx = min(i + horizon, n - 1)
        exit_price = close[exit_idx]
        exit_reason = 'time'
        for j in range(i + 1, exit_idx + 1):
            if stop_price is not None:
                if direction == 'long' and low[j] <= stop_price:
                    exit_price, exit_idx, exit_reason = stop_price, j, 'stop'
                    break
                if direction == 'short' and high[j] >= stop_price:
                    exit_price, exit_idx, exit_reason = stop_price, j, 'stop'
                    break

        if direction == 'long':
            gross_pct = (exit_price - entry_price) / entry_price * 100 * leverage
        else:
            gross_pct = (entry_price - exit_price) / entry_price * 100 * leverage
        net_pct = gross_pct - fee_cost_pct

        equity *= (1 + net_pct / 100)
        equity_curve.append(equity)

        trades.append({
            'symbol': symbol,
            'direction': direction,
            'prob_up': round(prob_up, 3),
            'entry_time': data.index[i],
            'exit_time': data.index[exit_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'net_pnl_pct': net_pct,
            'exit_reason': exit_reason,
            'bars_held': exit_idx - i,
        })

        i = exit_idx + 1  # non-overlapping: resume after the trade closes

    # Buy & hold benchmark over the same out-of-sample window.
    bh_return = (close[-1] / oos_start_price - 1) * 100

    return summarize(symbol, trades, equity_curve, starting_equity, bh_return)


def summarize(symbol, trades, equity_curve, starting_equity, bh_return) -> dict:
    """Compute the headline stats for one coin."""
    if not trades:
        return {'symbol': symbol, 'n_trades': 0, 'bh_return': bh_return}

    pnls = np.array([t['net_pnl_pct'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    peak = starting_equity
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak * 100 if peak > 0 else 0.0)

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    return {
        'symbol': symbol,
        'n_trades': len(trades),
        'win_rate': (pnls > 0).mean() * 100,
        'avg_win': wins.mean() if len(wins) else 0.0,
        'avg_loss': losses.mean() if len(losses) else 0.0,
        'expectancy': pnls.mean(),
        'profit_factor': (gross_win / gross_loss) if gross_loss else float('inf'),
        'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
        'max_dd': max_dd,
        'bh_return': bh_return,
        'trades': trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the longer-horizon ML strategy.")
    parser.add_argument('--symbols', default='XRP-USD,ADA-USD,SOL-USD,BTC-USD,ETH-USD')
    parser.add_argument('--period', default='720d')
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--horizon', type=int, default=48, help="Hold this many bars (~2 days at 1h).")
    parser.add_argument('--buy-thr', type=float, default=0.55)
    parser.add_argument('--sell-thr', type=float, default=0.45)
    parser.add_argument('--fee', type=float, default=0.001, help="Maker fee per side (0.001 = 0.10%%).")
    parser.add_argument('--leverage', type=float, default=2.0)
    parser.add_argument('--model', choices=['logistic', 'gbm'], default='logistic')
    parser.add_argument('--train-min', type=int, default=4000, help="Bars before trading starts.")
    parser.add_argument('--retrain-every', type=int, default=720, help="Retrain cadence in bars.")
    parser.add_argument('--atr-stop-mult', type=float, default=0.0, help="ATR stop distance (0 = off).")
    parser.add_argument('--atr-period', type=int, default=14)
    parser.add_argument('--out', default='ml_strategy_trades.csv')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    print(f"ML strategy backtest | model={args.model} | horizon={args.horizon}b | "
          f"lev={args.leverage}x | maker fee={args.fee * 100:.2f}%/side | "
          f"buy>{args.buy_thr} sell<{args.sell_thr} | atr_stop={args.atr_stop_mult or 'off'}")
    print("Running walk-forward (retrain on past only)...\n")

    results = []
    all_trades = []
    for symbol in symbols:
        data = get_history(symbol, args.period, args.interval)
        if data.empty or len(data) <= args.train_min + args.horizon + 50:
            print(f"  {symbol}: not enough data, skipping")
            continue
        r = backtest_symbol(
            symbol, data, args.horizon, args.buy_thr, args.sell_thr, args.fee,
            args.leverage, args.model, args.train_min, args.retrain_every,
            args.atr_stop_mult, args.atr_period,
        )
        results.append(r)
        all_trades.extend(r.get('trades', []))
        if r['n_trades'] == 0:
            print(f"  {symbol:9s} no trades (model never confident enough)")
        else:
            print(f"  {symbol:9s} trades={r['n_trades']:4d}  win={r['win_rate']:4.1f}%  "
                  f"exp/trade={r['expectancy']:+.2f}%  PF={r['profit_factor']:.2f}  "
                  f"return={r['total_return']:+7.1f}%  (buy&hold {r['bh_return']:+.1f}%)  "
                  f"maxDD={r['max_dd']:.1f}%")

    pooled = [t['net_pnl_pct'] for t in all_trades]
    print("\n" + "=" * 70)
    print("OVERALL (all coins pooled)")
    print("=" * 70)
    if not pooled:
        print("  No trades. Try lowering the confidence thresholds.")
        return
    pooled = np.array(pooled)
    wins = pooled[pooled > 0]
    losses = pooled[pooled <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float('inf')
    print(f"  Trades:        {len(pooled)}")
    print(f"  Win rate:      {(pooled > 0).mean() * 100:.1f}%")
    print(f"  Expectancy:    {pooled.mean():+.2f}% per trade (after fees)")
    print(f"  Profit factor: {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    reasons = {}
    for t in all_trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
    print("  Exit reasons:  " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print("=" * 70)

    # ---- Temporal robustness: does the edge hold in EACH time window, or is it
    # one lucky period? We split all trades into 3 equal time windows by entry
    # date and report each. A real edge should be positive in most windows. ----
    sorted_trades = sorted(all_trades, key=lambda t: t['entry_time'])
    if len(sorted_trades) >= 30:
        print("\nTemporal robustness (trades split into 3 time windows):")
        third = len(sorted_trades) // 3
        windows = [sorted_trades[:third], sorted_trades[third:2 * third], sorted_trades[2 * third:]]
        for w_idx, w in enumerate(windows, 1):
            if not w:
                continue
            wp = np.array([t['net_pnl_pct'] for t in w])
            start = pd.Timestamp(w[0]['entry_time']).date()
            end = pd.Timestamp(w[-1]['entry_time']).date()
            wl = wp[wp <= 0]
            wpf = (wp[wp > 0].sum() / abs(wl.sum())) if len(wl) and wl.sum() != 0 else float('inf')
            print(f"  Window {w_idx} [{start}..{end}]: n={len(w):4d}  "
                  f"win={ (wp > 0).mean() * 100:4.1f}%  exp={wp.mean():+.2f}%  "
                  f"PF={'inf' if wpf == float('inf') else f'{wpf:.2f}'}")
        print("=" * 70)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(args.out, index=False)
        print(f"\nSaved {len(all_trades)} trades to {args.out}")


if __name__ == "__main__":
    main()
