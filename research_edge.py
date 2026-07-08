"""
EDGE RESEARCH (Step 1 of "build a real edge")

The big question
----------------
Our backtests showed the current swing strategy is a coin flip (no edge). Before
we invest in a new strategy, we need to answer honestly:

    "Is there ANY predictive signal in this price data that a model can learn?"

This script builds standard technical features, trains a simple model to predict
whether price goes UP over the next few hours, and evaluates it *out-of-sample*
using a walk-forward (time-ordered) split. It also checks whether any edge
survives trading fees.

Why walk-forward / time-ordered?
--------------------------------
With time series you must NEVER shuffle rows or test on the past using a model
trained on the future. That "leaks" the answer and produces fake, over-optimistic
results. We always train on earlier data and test on strictly later data.

How to read the result
-----------------------
- AUC (area under ROC curve): 0.50 = no skill (coin flip). > 0.55 out-of-sample
  and consistent across coins = a real, if small, edge worth building on.
- Net expectancy after fees: if it's clearly positive out-of-sample, we have
  something. If not, the honest conclusion is these features hold no edge and we
  need different data/features, not more tuning.

This script does NOT touch the live bot. It only measures.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

# Reuse the exact same data download the bot/backtester uses.
from backtest import get_history

warnings.filterwarnings("ignore")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw OHLCV candles into model inputs (features). Every feature uses ONLY
    past/current information at each row, so there is no look-ahead leakage.
    """
    df = pd.DataFrame(index=data.index)
    close = data['Close']
    ret = close.pct_change()

    # Past returns over several horizons (momentum at different speeds).
    for n in (1, 3, 6, 12, 24):
        df[f'ret_{n}'] = close.pct_change(n)

    # RSI (14): momentum oscillator, 0-100.
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Distance from moving averages (where are we vs the trend?).
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    df['dist_ema50'] = (close - ema_50) / ema_50
    df['dist_ema200'] = (close - ema_200) / ema_200
    df['ema50_vs_200'] = (ema_50 - ema_200) / ema_200

    # Volatility: how choppy has it been recently?
    df['volatility_14'] = ret.rolling(14).std()

    # Volume pressure: current volume vs its recent average.
    vol_ma = data['Volume'].rolling(20).mean()
    df['volume_ratio'] = data['Volume'] / vol_ma

    # Candle range as a fraction of price.
    df['range'] = (data['High'] - data['Low']) / close

    # Bollinger position: where is price inside its band (0=lower, 1=upper)?
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pos'] = (close - (sma20 - 2 * std20)) / (4 * std20)

    return df


def build_labels(data: pd.DataFrame, horizon: int) -> pd.Series:
    """
    The thing we try to predict: will price be higher `horizon` bars from now?
    1 = up, 0 = down. This is the FUTURE, so it's only used as the training
    target, never as a feature.
    """
    future_return = data['Close'].shift(-horizon) / data['Close'] - 1
    labels = pd.Series(np.where(future_return > 0, 1.0, 0.0), index=data.index)
    labels[future_return.isna()] = np.nan
    return labels, future_return


def evaluate_symbol(symbol: str, data: pd.DataFrame, horizon: int,
                    fee_roundtrip: float, model_name: str, n_splits: int) -> dict:
    """Train + walk-forward test one coin, returning its out-of-sample scores."""
    features = build_features(data)
    labels, future_return = build_labels(data, horizon)

    # Align everything and drop rows with missing values (warmup + last `horizon`).
    dataset = features.copy()
    dataset['label'] = labels
    dataset['future_return'] = future_return
    dataset = dataset.dropna()

    if len(dataset) < 500:
        return {'symbol': symbol, 'error': 'not enough data'}

    feature_cols = [c for c in features.columns]
    X = dataset[feature_cols].values
    y = dataset['label'].values
    fwd = dataset['future_return'].values

    # Walk-forward: each fold trains on the past, tests on the next chunk.
    splitter = TimeSeriesSplit(n_splits=n_splits)

    oos_auc = []
    oos_acc = []
    strat_returns = []  # net returns of "trade when model is confident"

    for train_idx, test_idx in splitter.split(X):
        model = make_model(model_name)
        model.fit(X[train_idx], y[train_idx])

        proba = model.predict_proba(X[test_idx])[:, 1]
        preds = (proba > 0.5).astype(int)

        # AUC needs both classes present in the test fold.
        if len(np.unique(y[test_idx])) > 1:
            oos_auc.append(roc_auc_score(y[test_idx], proba))
        oos_acc.append(accuracy_score(y[test_idx], preds))

        # Simple confidence-gated strategy on the test fold:
        #   go long when prob > 0.55, short when prob < 0.45, else stand aside.
        # Return per trade = forward return in our direction, minus fees.
        long_mask = proba > 0.55
        short_mask = proba < 0.45
        trade_ret = np.where(long_mask, fwd[test_idx],
                     np.where(short_mask, -fwd[test_idx], np.nan))
        trade_ret = trade_ret[~np.isnan(trade_ret)] - fee_roundtrip
        if len(trade_ret) > 0:
            strat_returns.append(trade_ret)

    all_trades = np.concatenate(strat_returns) if strat_returns else np.array([])

    return {
        'symbol': symbol,
        'rows': len(dataset),
        'auc': float(np.mean(oos_auc)) if oos_auc else float('nan'),
        'acc': float(np.mean(oos_acc)) if oos_acc else float('nan'),
        'n_trades': int(len(all_trades)),
        'net_expectancy_pct': float(np.mean(all_trades) * 100) if len(all_trades) else float('nan'),
        'win_rate_pct': float((all_trades > 0).mean() * 100) if len(all_trades) else float('nan'),
    }


def make_model(model_name: str):
    """Create a fresh model. Logistic = simple/fast; GBM = captures nonlinearity."""
    if model_name == 'gbm':
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8
        )
    # Default: logistic regression with feature scaling.
    return Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=0.5)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether a learnable price edge exists.")
    parser.add_argument('--symbols', default='BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD')
    parser.add_argument('--period', default='720d', help="History window (more is better for ML).")
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--horizon', type=int, default=6, help="Predict return this many bars ahead.")
    parser.add_argument('--fee', type=float, default=0.002,
                        help="Round-trip fee as a fraction (0.002 = 0.2%%, ~maker both sides).")
    parser.add_argument('--model', choices=['logistic', 'gbm'], default='logistic')
    parser.add_argument('--splits', type=int, default=5, help="Walk-forward folds.")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    print(f"Edge research | model={args.model} | horizon={args.horizon} bars | "
          f"period={args.period} | fee_roundtrip={args.fee * 100:.2f}%")
    print("Downloading data and running walk-forward evaluation...\n")

    results = []
    for symbol in symbols:
        data = get_history(symbol, args.period, args.interval)
        if data.empty:
            print(f"  {symbol}: no data, skipping")
            continue
        res = evaluate_symbol(symbol, data, args.horizon, args.fee, args.model, args.splits)
        results.append(res)
        if 'error' in res:
            print(f"  {symbol}: {res['error']}")
        else:
            print(f"  {symbol:9s} rows={res['rows']:5d}  OOS AUC={res['auc']:.3f}  "
                  f"acc={res['acc'] * 100:4.1f}%  trades={res['n_trades']:5d}  "
                  f"net exp/trade={res['net_expectancy_pct']:+.3f}%  "
                  f"win={res['win_rate_pct']:4.1f}%")

    valid = [r for r in results if 'error' not in r and not np.isnan(r['auc'])]
    if not valid:
        print("\nNo valid results.")
        return

    mean_auc = np.mean([r['auc'] for r in valid])
    mean_exp = np.nanmean([r['net_expectancy_pct'] for r in valid])

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Average out-of-sample AUC:      {mean_auc:.3f}   (0.50 = no skill)")
    print(f"Average net expectancy/trade:   {mean_exp:+.3f}%  (after {args.fee * 100:.2f}% fees)")
    if mean_auc >= 0.55 and mean_exp > 0:
        print("=> Promising: a clear, real edge to build a strategy on.")
    elif mean_auc > 0.51 and mean_exp > 0:
        print("=> Weak but POSITIVE edge after fees. Worth building into a proper")
        print("   (non-overlapping) backtest to confirm it is tradeable.")
    elif mean_auc >= 0.53:
        print("=> Weak signal but not profitable after fees at this horizon.")
    else:
        print("=> No usable edge in these features. Tuning the old strategy won't help;")
        print("   we need different inputs (e.g. order-flow, cross-asset, alt-data).")
    print("=" * 70)


if __name__ == "__main__":
    main()
