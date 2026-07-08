"""
BASELINE on CryptoCompare long history (research only, does NOT touch live).

Runs the EXACT current live configuration (ml_live_trade.py settings) over the
full cached CryptoCompare hourly history and reports expectancy by calendar year.
This tells us honestly how the strategy the bot trades today would have done
across every past market regime (2018 bear, 2020 crash, 2021 mania, 2022 bear...).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import ml_strategy_backtest as msb

CACHE_DIR = Path('data_cc')

# ── Current live configuration (mirrors ml_live_trade.py) ──
HORIZON = 48
BUY_THR = 0.65
SELL_THR = 0.0        # long-only
FEE_PER_SIDE = 0.0021
LEVERAGE = 2.0
MODEL = 'logistic'
TRAIN_MIN = 4000
RETRAIN_EVERY = 720


def load_cached(coin: str) -> pd.DataFrame:
    """Read one coin's cached CryptoCompare hourly CSV."""
    df = pd.read_csv(CACHE_DIR / f"{coin}_1h.csv", index_col=0, parse_dates=True)
    return df


def run(symbols_map: dict) -> pd.DataFrame:
    """Backtest every cached coin with the live config; return pooled trades."""
    all_trades = []
    for label, coin in symbols_map.items():
        df = load_cached(coin)
        print(f"  {coin}: {len(df)} bars ({df.index.min().date()} .. {df.index.max().date()})")
        r = msb.backtest_symbol(
            label, df, horizon=HORIZON, buy_thr=BUY_THR, sell_thr=SELL_THR,
            fee_rate=FEE_PER_SIDE, leverage=LEVERAGE, model_name=MODEL,
            train_min=TRAIN_MIN, retrain_every=RETRAIN_EVERY,
            atr_stop_mult=0.0, atr_period=14,
        )
        t = r.get('trades', [])
        all_trades.extend(t)
        p = np.array([x['net_pnl_pct'] for x in t]) if t else np.array([])
        print(f"    -> trades={len(t):4d}  exp/trade={p.mean() if len(p) else float('nan'):+6.2f}%  "
              f"win={(p > 0).mean() * 100 if len(p) else float('nan'):4.1f}%  "
              f"total={r.get('total_return', float('nan')):+8.1f}%  "
              f"(buy&hold {r.get('bh_return', float('nan')):+.0f}%)  maxDD={r.get('max_dd', float('nan')):.1f}%")
    return pd.DataFrame(all_trades)


def report_by_year(df_t: pd.DataFrame) -> None:
    """Print pooled expectancy per calendar year - the key robustness check."""
    if df_t.empty:
        print("No trades.")
        return
    df_t = df_t.copy()
    df_t['year'] = pd.to_datetime(df_t['entry_time']).dt.year
    print("\n── Expectancy by calendar year (all coins pooled) ──")
    print(f"  {'year':6s} {'trades':>6s} {'exp/trade':>10s} {'win%':>6s} {'PF':>6s}")
    for year, g in df_t.groupby('year'):
        p = g['net_pnl_pct'].values
        losses = p[p <= 0]
        pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
        print(f"  {year:<6d} {len(p):6d} {p.mean():+9.2f}% {(p > 0).mean() * 100:5.1f}% "
              f"{'inf' if pf == float('inf') else f'{pf:6.2f}'}")
    p = df_t['net_pnl_pct'].values
    losses = p[p <= 0]
    pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
    print(f"  {'ALL':6s} {len(p):6d} {p.mean():+9.2f}% {(p > 0).mean() * 100:5.1f}% "
          f"{'inf' if pf == float('inf') else f'{pf:6.2f}'}")


if __name__ == "__main__":
    symbols = {'XRP-USD': 'XRP', 'ADA-USD': 'ADA'}
    print(f"BASELINE (live config: long-only, h={HORIZON}, thr={BUY_THR}, {LEVERAGE:.0f}x, "
          f"fee={FEE_PER_SIDE * 100:.2f}%/side, model={MODEL})\n")
    trades = run(symbols)
    report_by_year(trades)
