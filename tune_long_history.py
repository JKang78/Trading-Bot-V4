"""
LONG-HISTORY VALIDATION via CryptoCompare (research only)

yfinance only provides ~2 years of hourly candles, so everything we validated
so far comes from 2025-2026. CryptoCompare's hourly archive goes back to 2018,
which lets us ask the most important question in this whole project:

    Does the strategy's edge survive COMPLETELY different market regimes -
    the 2018/2022 bear markets, the 2020 crash, the 2021 mania?

An edge that only exists in one regime is a bet on that regime continuing.
An edge that holds across regimes is much more likely to be real.

Downloads are cached in data_cc/ so we only pay the (rate-limited) download
cost once. Output: per-year expectancy of the exact live configuration.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import ml_strategy_backtest as msb

CACHE_DIR = Path('data_cc')
API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', '')
SYMBOLS = {'XRP-USD': 'XRP', 'ADA-USD': 'ADA', 'SOL-USD': 'SOL',
           'LINK-USD': 'LINK', 'DOGE-USD': 'DOGE'}

# Live configuration (what ml_live_trade.py runs).
HORIZON = 48
BUY_THR = 0.65
FEE_PER_SIDE = 0.0021
LEVERAGE = 2.0


def fetch_cc_hourly(coin: str, start_ts: int = 1514764800) -> pd.DataFrame:
    """
    Download full hourly OHLCV history for one coin from CryptoCompare,
    paging backwards 2000 bars at a time. Cached to CSV after first download.
    start_ts default = 2018-01-01.
    """
    cache = CACHE_DIR / f"{coin}_1h.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  {coin}: {len(df)} bars from cache ({df.index.min().date()} .. {df.index.max().date()})")
        return df

    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    to_ts = int(time.time())
    rate_limit_waits = 0
    max_rate_limit_waits = 6
    while to_ts > start_ts:
        r = requests.get(
            'https://min-api.cryptocompare.com/data/v2/histohour',
            params={'fsym': coin, 'tsym': 'USD', 'limit': 2000,
                    'toTs': to_ts, 'api_key': API_KEY},
            timeout=30).json()
        if r.get('Response') != 'Success':
            msg = r.get('Message', '')
            if 'rate limit' in msg.lower():
                rate_limit_waits += 1
                if rate_limit_waits >= max_rate_limit_waits:
                    print(f"  {coin}: rate limit persists after {max_rate_limit_waits} waits, stopping download", flush=True)
                    break
                # The free tier has an HOURLY quota, and every retry burns more
                # of it. Back off hard and wait for the window to reset.
                print(f"  {coin}: hourly rate limit hit ({rate_limit_waits}/{max_rate_limit_waits}), sleeping 10 min...", flush=True)
                time.sleep(600)
                continue
            print(f"  {coin}: API error: {msg[:120]}")
            break
        rows = r['Data']['Data']
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        oldest = rows[0]['time']
        # All-zero chunks mean we've paged past the coin's first listing.
        if all(x['close'] == 0 for x in rows):
            break
        to_ts = oldest - 3600
        time.sleep(1.2)  # stay under the free-tier rate limit

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).drop_duplicates('time').sort_values('time')
    df = df[df['close'] > 0]
    df.index = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volumefrom': 'Volume'})
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.to_csv(cache)
    print(f"  {coin}: downloaded {len(df)} bars ({df.index.min().date()} .. {df.index.max().date()})")
    return df


def main() -> None:
    print("Fetching long hourly history from CryptoCompare (cached after first run)...")
    data = {}
    for yf_sym, cc_sym in SYMBOLS.items():
        df = fetch_cc_hourly(cc_sym)
        if len(df) > 6000:
            data[yf_sym] = df

    print(f"\nRunning walk-forward (live config: long-only, h={HORIZON}, thr={BUY_THR}, "
          f"{LEVERAGE:.0f}x, honest fees) over the full history...\n")

    all_trades = []
    for sym, df in data.items():
        r = msb.backtest_symbol(
            sym, df, horizon=HORIZON, buy_thr=BUY_THR, sell_thr=0.0,
            fee_rate=FEE_PER_SIDE, leverage=LEVERAGE, model_name='logistic',
            train_min=4000, retrain_every=720, atr_stop_mult=0.0, atr_period=14,
        )
        t = r.get('trades', [])
        all_trades.extend(t)
        p = np.array([x['net_pnl_pct'] for x in t]) if t else np.array([])
        print(f"  {sym:9s} trades={len(t):4d}  exp={p.mean() if len(p) else float('nan'):+6.2f}%  "
              f"win={(p > 0).mean() * 100 if len(p) else float('nan'):4.1f}%")

    if not all_trades:
        print("No trades.")
        return

    df_t = pd.DataFrame(all_trades)
    df_t['year'] = pd.to_datetime(df_t['entry_time']).dt.year
    df_t.to_csv('long_history_trades.csv', index=False)

    print("\n── Expectancy by calendar year (all coins pooled) ──")
    print(f"  {'year':6s} {'trades':>6s} {'exp/trade':>9s} {'win%':>6s} {'PF':>6s}")
    for year, g in df_t.groupby('year'):
        p = g['net_pnl_pct'].values
        losses = p[p <= 0]
        pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
        print(f"  {year:<6d} {len(p):6d} {p.mean():+8.2f}% {(p > 0).mean() * 100:5.1f}% "
              f"{'inf' if pf == float('inf') else f'{pf:6.2f}'}")

    p = df_t['net_pnl_pct'].values
    losses = p[p <= 0]
    pf = p[p > 0].sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float('inf')
    print(f"\n  ALL    {len(p):6d} {p.mean():+8.2f}% {(p > 0).mean() * 100:5.1f}% {pf:6.2f}")
    print("\nSaved trades to long_history_trades.csv")


if __name__ == "__main__":
    main()
