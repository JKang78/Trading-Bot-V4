"""
Run ML V2 walk-forward backtest on long hourly history.

Uses CryptoCompare cache when available, otherwise falls back to Binance
hourly klines (research only — no API key, no rate-limit issues).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import ml_strategy_backtest as msb
from ml_strategy import V2_PROFILE, KrakenCostModel
from tune_long_history import CACHE_DIR, SYMBOLS, fetch_cc_hourly

OUT = Path("ml_v2_extended.csv")
LOG = Path("ml_v2_extended_run.log")
BINANCE_CACHE_DIR = Path("data_binance")

# yfinance symbol -> Binance USDT pair
BINANCE_SYMBOLS = {
    "XRP-USD": "XRPUSDT",
    "ADA-USD": "ADAUSDT",
    "SOL-USD": "SOLUSDT",
    "LINK-USD": "LINKUSDT",
    "DOGE-USD": "DOGEUSDT",
}


def log(msg: str) -> None:
    """Print and append to the run log."""
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def fetch_binance_hourly(symbol: str, start_year: int = 2018) -> pd.DataFrame:
    """Download hourly OHLCV from Binance and cache to CSV."""
    cache = BINANCE_CACHE_DIR / f"{symbol}_1h.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        log(f"  {symbol}: {len(df)} bars from Binance cache ({df.index.min().date()} .. {df.index.max().date()})")
        return df

    BINANCE_CACHE_DIR.mkdir(exist_ok=True)
    start_ms = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    frames: list[pd.DataFrame] = []

    while True:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 1000, "startTime": start_ms},
            timeout=30,
        )
        rows = response.json()
        if not rows or isinstance(rows, dict):
            break

        batch = pd.DataFrame(
            rows,
            columns=[
                "open_time", "Open", "High", "Low", "Close", "Volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ],
        )
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            batch[col] = batch[col].astype(float)
        batch.index = pd.to_datetime(batch["open_time"], unit="ms")
        frames.append(batch[["Open", "High", "Low", "Close", "Volume"]])

        start_ms = int(rows[-1][0]) + 1
        if len(rows) < 1000:
            break
        time.sleep(0.05)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(cache)
    log(f"  {symbol}: downloaded {len(df)} bars from Binance ({df.index.min().date()} .. {df.index.max().date()})")
    return df


def load_hourly_history(yf_sym: str, cc_sym: str) -> pd.DataFrame:
    """Prefer CryptoCompare cache; fall back to Binance for missing coins."""
    cc_cache = CACHE_DIR / f"{cc_sym}_1h.csv"
    if cc_cache.exists():
        return fetch_cc_hourly(cc_sym)

    log(f"  {yf_sym}: no CryptoCompare cache, using Binance instead...")
    return fetch_binance_hourly(BINANCE_SYMBOLS[yf_sym])


def portfolio_by_year(
    trades: pd.DataFrame,
    position_frac: float = 0.25,
    start_capital: float = 1000.0,
) -> None:
    """Print year-by-year portfolio results using fixed position sizing."""
    log(f"\n── Portfolio simulation ($1000 start, {position_frac:.0%} per trade) ──")
    for year in sorted(trades["year"].unique()):
        year_trades = trades[trades["year"] == year].sort_values("entry_time")
        equity = start_capital
        min_equity = equity
        for pnl in year_trades["net_pnl_pct"]:
            equity *= 1 + (pnl / 100) * position_frac
            min_equity = min(min_equity, equity)
        log(
            f"  {year}: ending ${equity:,.0f}  profit ${equity - start_capital:+,.0f}  "
            f"worst dip ${min_equity:,.0f}"
        )

    equity = start_capital
    for pnl in trades.sort_values("entry_time")["net_pnl_pct"]:
        equity *= 1 + (pnl / 100) * position_frac
    log(f"  ALL YEARS: ending ${equity:,.0f}  profit ${equity - start_capital:+,.0f}")


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    profile = V2_PROFILE
    cost_model = KrakenCostModel(
        maker_entry_fee=0.0023,
        taker_entry_fee=0.0040,
        taker_exit_fee=0.0040,
        margin_open_fee=profile.margin_open_fee,
        margin_rollover_fee_4h=profile.rollover_fee_4h,
        spread_buffer=0.0005,
        slippage_buffer=0.0010,
        minimum_edge=profile.minimum_edge,
    )

    log("Loading long hourly history (CryptoCompare cache or Binance fallback)...")
    data: dict[str, pd.DataFrame] = {}
    min_bars = profile.horizon + 4050
    for yf_sym, cc_sym in SYMBOLS.items():
        df = load_hourly_history(yf_sym, cc_sym)
        if len(df) > min_bars:
            data[yf_sym] = df
            log(f"  {yf_sym}: using {len(df)} bars ({df.index.min().date()} .. {df.index.max().date()})")
        else:
            log(f"  {yf_sym}: skipped, only {len(df)} bars")

    if not data:
        log("No symbols with enough history.")
        return 1

    log(f"\nRunning ML V2 walk-forward on {len(data)} symbols...")
    log(
        f"  horizon={profile.horizon}, buy_thr={profile.buy_thr}, long_only, "
        f"fng=on, leverage=2x"
    )

    all_trades: list[dict] = []
    for sym, df in data.items():
        result = msb.backtest_symbol(
            sym,
            df,
            horizon=profile.horizon,
            buy_thr=profile.buy_thr,
            sell_thr=0.0,
            fee_rate=0.0023,
            leverage=2.0,
            model_name="logistic",
            train_min=4000,
            retrain_every=720,
            atr_stop_mult=0.0,
            atr_period=14,
            exit_thr=profile.exit_thr,
            use_fng_features=profile.use_fng_features,
            use_fng_filter=profile.use_fng_filter,
            margin_open_fee=profile.margin_open_fee,
            rollover_fee=profile.rollover_fee_4h,
            cost_model=cost_model,
            use_cost_aware_labels=profile.use_cost_aware_labels,
            entry_fee_rate=0.0023,
            exit_fee_rate=0.0040,
        )
        trades = result.get("trades", [])
        all_trades.extend(trades)
        pnls = np.array([t["net_pnl_pct"] for t in trades]) if trades else np.array([])
        first = str(trades[0]["entry_time"])[:10] if trades else "n/a"
        last = str(trades[-1]["entry_time"])[:10] if trades else "n/a"
        log(
            f"  {sym:9s} trades={len(trades):4d}  exp={pnls.mean() if len(pnls) else float('nan'):+6.2f}%  "
            f"win={(pnls > 0).mean() * 100 if len(pnls) else float('nan'):4.1f}%  "
            f"range={first} .. {last}"
        )

    if not all_trades:
        log("No trades generated.")
        return 1

    trades_df = pd.DataFrame(all_trades)
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df = trades_df.sort_values("entry_time")
    trades_df["year"] = trades_df["entry_time"].dt.year
    trades_df.to_csv(OUT, index=False)

    log(f"\nSaved {len(trades_df)} trades to {OUT}")
    log(f"Date range: {trades_df['entry_time'].min().date()} to {trades_df['entry_time'].max().date()}")

    log("\n── By calendar year (all coins pooled) ──")
    log(f"  {'year':6s} {'trades':>6s} {'win%':>6s} {'exp/trade':>10s} {'PF':>6s} {'sum_pnl':>9s}")
    for year, group in trades_df.groupby("year"):
        pnls = group["net_pnl_pct"].values
        losses = pnls[pnls <= 0]
        if len(losses) and losses.sum() != 0:
            pf = pnls[pnls > 0].sum() / abs(losses.sum())
            pf_text = f"{pf:6.2f}"
        else:
            pf_text = "   inf"
        log(
            f"  {year:<6d} {len(pnls):6d} {(pnls > 0).mean() * 100:5.1f}% "
            f"{pnls.mean():+9.2f}% {pf_text:>6s} {pnls.sum():+8.1f}%"
        )

    portfolio_by_year(trades_df)
    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
