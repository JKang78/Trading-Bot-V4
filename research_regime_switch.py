"""
Research only: compare three market-regime policies on the same V2 history.

  1. long_only   - current live style (always allow longs; no bear switch)
  2. cash_bear   - same longs, but skip new entries while BTC regime is weak
  3. short_bear  - longs in non-weak regimes; shorts only while BTC is weak

Uses the same long hourly caches as run_v2_long_history.py and the existing
BTC regime classifier in ml_strategy.py. Does NOT change the live bot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import ml_strategy_backtest as msb
from ml_strategy import V2_PROFILE, KrakenCostModel
from tune_long_history import CACHE_DIR, SYMBOLS, fetch_cc_hourly

OUT_DIR = Path(".")
LOG = Path("research_regime_switch.log")
BTC_CACHE = Path("data_binance/BTCUSDT_1h.csv")
BINANCE_CACHE_DIR = Path("data_binance")
BINANCE_SYMBOLS = {
    "XRP-USD": "XRPUSDT",
    "ADA-USD": "ADAUSDT",
    "SOL-USD": "SOLUSDT",
    "LINK-USD": "LINKUSDT",
    "DOGE-USD": "DOGEUSDT",
}
SELL_THR_SHORT = 0.35
POSITION_FRAC = 0.25
START_CAPITAL = 1000.0


def log(msg: str) -> None:
    """Print and append to the run log."""
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def load_binance_cache(symbol: str) -> pd.DataFrame:
    """Read one Binance hourly CSV from the local cache."""
    cache = BINANCE_CACHE_DIR / f"{symbol}_1h.csv"
    df = pd.read_csv(cache, index_col=0, parse_dates=True)
    return df


def load_hourly_history(yf_sym: str, cc_sym: str) -> pd.DataFrame:
    """Prefer CryptoCompare cache; fall back to Binance cache."""
    cc_cache = CACHE_DIR / f"{cc_sym}_1h.csv"
    if cc_cache.exists():
        return fetch_cc_hourly(cc_sym)
    return load_binance_cache(BINANCE_SYMBOLS[yf_sym])


def load_btc() -> pd.DataFrame:
    """Load cached BTC hourly bars used for the regime classifier."""
    if not BTC_CACHE.exists():
        raise FileNotFoundError(f"Missing BTC cache: {BTC_CACHE}")
    df = pd.read_csv(BTC_CACHE, index_col=0, parse_dates=True)
    log(f"  BTC: {len(df)} bars ({df.index.min().date()} .. {df.index.max().date()})")
    return df


def load_coin_data(profile) -> dict[str, pd.DataFrame]:
    """Load the same altcoin histories used by the V2 extended backtest."""
    data: dict[str, pd.DataFrame] = {}
    min_bars = profile.horizon + 4050
    for yf_sym, cc_sym in SYMBOLS.items():
        df = load_hourly_history(yf_sym, cc_sym)
        if len(df) > min_bars:
            data[yf_sym] = df
            log(f"  {yf_sym}: using {len(df)} bars ({df.index.min().date()} .. {df.index.max().date()})")
        else:
            log(f"  {yf_sym}: skipped, only {len(df)} bars")
    return data


def run_policy(
    name: str,
    data: dict[str, pd.DataFrame],
    btc_data: pd.DataFrame,
    profile,
    cost_model: KrakenCostModel,
    *,
    sell_thr: float,
    bear_policy: str | None,
) -> pd.DataFrame:
    """Walk-forward every coin under one regime policy; return pooled trades."""
    log(f"\n══ Policy: {name} ══")
    all_trades: list[dict] = []
    for sym, df in data.items():
        result = msb.backtest_symbol(
            sym,
            df,
            horizon=profile.horizon,
            buy_thr=profile.buy_thr,
            sell_thr=sell_thr,
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
            btc_data=btc_data,
            cost_model=cost_model,
            use_cost_aware_labels=profile.use_cost_aware_labels,
            entry_fee_rate=0.0023,
            exit_fee_rate=0.0040,
            bear_policy=bear_policy,
        )
        trades = result.get("trades", [])
        for trade in trades:
            trade["policy"] = name
        all_trades.extend(trades)
        pnls = np.array([t["net_pnl_pct"] for t in trades]) if trades else np.array([])
        log(
            f"  {sym:9s} trades={len(trades):4d}  "
            f"exp={pnls.mean() if len(pnls) else float('nan'):+6.2f}%  "
            f"win={(pnls > 0).mean() * 100 if len(pnls) else float('nan'):4.1f}%"
        )

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["year"] = trades_df["entry_time"].dt.year
        trades_df.to_csv(OUT_DIR / f"regime_switch_{name}.csv", index=False)
    return trades_df


def trade_stats(trades: pd.DataFrame) -> dict:
    """Headline expectancy stats for a trade list."""
    if trades.empty:
        return {"n": 0, "exp": float("nan"), "win": float("nan"), "pf": float("nan")}
    pnls = trades["net_pnl_pct"].values
    losses = pnls[pnls <= 0]
    if len(losses) and losses.sum() != 0:
        pf = float(pnls[pnls > 0].sum() / abs(losses.sum()))
    else:
        pf = float("inf")
    return {
        "n": len(pnls),
        "exp": float(pnls.mean()),
        "win": float((pnls > 0).mean() * 100),
        "pf": pf,
    }


def portfolio_sim(trades: pd.DataFrame, start: float = START_CAPITAL) -> dict:
    """Compound $start with fixed fraction sizing, matching the V2 extended sim."""
    equity = start
    min_equity = start
    if trades.empty:
        return {"end": equity, "profit": 0.0, "worst": min_equity, "cagr": 0.0}
    ordered = trades.sort_values("entry_time")
    for pnl in ordered["net_pnl_pct"]:
        equity *= 1 + (pnl / 100.0) * POSITION_FRAC
        min_equity = min(min_equity, equity)
    t0 = ordered["entry_time"].iloc[0]
    t1 = ordered["entry_time"].iloc[-1]
    years = max((t1 - t0).days / 365.25, 1 / 365.25)
    cagr = (equity / start) ** (1 / years) - 1
    return {
        "end": equity,
        "profit": equity - start,
        "worst": min_equity,
        "cagr": cagr,
        "years": years,
    }


def report_policy(name: str, trades: pd.DataFrame) -> None:
    """Print year-by-year and portfolio results for one policy."""
    stats = trade_stats(trades)
    port = portfolio_sim(trades)
    pf_text = "inf" if stats["pf"] == float("inf") else f"{stats['pf']:.2f}"
    log(
        f"\n── {name} summary ──\n"
        f"  trades={stats['n']}  exp/trade={stats['exp']:+.2f}%  "
        f"win={stats['win']:.1f}%  PF={pf_text}\n"
        f"  $1000 -> ${port['end']:,.0f}  profit ${port['profit']:+,.0f}  "
        f"worst dip ${port['worst']:,.0f}  CAGR={port.get('cagr', 0)*100:.1f}%"
    )

    if trades.empty:
        return

    log(f"  {'year':6s} {'trades':>6s} {'win%':>6s} {'exp':>8s} {'$end':>8s} {'profit':>9s}")
    for year, group in trades.groupby("year"):
        s = trade_stats(group)
        # Year sims always restart at $1000 (same as extended V2 report).
        yport = portfolio_sim(group)
        log(
            f"  {year:<6d} {s['n']:6d} {s['win']:5.1f}% {s['exp']:+7.2f}% "
            f"${yport['end']:7,.0f} ${yport['profit']:+8,.0f}"
        )

    if "direction" in trades.columns:
        for direction, group in trades.groupby("direction"):
            s = trade_stats(group)
            log(
                f"  direction={direction:5s} n={s['n']:3d} "
                f"exp={s['exp']:+.2f}% win={s['win']:.1f}%"
            )
    if "btc_regime" in trades.columns:
        for regime, group in trades.groupby("btc_regime"):
            s = trade_stats(group)
            log(
                f"  regime={regime:7s} n={s['n']:3d} "
                f"exp={s['exp']:+.2f}% win={s['win']:.1f}%"
            )


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

    log("Loading histories...")
    # Silence nested loaders' prints into our log by calling them directly.
    data = load_coin_data(profile)
    btc_data = load_btc()
    if not data:
        log("No coin data loaded.")
        return 1

    policies = [
        ("long_only", 0.0, None),
        ("cash_bear", 0.0, "cash"),
        ("short_bear", SELL_THR_SHORT, "short"),
    ]

    results: dict[str, pd.DataFrame] = {}
    for name, sell_thr, bear_policy in policies:
        results[name] = run_policy(
            name, data, btc_data, profile, cost_model,
            sell_thr=sell_thr, bear_policy=bear_policy,
        )

    log("\n════════════════ COMPARISON ════════════════")
    log(
        f"{'policy':12s} {'trades':>6s} {'exp':>8s} {'win%':>6s} "
        f"{'$end':>8s} {'profit':>9s} {'CAGR':>7s} {'worst':>8s}"
    )
    rows = []
    for name, _sell, _pol in policies:
        trades = results[name]
        s = trade_stats(trades)
        p = portfolio_sim(trades)
        rows.append((name, s, p))
        log(
            f"{name:12s} {s['n']:6d} {s['exp']:+7.2f}% {s['win']:5.1f}% "
            f"${p['end']:7,.0f} ${p['profit']:+8,.0f} "
            f"{p.get('cagr', 0)*100:6.1f}% ${p['worst']:7,.0f}"
        )

    for name, trades in results.items():
        report_policy(name, trades)

    # Explicit verdict vs long_only.
    base_end = portfolio_sim(results["long_only"])["end"]
    cash_end = portfolio_sim(results["cash_bear"])["end"]
    short_end = portfolio_sim(results["short_bear"])["end"]
    log("\n── Verdict vs long_only ──")
    log(f"  cash_bear:  {cash_end - base_end:+,.0f} dollars on $1000 compound path")
    log(f"  short_bear: {short_end - base_end:+,.0f} dollars on $1000 compound path")

    best = max(rows, key=lambda r: r[2]["end"])
    log(f"  Best ending equity: {best[0]} (${best[2]['end']:,.0f})")
    log("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
