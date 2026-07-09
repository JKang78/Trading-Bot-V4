"""
Long-history profit improvement harness.

Research-only: this script does not place orders and does not modify live state.

It answers the practical questions raised by the ML V2 long-history run:
- Which exit variants beat the current 72h V2 profile?
- Does V3 work better as a full replacement or as a V2 sizing overlay?
- Should weak symbols such as XRP be underweighted or removed?
- How sensitive are results to Kraken fee tier and margin rollover assumptions?
- Do expanded symbols deserve promotion into the live universe?

The heavy step is the walk-forward backtest for each strategy case. After trade
lists are generated, the fee/sizing/symbol portfolio sweeps are fast.

Default usage:

    venv/bin/python research_long_history_profit_harness.py

Focused smoke test:

    venv/bin/python research_long_history_profit_harness.py \
      --symbols DOGE-USD --cases v2_baseline,v3_ev --out-dir research_outputs/smoke
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

import ml_strategy_backtest as msb
from ml_strategy import KrakenCostModel, StrategyProfile, V2_PROFILE, V3_PROFILE
from tune_long_history import CACHE_DIR


LIVE_SYMBOLS = "XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD"
EXPANDED_SYMBOLS = (
    "ALGO-USD,ATOM-USD,AVAX-USD,BCH-USD,DOT-USD,ETC-USD,FIL-USD,"
    "LTC-USD,NEAR-USD,TRX-USD,XLM-USD"
)
BINANCE_CACHE_DIR = Path("data_binance")
START_CAPITAL = 1000.0
LEVERAGE = 2.0


@dataclass(frozen=True)
class StrategyCase:
    name: str
    profile: StrategyProfile
    horizon: int
    buy_thr: float
    exit_thr: float
    atr_stop_mult: float = 0.0
    use_cost_aware_labels: bool = False
    use_expected_value_filter: bool = False
    use_btc_features: bool = False
    use_btc_regime_filter: bool = False
    use_relative_strength_filter: bool = False
    ev_cost_multiplier: float = 1.5
    sell_thr: float = 0.0
    bear_policy: str | None = None


@dataclass(frozen=True)
class FeeScenario:
    name: str
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    margin_open_fee: float = 0.0
    rollover_fee_4h: float = 0.0
    spread_slippage: float = 0.0
    use_existing_net: bool = False


def symbol_base(symbol: str) -> str:
    """Convert XRP-USD -> XRP."""
    return symbol.split("-", 1)[0].upper()


def binance_pair(symbol: str) -> str:
    return f"{symbol_base(symbol)}USDT"


def log(msg: str) -> None:
    print(msg, flush=True)


def pct(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:+.2f}%"


def fetch_binance_hourly(pair: str, start_year: int = 2018) -> pd.DataFrame:
    """Download hourly Binance klines and cache them locally."""
    cache = BINANCE_CACHE_DIR / f"{pair}_1h.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    BINANCE_CACHE_DIR.mkdir(exist_ok=True)
    start_ms = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    frames: list[pd.DataFrame] = []

    while True:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": pair, "interval": "1h", "limit": 1000, "startTime": start_ms},
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
    return df


def load_hourly_history(symbol: str) -> pd.DataFrame:
    """Prefer local CryptoCompare cache; fall back to Binance USDT history."""
    base = symbol_base(symbol)
    cc_cache = CACHE_DIR / f"{base}_1h.csv"
    if cc_cache.exists():
        return pd.read_csv(cc_cache, index_col=0, parse_dates=True)
    return fetch_binance_hourly(binance_pair(symbol))


def load_symbols(symbols: list[str], min_bars: int) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_hourly_history(symbol)
        if len(df) <= min_bars:
            log(f"  {symbol:9s} skipped: only {len(df)} bars")
            continue
        data[symbol] = df
        log(f"  {symbol:9s} {len(df):6d} bars ({df.index.min().date()} .. {df.index.max().date()})")
    return data


def load_btc_if_needed(cases: list[StrategyCase]) -> pd.DataFrame | None:
    if not any(
        c.use_btc_features
        or c.use_btc_regime_filter
        or c.use_relative_strength_filter
        or c.bear_policy
        for c in cases
    ):
        return None
    btc = fetch_binance_hourly("BTCUSDT")
    if btc.empty:
        raise RuntimeError("BTCUSDT history is required for BTC-aware cases but could not be loaded.")
    log(f"  BTC-USD   {len(btc):6d} bars ({btc.index.min().date()} .. {btc.index.max().date()})")
    return btc


def build_cost_model(profile: StrategyProfile) -> KrakenCostModel:
    return KrakenCostModel(
        maker_entry_fee=0.0023,
        taker_entry_fee=0.0040,
        taker_exit_fee=0.0040,
        margin_open_fee=profile.margin_open_fee,
        margin_rollover_fee_4h=profile.rollover_fee_4h,
        spread_buffer=0.0005,
        slippage_buffer=0.0010,
        minimum_edge=profile.minimum_edge,
    )


def all_strategy_cases() -> dict[str, StrategyCase]:
    v2 = V2_PROFILE
    v3 = V3_PROFILE
    return {
        "v2_baseline": StrategyCase(
            "v2_baseline", v2, v2.horizon, v2.buy_thr, v2.exit_thr,
            use_cost_aware_labels=v2.use_cost_aware_labels,
        ),
        "v2_exit35": StrategyCase("v2_exit35", v2, v2.horizon, v2.buy_thr, 0.35),
        "v2_exit45": StrategyCase("v2_exit45", v2, v2.horizon, v2.buy_thr, 0.45),
        "v2_exit50": StrategyCase("v2_exit50", v2, v2.horizon, v2.buy_thr, 0.50),
        "v2_h48": StrategyCase("v2_h48", v2, 48, v2.buy_thr, v2.exit_thr),
        "v2_h96": StrategyCase("v2_h96", v2, 96, v2.buy_thr, v2.exit_thr),
        "v2_atr2": StrategyCase("v2_atr2", v2, v2.horizon, v2.buy_thr, v2.exit_thr, atr_stop_mult=2.0),
        "v2_atr3": StrategyCase("v2_atr3", v2, v2.horizon, v2.buy_thr, v2.exit_thr, atr_stop_mult=3.0),
        "v2_btc_features": StrategyCase(
            "v2_btc_features", v2, v2.horizon, v2.buy_thr, v2.exit_thr,
            use_btc_features=True,
        ),
        "v2_btc_regime": StrategyCase(
            "v2_btc_regime", v2, v2.horizon, v2.buy_thr, v2.exit_thr,
            use_btc_regime_filter=True,
        ),
        "v2_relative_strength": StrategyCase(
            "v2_relative_strength", v2, v2.horizon, v2.buy_thr, v2.exit_thr,
            use_relative_strength_filter=True,
        ),
        "v2_cash_bear": StrategyCase(
            "v2_cash_bear", v2, v2.horizon, v2.buy_thr, v2.exit_thr,
            bear_policy="cash",
        ),
        "v2_short_bear": StrategyCase(
            "v2_short_bear", v2, v2.horizon, v2.buy_thr, v2.exit_thr,
            sell_thr=0.35, bear_policy="short",
        ),
        "v3_ev": StrategyCase(
            "v3_ev", v3, v3.horizon, v3.buy_thr, v3.exit_thr,
            use_cost_aware_labels=True,
            use_expected_value_filter=True,
            ev_cost_multiplier=v3.ev_cost_multiplier,
        ),
        "v3_btc_features": StrategyCase(
            "v3_btc_features", v3, v3.horizon, v3.buy_thr, v3.exit_thr,
            use_cost_aware_labels=True,
            use_expected_value_filter=True,
            use_btc_features=True,
            ev_cost_multiplier=v3.ev_cost_multiplier,
        ),
    }


def fee_scenarios() -> dict[str, FeeScenario]:
    return {
        "case_assumed": FeeScenario("case_assumed", use_existing_net=True),
        "repo_v2_no_exec_buffer": FeeScenario(
            "repo_v2_no_exec_buffer",
            entry_fee=0.0023,
            exit_fee=0.0040,
            margin_open_fee=0.0002,
            rollover_fee_4h=0.0002,
        ),
        "kraken_0_volume_high_margin": FeeScenario(
            "kraken_0_volume_high_margin",
            entry_fee=0.0040,
            exit_fee=0.0080,
            margin_open_fee=0.0004,
            rollover_fee_4h=0.0004,
            spread_slippage=0.0015,
        ),
        "kraken_10k_high_margin": FeeScenario(
            "kraken_10k_high_margin",
            entry_fee=0.0022,
            exit_fee=0.0038,
            margin_open_fee=0.0004,
            rollover_fee_4h=0.0004,
            spread_slippage=0.0015,
        ),
        "kraken_50k_high_margin": FeeScenario(
            "kraken_50k_high_margin",
            entry_fee=0.0014,
            exit_fee=0.0024,
            margin_open_fee=0.0004,
            rollover_fee_4h=0.0004,
            spread_slippage=0.0015,
        ),
    }


def run_strategy_case(
    case: StrategyCase,
    data_by_symbol: dict[str, pd.DataFrame],
    btc_data: pd.DataFrame | None,
    train_min: int,
    retrain_every: int,
) -> pd.DataFrame:
    all_trades: list[dict] = []
    cost_model = build_cost_model(case.profile)
    for symbol, data in data_by_symbol.items():
        result = msb.backtest_symbol(
            symbol=symbol,
            data=data,
            horizon=case.horizon,
            buy_thr=case.buy_thr,
            sell_thr=case.sell_thr,
            fee_rate=0.0023,
            leverage=LEVERAGE,
            model_name="logistic",
            train_min=train_min,
            retrain_every=retrain_every,
            atr_stop_mult=case.atr_stop_mult,
            atr_period=14,
            exit_thr=case.exit_thr,
            use_fng_features=case.profile.use_fng_features,
            use_fng_filter=case.profile.use_fng_filter,
            margin_open_fee=case.profile.margin_open_fee,
            rollover_fee=case.profile.rollover_fee_4h,
            btc_data=btc_data,
            cost_model=cost_model,
            use_cost_aware_labels=case.use_cost_aware_labels,
            use_btc_features=case.use_btc_features,
            use_btc_regime_filter=case.use_btc_regime_filter,
            use_relative_strength_filter=case.use_relative_strength_filter,
            use_expected_value_filter=case.use_expected_value_filter,
            ev_cost_multiplier=case.ev_cost_multiplier,
            entry_fee_rate=0.0023,
            exit_fee_rate=0.0040,
            bear_policy=case.bear_policy,
        )
        trades = result.get("trades", [])
        all_trades.extend(trades)
        pnls = np.array([t["net_pnl_pct"] for t in trades]) if trades else np.array([])
        exp = float(pnls.mean()) if len(pnls) else float("nan")
        win = float((pnls > 0).mean() * 100) if len(pnls) else float("nan")
        log(f"    {symbol:9s} trades={len(trades):4d} exp={pct(exp):>8s} win={win:5.1f}%")

    if not all_trades:
        return pd.DataFrame()
    trades_df = pd.DataFrame(all_trades)
    trades_df["case"] = case.name
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    return trades_df.sort_values("entry_time").reset_index(drop=True)


def gross_pnl_pct(trades: pd.DataFrame) -> pd.Series:
    direction = trades["direction"].fillna("long")
    long_gross = (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"] * 100 * LEVERAGE
    short_gross = (trades["entry_price"] - trades["exit_price"]) / trades["entry_price"] * 100 * LEVERAGE
    return pd.Series(np.where(direction == "short", short_gross, long_gross), index=trades.index)


def apply_fee_scenario(trades: pd.DataFrame, scenario: FeeScenario) -> pd.DataFrame:
    df = trades.copy()
    if scenario.use_existing_net:
        df["scenario_net_pnl_pct"] = df["net_pnl_pct"].astype(float)
        return df

    rollovers = (df["bars_held"].astype(float) // 4).clip(lower=0)
    total_cost = (
        scenario.entry_fee
        + scenario.exit_fee
        + scenario.margin_open_fee
        + rollovers * scenario.rollover_fee_4h
        + scenario.spread_slippage
    )
    df["gross_pnl_pct"] = gross_pnl_pct(df)
    df["scenario_net_pnl_pct"] = df["gross_pnl_pct"] - total_cost * 100 * LEVERAGE
    return df


def profit_factor(pnls: np.ndarray) -> float:
    losses = pnls[pnls <= 0]
    if len(losses) == 0 or losses.sum() == 0:
        return float("inf")
    return float(pnls[pnls > 0].sum() / abs(losses.sum()))


def symbol_multiplier(policy: str, symbol: str) -> float:
    if policy == "all":
        return 1.0
    if policy == "no_xrp":
        return 0.0 if symbol == "XRP-USD" else 1.0
    if policy == "half_xrp":
        return 0.5 if symbol == "XRP-USD" else 1.0
    if policy == "no_xrp_ada":
        return 0.0 if symbol in {"XRP-USD", "ADA-USD"} else 1.0
    if policy == "core_momentum":
        return 1.0 if symbol in {"DOGE-USD", "SOL-USD", "LINK-USD"} else 0.0
    raise ValueError(f"Unknown symbol policy: {policy}")


def add_v3_confirmation(
    trades: pd.DataFrame,
    v3_trades: pd.DataFrame | None,
    confirm_hours: float,
) -> pd.DataFrame:
    df = trades.copy()
    df["v3_confirmed"] = False
    if v3_trades is None or v3_trades.empty or df.empty:
        return df

    window = pd.Timedelta(hours=confirm_hours)
    for symbol, group_idx in df.groupby("symbol").groups.items():
        v3_times = v3_trades.loc[v3_trades["symbol"] == symbol, "entry_time"].sort_values().reset_index(drop=True)
        if v3_times.empty:
            continue
        values = v3_times.values
        for idx in group_idx:
            ts = df.at[idx, "entry_time"]
            pos = np.searchsorted(values, np.datetime64(ts), side="left")
            confirmed = False
            for check_pos in (pos - 1, pos):
                if 0 <= check_pos < len(v3_times):
                    if abs(v3_times.iloc[check_pos] - ts) <= window:
                        confirmed = True
                        break
            df.at[idx, "v3_confirmed"] = confirmed
    return df


def size_fraction(
    row: pd.Series,
    sizing_policy: str,
    symbol_policy: str,
    base_fraction: float,
) -> float:
    mult = symbol_multiplier(symbol_policy, str(row["symbol"]))
    if mult <= 0:
        return 0.0

    if sizing_policy == "fixed_25":
        frac = base_fraction
    elif sizing_policy == "confidence":
        prob = float(row.get("prob_up", 0.0) or 0.0)
        if prob >= 0.78:
            frac = 0.35
        elif prob >= 0.72:
            frac = base_fraction
        else:
            frac = 0.15
    elif sizing_policy == "v3_confirm":
        frac = 0.35 if bool(row.get("v3_confirmed", False)) else 0.15
    else:
        raise ValueError(f"Unknown sizing policy: {sizing_policy}")

    return frac * mult


def max_open_exposure(trades: pd.DataFrame) -> tuple[int, float, float]:
    events: list[tuple[pd.Timestamp, int, float]] = []
    for row in trades.itertuples(index=False):
        frac = float(getattr(row, "size_frac"))
        if frac <= 0:
            continue
        events.append((getattr(row, "entry_time"), 1, frac))
        events.append((getattr(row, "exit_time"), -1, -frac))
    events.sort(key=lambda x: (x[0], -x[1]))

    open_count = 0
    margin_frac = 0.0
    max_open = 0
    max_margin_frac = 0.0
    for _ts, count_delta, frac_delta in events:
        open_count += count_delta
        margin_frac += frac_delta
        max_open = max(max_open, open_count)
        max_margin_frac = max(max_margin_frac, margin_frac)
    return max_open, max_margin_frac, max_margin_frac * LEVERAGE


def simulate_portfolio(
    trades: pd.DataFrame,
    sizing_policy: str,
    symbol_policy: str,
    base_fraction: float,
    start_capital: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = trades.copy()
        return {
            "n": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "profit_factor": float("nan"),
            "ending_equity": start_capital,
            "profit": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "max_open": 0,
            "max_margin_frac": 0.0,
            "max_notional_frac": 0.0,
            "positive_years": 0,
            "worst_year_return": float("nan"),
        }, empty, pd.DataFrame()

    df = trades.sort_values("entry_time").copy()
    df["size_frac"] = [
        size_fraction(row, sizing_policy, symbol_policy, base_fraction)
        for _, row in df.iterrows()
    ]
    df = df[df["size_frac"] > 0].copy()
    if df.empty:
        return simulate_portfolio(df, "fixed_25", "all", base_fraction, start_capital)

    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    equity_curve = []
    for row in df.itertuples(index=False):
        pnl = float(getattr(row, "scenario_net_pnl_pct"))
        frac = float(getattr(row, "size_frac"))
        equity *= 1.0 + (pnl / 100.0) * frac
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        equity_curve.append(equity)

    pnls = df["scenario_net_pnl_pct"].to_numpy(dtype=float)
    years = max((df["entry_time"].max() - df["entry_time"].min()).days / 365.25, 1 / 365.25)
    cagr = (equity / start_capital) ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    max_open, max_margin_frac, max_notional_frac = max_open_exposure(df)

    yearly_rows = []
    for year, group in df.groupby(df["entry_time"].dt.year):
        year_equity = start_capital
        for row in group.sort_values("entry_time").itertuples(index=False):
            year_equity *= 1.0 + (float(getattr(row, "scenario_net_pnl_pct")) / 100.0) * float(getattr(row, "size_frac"))
        yearly_rows.append({
            "year": int(year),
            "trades": int(len(group)),
            "ending_equity": year_equity,
            "profit": year_equity - start_capital,
            "return_pct": (year_equity / start_capital - 1.0) * 100.0,
            "expectancy": float(group["scenario_net_pnl_pct"].mean()),
            "win_rate": float((group["scenario_net_pnl_pct"] > 0).mean() * 100.0),
        })
    yearly = pd.DataFrame(yearly_rows)
    positive_years = int((yearly["return_pct"] > 0).sum()) if not yearly.empty else 0
    worst_year_return = float(yearly["return_pct"].min()) if not yearly.empty else float("nan")

    metrics = {
        "n": int(len(df)),
        "win_rate": float((pnls > 0).mean() * 100.0),
        "expectancy": float(pnls.mean()),
        "profit_factor": profit_factor(pnls),
        "ending_equity": float(equity),
        "profit": float(equity - start_capital),
        "cagr": float(cagr * 100.0),
        "max_drawdown": float(max_dd * 100.0),
        "max_open": int(max_open),
        "max_margin_frac": float(max_margin_frac),
        "max_notional_frac": float(max_notional_frac),
        "margin_feasible": bool(max_margin_frac <= 1.000001),
        "positive_years": positive_years,
        "worst_year_return": worst_year_return,
    }
    df["sim_equity"] = equity_curve
    return metrics, df, yearly


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    cases = all_strategy_cases()
    fees = fee_scenarios()
    parser = argparse.ArgumentParser(description="Research long-history ML profit improvements.")
    parser.add_argument("--symbols", default=LIVE_SYMBOLS)
    parser.add_argument("--include-expanded", action="store_true")
    parser.add_argument("--expanded-symbols", default=EXPANDED_SYMBOLS)
    parser.add_argument(
        "--cases",
        default="v2_baseline,v2_exit45,v2_exit50,v2_atr3,v3_ev",
        help=f"Comma-separated case names. Available: {','.join(cases)}",
    )
    parser.add_argument(
        "--fee-scenarios",
        default="case_assumed,kraken_10k_high_margin,kraken_0_volume_high_margin",
        help=f"Comma-separated fee scenario names. Available: {','.join(fees)}",
    )
    parser.add_argument(
        "--portfolio-policies",
        default="fixed_25,confidence,v3_confirm",
        help="Comma-separated sizing policies: fixed_25,confidence,v3_confirm",
    )
    parser.add_argument(
        "--symbol-policies",
        default="all,no_xrp,half_xrp,core_momentum",
        help="Comma-separated symbol policies: all,no_xrp,half_xrp,no_xrp_ada,core_momentum",
    )
    parser.add_argument("--train-min", type=int, default=4000)
    parser.add_argument("--retrain-every", type=int, default=720)
    parser.add_argument("--position-frac", type=float, default=0.25)
    parser.add_argument("--start-capital", type=float, default=START_CAPITAL)
    parser.add_argument("--hybrid-confirm-hours", type=float, default=12.0)
    parser.add_argument("--out-dir", default="research_outputs/long_profit")
    parser.add_argument("--reuse-trades", action="store_true", help="Reuse existing per-case trade CSVs in out-dir.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_map = all_strategy_cases()
    selected_cases = parse_csv_arg(args.cases)
    unknown_cases = sorted(set(selected_cases) - set(case_map))
    if unknown_cases:
        raise SystemExit(f"Unknown cases: {', '.join(unknown_cases)}")

    fee_map = fee_scenarios()
    selected_fees = parse_csv_arg(args.fee_scenarios)
    unknown_fees = sorted(set(selected_fees) - set(fee_map))
    if unknown_fees:
        raise SystemExit(f"Unknown fee scenarios: {', '.join(unknown_fees)}")

    sizing_policies = parse_csv_arg(args.portfolio_policies)
    symbol_policies = parse_csv_arg(args.symbol_policies)

    symbols = parse_csv_arg(args.symbols)
    if args.include_expanded:
        symbols.extend(s for s in parse_csv_arg(args.expanded_symbols) if s not in symbols)

    cases = [case_map[name] for name in selected_cases]
    min_bars = max(c.horizon for c in cases) + args.train_min + 50

    log("Loading long hourly histories...")
    data_by_symbol = load_symbols(symbols, min_bars)
    btc_data = load_btc_if_needed(cases)
    if not data_by_symbol:
        raise SystemExit("No symbols had enough data.")

    trades_by_case: dict[str, pd.DataFrame] = {}
    for case in cases:
        path = out_dir / f"{case.name}_trades.csv"
        if args.reuse_trades and path.exists():
            log(f"\nReusing {case.name}: {path}")
            trades = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
        else:
            log(f"\nRunning case: {case.name}")
            trades = run_strategy_case(case, data_by_symbol, btc_data, args.train_min, args.retrain_every)
            if not trades.empty:
                trades.to_csv(path, index=False)
                log(f"  saved {len(trades)} trades -> {path}")
            else:
                log("  no trades")
        trades_by_case[case.name] = trades

    v3_for_confirmation = trades_by_case.get("v3_ev")
    summary_rows: list[dict] = []
    yearly_rows: list[dict] = []
    symbol_rows: list[dict] = []

    for case_name, raw_trades in trades_by_case.items():
        if raw_trades.empty:
            continue
        for fee_name in selected_fees:
            repriced = apply_fee_scenario(raw_trades, fee_map[fee_name])
            if case_name.startswith("v2"):
                repriced = add_v3_confirmation(repriced, v3_for_confirmation, args.hybrid_confirm_hours)
            else:
                repriced["v3_confirmed"] = False

            for symbol_policy in symbol_policies:
                for sizing_policy in sizing_policies:
                    if sizing_policy == "v3_confirm" and not case_name.startswith("v2"):
                        continue
                    metrics, sim_trades, yearly = simulate_portfolio(
                        repriced,
                        sizing_policy,
                        symbol_policy,
                        args.position_frac,
                        args.start_capital,
                    )
                    row = {
                        "case": case_name,
                        "fee_scenario": fee_name,
                        "sizing_policy": sizing_policy,
                        "symbol_policy": symbol_policy,
                        **metrics,
                    }
                    summary_rows.append(row)

                    for yr in yearly.to_dict("records"):
                        yearly_rows.append({
                            "case": case_name,
                            "fee_scenario": fee_name,
                            "sizing_policy": sizing_policy,
                            "symbol_policy": symbol_policy,
                            **yr,
                        })

                    if not sim_trades.empty:
                        for symbol, group in sim_trades.groupby("symbol"):
                            pnls = group["scenario_net_pnl_pct"].to_numpy(dtype=float)
                            symbol_rows.append({
                                "case": case_name,
                                "fee_scenario": fee_name,
                                "sizing_policy": sizing_policy,
                                "symbol_policy": symbol_policy,
                                "symbol": symbol,
                                "trades": int(len(group)),
                                "expectancy": float(pnls.mean()),
                                "win_rate": float((pnls > 0).mean() * 100.0),
                                "profit_factor": profit_factor(pnls),
                                "avg_size_frac": float(group["size_frac"].mean()),
                                "confirmed_rate": float(group["v3_confirmed"].mean() * 100.0)
                                if "v3_confirmed" in group else 0.0,
                            })

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    symbol_summary = pd.DataFrame(symbol_rows)

    summary_path = out_dir / "summary.csv"
    yearly_path = out_dir / "yearly.csv"
    symbol_path = out_dir / "symbol_summary.csv"
    summary.to_csv(summary_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    symbol_summary.to_csv(symbol_path, index=False)

    if not summary.empty:
        ordered = summary.sort_values(
            ["ending_equity", "cagr", "max_drawdown"],
            ascending=[False, False, True],
        )
        log("\nTop portfolio variants by ending equity:")
        cols = [
            "case", "fee_scenario", "sizing_policy", "symbol_policy",
            "n", "ending_equity", "cagr", "max_drawdown", "margin_feasible",
            "positive_years", "max_margin_frac", "max_notional_frac",
        ]
        print(ordered[cols].head(15).to_string(index=False, formatters={
            "ending_equity": "{:,.0f}".format,
            "cagr": "{:.1f}".format,
            "max_drawdown": "{:.1f}".format,
            "max_margin_frac": "{:.2f}".format,
            "max_notional_frac": "{:.2f}".format,
        }))

        feasible = ordered[ordered["margin_feasible"]]
        if not feasible.empty:
            log("\nTop margin-feasible variants by ending equity:")
            print(feasible[cols].head(15).to_string(index=False, formatters={
                "ending_equity": "{:,.0f}".format,
                "cagr": "{:.1f}".format,
                "max_drawdown": "{:.1f}".format,
                "max_margin_frac": "{:.2f}".format,
                "max_notional_frac": "{:.2f}".format,
            }))

    log(f"\nSaved summary -> {summary_path}")
    log(f"Saved yearly results -> {yearly_path}")
    log(f"Saved symbol results -> {symbol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
