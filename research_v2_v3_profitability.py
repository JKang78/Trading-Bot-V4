"""
V2/V3 profitability research harness.

This is research-only: it does not place orders and does not modify live state.
It reuses the production walk-forward backtest, then compares V2/V3 variants
under multiple fee assumptions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backtest import get_history
from ml_strategy import KrakenCostModel, StrategyProfile, get_strategy_profile
from ml_strategy_backtest import LIVE_ML_SYMBOLS, backtest_symbol


@dataclass(frozen=True)
class FeeScenario:
    name: str
    maker_entry_fee: float
    taker_exit_fee: float
    margin_open_fee: float | None
    rollover_fee_4h: float | None
    spread_buffer: float = 0.0005
    slippage_buffer: float = 0.0010


@dataclass(frozen=True)
class StrategyCase:
    name: str
    profile: StrategyProfile
    buy_thr: float | None = None
    use_expected_value_filter: bool | None = None
    use_btc_features: bool | None = None
    use_btc_regime_filter: bool | None = None
    use_relative_strength_filter: bool | None = None
    ev_cost_multiplier: float | None = None


def pct(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:+.2f}%"


def profit_factor(pnls: np.ndarray) -> float:
    losses = pnls[pnls <= 0]
    loss_sum = losses.sum()
    if len(losses) == 0 or loss_sum == 0:
        return float("inf")
    return float(pnls[pnls > 0].sum() / abs(loss_sum))


def summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
            "w1_expectancy": np.nan,
            "w2_expectancy": np.nan,
            "w3_expectancy": np.nan,
            "robust_windows": 0,
        }

    pnls = np.array([float(t["net_pnl_pct"]) for t in trades])
    sorted_trades = sorted(trades, key=lambda t: pd.Timestamp(t["entry_time"]))
    third = len(sorted_trades) // 3
    windows = [
        sorted_trades[:third],
        sorted_trades[third:2 * third],
        sorted_trades[2 * third:],
    ]
    window_exp = []
    robust_windows = 0
    for window in windows:
        if len(window) == 0:
            window_exp.append(np.nan)
            continue
        w_pnls = np.array([float(t["net_pnl_pct"]) for t in window])
        exp = float(w_pnls.mean())
        window_exp.append(exp)
        if exp > 0:
            robust_windows += 1

    return {
        "trades": int(len(pnls)),
        "win_rate": float((pnls > 0).mean() * 100),
        "expectancy": float(pnls.mean()),
        "profit_factor": profit_factor(pnls),
        "w1_expectancy": window_exp[0],
        "w2_expectancy": window_exp[1],
        "w3_expectancy": window_exp[2],
        "robust_windows": robust_windows,
    }


def scenario_cost_model(profile: StrategyProfile, scenario: FeeScenario) -> KrakenCostModel:
    margin_open_fee = (
        profile.margin_open_fee
        if scenario.margin_open_fee is None
        else scenario.margin_open_fee
    )
    rollover_fee_4h = (
        profile.rollover_fee_4h
        if scenario.rollover_fee_4h is None
        else scenario.rollover_fee_4h
    )
    return KrakenCostModel(
        maker_entry_fee=scenario.maker_entry_fee,
        taker_entry_fee=scenario.taker_exit_fee,
        taker_exit_fee=scenario.taker_exit_fee,
        margin_open_fee=margin_open_fee,
        margin_rollover_fee_4h=rollover_fee_4h,
        spread_buffer=scenario.spread_buffer,
        slippage_buffer=scenario.slippage_buffer,
        minimum_edge=profile.minimum_edge,
    )


def run_case(
    case: StrategyCase,
    scenario: FeeScenario,
    data_by_symbol: dict[str, pd.DataFrame],
    btc_data: pd.DataFrame | None,
    train_min: int,
    retrain_every: int,
    leverage: float,
) -> tuple[dict, list[dict]]:
    profile = case.profile
    margin_open_fee = (
        profile.margin_open_fee
        if scenario.margin_open_fee is None
        else scenario.margin_open_fee
    )
    rollover_fee_4h = (
        profile.rollover_fee_4h
        if scenario.rollover_fee_4h is None
        else scenario.rollover_fee_4h
    )
    cost_model = scenario_cost_model(profile, scenario)
    use_btc_features = profile.use_btc_features if case.use_btc_features is None else case.use_btc_features
    use_btc_regime_filter = (
        profile.use_btc_regime_filter
        if case.use_btc_regime_filter is None
        else case.use_btc_regime_filter
    )
    use_relative_strength_filter = (
        profile.use_relative_strength_filter
        if case.use_relative_strength_filter is None
        else case.use_relative_strength_filter
    )
    use_expected_value_filter = (
        profile.use_expected_value_filter
        if case.use_expected_value_filter is None
        else case.use_expected_value_filter
    )
    ev_cost_multiplier = (
        profile.ev_cost_multiplier
        if case.ev_cost_multiplier is None
        else case.ev_cost_multiplier
    )

    all_trades = []
    for symbol, data in data_by_symbol.items():
        result = backtest_symbol(
            symbol=symbol,
            data=data,
            horizon=profile.horizon,
            buy_thr=profile.buy_thr if case.buy_thr is None else case.buy_thr,
            sell_thr=0.0 if profile.long_only else 0.35,
            fee_rate=scenario.maker_entry_fee,
            leverage=leverage,
            model_name="logistic",
            train_min=train_min,
            retrain_every=retrain_every,
            atr_stop_mult=0.0,
            atr_period=14,
            exit_thr=profile.exit_thr,
            use_fng_features=profile.use_fng_features,
            use_fng_filter=profile.use_fng_filter,
            margin_open_fee=margin_open_fee,
            rollover_fee=rollover_fee_4h,
            btc_data=btc_data if (use_btc_features or use_btc_regime_filter or use_relative_strength_filter) else None,
            cost_model=cost_model,
            use_cost_aware_labels=profile.use_cost_aware_labels,
            use_btc_features=use_btc_features,
            use_btc_regime_filter=use_btc_regime_filter,
            use_relative_strength_filter=use_relative_strength_filter,
            use_expected_value_filter=use_expected_value_filter,
            ev_cost_multiplier=ev_cost_multiplier,
            entry_fee_rate=scenario.maker_entry_fee,
            exit_fee_rate=scenario.taker_exit_fee,
        )
        for trade in result.get("trades", []):
            trade = dict(trade)
            trade["case"] = case.name
            trade["fee_scenario"] = scenario.name
            all_trades.append(trade)

    summary = summarize_trades(all_trades)
    summary.update(
        {
            "case": case.name,
            "profile": profile.version,
            "fee_scenario": scenario.name,
            "maker_entry_fee": scenario.maker_entry_fee,
            "taker_exit_fee": scenario.taker_exit_fee,
            "margin_open_fee": margin_open_fee,
            "rollover_fee_4h": rollover_fee_4h,
        }
    )
    return summary, all_trades


def build_cases() -> list[StrategyCase]:
    v2 = get_strategy_profile("v2")
    v3 = get_strategy_profile("v3")
    return [
        StrategyCase("live_v2", v2),
        StrategyCase("live_v2_thr70", replace(v2, buy_thr=0.70)),
        StrategyCase("live_v3", v3),
        StrategyCase("v3_ev_filter", replace(v3, use_expected_value_filter=True)),
        StrategyCase("v3_btc_features", replace(v3, use_btc_features=True)),
        StrategyCase(
            "v3_btc_ev",
            replace(v3, use_btc_features=True, use_expected_value_filter=True),
        ),
        StrategyCase(
            "v3_full_gates",
            replace(
                v3,
                use_btc_features=True,
                use_btc_regime_filter=True,
                use_relative_strength_filter=True,
                use_expected_value_filter=True,
            ),
        ),
    ]


def build_fee_scenarios() -> list[FeeScenario]:
    return [
        FeeScenario(
            "repo_profile_costs",
            maker_entry_fee=0.0023,
            taker_exit_fee=0.0040,
            margin_open_fee=None,
            rollover_fee_4h=None,
        ),
        FeeScenario(
            "current_standard_spot",
            maker_entry_fee=0.0025,
            taker_exit_fee=0.0040,
            margin_open_fee=0.0004,
            rollover_fee_4h=0.0004,
        ),
        FeeScenario(
            "jul9_tier1_high_fee",
            maker_entry_fee=0.0040,
            taker_exit_fee=0.0080,
            margin_open_fee=0.0004,
            rollover_fee_4h=0.0004,
        ),
        FeeScenario(
            "jul9_tier3_10k_volume_or_20k_aop",
            maker_entry_fee=0.0022,
            taker_exit_fee=0.0038,
            margin_open_fee=0.0004,
            rollover_fee_4h=0.0004,
        ),
    ]


def print_summary(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values(
        ["fee_scenario", "robust_windows", "expectancy", "trades"],
        ascending=[True, False, False, False],
    )
    print("\nV2/V3 sweep summary")
    print(
        "  "
        + f"{'fee':30s} {'case':18s} {'n':>4s} {'win':>6s} {'exp':>8s} "
        + f"{'PF':>6s} {'w+':>3s} {'w1':>8s} {'w2':>8s} {'w3':>8s}"
    )
    for _, row in ordered.iterrows():
        pf = row["profit_factor"]
        pf_text = "inf" if np.isinf(pf) else f"{pf:.2f}"
        print(
            "  "
            + f"{row['fee_scenario']:30s} {row['case']:18s} {int(row['trades']):4d} "
            + f"{row['win_rate']:5.1f}% {pct(row['expectancy']):>8s} "
            + f"{pf_text:>6s} {int(row['robust_windows']):3d} "
            + f"{pct(row['w1_expectancy']):>8s} {pct(row['w2_expectancy']):>8s} "
            + f"{pct(row['w3_expectancy']):>8s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research V2/V3 profitability variants.")
    parser.add_argument("--symbols", default=LIVE_ML_SYMBOLS)
    parser.add_argument("--period", default="720d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--train-min", type=int, default=4000)
    parser.add_argument("--retrain-every", type=int, default=720)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--summary-out", default="v2_v3_research_summary.csv")
    parser.add_argument("--trades-out", default="v2_v3_research_trades.csv")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Downloading {len(symbols)} symbols over {args.period} {args.interval}...")
    data_by_symbol = {
        symbol: get_history(symbol, args.period, args.interval)
        for symbol in symbols
    }
    data_by_symbol = {
        symbol: data
        for symbol, data in data_by_symbol.items()
        if len(data) > args.train_min + 100
    }
    btc_data = get_history("BTC-USD", args.period, args.interval)

    summaries = []
    all_trades = []
    for scenario in build_fee_scenarios():
        for case in build_cases():
            print(f"Running {case.name} under {scenario.name}...")
            summary, trades = run_case(
                case,
                scenario,
                data_by_symbol,
                btc_data,
                args.train_min,
                args.retrain_every,
                args.leverage,
            )
            summaries.append(summary)
            all_trades.extend(trades)

    summary_df = pd.DataFrame(summaries)
    trades_df = pd.DataFrame(all_trades)
    summary_df.to_csv(args.summary_out, index=False)
    if not trades_df.empty:
        trades_df.to_csv(args.trades_out, index=False)
    print_summary(summary_df)
    print(f"\nSaved summary to {args.summary_out}")
    print(f"Saved trades to {args.trades_out}")


if __name__ == "__main__":
    main()
