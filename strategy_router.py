"""
AI-routed live trader for the old V4 swing bot and ML V2/V3 strategy.

The model is only a meta-controller:
- it chooses one already-computed candidate strategy/symbol, or no trade
- it suggests a budget fraction and leverage

Hard rules in this script clamp or reject every model decision before any
Kraken order can be placed.
"""

import html
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from backtest import get_history
from kraken_bot_v4_advanced import (
    Config,
    CorrelationManager,
    KrakenClient,
    RegimeDetector,
    SwingDetectorV3,
    Telegram,
    TradingBotV4,
)
from ml_live_trade import enter_position
from ml_strategy import (
    build_cost_model,
    create_ml_strategy,
    get_strategy_profile,
)


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ROUTER_STATE_FILE = env_str("ROUTER_STATE_FILE", "strategy_router_state.json")
ML_STATE_FILE = env_str("ML_LIVE_STATE_FILE", "ml_live_state.json")
DRY_RUN = env_bool("ROUTER_DRY_RUN", False)
AI_ENABLED = env_bool("ROUTER_AI_ENABLED", True)
OPENAI_MODEL = env_str("OPENAI_ROUTER_MODEL", "gpt-5.5")
OPENAI_TIMEOUT = env_int("OPENAI_ROUTER_TIMEOUT", 30)
OPENAI_REASONING_EFFORT = env_str("OPENAI_ROUTER_REASONING_EFFORT", "medium")
NOTIFY_NO_TRADE = env_bool("ROUTER_NOTIFY_NO_TRADE", False)

MAX_OPEN_POSITIONS = env_int("ROUTER_MAX_OPEN_POSITIONS", 1)
MAX_TRADE_MARGIN_FRACTION = env_float("ROUTER_MAX_TRADE_MARGIN_FRACTION", 0.25)
MAX_TOTAL_MARGIN_FRACTION = env_float("ROUTER_MAX_TOTAL_MARGIN_FRACTION", 0.35)
MIN_TRADE_MARGIN = env_float("ROUTER_MIN_TRADE_MARGIN_USD", 1.0)
DEFAULT_BUDGET_FRACTION = env_float("ROUTER_DEFAULT_BUDGET_FRACTION", 0.15)
MAX_LEVERAGE = env_int("ROUTER_MAX_LEVERAGE", 2)
DEFAULT_LEVERAGE = env_int("ROUTER_DEFAULT_LEVERAGE", 2)

ML_STRATEGY_VERSIONS = [
    version.strip().lower()
    for version in env_str("ROUTER_ML_STRATEGIES", env_str("ROUTER_ML_STRATEGY", env_str("ML_LIVE_STRATEGY", "v2,v3"))).split(",")
    if version.strip()
]
if not ML_STRATEGY_VERSIONS:
    ML_STRATEGY_VERSIONS = ["v2"]
ML_SYMBOLS = [
    symbol.strip()
    for symbol in env_str("ROUTER_ML_SYMBOLS", env_str("ML_LIVE_SYMBOLS", "XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD")).split(",")
    if symbol.strip()
]
DATA_PERIOD = env_str("ROUTER_DATA_PERIOD", env_str("ML_LIVE_PERIOD", "720d"))
DATA_INTERVAL = env_str("ROUTER_DATA_INTERVAL", env_str("ML_LIVE_INTERVAL", "1h"))


@dataclass
class Candidate:
    strategy: str
    symbol: str
    kraken_pair: str
    signal: str
    direction: str
    confidence: float
    score: float
    current_price: float
    min_volume: float
    data: pd.DataFrame
    details: dict


def load_json(path: str, default: dict) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return default


def save_json(path: str, data: dict) -> None:
    data["updated_at"] = pd.Timestamp.utcnow().isoformat()
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def pair_map(config: Config) -> dict:
    return {pair.yf_symbol: pair for pair in config.TRADING_PAIRS}


def load_market_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in symbols:
        try:
            df = get_history(symbol, DATA_PERIOD, DATA_INTERVAL)
            if not df.empty:
                data[symbol] = df
        except Exception as exc:
            print(f"  ⚠️ {symbol}: data error {exc}")
    return data


def safe_trade_balance(kraken: KrakenClient) -> dict:
    try:
        return kraken._request("/0/private/TradeBalance", private=True)
    except Exception as exc:
        print(f"⚠️ Could not read trade balance: {exc}")
        return {}


def margin_budget_cap(trade_balance: dict) -> tuple[float, float]:
    free_margin = float(trade_balance.get("mf", 0) or 0)
    equity = float(trade_balance.get("e", 0) or 0)
    basis = equity if equity > 0 else free_margin
    total_cap = basis * MAX_TOTAL_MARGIN_FRACTION if basis > 0 else free_margin
    single_cap = free_margin * MAX_TRADE_MARGIN_FRACTION
    return free_margin, max(0.0, min(total_cap, single_cap, free_margin))


def manage_open_positions(
    config: Config,
    kraken: KrakenClient,
    telegram: Telegram,
    market_data: dict[str, pd.DataFrame],
) -> list[str]:
    """
    Run deterministic exit checks first. New entries are blocked while any
    position remains open after this function.
    """
    ml_state = load_json(ML_STATE_FILE, {"open": {}, "closed": []})
    pairs = pair_map(config)
    actions = []

    # Close ML-tracked positions on their own time/model rules.
    for symbol in list(ml_state.get("open", {}).keys()):
        pos = ml_state["open"][symbol]
        df = market_data.get(symbol)
        kp = pairs.get(symbol)
        if df is None or kp is None:
            continue
        now = df.index[-1]
        time_due = now >= pd.Timestamp(pos["exit_due"])
        current_price = float(df["Close"].iloc[-1])
        if not time_due:
            continue

        try:
            if not DRY_RUN:
                kraken.close_position(
                    kp.kraken_pair,
                    pos["direction"],
                    float(pos["volume"]),
                    int(pos.get("leverage", DEFAULT_LEVERAGE)),
                )
            pnl_pct = (
                (current_price - pos["entry_price"]) / pos["entry_price"] * 100 * int(pos.get("leverage", DEFAULT_LEVERAGE))
                if pos["direction"] == "long"
                else (pos["entry_price"] - current_price) / pos["entry_price"] * 100 * int(pos.get("leverage", DEFAULT_LEVERAGE))
            )
            ml_state.setdefault("closed", []).append({
                **pos,
                "symbol": symbol,
                "exit_price": current_price,
                "exit_time": str(now),
                "pnl_pct": round(pnl_pct, 3),
                "exit_reason": "router_time",
            })
            del ml_state["open"][symbol]
            actions.append(f"Closed ML {symbol} {pos['direction']} @ {current_price:.4f} ({pnl_pct:+.2f}%)")
        except Exception as exc:
            actions.append(f"⚠️ ML close failed for {symbol}: {exc}")

    save_json(ML_STATE_FILE, ml_state)

    # Close non-ML positions using old V4 stop/take-profit/trailing logic.
    refreshed_positions = kraken.get_open_positions()
    if refreshed_positions:
        bot = TradingBotV4(config)
        active_ids = []
        for pair_key, pos_data in refreshed_positions.items():
            trading_pair = next((tp for tp in config.TRADING_PAIRS if tp.kraken_pair == pair_key), None)
            if trading_pair is None:
                active_ids.append(pair_key)
                continue
            df = market_data.get(trading_pair.yf_symbol)
            if df is None:
                active_ids.append(pair_key)
                continue
            current_price = float(df["Close"].iloc[-1])
            regime = RegimeDetector.detect(df, config.REGIME_LOOKBACK)
            regime_params = RegimeDetector.get_adapted_params(
                regime,
                config.BASE_STOP_LOSS,
                config.BASE_TAKE_PROFIT,
                config.BASE_TRAILING_STOP,
            )
            should_close, reason = bot.position_mgr.check_position(pair_key, pos_data, current_price, regime_params)
            if should_close:
                pos_type = bot.position_mgr.normalize_position_type(pos_data.get("type", "long"))
                volume = float(pos_data.get("vol", 0) or 0)
                closed = bot.position_mgr.close_position(pair_key, pos_type, volume, reason, pos_data, current_price)
                if closed:
                    actions.append(f"Closed V4 {trading_pair.yf_symbol} {pos_type} @ {current_price:.4f}: {reason}")
                else:
                    active_ids.append(pair_key)
            else:
                active_ids.append(pair_key)
        bot.position_mgr.sync_active_positions(active_ids)

    if actions:
        telegram.send("<b>Strategy Router exits</b>\n" + "\n".join(f"• {html.escape(a)}" for a in actions))
    return actions


def collect_old_v4_candidates(config: Config, market_data: dict[str, pd.DataFrame]) -> list[Candidate]:
    bot = TradingBotV4(config)
    old_symbols = [pair.yf_symbol for pair in config.TRADING_PAIRS]
    data_subset = {symbol: market_data[symbol] for symbol in old_symbols if symbol in market_data}
    corr_matrix = CorrelationManager.calculate_correlation_matrix(data_subset, lookback=30)
    candidates = []

    for pair in config.TRADING_PAIRS:
        df = market_data.get(pair.yf_symbol)
        if df is None:
            continue
        detector = SwingDetectorV3(
            df,
            volume_filter=config.USE_VOLUME_FILTER,
            use_ml=config.USE_ML_VALIDATION,
            ml_threshold=config.ML_CONFIDENCE_THRESHOLD,
        )
        signal, signal_price, confidence = detector.get_signal()
        if not signal:
            continue

        analysis = bot.analyze_trading_opportunity(pair, df, (signal, signal_price, confidence))
        if not analysis.get("can_trade"):
            continue
        can_open, max_corr = CorrelationManager.check_position_correlation(
            [],
            pair.yf_symbol,
            corr_matrix,
            config.MAX_CORRELATION,
        )
        if not can_open:
            continue

        current_price = float(df["Close"].iloc[-1])
        candidates.append(Candidate(
            strategy="old_v4",
            symbol=pair.yf_symbol,
            kraken_pair=pair.kraken_pair,
            signal=signal,
            direction="long" if signal == "BUY" else "short",
            confidence=float(analysis.get("confidence", confidence) or 0.0),
            score=float(analysis.get("confidence", confidence) or 0.0),
            current_price=current_price,
            min_volume=pair.min_volume,
            data=df,
            details={
                "signal_price": float(signal_price) if signal_price is not None else current_price,
                "reasons": analysis.get("reasons", []),
                "max_correlation": max_corr,
                "v4_data": analysis.get("v4_data", {}),
                "analysis": analysis,
            },
        ))
    return candidates


def collect_ml_candidates(config: Config, market_data: dict[str, pd.DataFrame]) -> list[Candidate]:
    pairs = pair_map(config)
    candidates = []

    btc_data = market_data.get("BTC-USD")
    if btc_data is None:
        try:
            btc_data = get_history("BTC-USD", DATA_PERIOD, DATA_INTERVAL)
        except Exception:
            btc_data = None

    seen_versions = set()
    for version in ML_STRATEGY_VERSIONS:
        profile = get_strategy_profile(version)
        if profile.version in seen_versions:
            continue
        seen_versions.add(profile.version)
        cost_model = build_cost_model(profile)
        strategy = create_ml_strategy(profile, cost_model)

        for symbol in ML_SYMBOLS:
            df = market_data.get(symbol)
            kp = pairs.get(symbol)
            if df is None or kp is None:
                continue
            sig = strategy.get_signal(df, btc_data)
            if sig.signal != "BUY":
                continue
            current_price = float(df["Close"].iloc[-1])
            candidates.append(Candidate(
                strategy=f"ml_{profile.version}",
                symbol=symbol,
                kraken_pair=kp.kraken_pair,
                signal="BUY",
                direction="long",
                confidence=float(sig.confidence),
                score=float(sig.score),
                current_price=current_price,
                min_volume=kp.min_volume,
                data=df,
                details={
                    "profile_version": profile.version,
                    "prob_up": sig.prob_up,
                    "dynamic_threshold": sig.dynamic_threshold,
                    "expected_value": sig.expected_value,
                    "estimated_cost": sig.estimated_cost,
                    "btc_regime": sig.btc_regime,
                    "relative_strength_7d": sig.relative_strength_7d,
                    "blocked_reason": sig.blocked_reason,
                    "horizon": sig.horizon,
                    "regime_size_multiplier": sig.regime_size_multiplier,
                    "signal": sig,
                },
            ))
    return candidates


def candidate_for_prompt(candidate: Candidate) -> dict:
    details = candidate.details.copy()
    details.pop("analysis", None)
    details.pop("signal", None)
    details.pop("v4_data", None)
    return {
        "strategy": candidate.strategy,
        "symbol": candidate.symbol,
        "signal": candidate.signal,
        "direction": candidate.direction,
        "confidence": round(candidate.confidence, 4),
        "score": round(candidate.score, 4),
        "current_price": round(candidate.current_price, 8),
        "min_volume": candidate.min_volume,
        "details": details,
    }


ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["no_trade", "old_v4", "ml_v2", "ml_v3"]},
        "symbol": {"type": "string"},
        "budget_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "leverage": {"type": "integer", "minimum": 1, "maximum": 10},
        "old_v4_budget_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "ml_budget_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "ml_v2_budget_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "ml_v3_budget_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "decision",
        "symbol",
        "budget_fraction",
        "leverage",
        "old_v4_budget_fraction",
        "ml_budget_fraction",
        "ml_v2_budget_fraction",
        "ml_v3_budget_fraction",
        "confidence",
        "rationale",
        "risk_notes",
    ],
    "additionalProperties": False,
}


def parse_response_text(response: dict) -> str:
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks).strip()


def ask_openai_router(candidates: list[Candidate], account: dict, context: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not AI_ENABLED:
        return deterministic_router(candidates)
    if not api_key:
        return {
            "decision": "no_trade",
            "symbol": "",
            "budget_fraction": 0.0,
            "leverage": 1,
            "old_v4_budget_fraction": 0.0,
            "ml_budget_fraction": 0.0,
            "ml_v2_budget_fraction": 0.0,
            "ml_v3_budget_fraction": 0.0,
            "confidence": 0.0,
            "rationale": "OPENAI_API_KEY is not configured, so the AI router refused to trade.",
            "risk_notes": ["Missing OpenAI API key"],
        }

    prompt = {
        "goal": (
            "Choose one strategy candidate to trade, or no_trade. Suggest a selected-trade budget_fraction "
            "and per-version budgets. Hard risk caps will be enforced by code."
        ),
        "hard_rules": {
            "max_open_positions": MAX_OPEN_POSITIONS,
            "max_trade_margin_fraction": MAX_TRADE_MARGIN_FRACTION,
            "max_total_margin_fraction": MAX_TOTAL_MARGIN_FRACTION,
            "max_leverage": MAX_LEVERAGE,
            "min_trade_margin_usd": MIN_TRADE_MARGIN,
            "no_trade_when_candidates_conflict_or_edge_is_weak": True,
        },
        "account": account,
        "market_context": context,
        "candidates": [candidate_for_prompt(candidate) for candidate in candidates],
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "developer",
                "content": (
                    "You are a cautious trading strategy router. You do not create new trades. "
                    "You may only choose one provided candidate, or no_trade. Prefer no_trade when "
                    "evidence is mixed, strategies conflict, budget is too small, or risk is unclear. "
                    "Budget fractions are fractions of currently free margin, not account equity. "
                    "Use old_v4_budget_fraction, ml_v2_budget_fraction, and ml_v3_budget_fraction "
                    "to express how much budget each bot version deserves in this market."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, default=str)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "strategy_router_decision",
                "strict": True,
                "schema": ROUTER_SCHEMA,
            }
        },
        "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        "store": False,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=OPENAI_TIMEOUT,
    )
    response.raise_for_status()
    text = parse_response_text(response.json())
    return json.loads(text)


def deterministic_router(candidates: list[Candidate]) -> dict:
    if not candidates:
        return {
            "decision": "no_trade",
            "symbol": "",
            "budget_fraction": 0.0,
            "leverage": 1,
            "old_v4_budget_fraction": 0.0,
            "ml_budget_fraction": 0.0,
            "ml_v2_budget_fraction": 0.0,
            "ml_v3_budget_fraction": 0.0,
            "confidence": 0.0,
            "rationale": "No candidates.",
            "risk_notes": [],
        }
    best = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[0]
    return {
        "decision": best.strategy,
        "symbol": best.symbol,
        "budget_fraction": DEFAULT_BUDGET_FRACTION,
        "leverage": DEFAULT_LEVERAGE,
        "old_v4_budget_fraction": DEFAULT_BUDGET_FRACTION if best.strategy == "old_v4" else 0.0,
        "ml_budget_fraction": DEFAULT_BUDGET_FRACTION if best.strategy.startswith("ml_") else 0.0,
        "ml_v2_budget_fraction": DEFAULT_BUDGET_FRACTION if best.strategy == "ml_v2" else 0.0,
        "ml_v3_budget_fraction": DEFAULT_BUDGET_FRACTION if best.strategy == "ml_v3" else 0.0,
        "confidence": best.confidence,
        "rationale": "Deterministic fallback chose the highest-scored candidate.",
        "risk_notes": ["AI router disabled"],
    }


def validate_decision(decision: dict, candidates: list[Candidate], budget_cap: float, free_margin: float) -> tuple[Optional[Candidate], float, int, list[str]]:
    notes = []
    selected_strategy = str(decision.get("decision", "no_trade"))
    selected_symbol = str(decision.get("symbol", ""))
    if selected_strategy == "no_trade":
        return None, 0.0, 1, notes

    selected = next(
        (
            candidate for candidate in candidates
            if candidate.strategy == selected_strategy and candidate.symbol == selected_symbol
        ),
        None,
    )
    if selected is None:
        notes.append("AI selected a strategy/symbol that was not in the candidate list.")
        return None, 0.0, 1, notes

    leverage = int(decision.get("leverage", DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    leverage = max(1, min(leverage, MAX_LEVERAGE))
    if selected.direction == "short" and leverage < 2:
        leverage = 2

    requested_fraction = float(decision.get("budget_fraction", DEFAULT_BUDGET_FRACTION) or 0)
    if selected.strategy == "old_v4":
        version_fraction = float(decision.get("old_v4_budget_fraction", requested_fraction) or 0)
    elif selected.strategy == "ml_v2":
        version_fraction = float(decision.get("ml_v2_budget_fraction", decision.get("ml_budget_fraction", requested_fraction)) or 0)
    elif selected.strategy == "ml_v3":
        version_fraction = float(decision.get("ml_v3_budget_fraction", decision.get("ml_budget_fraction", requested_fraction)) or 0)
    else:
        version_fraction = requested_fraction
    requested_fraction = min(requested_fraction, version_fraction)
    requested_fraction = max(0.0, requested_fraction)
    margin_usd = min(free_margin * requested_fraction, budget_cap)
    if margin_usd < MIN_TRADE_MARGIN:
        notes.append(f"Budget ${margin_usd:.2f} below minimum ${MIN_TRADE_MARGIN:.2f}.")
        return None, 0.0, leverage, notes

    volume = (margin_usd * leverage) / selected.current_price
    if volume < selected.min_volume:
        notes.append(f"Volume {volume:.8f} below Kraken minimum {selected.min_volume}.")
        return None, 0.0, leverage, notes

    return selected, margin_usd, leverage, notes


def execute_old_v4(config: Config, candidate: Candidate, margin_usd: float, leverage: int) -> str:
    bot = TradingBotV4(config)
    analysis = candidate.details.get("analysis", {
        "final_signal": candidate.signal,
        "confidence": candidate.confidence,
        "reasons": candidate.details.get("reasons", []),
        "v4_data": {},
    })
    analysis["final_signal"] = candidate.signal
    bot.config.DRY_RUN = DRY_RUN
    trading_pair = next(pair for pair in config.TRADING_PAIRS if pair.yf_symbol == candidate.symbol)
    bot.open_position(trading_pair, analysis, candidate.data, candidate.current_price, margin_usd, leverage)
    return f"old_v4 {candidate.symbol} {candidate.direction} margin ${margin_usd:.2f} @ {leverage}x"


def execute_ml(config: Config, kraken: KrakenClient, candidate: Candidate, margin_usd: float, leverage: int) -> str:
    state = load_json(ML_STATE_FILE, {"open": {}, "closed": []})
    profile = get_strategy_profile(candidate.details.get("profile_version", ML_STRATEGY_VERSIONS[0]))
    sig = candidate.details["signal"]
    volume = (margin_usd * leverage) / candidate.current_price
    entry_price = candidate.current_price
    fill_how = "dry-run"
    filled_volume = volume

    if not DRY_RUN:
        entry_price, fill_how, filled_volume = enter_position(
            kraken,
            candidate.kraken_pair,
            "buy",
            volume,
            leverage,
            candidate.current_price,
            True,
        )
    if entry_price is None or filled_volume <= 0:
        raise RuntimeError("ML entry did not fill")

    now = candidate.data.index[-1]
    state.setdefault("open", {})[candidate.symbol] = {
        "direction": "long",
        "entry_price": entry_price,
        "entry_time": str(now),
        "exit_due": str(now + timedelta(hours=profile.horizon)),
        "volume": round(filled_volume, 8),
        "prob_up": round(sig.prob_up, 3),
        "expected_value": round(sig.expected_value, 5),
        "dynamic_threshold": round(sig.dynamic_threshold, 3),
        "estimated_cost": round(sig.estimated_cost, 5),
        "btc_regime": sig.btc_regime,
        "relative_strength_7d": round(sig.relative_strength_7d, 5),
        "margin_usd": round(margin_usd, 2),
        "leverage": leverage,
        "entry_fill": fill_how,
        "model_version": profile.version,
        "opened_by": "strategy_router",
    }
    save_json(ML_STATE_FILE, state)
    return f"ml_{profile.version} {candidate.symbol} long margin ${margin_usd:.2f} @ {leverage}x ({fill_how})"


def notify_decision(telegram: Telegram, title: str, decision: dict, lines: list[str]) -> None:
    body = [f"<b>{html.escape(title)}</b>"]
    body.append(f"Decision: {html.escape(str(decision.get('decision', 'unknown')))} {html.escape(str(decision.get('symbol', '')))}")
    body.append(f"Budget: {float(decision.get('budget_fraction', 0) or 0):.1%} | leverage {decision.get('leverage', 'n/a')}")
    body.append(
        "Version budgets: "
        f"V4 {float(decision.get('old_v4_budget_fraction', 0) or 0):.1%}, "
        f"ML V2 {float(decision.get('ml_v2_budget_fraction', 0) or 0):.1%}, "
        f"ML V3 {float(decision.get('ml_v3_budget_fraction', 0) or 0):.1%}"
    )
    body.append(f"Reason: {html.escape(str(decision.get('rationale', '')))}")
    for note in decision.get("risk_notes", [])[:5]:
        body.append(f"Risk: {html.escape(str(note))}")
    for line in lines:
        body.append(f"• {html.escape(line)}")
    telegram.send("\n".join(body))


def main() -> int:
    config = Config()
    kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, config.KRAKEN_API_URL)
    telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("Missing Kraken credentials.")
        return 1

    print(f"STRATEGY ROUTER | {'DRY-RUN' if DRY_RUN else 'LIVE'} | AI={'on' if AI_ENABLED else 'off'} | model={OPENAI_MODEL}")

    symbols = sorted(set([pair.yf_symbol for pair in config.TRADING_PAIRS] + ML_SYMBOLS + ["BTC-USD"]))
    market_data = load_market_data(symbols)
    if not market_data:
        print("No market data available.")
        return 1

    manage_open_positions(config, kraken, telegram, market_data)

    open_positions = kraken.get_open_positions()
    open_orders = kraken.get_open_orders()
    trade_balance = safe_trade_balance(kraken)
    free_margin, budget_cap = margin_budget_cap(trade_balance)
    if len(open_positions) >= MAX_OPEN_POSITIONS or open_orders:
        print(f"Existing exposure blocks new entries: positions={len(open_positions)}, orders={len(open_orders)}")
        return 0

    old_candidates = collect_old_v4_candidates(config, market_data)
    ml_candidates = collect_ml_candidates(config, market_data)
    candidates = old_candidates + ml_candidates
    print(f"Candidates: old_v4={len(old_candidates)}, ml={len(ml_candidates)}")

    if not candidates:
        if NOTIFY_NO_TRADE:
            telegram.send("<b>Strategy Router</b>\nNo valid candidates this run.")
        print("No valid candidates.")
        return 0

    account_context = {
        "free_margin": free_margin,
        "equity": float(trade_balance.get("e", 0) or 0),
        "margin_used": float(trade_balance.get("m", 0) or 0),
        "budget_cap_usd": budget_cap,
        "dry_run": DRY_RUN,
    }
    market_context = {
        "timestamp": max(df.index[-1] for df in market_data.values()).isoformat(),
        "candidate_count": len(candidates),
        "old_v4_candidate_count": len(old_candidates),
        "ml_candidate_count": len(ml_candidates),
    }

    try:
        decision = ask_openai_router(candidates, account_context, market_context)
    except Exception as exc:
        decision = {
            "decision": "no_trade",
            "symbol": "",
            "budget_fraction": 0.0,
            "leverage": 1,
            "old_v4_budget_fraction": 0.0,
            "ml_budget_fraction": 0.0,
            "ml_v2_budget_fraction": 0.0,
            "ml_v3_budget_fraction": 0.0,
            "confidence": 0.0,
            "rationale": f"OpenAI router error: {exc}",
            "risk_notes": ["AI error caused no-trade fallback"],
        }

    selected, margin_usd, leverage, validation_notes = validate_decision(decision, candidates, budget_cap, free_margin)
    if selected is None:
        lines = validation_notes + [f"Candidates available: {len(candidates)}"]
        if NOTIFY_NO_TRADE or validation_notes or decision.get("decision") != "no_trade":
            notify_decision(telegram, "Strategy Router no-trade", decision, lines)
        print("No trade:", decision.get("rationale", ""), validation_notes)
        save_json(ROUTER_STATE_FILE, {"last_decision": decision, "validation_notes": validation_notes})
        return 0

    result_line = ""
    try:
        if selected.strategy == "old_v4":
            result_line = execute_old_v4(config, selected, margin_usd, leverage)
        elif selected.strategy.startswith("ml_"):
            result_line = execute_ml(config, kraken, selected, margin_usd, leverage)
        else:
            raise RuntimeError(f"Unsupported strategy {selected.strategy}")
    except Exception as exc:
        decision["rationale"] = f"Execution failed after router decision: {exc}"
        notify_decision(telegram, "Strategy Router execution failed", decision, validation_notes)
        raise

    lines = validation_notes + [result_line]
    notify_decision(telegram, "Strategy Router trade", decision, lines)
    save_json(ROUTER_STATE_FILE, {
        "last_decision": decision,
        "selected": candidate_for_prompt(selected),
        "margin_usd": margin_usd,
        "leverage": leverage,
        "dry_run": DRY_RUN,
    })
    print(result_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
