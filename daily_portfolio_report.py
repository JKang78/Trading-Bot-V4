"""
Read-only daily Kraken portfolio report sent to Telegram.

This script does not place, cancel, or modify orders. It reads account balance,
trade balance, open positions, and open orders, then sends a concise summary to
the configured Telegram chat.
"""

import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kraken_bot_v4_advanced import Config, KrakenClient, Telegram


DEFAULT_SYMBOLS = "BTC-USD,ETH-USD,XRP-USD,ADA-USD,SOL-USD,LINK-USD,DOGE-USD"

ASSET_INFO = {
    "ZUSD": ("USD", None),
    "USD": ("USD", None),
    "ZEUR": ("EUR", None),
    "EUR": ("EUR", None),
    "XXBT": ("BTC", "XBTUSD"),
    "XBT": ("BTC", "XBTUSD"),
    "BTC": ("BTC", "XBTUSD"),
    "XETH": ("ETH", "ETHUSD"),
    "ETH": ("ETH", "ETHUSD"),
    "XXRP": ("XRP", "XRPUSD"),
    "XRP": ("XRP", "XRPUSD"),
    "ADA": ("ADA", "ADAUSD"),
    "SOL": ("SOL", "SOLUSD"),
    "LINK": ("LINK", "LINKUSD"),
    "XLINK": ("LINK", "LINKUSD"),
    "XDG": ("DOGE", "XDGUSD"),
    "XXDG": ("DOGE", "XDGUSD"),
    "DOGE": ("DOGE", "XDGUSD"),
}

PAIR_DISPLAY = {
    "XBTUSD": "BTC/USD",
    "XXBTZUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
    "XETHZUSD": "ETH/USD",
    "XRPUSD": "XRP/USD",
    "XXRPZUSD": "XRP/USD",
    "ADAUSD": "ADA/USD",
    "SOLUSD": "SOL/USD",
    "LINKUSD": "LINK/USD",
    "XLINKZUSD": "LINK/USD",
    "XDGUSD": "DOGE/USD",
    "XXDGZUSD": "DOGE/USD",
}


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def as_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def money(value: Optional[float], currency: str = "USD") -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}{currency} {abs(value):,.2f}"


def quantity(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return f"{value:,.8f}".rstrip("0").rstrip(".")


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def load_json_file(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return {}


def local_now() -> datetime:
    timezone = os.getenv("PORTFOLIO_REPORT_TIMEZONE", "Asia/Seoul")
    try:
        return datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return datetime.now().astimezone()


class TickerCache:
    def __init__(self, kraken: KrakenClient):
        self.kraken = kraken
        self.cache = {}

    def get(self, pair: str) -> Optional[dict]:
        if not pair:
            return None
        if pair in self.cache:
            return self.cache[pair]
        try:
            result = self.kraken._request("/0/public/Ticker", data={"pair": pair})
            for _, info in result.items():
                ticker = {
                    "last": as_float(info.get("c", [None])[0]),
                    "open": as_float(info.get("o")),
                    "high": as_float(info.get("h", [None, None])[1]),
                    "low": as_float(info.get("l", [None, None])[1]),
                }
                self.cache[pair] = ticker
                return ticker
        except Exception:
            self.cache[pair] = None
            return None
        return None


def symbol_to_pair(symbol: str, config: Config) -> Optional[str]:
    for trading_pair in config.TRADING_PAIRS:
        if trading_pair.yf_symbol == symbol:
            return trading_pair.kraken_pair

    fallback = {
        "BTC-USD": "XBTUSD",
        "ETH-USD": "ETHUSD",
        "XRP-USD": "XRPUSD",
        "ADA-USD": "ADAUSD",
        "SOL-USD": "SOLUSD",
        "LINK-USD": "LINKUSD",
        "DOGE-USD": "XDGUSD",
    }
    return fallback.get(symbol)


def display_pair(pair: str) -> str:
    if not pair:
        return "UNKNOWN"
    return PAIR_DISPLAY.get(pair, pair)


def nonzero_balances(kraken: KrakenClient) -> dict:
    result = kraken._request("/0/private/Balance", private=True)
    balances = {}
    for asset, amount in result.items():
        value = as_float(amount, 0.0) or 0.0
        if abs(value) > 1e-12:
            balances[asset] = value
    return balances


def trade_balance(kraken: KrakenClient) -> dict:
    result = kraken._request("/0/private/TradeBalance", private=True)
    return {key: as_float(value) for key, value in result.items()}


def open_positions(kraken: KrakenClient) -> dict:
    try:
        return kraken._request(
            "/0/private/OpenPositions",
            data={"docalcs": "true"},
            private=True,
        )
    except Exception as exc:
        if "No open positions" in str(exc):
            return {}
        raise


def open_orders(kraken: KrakenClient) -> dict:
    result = kraken._request("/0/private/OpenOrders", private=True)
    return result.get("open", {}) if result else {}


def balance_rows(balances: dict, tickers: TickerCache, limit: int) -> tuple[list[str], int]:
    rows = []
    sortable = []

    for raw_asset, amount in balances.items():
        asset, pair = ASSET_INFO.get(raw_asset, (raw_asset, None))
        usd_value = None
        if asset == "USD":
            usd_value = amount
        elif pair:
            ticker = tickers.get(pair)
            if ticker and ticker["last"] is not None:
                usd_value = amount * ticker["last"]

        sortable.append((usd_value if usd_value is not None else -1.0, asset, raw_asset, amount, usd_value))

    for usd_value, asset, raw_asset, amount, value in sorted(sortable, reverse=True)[:limit]:
        name = asset if asset != raw_asset else raw_asset
        if value is None:
            rows.append(f"- {name}: {quantity(amount)}")
        else:
            rows.append(f"- {name}: {quantity(amount)} ~= {money(value)}")

    return rows, len(sortable)


def position_rows(positions: dict, tickers: TickerCache, limit: int) -> tuple[list[str], int]:
    rows = []
    items = list(positions.items())

    for pos_id, pos in items[:limit]:
        pair = pos.get("pair") or pos_id
        raw_side = str(pos.get("type", "?")).lower()
        side = {"buy": "LONG", "sell": "SHORT"}.get(raw_side, raw_side.upper())
        vol = (as_float(pos.get("vol"), 0.0) or 0.0) - (as_float(pos.get("vol_closed"), 0.0) or 0.0)
        cost = as_float(pos.get("cost"), 0.0) or 0.0
        margin = as_float(pos.get("margin"), 0.0) or 0.0
        leverage = as_float(pos.get("leverage"))
        entry = cost / vol if vol else None

        ticker = tickers.get(pair)
        last = ticker["last"] if ticker else None
        pnl_value = as_float(pos.get("net"))

        if pnl_value is None and last is not None and entry is not None:
            if side == "LONG":
                pnl_value = (last - entry) * vol
            elif side == "SHORT":
                pnl_value = (entry - last) * vol

        if margin:
            pnl_pct = (pnl_value / margin * 100) if pnl_value is not None else None
        elif cost:
            pnl_pct = (pnl_value / cost * 100) if pnl_value is not None else None
        else:
            pnl_pct = None

        entry_text = money(entry) if entry is not None else "n/a"
        last_text = money(last) if last is not None else "n/a"
        pnl_text = f"{money(pnl_value)} ({pct(pnl_pct)})" if pnl_value is not None else "n/a"
        margin_text = money(margin) if margin else "n/a"
        if leverage is not None:
            exposure_text = f"lev {leverage:g}x"
        elif margin:
            exposure_text = f"margin position, notional/margin {cost / margin:.2f}x"
        else:
            exposure_text = "margin position"

        terms = pos.get("terms")
        terms_text = f"; terms {terms}" if terms else ""

        rows.append(
            f"- {display_pair(pair)} {side} {quantity(vol)} @ {entry_text} -> {last_text}; "
            f"P/L {pnl_text}; margin {margin_text}; {exposure_text}; id {pos_id}{terms_text}"
        )

    return rows, len(items)


def market_rows(symbols: list[str], config: Config, tickers: TickerCache) -> list[str]:
    rows = []
    for symbol in symbols:
        pair = symbol_to_pair(symbol, config)
        if not pair:
            continue
        ticker = tickers.get(pair)
        if not ticker or ticker["last"] is None:
            continue

        change = None
        if ticker["open"] and ticker["open"] > 0:
            change = (ticker["last"] - ticker["open"]) / ticker["open"] * 100
        label = display_pair(pair).replace("/USD", "")
        rows.append(f"- {label}: {money(ticker['last'])} today {pct(change)}")
    return rows


def ml_state_rows(state_file: str) -> list[str]:
    state = load_json_file(state_file)
    if not state:
        return [f"- No ML live state file found at {state_file}"]

    open_state = state.get("open", {})
    closed = state.get("closed", [])
    wins = [trade for trade in closed if (trade.get("pnl_pct", 0) or 0) > 0]
    win_rate = (len(wins) / len(closed) * 100) if closed else 0.0

    rows = [
        f"- Tracked open: {len(open_state)}",
        f"- Closed trades: {len(closed)} | Win rate: {win_rate:.1f}%",
    ]
    if state.get("updated_at"):
        rows.append(f"- Last state update: {state['updated_at']}")

    for symbol, pos in list(open_state.items())[:5]:
        direction = str(pos.get("direction", "?")).upper()
        entry = as_float(pos.get("entry_price"))
        due = pos.get("exit_due", "n/a")
        rows.append(
            f"- {symbol} {direction}: entry {money(entry)}; "
            f"p_up {pos.get('prob_up', 'n/a')}; exit due {due}"
        )

    return rows


def order_rows(orders: dict, limit: int = 5) -> list[str]:
    if not orders:
        return ["- None"]

    rows = []
    for order_id, order in list(orders.items())[:limit]:
        descr = order.get("descr", {})
        pair = display_pair(descr.get("pair", order.get("pair", "UNKNOWN")))
        order_type = descr.get("type", "?")
        order_kind = descr.get("ordertype", "?")
        price = descr.get("price") or order.get("price") or "market"
        volume = order.get("vol", "n/a")
        rows.append(f"- {pair} {order_type} {order_kind} vol {volume} price {price} ({order_id})")

    remaining = len(orders) - limit
    if remaining > 0:
        rows.append(f"- ... plus {remaining} more")
    return rows


def html_section(title: str, rows: list[str]) -> list[str]:
    escaped_rows = [html.escape(row) for row in rows]
    return [f"<b>{html.escape(title)}</b>", *escaped_rows, ""]


def build_report(config: Config, kraken: KrakenClient) -> tuple[str, dict]:
    tickers = TickerCache(kraken)
    balances = nonzero_balances(kraken)
    trade = trade_balance(kraken)
    positions = open_positions(kraken)
    orders = open_orders(kraken)

    state_file = os.getenv("ML_LIVE_STATE_FILE", "ml_live_state.json")
    symbols_raw = os.getenv("PORTFOLIO_REPORT_SYMBOLS", os.getenv("ML_LIVE_SYMBOLS", DEFAULT_SYMBOLS))
    symbols = [symbol.strip() for symbol in symbols_raw.split(",") if symbol.strip()]
    balance_limit = env_int("PORTFOLIO_REPORT_MAX_BALANCES", 10)
    position_limit = env_int("PORTFOLIO_REPORT_MAX_POSITIONS", 10)

    now = local_now().strftime("%Y-%m-%d %H:%M %Z")
    account_rows = [
        f"Time: {now}",
        f"Equity: {money(trade.get('e'))}",
        f"Trade balance: {money(trade.get('tb'))}",
        f"Free margin: {money(trade.get('mf'))}",
        f"Margin used: {money(trade.get('m'))}",
        f"Unrealized P/L: {money(trade.get('n'))}",
    ]
    margin_level = trade.get("ml")
    if margin_level:
        account_rows.append(f"Margin level: {margin_level:.2f}%")

    bal_rows, balance_count = balance_rows(balances, tickers, balance_limit)
    if balance_count > balance_limit:
        bal_rows.append(f"- ... plus {balance_count - balance_limit} more")

    pos_rows, position_count = position_rows(positions, tickers, position_limit)
    if not pos_rows:
        pos_rows = ["- None"]
    elif position_count > position_limit:
        pos_rows.append(f"- ... plus {position_count - position_limit} more")

    market = market_rows(symbols, config, tickers)
    if not market:
        market = ["- No tracked market prices available"]

    message_lines = [
        "<b>Daily Portfolio Update</b>",
        "",
        *html_section("Account", account_rows),
        *html_section("Open Kraken Margin Positions", pos_rows),
        *html_section("Balances", bal_rows or ["- None"]),
        *html_section("Open Orders", order_rows(orders)),
        *html_section("ML Bot State", ml_state_rows(state_file)),
        *html_section("Tracked Markets", market),
    ]

    stats = {
        "balance_count": balance_count,
        "position_count": position_count,
        "order_count": len(orders),
    }
    return "\n".join(message_lines).strip(), stats


def plain_text(message: str) -> str:
    text = re.sub(r"</?b>", "", message)
    return html.unescape(text)


def main() -> int:
    config = Config()
    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET.")
        return 1

    kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, config.KRAKEN_API_URL)
    message, stats = build_report(config, kraken)

    if env_bool("PORTFOLIO_REPORT_PRINT_STDOUT", False):
        print(plain_text(message))
    else:
        print(
            "Generated portfolio report "
            f"({stats['position_count']} open positions, "
            f"{stats['balance_count']} balances, {stats['order_count']} open orders)."
        )

    if not env_bool("PORTFOLIO_REPORT_SEND_TELEGRAM", True):
        print("Telegram send skipped because PORTFOLIO_REPORT_SEND_TELEGRAM=false.")
        return 0

    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
        return 1

    telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    if not telegram.send(message):
        print("Telegram send failed.")
        return 1

    print("Telegram portfolio report sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
