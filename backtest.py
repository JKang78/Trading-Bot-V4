"""
BACKTESTER FOR THE KRAKEN SWING BOT (V4)

Purpose
-------
This script measures how the bot's CORE strategy would have performed on past
price data. It reuses the *exact same* strategy code as the live bot
(SwingDetectorV3 + RegimeDetector), so the results reflect the real logic and
not a re-guessed copy of it.

What it measures
----------------
- Win rate (percentage of trades that ended in profit)
- Expectancy (average profit per trade, after fees)
- Profit factor (gross profit / gross loss)
- Max drawdown (worst peak-to-trough drop of the equity curve)
- Total return

What it does NOT include (on purpose, to keep step 1 simple)
------------------------------------------------------------
- Sentiment / on-chain filters: these call live APIs that return only the
  *current* value, so they cannot be replayed historically.
- Ensemble / RL layers (can be added later once the baseline is trusted).
- A full multi-coin portfolio simulation: each coin is backtested on its own
  independent equity curve. This is enough to judge win rate and expectancy,
  which is the goal of this step.

Important honesty note
----------------------
This is an approximation. Real fills, funding/rollover fees, and slippage will
differ. Treat the numbers as a *relative* guide (is change A better than B?),
not an exact prediction of future profit.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Reuse the live bot's real strategy classes so the backtest matches production.
from kraken_bot_v4_advanced import Config, SwingDetectorV3, RegimeDetector
from ensemble_strategies import EnsembleSystem, StrategyType


@dataclass
class Trade:
    """One completed trade, with everything we need to score it."""
    symbol: str
    direction: str          # 'long' or 'short'
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    leverage: float
    regime: str
    gross_pnl_pct: float    # profit % on margin, before fees (includes leverage)
    net_pnl_pct: float      # profit % on margin, after fees
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    """Summary numbers for one coin (or the whole run)."""
    symbol: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> List[Trade]:
        return [t for t in self.trades if t.net_pnl_pct > 0]

    @property
    def losses(self) -> List[Trade]:
        return [t for t in self.trades if t.net_pnl_pct <= 0]

    @property
    def win_rate(self) -> float:
        """Fraction of trades that made money (0..1)."""
        if not self.trades:
            return 0.0
        return len(self.wins) / len(self.trades)

    @property
    def avg_win(self) -> float:
        return sum(t.net_pnl_pct for t in self.wins) / len(self.wins) if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return sum(t.net_pnl_pct for t in self.losses) / len(self.losses) if self.losses else 0.0

    @property
    def expectancy(self) -> float:
        """Average net profit % per trade (this is what really matters)."""
        if not self.trades:
            return 0.0
        return sum(t.net_pnl_pct for t in self.trades) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Total winnings divided by total losses. > 1 means profitable."""
        gross_win = sum(t.net_pnl_pct for t in self.wins)
        gross_loss = abs(sum(t.net_pnl_pct for t in self.losses))
        if gross_loss == 0:
            return float('inf') if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def total_return_pct(self) -> float:
        """How much the starting equity grew/shrank overall."""
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100

    @property
    def max_drawdown_pct(self) -> float:
        """Worst drop from a peak in the equity curve (as a positive %)."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            dd = (peak - value) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd


def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download historical OHLCV candles from yfinance (same source the bot uses)."""
    data = yf.Ticker(symbol).history(period=period, interval=interval)
    if data.empty:
        return data
    # yfinance returns a timezone-aware index. The strategy compares timestamps
    # against pd.Timestamp.min (timezone-naive), which crashes on tz-aware data,
    # so we drop the timezone here to keep the comparison valid.
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    # Keep only the columns the strategy needs and drop rows with gaps.
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data


def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range: a measure of how much price typically moves per bar.
    We use it to size stops/targets to each coin's real volatility instead of
    a fixed percentage. Returned as a price-unit Series aligned to `data`.
    """
    high = data['High']
    low = data['Low']
    prev_close = data['Close'].shift(1)
    # True Range = the largest of these three ranges for each bar.
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def passes_trend_filter(window: pd.DataFrame, direction: str, ema_period: int) -> bool:
    """
    Higher-timeframe trend filter. Only allow longs when price is above a long
    EMA (uptrend) and shorts when below it (downtrend). ema_period == 0 disables
    the filter. Using an EMA on the 1h data (e.g. 200 = ~8 days) approximates a
    higher-timeframe trend without downloading a second dataset.
    """
    if ema_period <= 0:
        return True
    ema = window['Close'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    price = float(window['Close'].iloc[-1])
    if direction == 'long':
        return price > ema
    return price < ema


def simulate_symbol(
    symbol: str,
    data: pd.DataFrame,
    config: Config,
    leverage: float,
    base_sl: float,
    base_tp: float,
    base_trail: float,
    fee_rate: float,
    use_regime: bool,
    warmup: int,
    risk_fraction: float,
    starting_equity: float,
    exit_mode: str = 'percent',
    atr_period: int = 14,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
    atr_trail_mult: float = 2.0,
    trend_ema: int = 0,
    use_ensemble: bool = False,
    ensemble_consensus: float = 0.6,
    ensemble_confidence: float = 0.6,
) -> BacktestResult:
    """
    Walk through the candles one bar at a time and simulate trades.

    Entry: only when flat. We ask the SAME SwingDetectorV3 the live bot uses,
    feeding it only the data available up to the current bar (no peeking ahead).
    An optional trend filter can reject entries that fight the higher-timeframe
    trend.

    Exit modes:
    - 'percent': the live bot's rules -> fixed PnL%% stop / take profit / trailing
      (optionally regime-adjusted).
    - 'atr': stop/target/trailing sized to the coin's volatility using ATR.
      stop  = entry -/+ atr_sl_mult * ATR
      target= entry +/- atr_tp_mult * ATR
      trail = chandelier stop at peak -/+ atr_trail_mult * ATR

    Stop/target are checked intrabar (bar High/Low); trailing is checked on the
    bar close. ATR is measured at entry and held fixed for the trade (no peeking).
    """
    result = BacktestResult(symbol=symbol)
    equity = starting_equity
    result.equity_curve.append(equity)

    # Fee cost, expressed as a % of the margin. Fees hit both entry and exit,
    # and apply to the leveraged notional, so leverage multiplies the fee too.
    fee_cost_pct = fee_rate * 100 * leverage * 2

    # Precompute ATR once over the whole series (each entry reads its own bar).
    atr_series = compute_atr(data, atr_period) if exit_mode == 'atr' else None

    # Build the ensemble once (same weights the live bot uses).
    ensemble = None
    if use_ensemble:
        ensemble = EnsembleSystem(weights={
            StrategyType.SWING: config.WEIGHT_SWING,
            StrategyType.MOMENTUM: config.WEIGHT_MOMENTUM,
            StrategyType.MEAN_REVERSION: config.WEIGHT_MEAN_REVERSION,
            StrategyType.TREND_FOLLOWING: config.WEIGHT_TREND_FOLLOWING,
        })

    i = warmup
    n = len(data)

    while i < n:
        # ---- Look for an entry signal using only past+current data ----
        window = data.iloc[: i + 1]
        detector = SwingDetectorV3(
            window,
            volume_filter=config.USE_VOLUME_FILTER,
            use_ml=config.USE_ML_VALIDATION,
            ml_threshold=config.ML_CONFIDENCE_THRESHOLD,
        )
        signal, _swing_price, _conf = detector.get_signal()

        if signal not in ('BUY', 'SELL'):
            i += 1
            result.equity_curve.append(equity)
            continue

        direction = 'long' if signal == 'BUY' else 'short'

        # Optional higher-timeframe trend filter: skip counter-trend entries.
        if not passes_trend_filter(window, direction, trend_ema):
            i += 1
            result.equity_curve.append(equity)
            continue

        # Optional ensemble filter: the 4-strategy vote must confirm the swing
        # direction AND clear the consensus/confidence thresholds. This is the
        # exact gate the live bot applies when USE_ENSEMBLE_SYSTEM=true.
        if ensemble is not None:
            decision = ensemble.get_ensemble_decision(window, (signal, _swing_price, _conf))
            if (decision.final_signal != signal
                    or decision.consensus_level < ensemble_consensus
                    or decision.confidence < ensemble_confidence):
                i += 1
                result.equity_curve.append(equity)
                continue

        entry_price = float(data['Close'].iloc[i])
        entry_time = data.index[i]

        # ---- Decide the stop / target / trailing distances in PRICE units ----
        if exit_mode == 'atr':
            atr = float(atr_series.iloc[i]) if atr_series is not None else 0.0
            if not atr or atr <= 0:
                # Not enough data / flat bar -> skip this entry safely.
                i += 1
                result.equity_curve.append(equity)
                continue
            regime = 'ATR'
            trail_distance = atr_trail_mult * atr  # price-unit trailing gap
            if direction == 'long':
                stop_price = entry_price - atr_sl_mult * atr
                target_price = entry_price + atr_tp_mult * atr
            else:
                stop_price = entry_price + atr_sl_mult * atr
                target_price = entry_price - atr_tp_mult * atr
        else:
            # Percent mode: optionally regime-adjust the base PnL%% levels.
            if use_regime:
                regime = RegimeDetector.detect(window, config.REGIME_LOOKBACK)
                params = RegimeDetector.get_adapted_params(regime, base_sl, base_tp, base_trail)
                sl_pnl, tp_pnl, trail_pnl = params['stop_loss'], params['take_profit'], params['trailing_stop']
            else:
                regime = 'OFF'
                sl_pnl, tp_pnl, trail_pnl = base_sl, base_tp, base_trail
            # Convert PnL-based levels into price levels.
            # pnl_pct = price_change_pct * leverage, so price_change = pnl / leverage.
            if direction == 'long':
                stop_price = entry_price * (1 - (sl_pnl / leverage) / 100)
                target_price = entry_price * (1 + (tp_pnl / leverage) / 100)
            else:
                stop_price = entry_price * (1 + (sl_pnl / leverage) / 100)
                target_price = entry_price * (1 - (tp_pnl / leverage) / 100)
            # Trailing gap in price units, derived from the PnL trailing %.
            trail_distance = entry_price * (trail_pnl / leverage) / 100

        peak_price = entry_price  # tracks best price seen (highest long / lowest short)
        exit_price = None
        exit_reason = None
        exit_index = None

        # ---- Manage the open position bar by bar ----
        j = i + 1
        while j < n:
            bar = data.iloc[j]
            high = float(bar['High'])
            low = float(bar['Low'])
            close = float(bar['Close'])

            if direction == 'long':
                # Conservative: assume the stop is touched before the target.
                if low <= stop_price:
                    exit_price, exit_reason, exit_index = stop_price, 'stop_loss', j
                    break
                if high >= target_price:
                    exit_price, exit_reason, exit_index = target_price, 'take_profit', j
                    break
                peak_price = max(peak_price, high)
                close_pnl = ((close - entry_price) / entry_price) * 100 * leverage
                if close_pnl >= config.MIN_PROFIT_FOR_TRAILING:
                    if (peak_price - close) >= trail_distance:
                        exit_price, exit_reason, exit_index = close, 'trailing', j
                        break
            else:  # short
                if high >= stop_price:
                    exit_price, exit_reason, exit_index = stop_price, 'stop_loss', j
                    break
                if low <= target_price:
                    exit_price, exit_reason, exit_index = target_price, 'take_profit', j
                    break
                peak_price = min(peak_price, low)
                close_pnl = ((entry_price - close) / entry_price) * 100 * leverage
                if close_pnl >= config.MIN_PROFIT_FOR_TRAILING:
                    if (close - peak_price) >= trail_distance:
                        exit_price, exit_reason, exit_index = close, 'trailing', j
                        break
            j += 1

        # If we ran out of data while still holding, close at the last price.
        if exit_price is None:
            exit_price = float(data['Close'].iloc[-1])
            exit_reason = 'end_of_data'
            exit_index = n - 1

        # ---- Score the trade ----
        if direction == 'long':
            gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
        else:
            gross_pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
        net_pnl_pct = gross_pnl_pct - fee_cost_pct

        # Update the equity curve. We only put `risk_fraction` of equity at risk
        # as margin, so the account moves by that fraction of the trade's return.
        equity *= (1 + risk_fraction * net_pnl_pct / 100)

        result.trades.append(Trade(
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            exit_time=data.index[exit_index],
            entry_price=entry_price,
            exit_price=exit_price,
            leverage=leverage,
            regime=regime,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_pct=net_pnl_pct,
            exit_reason=exit_reason,
            bars_held=exit_index - i,
        ))
        result.equity_curve.append(equity)

        # Continue scanning for the next entry from the bar after the exit.
        i = exit_index + 1

    return result


def print_report(results: List[BacktestResult], fee_rate: float, leverage: float) -> None:
    """Print a readable per-coin and overall summary to the terminal."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Leverage: {leverage}x   |   Fee per side: {fee_rate * 100:.3f}%")

    all_trades: List[Trade] = []
    for r in results:
        all_trades.extend(r.trades)
        print(f"\n--- {r.symbol} ---")
        if r.num_trades == 0:
            print("  No trades generated in this period.")
            continue
        print(f"  Trades:        {r.num_trades}")
        print(f"  Win rate:      {r.win_rate * 100:.1f}%")
        print(f"  Avg win:       {r.avg_win:+.2f}%   (on margin, after fees)")
        print(f"  Avg loss:      {r.avg_loss:+.2f}%")
        print(f"  Expectancy:    {r.expectancy:+.2f}% per trade")
        pf = r.profit_factor
        print(f"  Profit factor: {'inf' if pf == float('inf') else f'{pf:.2f}'}")
        print(f"  Total return:  {r.total_return_pct:+.1f}%")
        print(f"  Max drawdown:  {r.max_drawdown_pct:.1f}%")

    # Overall (pooled across all coins) win rate + expectancy.
    print("\n" + "=" * 70)
    print("OVERALL (all coins pooled)")
    print("=" * 70)
    if not all_trades:
        print("  No trades generated. Try a longer --period or check the data source.")
        return

    wins = [t for t in all_trades if t.net_pnl_pct > 0]
    losses = [t for t in all_trades if t.net_pnl_pct <= 0]
    win_rate = len(wins) / len(all_trades) * 100
    expectancy = sum(t.net_pnl_pct for t in all_trades) / len(all_trades)
    gross_win = sum(t.net_pnl_pct for t in wins)
    gross_loss = abs(sum(t.net_pnl_pct for t in losses))
    pf = (gross_win / gross_loss) if gross_loss else float('inf')

    print(f"  Total trades:  {len(all_trades)}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Expectancy:    {expectancy:+.2f}% per trade (after fees)")
    print(f"  Profit factor: {'inf' if pf == float('inf') else f'{pf:.2f}'}")

    # Break down why trades exited - useful for spotting a too-tight stop.
    reasons = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print("  Exit reasons:  " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print("=" * 70)


def save_trades_csv(results: List[BacktestResult], path: str) -> int:
    """Save every trade to a CSV so we can inspect and compare runs later."""
    rows = []
    for r in results:
        for t in r.trades:
            rows.append({
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': round(t.entry_price, 6),
                'exit_price': round(t.exit_price, 6),
                'leverage': t.leverage,
                'regime': t.regime,
                'gross_pnl_pct': round(t.gross_pnl_pct, 4),
                'net_pnl_pct': round(t.net_pnl_pct, 4),
                'exit_reason': t.exit_reason,
                'bars_held': t.bars_held,
            })
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def main() -> None:
    config = Config()

    parser = argparse.ArgumentParser(description="Backtest the Kraken swing bot's core strategy.")
    parser.add_argument('--symbols', default=None,
                        help="Comma-separated yfinance symbols. Default: the bot's trading pairs.")
    parser.add_argument('--period', default=config.LOOKBACK_PERIOD,
                        help="History window, e.g. 180d, 365d, 730d (default from bot config).")
    parser.add_argument('--interval', default=config.CANDLE_INTERVAL,
                        help="Candle size, e.g. 1h, 1d (default from bot config).")
    parser.add_argument('--leverage', type=float, default=float(config.LEVERAGE))
    parser.add_argument('--stop-loss', type=float, default=config.BASE_STOP_LOSS,
                        help="Base stop-loss in PnL%% (includes leverage).")
    parser.add_argument('--take-profit', type=float, default=config.BASE_TAKE_PROFIT)
    parser.add_argument('--trailing', type=float, default=config.BASE_TRAILING_STOP)
    parser.add_argument('--fee', type=float, default=0.0026,
                        help="Fee per side as a fraction (Kraken taker ~0.0026 = 0.26%%).")
    parser.add_argument('--no-regime', action='store_true',
                        help="Disable regime-based SL/TP/trailing multipliers.")
    parser.add_argument('--exit-mode', choices=['percent', 'atr'], default='percent',
                        help="'percent' = live bot's fixed PnL%% rules; "
                             "'atr' = volatility-based stops/targets.")
    parser.add_argument('--atr-period', type=int, default=14)
    parser.add_argument('--atr-sl-mult', type=float, default=1.5,
                        help="Stop distance = this * ATR (atr mode).")
    parser.add_argument('--atr-tp-mult', type=float, default=3.0,
                        help="Target distance = this * ATR (atr mode).")
    parser.add_argument('--atr-trail-mult', type=float, default=2.0,
                        help="Trailing (chandelier) distance = this * ATR (atr mode).")
    parser.add_argument('--trend-ema', type=int, default=0,
                        help="Trend filter EMA period (e.g. 200). 0 = off. "
                             "Longs only above the EMA, shorts only below.")
    parser.add_argument('--use-ensemble', action='store_true',
                        help="Require the 4-strategy ensemble to confirm each entry.")
    parser.add_argument('--ensemble-consensus', type=float, default=config.MIN_ENSEMBLE_CONSENSUS)
    parser.add_argument('--ensemble-confidence', type=float, default=config.MIN_ENSEMBLE_CONFIDENCE)
    parser.add_argument('--warmup', type=int, default=60,
                        help="Bars to skip at the start so indicators have data.")
    parser.add_argument('--out', default='backtest_trades.csv',
                        help="Where to save the per-trade CSV log.")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        allocations = {s: 1.0 for s in symbols}
    else:
        symbols = [p.yf_symbol for p in config.TRADING_PAIRS]
        # Use each pair's allocation (capped at 1.0) as its risk fraction.
        allocations = {p.yf_symbol: min(p.allocation, 1.0) for p in config.TRADING_PAIRS}

    print(f"Backtesting {len(symbols)} symbol(s): {', '.join(symbols)}")
    print(f"Period={args.period}  Interval={args.interval}  Leverage={args.leverage}x")
    if args.exit_mode == 'atr':
        print(f"Exit=ATR  period={args.atr_period}  SL={args.atr_sl_mult}xATR  "
              f"TP={args.atr_tp_mult}xATR  Trail={args.atr_trail_mult}xATR")
    else:
        print(f"Exit=percent  SL={args.stop_loss}%  TP={args.take_profit}%  "
              f"Trail={args.trailing}%  Regime={'OFF' if args.no_regime else 'ON'}")
    print(f"Trend filter EMA={args.trend_ema if args.trend_ema > 0 else 'OFF'}")
    if args.use_ensemble:
        print(f"Ensemble filter=ON  consensus>={args.ensemble_consensus}  "
              f"confidence>={args.ensemble_confidence}")
    else:
        print("Ensemble filter=OFF")
    print("Downloading data and simulating (this can take a minute)...")

    results: List[BacktestResult] = []
    for symbol in symbols:
        data = get_history(symbol, args.period, args.interval)
        if data.empty or len(data) <= args.warmup + 5:
            print(f"  ⚠️ {symbol}: not enough data, skipping.")
            results.append(BacktestResult(symbol=symbol))
            continue
        print(f"  {symbol}: {len(data)} candles from {data.index[0].date()} to {data.index[-1].date()}")
        result = simulate_symbol(
            symbol=symbol,
            data=data,
            config=config,
            leverage=args.leverage,
            base_sl=args.stop_loss,
            base_tp=args.take_profit,
            base_trail=args.trailing,
            fee_rate=args.fee,
            use_regime=not args.no_regime,
            warmup=args.warmup,
            risk_fraction=allocations.get(symbol, 1.0),
            starting_equity=1000.0,
            exit_mode=args.exit_mode,
            atr_period=args.atr_period,
            atr_sl_mult=args.atr_sl_mult,
            atr_tp_mult=args.atr_tp_mult,
            atr_trail_mult=args.atr_trail_mult,
            trend_ema=args.trend_ema,
            use_ensemble=args.use_ensemble,
            ensemble_consensus=args.ensemble_consensus,
            ensemble_confidence=args.ensemble_confidence,
        )
        results.append(result)

    print_report(results, fee_rate=args.fee, leverage=args.leverage)
    saved = save_trades_csv(results, args.out)
    if saved:
        print(f"\nSaved {saved} trades to {args.out}")
    else:
        print("\nNo trades to save.")


if __name__ == "__main__":
    main()
