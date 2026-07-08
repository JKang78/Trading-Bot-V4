"""
ENSEMBLE STRATEGIES SYSTEM
Combines multiple trading strategies with weighted voting

Strategies included:
1. Swing Trading (existing)
2. Momentum Strategy
3. Mean Reversion Strategy
4. Trend Following Strategy
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class StrategyType(Enum):
    SWING = "swing"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"


@dataclass
class StrategySignal:
    """Signal from an individual strategy."""
    strategy: StrategyType
    signal: Optional[str]  # 'BUY', 'SELL', None
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float]
    reason: str

    def __repr__(self):
        return f"{self.strategy.value}: {self.signal} ({self.confidence:.2f})"


@dataclass
class EnsembleDecision:
    """Final ensemble decision."""
    final_signal: Optional[str]
    confidence: float
    votes: Dict[StrategyType, StrategySignal]
    consensus_level: float  # What % of strategies agree

    def is_strong_consensus(self, threshold: float = 0.6) -> bool:
        return self.consensus_level >= threshold


class MomentumStrategy:
    """Momentum-based strategy."""

    @staticmethod
    def get_signal(data: pd.DataFrame, lookback: int = 10) -> StrategySignal:
        """
        Detect strong momentum using RSI and rate of change.
        """
        if len(data) < lookback + 5:
            return StrategySignal(
                strategy=StrategyType.MOMENTUM,
                signal=None,
                confidence=0.0,
                entry_price=None,
                reason="Insufficient data"
            )

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Calculate Rate of Change
        roc = ((data['Close'].iloc[-1] - data['Close'].iloc[-lookback]) /
               data['Close'].iloc[-lookback] * 100)

        # Volume confirmation
        avg_volume = data['Volume'].tail(20).mean()
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Decision logic
        signal = None
        confidence = 0.0
        reason = ""

        # Strong momentum buy
        if rsi.iloc[-1] > 55 and roc > 3 and volume_ratio > 1.2:
            signal = 'BUY'
            confidence = min(0.9, (roc / 10) * volume_ratio * 0.3)
            reason = f"Strong momentum: RSI={current_rsi:.1f}, ROC={roc:.1f}%"

        # Oversold momentum reversal
        elif current_rsi < 30 and roc < -5:
            signal = 'BUY'
            confidence = min(0.8, (abs(roc) / 10) * 0.4)
            reason = f"Oversold reversal: RSI={current_rsi:.1f}"

        # Strong momentum sell
        elif current_rsi > 70 and roc > 5:
            signal = 'SELL'
            confidence = min(0.8, (roc / 10) * 0.4)
            reason = f"Overbought momentum: RSI={current_rsi:.1f}"

        else:
            reason = f"No clear momentum: RSI={current_rsi:.1f}, ROC={roc:.1f}%"

        return StrategySignal(
            strategy=StrategyType.MOMENTUM,
            signal=signal,
            confidence=confidence,
            entry_price=data['Close'].iloc[-1] if signal else None,
            reason=reason
        )


class MeanReversionStrategy:
    """Mean reversion strategy."""

    @staticmethod
    def get_signal(data: pd.DataFrame, bb_period: int = 10,
                  bb_std: float = 2.0) -> StrategySignal:
        """
        Detect mean reversion opportunities using Bollinger Bands.
        """
        if len(data) < bb_period + 5:
            return StrategySignal(
                strategy=StrategyType.MEAN_REVERSION,
                signal=None,
                confidence=0.0,
                entry_price=None,
                reason="Insufficient data"
            )

        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=bb_period).mean()
        std = data['Close'].rolling(window=bb_period).std()

        upper_band = sma + (std * bb_std)
        lower_band = sma - (std * bb_std)

        current_price = data['Close'].iloc[-1]
        current_sma = sma.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]

        # BB Width (volatility)
        bb_width = (current_upper - current_lower) / current_sma

        # Distance from bands
        distance_to_lower = (current_price - current_lower) / current_lower
        distance_to_upper = (current_upper - current_price) / current_upper

        signal = None
        confidence = 0.0
        reason = ""

        # Oversold - near lower band
        if distance_to_lower < 0.02:  # Within 2% of lower band
            signal = 'BUY'
            confidence = min(0.85, (0.02 - distance_to_lower) * 40)
            reason = f"Oversold at lower band (dist: {distance_to_lower:.3f})"

        # Overbought - near upper band
        elif distance_to_upper < 0.02:
            signal = 'SELL'
            confidence = min(0.85, (0.02 - distance_to_upper) * 40)
            reason = f"Overbought at upper band (dist: {distance_to_upper:.3f})"

        else:
            reason = f"Price within bands (BB width: {bb_width:.3f})"

        return StrategySignal(
            strategy=StrategyType.MEAN_REVERSION,
            signal=signal,
            confidence=confidence,
            entry_price=current_price if signal else None,
            reason=reason
        )


class TrendFollowingStrategy:
    """Trend-following strategy."""

    @staticmethod
    def get_signal(data: pd.DataFrame, fast_ma: int = 10,
                  slow_ma: int = 30) -> StrategySignal:
        """
        Follow trends using moving average crossovers.
        """
        if len(data) < slow_ma + 5:
            return StrategySignal(
                strategy=StrategyType.TREND_FOLLOWING,
                signal=None,
                confidence=0.0,
                entry_price=None,
                reason="Insufficient data"
            )

        # Calculate moving averages
        ma_fast = data['Close'].rolling(window=fast_ma).mean()
        ma_slow = data['Close'].rolling(window=slow_ma).mean()

        current_price = data['Close'].iloc[-1]
        current_fast = ma_fast.iloc[-1]
        current_slow = ma_slow.iloc[-1]
        prev_fast = ma_fast.iloc[-2]
        prev_slow = ma_slow.iloc[-2]

        # Trend strength
        trend_strength = abs(current_fast - current_slow) / current_slow

        # Detect crossovers
        golden_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
        death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

        signal = None
        confidence = 0.0
        reason = ""

        # Golden cross (bullish)
        if golden_cross:
            signal = 'BUY'
            confidence = min(0.9, trend_strength * 20)
            reason = f"Golden Cross detected (strength: {trend_strength:.3f})"

        # Death cross (bearish)
        elif death_cross:
            signal = 'SELL'
            confidence = min(0.9, trend_strength * 20)
            reason = f"Death Cross detected (strength: {trend_strength:.3f})"

        # Strong uptrend continuation
        elif current_fast > current_slow and trend_strength > 0.05:
            signal = 'BUY'
            confidence = min(0.7, trend_strength * 10)
            reason = f"Strong uptrend (strength: {trend_strength:.3f})"

        # Strong downtrend continuation
        elif current_fast < current_slow and trend_strength > 0.05:
            signal = 'SELL'
            confidence = min(0.7, trend_strength * 10)
            reason = f"Strong downtrend (strength: {trend_strength:.3f})"

        else:
            reason = f"No clear trend (strength: {trend_strength:.3f})"

        return StrategySignal(
            strategy=StrategyType.TREND_FOLLOWING,
            signal=signal,
            confidence=confidence,
            entry_price=current_price if signal else None,
            reason=reason
        )


class EnsembleSystem:
    """
    Ensemble system that combines multiple strategies.
    """

    def __init__(self, weights: Optional[Dict[StrategyType, float]] = None):
        """
        Args:
            weights: Weights for each strategy. If None, uses equal weights.
        """
        self.weights = weights or {
            StrategyType.SWING: 0.30,
            StrategyType.MOMENTUM: 0.25,
            StrategyType.MEAN_REVERSION: 0.25,
            StrategyType.TREND_FOLLOWING: 0.20
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def get_ensemble_decision(self, data: pd.DataFrame,
                            swing_signal: Optional[Tuple] = None) -> EnsembleDecision:
        """
        Get ensemble decision by combining all strategies.

        Args:
            data: DataFrame with OHLCV data
            swing_signal: Tuple (signal, price, confidence) from swing detector

        Returns:
            EnsembleDecision with the final decision
        """
        votes = {}

        # 1. Swing strategy (if provided)
        if swing_signal and swing_signal[0]:
            votes[StrategyType.SWING] = StrategySignal(
                strategy=StrategyType.SWING,
                signal=swing_signal[0],
                confidence=swing_signal[2] if len(swing_signal) > 2 else 0.5,
                entry_price=swing_signal[1],
                reason="Swing point detected"
            )

        # 2. Momentum strategy
        votes[StrategyType.MOMENTUM] = MomentumStrategy.get_signal(data)

        # 3. Mean reversion strategy
        votes[StrategyType.MEAN_REVERSION] = MeanReversionStrategy.get_signal(data)

        # 4. Trend following strategy
        votes[StrategyType.TREND_FOLLOWING] = TrendFollowingStrategy.get_signal(data)

        # Calculate final decision
        final_signal, confidence, consensus = self._aggregate_votes(votes)

        return EnsembleDecision(
            final_signal=final_signal,
            confidence=confidence,
            votes=votes,
            consensus_level=consensus
        )

    def _aggregate_votes(self, votes: Dict[StrategyType, StrategySignal]) -> Tuple:
        """
        Aggregate votes from all strategies.

        Returns:
            (final_signal, confidence, consensus_level)
        """
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        buy_count = 0
        sell_count = 0
        total_count = 0

        for strategy_type, signal_obj in votes.items():
            if signal_obj.signal is None:
                continue

            weight = self.weights.get(strategy_type, 0.0)
            weighted_confidence = signal_obj.confidence * weight

            total_count += 1

            if signal_obj.signal == 'BUY':
                buy_score += weighted_confidence
                buy_count += 1
            elif signal_obj.signal == 'SELL':
                sell_score += weighted_confidence
                sell_count += 1

            total_weight += weight

        if total_count == 0:
            return None, 0.0, 0.0

        # Calculate consensus
        max_votes = max(buy_count, sell_count) if total_count > 0 else 0
        consensus_level = max_votes / total_count if total_count > 0 else 0.0

        # Final decision based on weighted scores
        if buy_score > sell_score and buy_score > 0.3:  # Minimum threshold
            final_signal = 'BUY'
            confidence = min(1.0, buy_score / total_weight) if total_weight > 0 else 0.0
        elif sell_score > buy_score and sell_score > 0.3:
            final_signal = 'SELL'
            confidence = min(1.0, sell_score / total_weight) if total_weight > 0 else 0.0
        else:
            final_signal = None
            confidence = 0.0

        return final_signal, confidence, consensus_level

    def print_decision_summary(self, decision: EnsembleDecision):
        """Print ensemble decision summary."""
        print(f"\n   📊 ENSEMBLE DECISION")
        print(f"   Signal: {decision.final_signal or 'NONE'}")
        print(f"   Confidence: {decision.confidence:.2%}")
        print(f"   Consensus: {decision.consensus_level:.2%}")
        print(f"\n   Individual votes:")

        for strategy_type, signal in decision.votes.items():
            status = "✓" if signal.signal == decision.final_signal else "✗"
            print(f"   {status} {signal}")


def integrate_ensemble_with_existing(swing_signal: Optional[Tuple],
                                    data: pd.DataFrame,
                                    min_consensus: float = 0.5,
                                    min_confidence: float = 0.5) -> Tuple[bool, float]:
    """
    Integrate ensemble with the existing system.

    Args:
        swing_signal: Signal from the swing detector
        data: DataFrame with market data
        min_consensus: Minimum required consensus
        min_confidence: Minimum required confidence

    Returns:
        (should_trade, ensemble_confidence)
    """
    ensemble = EnsembleSystem()
    decision = ensemble.get_ensemble_decision(data, swing_signal)

    # Check if ensemble confirms the trade
    if swing_signal and swing_signal[0]:
        swing_dir = swing_signal[0]

        # Ensemble must confirm direction
        confirms_direction = (decision.final_signal == swing_dir)

        # Check thresholds
        meets_consensus = decision.consensus_level >= min_consensus
        meets_confidence = decision.confidence >= min_confidence

        should_trade = confirms_direction and meets_consensus and meets_confidence

        return should_trade, decision.confidence

    return False, 0.0
