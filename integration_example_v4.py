"""
V4 INTEGRATION EXAMPLE
Shows how to integrate all new modules into the existing V3 bot
"""

import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd

# Import V4 modules
from sentiment_analyzer import (
    SentimentAnalyzer,
    should_trade_based_on_sentiment
)
from onchain_metrics import (
    OnChainAnalyzer,
    should_trade_based_on_onchain
)
from ensemble_strategies import (
    EnsembleSystem,
    integrate_ensemble_with_existing
)
from rl_position_sizing import (
    RLPositionSizer,
    PositionSizeCalculator
)

# ═══════════════════════════════════════════════════════════════════════════
#                   MODIFICATIONS TO THE V3 BOT
# ═══════════════════════════════════════════════════════════════════════════

class TradingBotV4:
    """
    V3 bot extension with V4 functionality.

    This class shows how to integrate:
    - Sentiment analysis
    - On-chain metrics
    - Ensemble strategies
    - RL position sizing
    """

    def __init__(self, config):
        """
        Initialize V4 bot with new components.
        """
        # Existing V3 components
        self.config = config
        # ... (kraken, telegram, position_mgr, etc.)

        # ══════════════════════════════════════════════════════
        #         NEW V4 COMPONENTS
        # ══════════════════════════════════════════════════════

        # 1. Sentiment Analyzer
        if getattr(config, 'USE_SENTIMENT_ANALYSIS', False):
            self.sentiment_analyzer = SentimentAnalyzer(
                api_key=config.CRYPTOCOMPARE_API_KEY
            )
            print("   ✓ Sentiment Analyzer enabled")
        else:
            self.sentiment_analyzer = None

        # 2. On-Chain Analyzer
        if getattr(config, 'USE_ONCHAIN_ANALYSIS', False):
            self.onchain_analyzer = OnChainAnalyzer(
                cryptocompare_api_key=config.CRYPTOCOMPARE_API_KEY
            )
            print("   ✓ On-Chain Analyzer enabled")
        else:
            self.onchain_analyzer = None

        # 3. Ensemble System
        if getattr(config, 'USE_ENSEMBLE_SYSTEM', False):
            weights = {
                'swing': getattr(config, 'WEIGHT_SWING', 0.30),
                'momentum': getattr(config, 'WEIGHT_MOMENTUM', 0.25),
                'mean_reversion': getattr(config, 'WEIGHT_MEAN_REVERSION', 0.25),
                'trend_following': getattr(config, 'WEIGHT_TREND_FOLLOWING', 0.20)
            }
            self.ensemble_system = EnsembleSystem(weights=weights)
            print("   ✓ Ensemble System enabled")
        else:
            self.ensemble_system = None

        # 4. RL Position Sizer
        if getattr(config, 'USE_RL_POSITION_SIZING', False):
            self.rl_sizer = RLPositionSizer(
                learning_rate=getattr(config, 'RL_LEARNING_RATE', 0.1),
                discount_factor=getattr(config, 'RL_DISCOUNT_FACTOR', 0.95),
                epsilon=getattr(config, 'RL_EPSILON', 0.1),
                state_file=getattr(config, 'RL_STATE_FILE', 'rl_state.json')
            )
            self.rl_calculator = PositionSizeCalculator(self.rl_sizer)
            print("   ✓ RL Position Sizing enabled")
        else:
            self.rl_sizer = None
            self.rl_calculator = None

    # ══════════════════════════════════════════════════════
    #      MODIFIED MAIN METHOD
    # ══════════════════════════════════════════════════════

    def analyze_trading_opportunity(self,
                                   pair,
                                   data: pd.DataFrame,
                                   swing_signal: Tuple) -> Dict:
        """
        Analyze a trading opportunity with ALL V4 layers.

        Args:
            pair: TradingPair object
            data: DataFrame with OHLCV data
            swing_signal: (signal, price, confidence) from the swing detector

        Returns:
            Dict with decision and metadata
        """
        symbol = pair.yf_symbol
        signal, signal_price, swing_confidence = swing_signal

        print(f"\n🔍 Multi-Layer Analysis: {symbol}")
        print(f"   Swing Signal: {signal} (conf: {swing_confidence:.2f})")

        result = {
            'can_trade': False,
            'final_signal': None,
            'confidence': 0.0,
            'reasons': [],
            'capital': 0.0,
            'leverage': self.config.LEVERAGE
        }

        # ══════════════════════════════════════════════════════
        #       LAYER 1: SENTIMENT ANALYSIS
        # ══════════════════════════════════════════════════════

        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.get_sentiment(symbol)

            can_trade_sentiment = should_trade_based_on_sentiment(
                sentiment,
                signal,
                min_confidence=getattr(self.config, 'MIN_SENTIMENT_CONFIDENCE', 0.5)
            )

            if not can_trade_sentiment:
                result['reasons'].append(
                    f"❌ Sentiment conflict: {sentiment.overall_score:.2f}"
                )
                return result

            result['reasons'].append(
                f"✓ Sentiment: {sentiment.overall_score:.2f} ({sentiment.signal_type if hasattr(sentiment, 'signal_type') else 'N/A'})"
            )

        # ══════════════════════════════════════════════════════
        #       LAYER 2: ON-CHAIN METRICS
        # ══════════════════════════════════════════════════════

        if self.onchain_analyzer:
            onchain = self.onchain_analyzer.get_onchain_signal(symbol)

            can_trade_onchain = should_trade_based_on_onchain(
                onchain,
                signal,
                min_strength=getattr(self.config, 'MIN_ONCHAIN_STRENGTH', 0.5)
            )

            if not can_trade_onchain:
                result['reasons'].append(
                    f"❌ On-Chain conflict: {onchain.signal_type}"
                )
                return result

            result['reasons'].append(
                f"✓ On-Chain: {onchain.signal_type} (strength: {onchain.strength:.2f})"
            )

        # ══════════════════════════════════════════════════════
        #       LAYER 3: ENSEMBLE STRATEGIES
        # ══════════════════════════════════════════════════════

        if self.ensemble_system:
            ensemble_decision = self.ensemble_system.get_ensemble_decision(
                data, swing_signal
            )

            self.ensemble_system.print_decision_summary(ensemble_decision)

            # Check consensus and confidence
            min_consensus = getattr(self.config, 'MIN_ENSEMBLE_CONSENSUS', 0.6)
            min_confidence = getattr(self.config, 'MIN_ENSEMBLE_CONFIDENCE', 0.6)

            if (ensemble_decision.final_signal != signal or
                ensemble_decision.consensus_level < min_consensus or
                ensemble_decision.confidence < min_confidence):

                result['reasons'].append(
                    f"❌ Ensemble: {ensemble_decision.final_signal} "
                    f"(consensus: {ensemble_decision.consensus_level:.2f}, "
                    f"conf: {ensemble_decision.confidence:.2f})"
                )
                return result

            result['reasons'].append(
                f"✓ Ensemble: {ensemble_decision.final_signal} "
                f"(consensus: {ensemble_decision.consensus_level:.2%})"
            )
            result['confidence'] = ensemble_decision.confidence
        else:
            result['confidence'] = swing_confidence

        # ══════════════════════════════════════════════════════
        #       LAYER 4: RL POSITION SIZING
        # ══════════════════════════════════════════════════════

        if self.rl_calculator:
            # Get optimal capital and leverage
            optimal_capital, optimal_leverage = self.rl_calculator.get_optimal_size(
                data=data,
                signal_confidence=result['confidence'],
                available_capital=self.get_available_capital(),
                base_leverage=self.config.LEVERAGE,
                open_positions=len(self.positions),
                recent_trades=self.get_recent_trades(),
                training=True  # Training mode
            )

            result['capital'] = optimal_capital
            result['leverage'] = optimal_leverage
            result['reasons'].append(
                f"✓ RL Sizing: ${optimal_capital:.2f} @ {optimal_leverage}x"
            )
        else:
            # Use traditional sizing
            allocation = 1.0 / self.config.MAX_POSITIONS
            result['capital'] = self.get_available_capital() * allocation
            result['leverage'] = self.config.LEVERAGE

        # ══════════════════════════════════════════════════════
        #       FINAL DECISION
        # ══════════════════════════════════════════════════════

        result['can_trade'] = True
        result['final_signal'] = signal

        print(f"\n✅ DECISION: {signal}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Capital: ${result['capital']:.2f}")
        print(f"   Leverage: {result['leverage']}x")
        print(f"   Reasons:")
        for reason in result['reasons']:
            print(f"   {reason}")

        return result

    # ══════════════════════════════════════════════════════
    #      MODIFY POSITION OPENING METHOD
    # ══════════════════════════════════════════════════════

    def open_position_v4(self, pair, analysis_result: Dict, current_price: float):
        """
        Open a position using V4 analysis results.

        Args:
            pair: TradingPair object
            analysis_result: Dict from analyze_trading_opportunity
            current_price: Current price
        """
        signal = analysis_result['final_signal']
        capital = analysis_result['capital']
        leverage = analysis_result['leverage']
        confidence = analysis_result['confidence']

        # Calculate volume based on RL capital
        volume = (capital * leverage) / current_price

        # Check minimum volume
        if volume < pair.min_volume:
            print(f"   ⚠️ Volume {volume:.8f} < minimum {pair.min_volume}")
            return

        try:
            print(f"\n🟢 Opening {signal} on {pair.yf_symbol}")
            print(f"   Price: ${current_price:.4f}")
            print(f"   RL Capital: ${capital:.2f}")
            print(f"   RL Leverage: {leverage}x")
            print(f"   Volume: {volume:.8f}")
            print(f"   Ensemble Confidence: {confidence:.2%}")

            if not self.config.DRY_RUN:
                order_type = 'buy' if signal == 'BUY' else 'sell'

                result = self.kraken.place_order(
                    pair=pair.kraken_pair,
                    order_type=order_type,
                    volume=volume,
                    leverage=leverage,
                    reduce_only=False
                )

                print(f"   ✓ Executed: {result}")

                # Save for later RL update
                if self.rl_calculator:
                    self._save_trade_for_rl_update({
                        'symbol': pair.yf_symbol,
                        'entry_price': current_price,
                        'volume': volume,
                        'leverage': leverage,
                        'capital': capital,
                        'confidence': confidence
                    })
            else:
                print(f"   🧪 [SIMULATION]")

            # Notify with V4 details
            self._send_v4_notification(pair, signal, current_price,
                                      volume, leverage, confidence,
                                      analysis_result['reasons'])

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # ══════════════════════════════════════════════════════
    #      RL UPDATE ON POSITION CLOSE
    # ══════════════════════════════════════════════════════

    def close_position_v4(self, pair, pos_data: Dict, exit_price: float, reason: str):
        """
        Close a position and update the RL agent.
        """
        # Close position normally (V3 code)
        # ...

        # Calculate PnL
        entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        leverage = float(pos_data.get('leverage', 1))
        pos_type = pos_data.get('type', 'long')

        if pos_type == 'long':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage

        # ══════════════════════════════════════════════════════
        #       UPDATE RL AGENT
        # ══════════════════════════════════════════════════════

        if self.rl_sizer:
            trade_result = {
                'closed': True,
                'pnl_pct': pnl_pct,
                'exit_reason': reason
            }

            reward = self.rl_sizer.calculate_reward(trade_result)

            # Get history for this position
            trade_history = self._get_trade_history(pair.yf_symbol)

            if trade_history:
                state = trade_history['state']
                action_idx = trade_history['action_idx']

                # Calculate next_state (current state)
                current_data = self.get_market_data(pair.yf_symbol)
                next_state = self.rl_calculator.calculate_market_state(
                    data=current_data,
                    signal_confidence=0.0,  # No signal now
                    open_positions=len(self.positions),
                    recent_trades=self.trades
                )

                # Update Q-values
                self.rl_sizer.update_q_value(state, action_idx, reward, next_state)

                print(f"   🤖 RL Updated: reward={reward:.3f}")

            # Save state periodically
            if len(self.trades) % 5 == 0:  # Every 5 trades
                self.rl_sizer.save_state()

    # ══════════════════════════════════════════════════════
    #      HELPERS
    # ══════════════════════════════════════════════════════

    def _send_v4_notification(self, pair, signal, price, volume,
                             leverage, confidence, reasons):
        """Telegram notification with V4 details."""

        reasons_text = "\n".join([f"• {r}" for r in reasons])

        msg = f"""
🟢 <b>NEW V4 POSITION</b>

<b>Pair:</b> {pair.yf_symbol} ({pair.kraken_pair})
<b>Signal:</b> {signal}
<b>Price:</b> ${price:.4f}
<b>Volume:</b> {volume:.8f}

<b>🤖 AI Decision:</b>
<b>Confidence:</b> {confidence:.1%}
<b>RL Capital:</b> ${volume * price / leverage:.2f}
<b>RL Leverage:</b> {leverage}x

<b>📊 Analysis:</b>
{reasons_text}

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULATION</b>\n" + msg

        self.telegram.send(msg)

    def _save_trade_for_rl_update(self, trade_data: Dict):
        """Save trade data for later RL update."""
        # Implement according to your persistence system
        pass

    def _get_trade_history(self, symbol: str) -> Optional[Dict]:
        """Retrieve trade history for RL update."""
        # Implement according to your persistence system
        pass

    def get_available_capital(self) -> float:
        """Get available capital."""
        # Implement according to your logic
        return 1000.0

    def get_recent_trades(self) -> list:
        """Get list of recent trades."""
        # Implement according to your logic
        return []

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data."""
        # Implement according to your logic
        pass


# ═══════════════════════════════════════════════════════════════════════════
#                   USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

def example_usage():
    """
    Example of how to use the V4 bot.
    """

    # Configuration
    class ConfigV4:
        # Existing V3 config
        KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
        KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
        CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')

        # V4 features
        USE_SENTIMENT_ANALYSIS = True
        USE_ONCHAIN_ANALYSIS = True
        USE_ENSEMBLE_SYSTEM = True
        USE_RL_POSITION_SIZING = True

        MIN_SENTIMENT_CONFIDENCE = 0.5
        MIN_ONCHAIN_STRENGTH = 0.5
        MIN_ENSEMBLE_CONSENSUS = 0.6
        MIN_ENSEMBLE_CONFIDENCE = 0.6

        WEIGHT_SWING = 0.30
        WEIGHT_MOMENTUM = 0.25
        WEIGHT_MEAN_REVERSION = 0.25
        WEIGHT_TREND_FOLLOWING = 0.20

        RL_LEARNING_RATE = 0.1
        RL_EPSILON = 0.1

        MAX_POSITIONS = 3
        LEVERAGE = 3
        DRY_RUN = True

    # Initialize V4 bot
    bot = TradingBotV4(ConfigV4())

    # In your main loop, replace:
    # if swing_signal:
    #     open_position(...)
    #
    # With:
    # if swing_signal:
    #     analysis = bot.analyze_trading_opportunity(pair, data, swing_signal)
    #     if analysis['can_trade']:
    #         bot.open_position_v4(pair, analysis, current_price)


if __name__ == "__main__":
    print("This is an integration example file.")
    print("Copy the relevant functions into your main bot.")
