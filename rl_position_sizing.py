"""
REINFORCEMENT LEARNING POSITION SIZING
RL system to determine optimal position size dynamically

Implements a simplified RL agent that learns:
- How much capital to allocate per trade based on market conditions
- Dynamic adjustment based on recent performance
- Adaptive risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class MarketState:
    """Market state for RL."""
    volatility: float  # Current volatility
    trend_strength: float  # Trend strength
    win_rate_recent: float  # Win rate of recent trades
    drawdown_current: float  # Current drawdown
    positions_open: int  # Number of open positions
    confidence_signal: float  # Confidence of the current signal

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([
            self.volatility,
            self.trend_strength,
            self.win_rate_recent,
            self.drawdown_current,
            self.positions_open / 5,  # Normalize (max 5 positions)
            self.confidence_signal
        ])


@dataclass
class PositionSizingAction:
    """Position sizing action."""
    allocation_pct: float  # % of available capital (0.0 - 1.0)
    leverage_multiplier: float  # Multiplier of base leverage

    def __repr__(self):
        return f"Alloc: {self.allocation_pct:.1%}, Lev: {self.leverage_multiplier:.1f}x"


class RLPositionSizer:
    """
    RL agent for dynamic position sizing.

    Uses simplified Q-Learning with discretized states.
    """

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 state_file: str = "rl_state.json"):
        """
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration probability
            state_file: File to persist Q-table
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_file = state_file

        # Q-table: state -> action -> Q-value
        self.q_table: Dict[str, Dict[int, float]] = {}

        # Available actions (discretized)
        self.actions = self._define_actions()

        # History
        self.history = []

        # Load previous state if it exists
        self.load_state()

    def _define_actions(self) -> List[PositionSizingAction]:
        """
        Define discrete action space.

        Returns:
            List of possible actions
        """
        actions = []

        # Allocation and leverage combinations
        allocations = [0.2, 0.33, 0.5, 0.7, 1.0]  # % of capital
        leverage_mults = [0.5, 1.0, 1.5]  # Base leverage multiplier

        for alloc in allocations:
            for lev_mult in leverage_mults:
                actions.append(PositionSizingAction(alloc, lev_mult))

        return actions

    def _discretize_state(self, state: MarketState) -> str:
        """
        Discretize continuous state to string for Q-table.
        """
        # Discretize each component
        vol_bucket = int(min(4, state.volatility * 100))  # 0-4
        trend_bucket = int(min(2, abs(state.trend_strength) * 10))  # 0-2
        winrate_bucket = int(state.win_rate_recent * 2)  # 0-2 (0-50%, 50-100%)
        dd_bucket = int(min(3, abs(state.drawdown_current) * 10))  # 0-3
        pos_bucket = min(3, state.positions_open)  # 0-3
        conf_bucket = int(state.confidence_signal * 3)  # 0-3

        return f"{vol_bucket}_{trend_bucket}_{winrate_bucket}_{dd_bucket}_{pos_bucket}_{conf_bucket}"

    def select_action(self, state: MarketState, training: bool = True) -> Tuple[int, PositionSizingAction]:
        """
        Select action using epsilon-greedy.

        Args:
            state: Market state
            training: If True, uses exploration

        Returns:
            (action_index, action)
        """
        state_key = self._discretize_state(state)

        # Exploration vs exploitation
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(len(self.actions))
        else:
            # Exploitation: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {i: 0.0 for i in range(len(self.actions))}

            q_values = self.q_table[state_key]
            action_idx = max(q_values.items(), key=lambda x: x[1])[0]

        return action_idx, self.actions[action_idx]

    def update_q_value(self, state: MarketState, action_idx: int,
                      reward: float, next_state: Optional[MarketState] = None):
        """
        Update Q-value using Q-learning.

        Q(s,a) = Q(s,a) + α * [reward + γ * max(Q(s',a')) - Q(s,a)]
        """
        state_key = self._discretize_state(state)

        # Initialize if new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(len(self.actions))}

        current_q = self.q_table[state_key][action_idx]

        # Calculate max Q of next state
        if next_state:
            next_state_key = self._discretize_state(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {i: 0.0 for i in range(len(self.actions))}
            max_next_q = max(self.q_table[next_state_key].values())
        else:
            max_next_q = 0.0

        # Update Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action_idx] = new_q

    def calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward based on trade result.

        Args:
            trade_result: Dict with 'pnl_pct', 'closed', etc.

        Returns:
            Reward value
        """
        if not trade_result.get('closed', False):
            return 0.0  # No reward if trade still open

        pnl_pct = trade_result.get('pnl_pct', 0.0)

        # Reward based on PnL with drawdown penalty
        base_reward = pnl_pct / 10  # Normalize

        # Bonus for large winning trades
        if pnl_pct > 5.0:
            base_reward *= 1.5

        # Penalty for large losing trades
        elif pnl_pct < -3.0:
            base_reward *= 1.5

        # Bonus for risk management (using stop loss)
        if trade_result.get('exit_reason') in ['Stop Loss', 'Trailing']:
            base_reward *= 1.1

        return base_reward

    def get_position_size(self,
                         state: MarketState,
                         available_capital: float,
                         base_leverage: int,
                         training: bool = False) -> Tuple[float, int]:
        """
        Determine position size and leverage using RL.

        Args:
            state: Current market state
            available_capital: Available capital
            base_leverage: Configured base leverage
            training: Whether in training mode

        Returns:
            (capital_to_use, adjusted_leverage)
        """
        action_idx, action = self.select_action(state, training=training)

        # Calculate size based on action
        capital_to_use = available_capital * action.allocation_pct
        adjusted_leverage = int(base_leverage * action.leverage_multiplier)

        # Safety limits
        adjusted_leverage = max(1, min(5, adjusted_leverage))

        # Record for later update
        self.history.append({
            'timestamp': datetime.now(),
            'state': state,
            'action_idx': action_idx,
            'action': action,
            'capital': capital_to_use,
            'leverage': adjusted_leverage
        })

        return capital_to_use, adjusted_leverage

    def save_state(self):
        """Save Q-table to file."""
        try:
            state_data = {
                'q_table': {k: dict(v) for k, v in self.q_table.items()},
                'metadata': {
                    'last_update': datetime.now().isoformat(),
                    'num_states': len(self.q_table),
                    'learning_rate': self.learning_rate,
                    'epsilon': self.epsilon
                }
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            print(f"   💾 RL state saved: {len(self.q_table)} states")

        except Exception as e:
            print(f"   ⚠️ Error saving RL state: {e}")

    def load_state(self):
        """Load Q-table from file."""
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # Restore Q-table
            self.q_table = {
                k: {int(action): value for action, value in actions.items()}
                for k, actions in state_data['q_table'].items()
            }

            metadata = state_data.get('metadata', {})
            print(f"   📁 RL state loaded: {metadata.get('num_states', 0)} states")
            print(f"      Last update: {metadata.get('last_update', 'unknown')}")

        except FileNotFoundError:
            print(f"   ℹ️ No previous RL state found, starting fresh")
        except Exception as e:
            print(f"   ⚠️ Error loading RL state: {e}")


class PositionSizeCalculator:
    """Helper to calculate position size with RL."""

    def __init__(self, rl_sizer: RLPositionSizer):
        self.rl_sizer = rl_sizer
        self.recent_trades = []

    def calculate_market_state(self,
                              data: pd.DataFrame,
                              signal_confidence: float,
                              open_positions: int,
                              recent_trades: List[Dict]) -> MarketState:
        """
        Calculate market state for RL.
        """
        # Volatility
        returns = data['Close'].pct_change().tail(20)
        volatility = returns.std()

        # Trend strength
        ma_fast = data['Close'].tail(10).mean()
        ma_slow = data['Close'].tail(50).mean()
        trend_strength = (ma_fast - ma_slow) / ma_slow if ma_slow > 0 else 0.0

        # Win rate recent
        if recent_trades:
            wins = sum(1 for t in recent_trades[-10:] if t.get('pnl_pct', 0) > 0)
            win_rate = wins / min(10, len(recent_trades))
        else:
            win_rate = 0.5  # Neutral

        # Drawdown
        equity_curve = [1000]  # Simulate from initial capital
        for trade in recent_trades:
            pnl = trade.get('pnl_dollars', 0)
            equity_curve.append(equity_curve[-1] + pnl)

        peak = max(equity_curve)
        current = equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0.0

        return MarketState(
            volatility=volatility,
            trend_strength=trend_strength,
            win_rate_recent=win_rate,
            drawdown_current=drawdown,
            positions_open=open_positions,
            confidence_signal=signal_confidence
        )

    def get_optimal_size(self,
                        data: pd.DataFrame,
                        signal_confidence: float,
                        available_capital: float,
                        base_leverage: int,
                        open_positions: int,
                        recent_trades: List[Dict],
                        training: bool = False) -> Tuple[float, int]:
        """
        Get optimal position size using RL.

        Returns:
            (capital_to_use, leverage_to_use)
        """
        # Calculate state
        state = self.calculate_market_state(
            data, signal_confidence, open_positions, recent_trades
        )

        # Get RL decision
        capital, leverage = self.rl_sizer.get_position_size(
            state, available_capital, base_leverage, training
        )

        print(f"   🤖 RL Position Sizing:")
        print(f"      State: vol={state.volatility:.3f}, trend={state.trend_strength:.2f}, "
              f"wr={state.win_rate_recent:.2f}")
        print(f"      Decision: ${capital:.2f} @ {leverage}x")

        return capital, leverage
