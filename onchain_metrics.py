"""
ON-CHAIN METRICS ANALYZER
Analyzes blockchain metrics for trading signals

Metrics included:
- Exchange flows (deposits/withdrawals)
- Active addresses
- Transaction volume
- Network hash rate (BTC)
- Whale activity
"""

import requests
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class OnChainSignal:
    """Signal derived from on-chain metrics."""
    signal_type: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    metrics: Dict[str, float]
    timestamp: datetime

    def is_strong_signal(self, threshold: float = 0.6) -> bool:
        return self.strength >= threshold


class OnChainAnalyzer:
    """
    Analyzes on-chain metrics to generate signals.

    Data sources:
    - CryptoCompare API (blockchain stats)
    - Public exchange data via APIs
    """

    def __init__(self, cryptocompare_api_key: str):
        self.api_key = cryptocompare_api_key
        self.base_url = "https://min-api.cryptocompare.com"
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 600  # 10 minutes

    def get_onchain_signal(self, symbol: str) -> Optional[OnChainSignal]:
        """
        Analyze on-chain metrics and generate a signal.

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)

        Returns:
            OnChainSignal or None
        """
        clean_symbol = symbol.split('-')[0].upper()

        # Check cache
        cache_key = f"{clean_symbol}_{int(time.time() / self.cache_duration)}"
        if cache_key in self.cache:
            print(f"   🔗 OnChain cache hit: {clean_symbol}")
            return self.cache[cache_key]

        try:
            metrics = {}

            # 1. Blockchain stats
            blockchain_data = self._get_blockchain_stats(clean_symbol)
            if blockchain_data:
                metrics.update(blockchain_data)

            # 2. Exchange flows (if available)
            exchange_flow = self._analyze_exchange_flows(clean_symbol)
            if exchange_flow:
                metrics['exchange_flow_signal'] = exchange_flow

            # 3. Active addresses trend
            address_trend = self._get_active_addresses_trend(clean_symbol)
            if address_trend:
                metrics['address_trend'] = address_trend

            if not metrics:
                return None

            # Calculate aggregated signal
            signal_type, strength = self._calculate_signal(metrics)

            onchain_signal = OnChainSignal(
                signal_type=signal_type,
                strength=strength,
                metrics=metrics,
                timestamp=datetime.now()
            )

            # Cache
            self.cache[cache_key] = onchain_signal

            print(f"   🔗 OnChain {clean_symbol}: {signal_type} (strength: {strength:.2f})")
            print(f"      Metrics: {metrics}")

            return onchain_signal

        except Exception as e:
            print(f"   ⚠️ On-chain analysis error: {e}")
            return None

    def _get_blockchain_stats(self, symbol: str) -> Optional[Dict]:
        """Get blockchain statistics."""
        try:
            url = f"{self.base_url}/data/blockchain/latest"
            params = {
                'fsym': symbol,
                'tsym': 'USD'
            }
            headers = {'authorization': f'Apikey {self.api_key}'}

            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') != 'Success':
                return None

            blockchain_data = data.get('Data', {})

            # Extract relevant metrics
            metrics = {}

            # Transaction count trend
            if 'transaction_count_24h' in blockchain_data:
                metrics['tx_count_24h'] = blockchain_data['transaction_count_24h']

            # Average transaction value
            if 'average_transaction_value' in blockchain_data:
                metrics['avg_tx_value'] = blockchain_data['average_transaction_value']

            # Hash rate (BTC specific)
            if symbol == 'BTC' and 'hashrate' in blockchain_data:
                metrics['hashrate'] = blockchain_data['hashrate']

            return metrics if metrics else None

        except Exception as e:
            print(f"   ⚠️ Blockchain stats error: {e}")
            return None

    def _analyze_exchange_flows(self, symbol: str) -> Optional[float]:
        """
        Analyze exchange flows.

        Returns:
            Score: +1 (mass outflow, bullish) to -1 (mass inflow, bearish)
        """
        # This is a simplified implementation
        # In production, use APIs like Glassnode, CryptoQuant, etc.

        try:
            # Use CryptoCompare data as a proxy
            url = f"{self.base_url}/data/exchange/histoday"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 7,  # Last week
                'aggregate': 1
            }
            headers = {'authorization': f'Apikey {self.api_key}'}

            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') != 'Success':
                return None

            exchange_data = data.get('Data', [])

            if len(exchange_data) < 2:
                return None

            # Compare recent volume vs historical
            recent_volume = sum(d.get('volumeto', 0) for d in exchange_data[-3:])
            historical_volume = sum(d.get('volumeto', 0) for d in exchange_data[:-3])

            if historical_volume == 0:
                return 0.0

            volume_ratio = recent_volume / historical_volume

            # Normalize to -1 to +1
            # Ratio > 1.5 = high volume (possible distribution) = bearish
            # Ratio < 0.7 = low volume (accumulation off exchanges) = bullish

            if volume_ratio > 1.5:
                return -0.3  # Mild bearish
            elif volume_ratio < 0.7:
                return 0.3   # Mild bullish
            else:
                return 0.0

        except Exception as e:
            print(f"   ⚠️ Exchange flows error: {e}")
            return None

    def _get_active_addresses_trend(self, symbol: str) -> Optional[float]:
        """
        Analyze active addresses trend.

        Returns:
            Score: -1 (decreasing) to +1 (increasing)
        """
        try:
            # CryptoCompare does not have this endpoint directly
            # In production, use specialized APIs
            # Here we use a proxy based on general activity

            url = f"{self.base_url}/data/blockchain/histo/day"
            params = {
                'fsym': symbol,
                'limit': 30,
                'aggregate': 1
            }
            headers = {'authorization': f'Apikey {self.api_key}'}

            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') != 'Success':
                return None

            hist_data = data.get('Data', {}).get('Data', [])

            if len(hist_data) < 15:
                return None

            # Compare recent activity vs past
            recent = hist_data[-7:]
            past = hist_data[-14:-7]

            recent_avg = sum(d.get('transaction_count', 0) for d in recent) / len(recent)
            past_avg = sum(d.get('transaction_count', 0) for d in past) / len(past)

            if past_avg == 0:
                return 0.0

            change = (recent_avg - past_avg) / past_avg

            # Normalize to -1 to +1
            normalized = max(-1.0, min(1.0, change * 2))  # *2 to amplify

            return normalized

        except Exception as e:
            print(f"   ⚠️ Active addresses error: {e}")
            return None

    def _calculate_signal(self, metrics: Dict) -> tuple:
        """
        Calculate aggregated signal from multiple metrics.

        Returns:
            (signal_type, strength)
        """
        signals = []
        weights = []

        # Exchange flow signal
        if 'exchange_flow_signal' in metrics:
            signals.append(metrics['exchange_flow_signal'])
            weights.append(0.4)

        # Active addresses trend
        if 'address_trend' in metrics:
            signals.append(metrics['address_trend'])
            weights.append(0.6)

        if not signals:
            return 'NEUTRAL', 0.0

        # Weighted average
        weighted_score = sum(s * w for s, w in zip(signals, weights)) / sum(weights)

        # Classify
        if weighted_score > 0.2:
            signal_type = 'BULLISH'
            strength = min(1.0, abs(weighted_score))
        elif weighted_score < -0.2:
            signal_type = 'BEARISH'
            strength = min(1.0, abs(weighted_score))
        else:
            signal_type = 'NEUTRAL'
            strength = 0.5

        return signal_type, strength


def should_trade_based_on_onchain(onchain_signal: Optional[OnChainSignal],
                                  trade_signal: str,
                                  min_strength: float = 0.5) -> bool:
    """
    Decide whether to trade based on on-chain metrics.

    Args:
        onchain_signal: OnChainSignal for the asset
        trade_signal: 'BUY' or 'SELL'
        min_strength: Minimum signal strength

    Returns:
        True if on-chain confirms the trade
    """
    if onchain_signal is None:
        return True  # Do not block if no data

    if not onchain_signal.is_strong_signal(min_strength):
        return True  # Do not block if signal is weak

    # Check alignment
    if trade_signal == 'BUY':
        return onchain_signal.signal_type != 'BEARISH'
    else:  # SELL
        return onchain_signal.signal_type != 'BULLISH'
