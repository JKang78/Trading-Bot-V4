"""
SENTIMENT ANALYZER - Fear & Greed + NewsData.io + optional CryptoCompare
Analyzes market mood from free and optional paid sources.
"""

import os
import requests
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SentimentScore:
    overall_score: float  # -1 (very negative) to +1 (very positive)
    news_score: float
    social_score: float
    newsdata_score: float
    fear_greed_score: float  # Free Fear & Greed Index (-1 to +1)
    confidence: float
    timestamp: datetime
    news_count: int = 0

    def is_bullish(self, threshold: float = 0.2) -> bool:
        return self.overall_score > threshold

    def is_bearish(self, threshold: float = -0.2) -> bool:
        return self.overall_score < threshold


class SentimentAnalyzer:
    """
    Combines sentiment from multiple sources:
    - Fear & Greed Index (free, no API key)
    - NewsData.io (optional free tier)
    - CryptoCompare (optional, if you have a key)
    """

    def __init__(self, cryptocompare_api_key: Optional[str] = None, newsdata_api_key: Optional[str] = None):
        self.cryptocompare_key = (cryptocompare_api_key or '').strip()
        self.newsdata_key = (newsdata_api_key or '').strip()
        self.base_url = "https://min-api.cryptocompare.com"
        self.newsdata_url = "https://newsdata.io/api/1/news"
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

        # Keywords for sentiment analysis
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'gain', 'pump', 'moon', 'adoption',
            'breakthrough', 'soar', 'skyrocket', 'boom', 'breakthrough',
            'positive', 'upgrade', 'partnership', 'growth', 'innovation',
            'institutional', 'buying', 'accumulation', 'breakout'
        ]

        self.negative_keywords = [
            'bearish', 'crash', 'dump', 'drop', 'fall', 'decline', 'regulation',
            'ban', 'crackdown', 'plunge', 'collapse', 'fear', 'panic',
            'lawsuit', 'hack', 'scam', 'fraud', 'bubble', 'selloff',
            'correction', 'weakness', 'concern', 'risk'
        ]

        # Crypto symbols mapping
        self.crypto_names = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'ADA': ['cardano', 'ada'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'MATIC': ['polygon', 'matic'],
            'AVAX': ['avalanche', 'avax'],
            'LINK': ['chainlink', 'link'],
            'DOT': ['polkadot', 'dot']
        }

    def get_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """
        Get aggregated sentiment score from all sources.

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)

        Returns:
            SentimentScore or None on error
        """
        # Normalize symbol (BTC-USD -> BTC)
        clean_symbol = symbol.split('-')[0].upper()

        # Check cache
        cache_key = f"{clean_symbol}_{int(time.time() / self.cache_duration)}"
        if cache_key in self.cache:
            print(f"   📋 Sentiment cache hit: {clean_symbol}")
            return self.cache[cache_key]

        try:
            scores = []
            weights = []
            news_count = 0
            cc_news_score = None
            social_score = None
            newsdata_score = None
            fear_greed_score = None

            # 1. Fear & Greed Index (free, no API key)
            fear_greed_score = self._get_fear_greed_sentiment()
            if fear_greed_score is not None:
                scores.append(fear_greed_score)
                weights.append(0.40)

            # 2. NewsData.io (optional)
            if self.newsdata_key:
                newsdata_score, news_count = self._get_newsdata_sentiment(clean_symbol)
                if newsdata_score is not None:
                    scores.append(newsdata_score)
                    weights.append(0.35)

            # 3. CryptoCompare (optional)
            if self.cryptocompare_key:
                cc_news_score = self._get_cryptocompare_news_sentiment(clean_symbol)
                if cc_news_score is not None:
                    scores.append(cc_news_score)
                    weights.append(0.15)

                social_score = self._get_social_sentiment(clean_symbol)
                if social_score is not None:
                    scores.append(social_score)
                    weights.append(0.10)

            if not scores:
                return None

            overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            max_sources = 4.0 if self.cryptocompare_key else (2.0 if self.newsdata_key else 1.0)
            confidence = min(1.0, len(scores) / max_sources)

            sentiment = SentimentScore(
                overall_score=overall,
                news_score=cc_news_score or 0.0,
                social_score=social_score or 0.0,
                newsdata_score=newsdata_score or 0.0,
                fear_greed_score=fear_greed_score or 0.0,
                confidence=confidence,
                timestamp=datetime.now(),
                news_count=news_count
            )

            self.cache[cache_key] = sentiment

            print(f"   💭 Sentiment {clean_symbol}: Overall={overall:.2f}")
            print(f"      FearGreed={fear_greed_score}, NewsData={newsdata_score}, CC_News={cc_news_score}, Social={social_score}")
            print(f"      Confidence={confidence:.2f}, News analyzed={news_count}")

            return sentiment

        except Exception as e:
            print(f"   ⚠️ Error getting sentiment for {clean_symbol}: {e}")
            return None

    def _get_fear_greed_sentiment(self) -> Optional[float]:
        """Fetch the free Crypto Fear & Greed Index (0-100 mapped to -1..+1)."""
        try:
            response = self.session.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            value = int(data["data"][0]["value"])
            classification = data["data"][0].get("value_classification", "")
            score = (value - 50) / 50.0
            print(f"   📊 Fear & Greed: {value} ({classification})")
            return max(-1.0, min(1.0, score))
        except Exception as e:
            print(f"   ⚠️ Fear & Greed error: {e}")
            return None

    def _get_cryptocompare_news_sentiment(self, symbol: str) -> Optional[float]:
        """
        Analyze sentiment from CryptoCompare news.

        Returns:
            Score from -1 to +1, or None on error
        """
        try:
            url = f"{self.base_url}/data/v2/news/"
            params = {
                'categories': symbol,
                'lang': 'EN'
            }

            headers = {
                'authorization': f'Apikey {self.cryptocompare_key}'
            }

            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') != 'Success':
                return None

            news_items = data.get('Data', [])

            if not news_items:
                return None

            sentiment_scores = []

            for item in news_items[:20]:  # Latest 20 news items
                title = item.get('title', '').lower()
                body = item.get('body', '').lower()
                text = title + ' ' + body

                score = self._calculate_text_sentiment(text)
                if score != 0.0:
                    sentiment_scores.append(score)

            if not sentiment_scores:
                return 0.0

            return sum(sentiment_scores) / len(sentiment_scores)

        except Exception as e:
            print(f"   ⚠️ CryptoCompare news sentiment error: {e}")
            return None

    def _get_newsdata_sentiment(self, symbol: str) -> tuple[Optional[float], int]:
        """
        Analyze sentiment from NewsData.io news.

        Returns:
            (score, news_count) or (None, 0) on error
        """
        if not self.newsdata_key:
            return None, 0

        try:
            # Get names associated with the symbol
            search_terms = self.crypto_names.get(symbol, [symbol.lower()])

            # Build query
            query = ' OR '.join(search_terms)

            params = {
                'apikey': self.newsdata_key,
                'q': query,
                'language': 'en',
                'category': 'business,technology',
                'size': 10  # Latest 10 news items
            }

            response = self.session.get(self.newsdata_url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if data.get('status') != 'success':
                return None, 0

            news_items = data.get('results', [])

            if not news_items:
                return 0.0, 0

            sentiment_scores = []

            for item in news_items:
                # Combine title and description
                title = item.get('title', '').lower()
                description = item.get('description', '').lower()
                content = item.get('content', '').lower() if item.get('content') else ''

                text = f"{title} {description} {content}"

                # Check relevance (does it mention the crypto?)
                is_relevant = any(term in text for term in search_terms)

                if is_relevant:
                    score = self._calculate_text_sentiment(text)
                    sentiment_scores.append(score)

            if not sentiment_scores:
                return 0.0, 0

            avg_score = sum(sentiment_scores) / len(sentiment_scores)

            return avg_score, len(sentiment_scores)

        except Exception as e:
            print(f"   ⚠️ NewsData sentiment error: {e}")
            return None, 0

    def _calculate_text_sentiment(self, text: str) -> float:
        """
        Calculate sentiment from text using keyword matching.

        Returns:
            Score from -1 to +1
        """
        text_lower = text.lower()

        pos_count = sum(1 for word in self.positive_keywords if word in text_lower)
        neg_count = sum(1 for word in self.negative_keywords if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.0

        # Normalized score
        score = (pos_count - neg_count) / (pos_count + neg_count)

        # Weight by keyword count (more keywords = higher confidence)
        total_keywords = pos_count + neg_count
        confidence_multiplier = min(1.0, total_keywords / 5.0)  # Max 5 keywords

        return score * confidence_multiplier

    def _get_social_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get social metrics (Twitter, Reddit, etc.) from CryptoCompare.

        Returns:
            Score from -1 to +1, or None on error
        """
        try:
            url = f"{self.base_url}/data/social/coin/latest"
            params = {
                'coinId': self._get_coin_id(symbol)
            }

            headers = {
                'authorization': f'Apikey {self.cryptocompare_key}'
            }

            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') != 'Success':
                return None

            social_data = data.get('Data', {})

            # Extract key metrics
            twitter_followers = social_data.get('Twitter', {}).get('followers', 0)
            reddit_subscribers = social_data.get('Reddit', {}).get('subscribers', 0)
            twitter_points = social_data.get('Twitter', {}).get('Points', 0)
            reddit_points = social_data.get('Reddit', {}).get('Points', 0)

            if twitter_followers == 0 and reddit_subscribers == 0:
                return None

            # Score based on engagement points
            total_points = twitter_points + reddit_points

            # Normalize
            if total_points > 10000:
                score = 0.5
            elif total_points > 5000:
                score = 0.3
            elif total_points > 1000:
                score = 0.1
            elif total_points < 100:
                score = -0.2
            else:
                score = 0.0

            return score

        except Exception as e:
            print(f"   ⚠️ Social sentiment error: {e}")
            return None

    def _get_coin_id(self, symbol: str) -> int:
        """
        Map symbols to CryptoCompare coin IDs.
        """
        coin_map = {
            'BTC': 1182,
            'ETH': 7605,
            'ADA': 5038,
            'SOL': 151340,
            'XRP': 4614,
            'MATIC': 202330,
            'AVAX': 166503,
            'LINK': 3808,
            'DOT': 165542
        }

        return coin_map.get(symbol, 1182)  # Default BTC

    def get_market_sentiment_summary(self, symbols: list) -> Dict[str, str]:
        """
        Get market sentiment summary for multiple symbols.

        Returns:
            Dict with symbol -> classification (BULLISH/NEUTRAL/BEARISH)
        """
        summary = {}

        for symbol in symbols:
            sentiment = self.get_sentiment(symbol)

            if sentiment is None:
                summary[symbol] = 'UNKNOWN'
            elif sentiment.is_bullish():
                summary[symbol] = 'BULLISH'
            elif sentiment.is_bearish():
                summary[symbol] = 'BEARISH'
            else:
                summary[symbol] = 'NEUTRAL'

        return summary


# Helper function for use in the bot
def should_trade_based_on_sentiment(sentiment: Optional[SentimentScore],
                                   signal: str,
                                   min_confidence: float = 0.5) -> bool:
    """
    Decide whether to trade based on sentiment and technical signal.

    Args:
        sentiment: SentimentScore for the asset
        signal: 'BUY' or 'SELL'
        min_confidence: Minimum required confidence

    Returns:
        True if sentiment confirms the signal
    """
    if sentiment is None:
        return True  # Do not block if no data

    if sentiment.confidence < min_confidence:
        return True  # Do not block if confidence is low

    # Confirm signal with sentiment
    if signal == 'BUY':
        return sentiment.overall_score >= 0.0  # Neutral or positive sentiment
    else:  # SELL
        return sentiment.overall_score <= 0.0  # Neutral or negative sentiment
