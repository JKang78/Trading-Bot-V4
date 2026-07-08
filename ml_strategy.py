"""
SHARED ML STRATEGY V2 (single source of truth for backtest + live)

Validated improvements (tune_v2_combined.py, walk-forward + holdout):
- Fear & Greed features (+ expectancy vs price-only baseline)
- Higher entry threshold 0.68 (fewer trades, +3.7% vs +0.8% at 0.65)
- 72-bar hold (~3 days) for higher per-trade edge
- Adaptive exit: close early if P(up) drops below 0.40 during the hold
- F&G entry filter: skip when index is 25-40 (that bucket loses money)

Combined V2 config backtest (720d, 5 coins, honest fees):
  +6.30% expectancy/trade, PF 2.18, positive in BOTH early and late halves.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

from research_edge import build_features, build_labels, make_model

# Module-level cache so we only download Fear & Greed once per process.
_FNG_CACHE: Optional[pd.Series] = None


def fetch_fear_greed() -> pd.Series:
    """Download full daily Fear & Greed history (0=extreme fear, 100=greed)."""
    global _FNG_CACHE
    if _FNG_CACHE is not None:
        return _FNG_CACHE
    r = requests.get(
        'https://api.alternative.me/fng/',
        params={'limit': 0, 'format': 'json'},
        timeout=30,
    ).json()
    rows = [
        (pd.Timestamp(int(d['timestamp']), unit='s'), float(d['value']))
        for d in r['data']
    ]
    _FNG_CACHE = pd.Series(dict(rows)).sort_index()
    return _FNG_CACHE


def make_fng_features(fng: pd.Series) -> pd.DataFrame:
    """Daily F&G -> feature columns, shifted 1 day to avoid look-ahead."""
    df = pd.DataFrame(index=fng.index)
    df['fng'] = fng / 100.0
    df['fng_chg_7'] = fng.diff(7) / 100.0
    df['fng_vs_ma30'] = (fng - fng.rolling(30).mean()) / 100.0
    return df.shift(1)


def build_enhanced_features(data: pd.DataFrame,
                            use_fng: bool = True) -> pd.DataFrame:
    """Price features plus optional Fear & Greed, aligned to hourly bars."""
    feats = build_features(data)
    if not use_fng:
        return feats
    fng_feats = make_fng_features(fetch_fear_greed())
    return feats.join(fng_feats.reindex(data.index, method='ffill'))


def fng_at_time(ts: pd.Timestamp) -> float:
    """F&G level visible at time ts (shifted 1 day, no look-ahead)."""
    fng = fetch_fear_greed().shift(1)
    day = pd.Timestamp(ts).normalize()
    val = fng.get(day, np.nan)
    return float(val) if not np.isnan(val) else np.nan


def passes_fng_filter(ts: pd.Timestamp) -> bool:
    """
    Return False for the 25-40 fear bucket that historically loses money.
    If F&G data is unavailable, allow the trade (don't block on missing data).
    """
    val = fng_at_time(ts)
    if np.isnan(val):
        return True
    return not (25 <= val < 40)


@dataclass
class MLSignal:
    """The strategy's decision for one coin at the latest bar."""
    signal: Optional[str]   # 'BUY', 'SELL', or None
    confidence: float       # 0..1, how far the probability is from 0.5
    prob_up: float          # raw model probability that price rises
    horizon: int            # intended hold length in bars
    blocked_reason: Optional[str] = None  # why a signal was suppressed


class MLSwingStrategy:
    """
    Longer-horizon ML entry model with V2 enhancements.
    Call get_signal(data) for entries; should_exit_early(data) for open positions.
    """

    def __init__(
        self,
        horizon: int = 72,
        buy_thr: float = 0.68,
        sell_thr: float = 0.35,
        exit_thr: float = 0.40,
        model_name: str = 'logistic',
        min_train_rows: int = 1000,
        use_fng_features: bool = True,
        use_fng_filter: bool = True,
        long_only: bool = True,
    ):
        self.horizon = horizon
        self.buy_thr = buy_thr
        self.sell_thr = sell_thr
        self.exit_thr = exit_thr
        self.model_name = model_name
        self.min_train_rows = min_train_rows
        self.use_fng_features = use_fng_features
        self.use_fng_filter = use_fng_filter
        self.long_only = long_only

    def _train_and_predict(self, data: pd.DataFrame) -> tuple:
        """
        Train on all past labelled rows and predict prob_up for the latest bar.
        Returns (prob_up, model) or (None, None) if not enough data.
        """
        feats = build_enhanced_features(data, self.use_fng_features)
        labels, _fwd = build_labels(data, self.horizon)

        X = feats.values
        y = labels.values
        valid_feat = ~np.isnan(X).any(axis=1)

        trainable = valid_feat & ~np.isnan(y)
        if trainable.sum() < self.min_train_rows or len(np.unique(y[trainable])) < 2:
            return None, None

        latest_idx = len(data) - 1
        if not valid_feat[latest_idx]:
            return None, None

        model = make_model(self.model_name)
        model.fit(X[trainable], y[trainable])
        prob_up = float(model.predict_proba(X[latest_idx:latest_idx + 1])[:, 1][0])
        return prob_up, model

    def get_signal(self, data: pd.DataFrame) -> MLSignal:
        """Return a BUY/SELL/None decision for the latest bar."""
        prob_up, _ = self._train_and_predict(data)
        if prob_up is None:
            return MLSignal(signal=None, confidence=0.0, prob_up=0.5,
                            horizon=self.horizon)

        if prob_up > self.buy_thr:
            signal = 'BUY'
        elif prob_up < self.sell_thr and not self.long_only:
            signal = 'SELL'
        else:
            signal = None

        blocked_reason = None
        if signal == 'BUY' and self.use_fng_filter:
            latest_ts = data.index[-1]
            if not passes_fng_filter(latest_ts):
                blocked_reason = 'fng_fear_bucket'
                signal = None
        if signal == 'SELL' and self.long_only:
            blocked_reason = 'long_only'
            signal = None

        confidence = min(1.0, abs(prob_up - 0.5) * 2)
        return MLSignal(signal=signal, confidence=confidence, prob_up=prob_up,
                        horizon=self.horizon, blocked_reason=blocked_reason)

    def should_exit_early(self, data: pd.DataFrame) -> tuple[bool, float]:
        """
        For an open long position: return (True, prob_up) if the model now
        thinks price will NOT rise (P(up) < exit_thr). Disabled when exit_thr=0.
        """
        if self.exit_thr <= 0:
            return False, 0.5
        prob_up, _ = self._train_and_predict(data)
        if prob_up is None:
            return False, 0.5
        return prob_up < self.exit_thr, prob_up
