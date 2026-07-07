"""
SHARED ML STRATEGY (the validated longer-horizon edge)

This is the single source of truth for the longer-horizon ML strategy that our
walk-forward backtest validated (edge held across 3 separate time windows).

Both the backtest and the live paper-trading runner import THIS class, so what
we test is exactly what we trade. No duplicated, drifting logic.

Locked spec (from validation):
- Predict direction 48 bars (~2 days at 1h) ahead.
- Trade only high-confidence signals: go long if P(up) > 0.65, short if < 0.35.
- Exit by time (hold 48 bars). No ATR stop (a stop cut winners and hurt results).
- Logistic regression, retrained on all past labelled data each time.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Reuse the exact feature/label/model code the research + backtest used.
from research_edge import build_features, build_labels, make_model


@dataclass
class MLSignal:
    """The strategy's decision for one coin at the latest bar."""
    signal: Optional[str]   # 'BUY', 'SELL', or None
    confidence: float       # 0..1, how far the probability is from 0.5, scaled
    prob_up: float          # raw model probability that price rises
    horizon: int            # how many bars we intend to hold


class MLSwingStrategy:
    """
    Longer-horizon ML entry model. Call get_signal(data) with a coin's OHLCV
    history; it trains on the past and returns a decision for the latest bar.
    """

    def __init__(self, horizon: int = 48, buy_thr: float = 0.65,
                 sell_thr: float = 0.35, model_name: str = 'logistic',
                 min_train_rows: int = 1000):
        self.horizon = horizon
        self.buy_thr = buy_thr
        self.sell_thr = sell_thr
        self.model_name = model_name
        self.min_train_rows = min_train_rows

    def get_signal(self, data: pd.DataFrame) -> MLSignal:
        """
        Train on all past bars whose 48h outcome is already known, then predict
        the direction for the most recent bar. This is causal: the latest bar's
        future is unknown, which is exactly what we are predicting.
        """
        feats = build_features(data)
        labels, _fwd = build_labels(data, self.horizon)

        X = feats.values
        y = labels.values
        valid_feat = ~np.isnan(X).any(axis=1)

        # The last `horizon` rows have no known outcome yet -> not trainable.
        # Train on every earlier row that has valid features and a known label.
        trainable = valid_feat & ~np.isnan(y)
        if trainable.sum() < self.min_train_rows or len(np.unique(y[trainable])) < 2:
            return MLSignal(signal=None, confidence=0.0, prob_up=0.5, horizon=self.horizon)

        # We must not train on the latest bar (its label is unknown), and the
        # latest bar itself must have valid features to predict on.
        latest_idx = len(data) - 1
        if not valid_feat[latest_idx]:
            return MLSignal(signal=None, confidence=0.0, prob_up=0.5, horizon=self.horizon)

        model = make_model(self.model_name)
        model.fit(X[trainable], y[trainable])

        prob_up = float(model.predict_proba(X[latest_idx:latest_idx + 1])[:, 1][0])

        if prob_up > self.buy_thr:
            signal = 'BUY'
        elif prob_up < self.sell_thr:
            signal = 'SELL'
        else:
            signal = None

        # Confidence = distance from a coin flip, scaled to roughly 0..1.
        confidence = min(1.0, abs(prob_up - 0.5) * 2)

        return MLSignal(signal=signal, confidence=confidence,
                        prob_up=prob_up, horizon=self.horizon)
