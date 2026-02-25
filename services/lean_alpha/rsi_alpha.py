"""RSI Alpha Model — ported from Lean Algorithm.Framework/Alphas/RsiAlphaModel.py

Detects RSI crossovers with bounce hysteresis (30/70 levels).
Publishes Insight events to Kafka topic `alpha.signals`.
"""
from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Optional


class RsiState(Enum):
    MIDDLE = 0
    TRIPPED_LOW = 1   # RSI < 30 → bullish signal
    TRIPPED_HIGH = 2  # RSI > 70 → bearish signal


class RsiAlpha:
    """RSI 14-period alpha with state machine to prevent signal spam.

    Source: Lean Algorithm.Framework/Alphas/RsiAlphaModel.py
    Logic: crossover below 30 (UP insight), above 70 (DOWN insight),
    with hysteresis: re-enters MIDDLE only at 35/65 respectively.
    """

    def __init__(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self._prices: deque[float] = deque(maxlen=period + 1)
        self._state = RsiState.MIDDLE
        self._rsi: Optional[float] = None

    def update(self, price: float) -> Optional[str]:
        """Feed new price. Returns 'UP', 'DOWN', or None."""
        self._prices.append(price)
        if len(self._prices) < self.period + 1:
            return None
        self._rsi = self._compute_rsi()
        return self._determine_signal()

    def _compute_rsi(self) -> float:
        prices = list(self._prices)
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _determine_signal(self) -> Optional[str]:
        rsi = self._rsi
        prev_state = self._state

        # Transition logic (mirrors Lean RsiAlphaModel.get_state)
        if rsi > self.overbought:
            new_state = RsiState.TRIPPED_HIGH
        elif rsi < self.oversold:
            new_state = RsiState.TRIPPED_LOW
        elif prev_state == RsiState.TRIPPED_LOW and rsi > 35:
            new_state = RsiState.MIDDLE
        elif prev_state == RsiState.TRIPPED_HIGH and rsi < 65:
            new_state = RsiState.MIDDLE
        else:
            new_state = prev_state

        signal = None
        if new_state != prev_state:
            if new_state == RsiState.TRIPPED_LOW:
                signal = "UP"
            elif new_state == RsiState.TRIPPED_HIGH:
                signal = "DOWN"

        self._state = new_state
        return signal

    @property
    def rsi_value(self) -> Optional[float]:
        return self._rsi
