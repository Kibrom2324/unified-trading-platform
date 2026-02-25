"""Market regime classifier — Bull / Bear / Sideways.

Uses the 20-bar and 50-bar Simple Moving Average relationship:
  - Bull:     SMA-20 > SMA-50 and price > SMA-20
  - Bear:     SMA-20 < SMA-50 and price < SMA-20
  - Sideways: everything else (range-bound, choppy)

Confidence is scaled per regime:
  - Bull:     1.0x (full confidence)
  - Bear:     0.7x (reduce size in downtrends)
  - Sideways: 0.5x (chop = lower conviction)
"""
from __future__ import annotations

import os
from collections import deque
from enum import Enum
from typing import Optional


class Regime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


# Configurable confidence multipliers per regime
REGIME_CONFIDENCE_SCALE = {
    Regime.BULL: float(os.getenv("REGIME_BULL_SCALE", "1.0")),
    Regime.BEAR: float(os.getenv("REGIME_BEAR_SCALE", "0.7")),
    Regime.SIDEWAYS: float(os.getenv("REGIME_SIDEWAYS_SCALE", "0.5")),
}


class RegimeClassifier:
    """Classify market regime per symbol using SMA 20/50 crossover.

    Usage:
        rc = RegimeClassifier()
        regime = rc.update("AAPL", 150.0)
        scale  = rc.confidence_scale("AAPL")
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: dict[str, deque[float]] = {}
        self._regime: dict[str, Regime] = {}

    def _get_prices(self, symbol: str) -> deque[float]:
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.slow_period)
        return self._prices[symbol]

    def update(self, symbol: str, price: float) -> Regime:
        """Feed a new price bar. Returns current regime classification."""
        prices = self._get_prices(symbol)
        prices.append(price)

        if len(prices) < self.slow_period:
            self._regime[symbol] = Regime.SIDEWAYS
            return Regime.SIDEWAYS

        price_list = list(prices)
        sma_fast = sum(price_list[-self.fast_period:]) / self.fast_period
        sma_slow = sum(price_list) / self.slow_period

        if sma_fast > sma_slow and price > sma_fast:
            regime = Regime.BULL
        elif sma_fast < sma_slow and price < sma_fast:
            regime = Regime.BEAR
        else:
            regime = Regime.SIDEWAYS

        self._regime[symbol] = regime
        return regime

    def get_regime(self, symbol: str) -> Regime:
        """Get latest regime for symbol."""
        return self._regime.get(symbol, Regime.SIDEWAYS)

    def confidence_scale(self, symbol: str) -> float:
        """Get confidence multiplier for current regime."""
        regime = self.get_regime(symbol)
        return REGIME_CONFIDENCE_SCALE.get(regime, 0.5)
