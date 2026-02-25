"""MACD Alpha Model — ported from Lean Algorithm.Framework/Alphas/MacdAlphaModel.py

Detects MACD histogram direction changes as buy/sell signals.
"""
from __future__ import annotations

from typing import Optional


def _ema(prev: Optional[float], price: float, mult: float) -> float:
    if prev is None:
        return price
    return price * mult + prev * (1.0 - mult)


class MacdAlpha:
    """MACD (12, 26, 9) alpha — histogram direction change signals.

    Source: Lean Algorithm.Framework/Alphas/MacdAlphaModel.py
    Logic: when MACD histogram crosses from negative→positive → UP;
    positive→negative → DOWN.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._signal_ema: Optional[float] = None
        self._fast_mult = 2.0 / (fast + 1)
        self._slow_mult = 2.0 / (slow + 1)
        self._signal_mult = 2.0 / (signal + 1)
        self._prev_histogram: Optional[float] = None
        self._bar_count = 0

    def update(self, price: float) -> Optional[str]:
        """Feed new price. Returns 'UP', 'DOWN', or None."""
        self._bar_count += 1
        self._fast_ema = _ema(self._fast_ema, price, self._fast_mult)
        self._slow_ema = _ema(self._slow_ema, price, self._slow_mult)

        if self._bar_count < self.slow:
            return None

        macd_line = self._fast_ema - self._slow_ema
        self._signal_ema = _ema(self._signal_ema, macd_line, self._signal_mult)

        if self._bar_count < self.slow + self.signal_period:
            self._prev_histogram = macd_line - self._signal_ema
            return None

        histogram = macd_line - self._signal_ema
        signal = None

        if self._prev_histogram is not None:
            if self._prev_histogram <= 0 and histogram > 0:
                signal = "UP"
            elif self._prev_histogram >= 0 and histogram < 0:
                signal = "DOWN"

        self._prev_histogram = histogram
        return signal

    @property
    def macd_line(self) -> Optional[float]:
        if self._fast_ema is None or self._slow_ema is None:
            return None
        return self._fast_ema - self._slow_ema

    @property
    def histogram(self) -> Optional[float]:
        ml = self.macd_line
        if ml is None or self._signal_ema is None:
            return None
        return ml - self._signal_ema
