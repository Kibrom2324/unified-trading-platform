"""EMA Cross Alpha Model — ported from Lean Algorithm.Framework/Alphas/EmaCrossAlphaModel.py

Detects EMA fast/slow crossover and emits UP/DOWN signals.
"""
from __future__ import annotations

from typing import Optional


def _ema_multiplier(period: int) -> float:
    return 2.0 / (period + 1.0)


class EmaCrossAlpha:
    """Exponential Moving Average crossover alpha.

    Source: Lean Algorithm.Framework/Alphas/EmaCrossAlphaModel.py
    Logic: when fast EMA crosses above slow EMA → UP; below → DOWN.

    Crossover is tracked by remembering whether fast was above slow on the
    PREVIOUS bar (a single bool), which correctly handles convergence cases
    where both EMAs are near-identical before the price swing.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        # Track the relationship on the previous bar (None = not yet determined)
        self._prev_fast_above: Optional[bool] = None
        self._fast_mult = _ema_multiplier(fast_period)
        self._slow_mult = _ema_multiplier(slow_period)
        self._bar_count = 0

    def update(self, price: float) -> Optional[str]:
        """Feed new price. Returns 'UP', 'DOWN', or None on crossover."""
        self._bar_count += 1

        if self._fast_ema is None:
            self._fast_ema = price
        else:
            self._fast_ema = price * self._fast_mult + self._fast_ema * (1 - self._fast_mult)

        if self._slow_ema is None:
            self._slow_ema = price
        else:
            self._slow_ema = price * self._slow_mult + self._slow_ema * (1 - self._slow_mult)

        if self._bar_count < self.slow_period:
            # Still warming up — record relationship but don't signal
            self._prev_fast_above = (self._fast_ema > self._slow_ema)
            return None

        is_above = self._fast_ema > self._slow_ema
        signal = None

        if self._prev_fast_above is not None:
            if not self._prev_fast_above and is_above:
                signal = "UP"
            elif self._prev_fast_above and not is_above:
                signal = "DOWN"

        self._prev_fast_above = is_above
        return signal

    @property
    def fast_ema(self) -> Optional[float]:
        return self._fast_ema

    @property
    def slow_ema(self) -> Optional[float]:
        return self._slow_ema
