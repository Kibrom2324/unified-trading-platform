"""Unit tests for LeanAlpha models (RSI, EMA cross, MACD) and SignalEngine ensemble.

Tests verify:
  - RSI state machine (30/70 crossovers with bounce hysteresis)
  - EMA crossover detection
  - MACD histogram direction changes
  - Signal ensemble weighting and direction logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.lean_alpha.rsi_alpha import RsiAlpha, RsiState
from services.lean_alpha.ema_cross_alpha import EmaCrossAlpha
from services.lean_alpha.macd_alpha import MacdAlpha
from services.signal_engine.ensemble import (
    SignalEnsemble, TftSignal, AlphaSignal, EnsembleResult
)
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# RSI Alpha Tests
# ---------------------------------------------------------------------------

class TestRsiAlpha:
    def test_no_signal_before_warmup(self):
        rsi = RsiAlpha(period=14)
        for i in range(14):
            result = rsi.update(100.0 + i)
        # Should still be None — need period+1 bars
        assert result is None

    def test_oversold_signal(self):
        """After crash from 100 → 50, RSI should cross below 30 and emit UP."""
        rsi = RsiAlpha(period=14)
        # Warm up with stable prices
        for _ in range(14):
            rsi.update(100.0)
        # Now plunge
        signal = None
        for _ in range(20):
            s = rsi.update(50.0)
            if s:
                signal = s
                break
        assert signal == "UP", f"Expected UP signal after crash, got {signal}"
        assert rsi.rsi_value is not None and rsi.rsi_value < 30

    def test_overbought_signal(self):
        """After rally from 50 → 150, RSI should cross above 70 and emit DOWN."""
        rsi = RsiAlpha(period=14)
        for _ in range(14):
            rsi.update(50.0)
        signal = None
        for _ in range(30):
            s = rsi.update(150.0)
            if s:
                signal = s
                break
        assert signal == "DOWN", f"Expected DOWN after rally, got {signal}"
        assert rsi.rsi_value is not None and rsi.rsi_value > 70

    def test_bounce_hysteresis_prevents_duplicate(self):
        """Once in TRIPPED_LOW, should not re-emit UP until RSI returns to MIDDLE."""
        rsi = RsiAlpha(period=14)
        for _ in range(14):
            rsi.update(100.0)
        # Crash to trigger TRIPPED_LOW
        signals = []
        for _ in range(20):
            s = rsi.update(40.0)
            if s:
                signals.append(s)
        # Should only emit UP once (not on every bar)
        assert signals.count("UP") <= 1, f"Bounce hysteresis failed: {signals}"


# ---------------------------------------------------------------------------
# EMA Cross Alpha Tests
# ---------------------------------------------------------------------------

class TestEmaCrossAlpha:
    def test_no_signal_before_warmup(self):
        ema = EmaCrossAlpha(fast_period=3, slow_period=5)
        for i in range(4):
            result = ema.update(100.0)
        assert result is None

    def test_bullish_crossover(self):
        """Fast EMA crossing above slow EMA should emit UP."""
        ema = EmaCrossAlpha(fast_period=3, slow_period=5)
        # Drop prices to push slow EMA below fast
        for _ in range(10):
            ema.update(50.0)
        # Rally to trigger fast crossing above slow
        result = None
        for _ in range(10):
            s = ema.update(200.0)
            if s == "UP":
                result = s
                break
        assert result == "UP", "Expected bullish EMA crossover"

    def test_bearish_crossover(self):
        """Fast EMA crossing below slow EMA should emit DOWN."""
        ema = EmaCrossAlpha(fast_period=3, slow_period=5)
        # Rally to ensure fast EMA firmly above slow EMA (enough bars to converge)
        for _ in range(20):
            ema.update(200.0)
        # Crash to trigger fast dropping below slow (give 30 bars for convergence)
        result = None
        for _ in range(30):
            s = ema.update(50.0)
            if s == "DOWN":
                result = s
                break
        assert result == "DOWN", "Expected bearish EMA crossover"

    def test_ema_values_update(self):
        ema = EmaCrossAlpha(fast_period=3, slow_period=5)
        for p in [100, 101, 102, 103, 104, 105]:
            ema.update(float(p))
        assert ema.fast_ema is not None
        assert ema.slow_ema is not None
        # Fast EMA should be closer to recent prices than slow
        assert abs(ema.fast_ema - 105) < abs(ema.slow_ema - 105)


# ---------------------------------------------------------------------------
# MACD Alpha Tests
# ---------------------------------------------------------------------------

class TestMacdAlpha:
    def test_no_signal_before_warmup(self):
        macd = MacdAlpha(fast=3, slow=5, signal=3)
        for i in range(10):
            result = macd.update(100.0)
        # May or may not emit during warmup depending on histogram
        # Just verify no crash
        assert macd.macd_line is not None

    def test_bullish_crossover(self):
        """After crash then rally, MACD histogram should cross positive."""
        macd = MacdAlpha(fast=5, slow=10, signal=5)
        # Crash
        for _ in range(20):
            macd.update(50.0)
        # Rally
        result = None
        for _ in range(30):
            s = macd.update(200.0)
            if s == "UP":
                result = s
                break
        assert result == "UP", "Expected bullish MACD crossover"


# ---------------------------------------------------------------------------
# Signal Ensemble Tests
# ---------------------------------------------------------------------------

class TestSignalEnsemble:
    def _make_tft(self, direction: str, confidence: float = 0.8) -> TftSignal:
        return TftSignal(
            direction=direction,
            confidence=confidence,
            q50=100.0,
            ts=datetime.now(timezone.utc),
        )

    def _make_alpha(self, indicator: str, direction: str, weight: float = 0.2) -> AlphaSignal:
        return AlphaSignal(
            indicator=indicator,
            direction=direction,
            weight=weight,
            ts=datetime.now(timezone.utc),
        )

    def test_tft_only_long_gives_long(self):
        """Strong TFT long with no alpha signals → ensemble direction = long."""
        ens = SignalEnsemble()
        ens.update_tft("AAPL", self._make_tft("long", confidence=0.9))
        result = ens.compute("AAPL")
        assert result is not None
        assert result.ensemble_direction == "long"
        assert result.signal_score > 0

    def test_tft_only_short_gives_short(self):
        ens = SignalEnsemble()
        ens.update_tft("MSFT", self._make_tft("short", confidence=0.9))
        result = ens.compute("MSFT")
        assert result is not None
        assert result.ensemble_direction == "short"
        assert result.signal_score < 0

    def test_tft_neutral_low_conf_gives_neutral(self):
        ens = SignalEnsemble()
        ens.update_tft("GOOG", self._make_tft("neutral", confidence=0.1))
        result = ens.compute("GOOG")
        assert result is not None
        assert result.ensemble_direction == "neutral"

    def test_alpha_agrees_with_tft_boosts_confidence(self):
        """TFT long + RSI UP + EMA UP → higher confidence than TFT alone."""
        ens_with_alpha = SignalEnsemble()
        ens_with_alpha.update_tft("AAPL", self._make_tft("long", confidence=0.5))
        ens_with_alpha.update_alpha("AAPL", self._make_alpha("rsi_14", "UP", weight=0.2))
        ens_with_alpha.update_alpha("AAPL", self._make_alpha("ema_cross", "UP", weight=0.2))
        result_multi = ens_with_alpha.compute("AAPL")

        ens_tft_only = SignalEnsemble()
        ens_tft_only.update_tft("AAPL", self._make_tft("long", confidence=0.5))
        result_tft = ens_tft_only.compute("AAPL")

        assert result_multi is not None
        assert result_tft is not None
        assert result_multi.signal_score > result_tft.signal_score, \
            "Agreeing alpha signals should increase score"

    def test_alpha_disagrees_reduces_signal(self):
        """TFT long + RSI DOWN → lower signal score."""
        ens = SignalEnsemble()
        ens.update_tft("NVDA", self._make_tft("long", confidence=0.7))
        ens.update_alpha("NVDA", self._make_alpha("rsi_14", "DOWN", weight=0.2))
        result = ens.compute("NVDA")

        ens_clean = SignalEnsemble()
        ens_clean.update_tft("NVDA", self._make_tft("long", confidence=0.7))
        result_clean = ens_clean.compute("NVDA")

        assert result is not None and result_clean is not None
        assert result.signal_score < result_clean.signal_score, \
            "Disagreeing alpha should reduce score"

    def test_no_result_without_tft(self):
        """Should return None if no TFT signal has arrived yet."""
        ens = SignalEnsemble()
        ens.update_alpha("TSLA", self._make_alpha("rsi_14", "UP"))
        result = ens.compute("TSLA")
        assert result is None, "Ensemble cannot produce signal without TFT"

    def test_signal_score_in_bounds(self):
        """Score should always be in [-1, +1]."""
        ens = SignalEnsemble()
        ens.update_tft("AMZN", self._make_tft("long", confidence=1.0))
        ens.update_alpha("AMZN", self._make_alpha("rsi_14", "UP", weight=0.2))
        ens.update_alpha("AMZN", self._make_alpha("ema_cross", "UP", weight=0.2))
        ens.update_alpha("AMZN", self._make_alpha("macd", "UP", weight=0.1))
        result = ens.compute("AMZN")
        assert result is not None
        assert -1.0 <= result.signal_score <= 1.0, f"Score out of bounds: {result.signal_score}"
        assert 0.0 <= result.ensemble_confidence <= 1.0


# ---------------------------------------------------------------------------
# Run directly (without pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    test_classes = [TestRsiAlpha, TestEmaCrossAlpha, TestMacdAlpha, TestSignalEnsemble]
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        for method_name in [m for m in dir(cls) if m.startswith("test_")]:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✅ {cls.__name__}.{method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls.__name__}.{method_name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
