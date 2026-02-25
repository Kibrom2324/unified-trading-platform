"""Unit tests for trade decision logic."""

import pytest
from datetime import datetime, timezone

from src.core.trade_decision import (
    DecisionThresholds,
    decide_trade,
    decide_trade_v2,
)


class TestDecisionThresholds:
    """Tests for DecisionThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values are sensible."""
        thresholds = DecisionThresholds()
        assert thresholds.buy_return_threshold == 0.005
        assert thresholds.sell_return_threshold == 0.005
        assert thresholds.min_confidence == 0.65

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = DecisionThresholds(
            buy_return_threshold=0.01,
            sell_return_threshold=0.02,
            min_confidence=0.8,
        )
        assert thresholds.buy_return_threshold == 0.01
        assert thresholds.sell_return_threshold == 0.02
        assert thresholds.min_confidence == 0.8

    def test_effective_confidence_normal_mode(self):
        """Effective confidence equals min_confidence in normal mode."""
        thresholds = DecisionThresholds(min_confidence=0.7, debug_mode=False)
        assert thresholds.effective_confidence() == 0.7

    def test_effective_confidence_debug_mode(self):
        """Effective confidence is lowered in debug mode."""
        thresholds = DecisionThresholds(min_confidence=0.7, debug_mode=True)
        assert thresholds.effective_confidence() == 0.1  # Debug threshold


class TestDecideTrade:
    """Tests for legacy decide_trade function (dict-based)."""

    def test_hold_when_low_confidence(self):
        """Hold when confidence below threshold."""
        prediction = {"direction": "long", "confidence": 0.3}
        decision = decide_trade(prediction, min_confidence=0.65)
        assert decision == "HOLD"

    def test_buy_when_direction_long_high_confidence(self):
        """Buy when direction is long with high confidence."""
        prediction = {"direction": "long", "confidence": 0.8}
        decision = decide_trade(prediction, min_confidence=0.65)
        assert decision == "BUY"

    def test_sell_when_direction_short_high_confidence(self):
        """Sell when direction is short with high confidence."""
        prediction = {"direction": "short", "confidence": 0.8}
        decision = decide_trade(prediction, min_confidence=0.65)
        assert decision == "SELL"

    def test_hold_when_unknown_direction(self):
        """Hold when direction is unknown."""
        prediction = {"direction": "neutral", "confidence": 0.8}
        decision = decide_trade(prediction, min_confidence=0.65)
        assert decision == "HOLD"


class TestDecideTradeV2:
    """Tests for enhanced decide_trade_v2 function."""

    def test_buy_with_positive_return_and_high_confidence(self):
        """Buy when predicted return positive and downside risk acceptable."""
        prediction = {
            "q10": 0.01,   # 1% worst case - acceptable
            "q50": 0.02,   # 2% median return - above threshold
            "q90": 0.05,   # 5% best case
            "confidence": 0.8,
        }
        thresholds = DecisionThresholds(buy_return_threshold=0.005)
        decision = decide_trade_v2(prediction, thresholds)
        assert decision == "BUY"

    def test_hold_when_return_below_threshold(self):
        """Hold when predicted return below threshold."""
        prediction = {
            "q10": 0.001,
            "q50": 0.003,  # 0.3% - below 0.5% threshold
            "q90": 0.01,
            "confidence": 0.8,
        }
        thresholds = DecisionThresholds(buy_return_threshold=0.005)
        decision = decide_trade_v2(prediction, thresholds)
        assert decision == "HOLD"

    def test_hold_when_low_confidence(self):
        """Hold when confidence below threshold."""
        prediction = {
            "q10": 0.01,
            "q50": 0.02,
            "q90": 0.05,
            "confidence": 0.3,  # Below threshold
        }
        thresholds = DecisionThresholds(min_confidence=0.65)
        decision = decide_trade_v2(prediction, thresholds)
        assert decision == "HOLD"

    def test_sell_with_negative_return(self):
        """Sell when predicted return is significantly negative."""
        prediction = {
            "q10": -0.05,
            "q50": -0.02,  # -2% median - below sell threshold
            "q90": 0.01,
            "confidence": 0.8,
        }
        thresholds = DecisionThresholds(sell_return_threshold=0.005)
        decision = decide_trade_v2(prediction, thresholds)
        assert decision == "SELL"

    def test_hold_when_prediction_uncertain(self):
        """Hold when predictions span both positive and negative."""
        prediction = {
            "q10": -0.01,  # Negative downside
            "q50": 0.001,  # Small positive - below threshold
            "q90": 0.02,
            "confidence": 0.8,
        }
        thresholds = DecisionThresholds(buy_return_threshold=0.005)
        decision = decide_trade_v2(prediction, thresholds)
        assert decision == "HOLD"

    def test_custom_thresholds_affect_decision(self):
        """Verify custom thresholds change the decision outcome."""
        prediction = {
            "q10": 0.002,
            "q50": 0.004,  # Below default 0.5%
            "q90": 0.01,
            "confidence": 0.8,
        }
        
        # With default thresholds - would be HOLD
        default_thresholds = DecisionThresholds()
        decision_default = decide_trade_v2(prediction, default_thresholds)
        assert decision_default == "HOLD"

        # With lower threshold - should be BUY
        low_thresholds = DecisionThresholds(buy_return_threshold=0.003)
        decision_low = decide_trade_v2(prediction, low_thresholds)
        assert decision_low == "BUY"

    def test_fallback_to_direction_when_no_quantiles(self):
        """Falls back to direction-based logic when no quantiles."""
        prediction = {
            "direction": "long",
            "confidence": 0.8,
        }
        decision = decide_trade_v2(prediction, DecisionThresholds())
        assert decision == "BUY"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_return_prediction(self):
        """Handle zero return prediction."""
        prediction = {
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
            "confidence": 0.8,
        }
        decision = decide_trade_v2(prediction, DecisionThresholds())
        assert decision == "HOLD"

    def test_symmetric_predictions(self):
        """Handle symmetric predictions around zero."""
        prediction = {
            "q10": -0.01,
            "q50": 0.0,
            "q90": 0.01,
            "confidence": 0.8,
        }
        decision = decide_trade_v2(prediction, DecisionThresholds())
        assert decision == "HOLD"

    def test_very_confident_prediction(self):
        """Handle very confident predictions (narrow quantile spread)."""
        prediction = {
            "q10": 0.019,  # All clustered around 2%
            "q50": 0.02,
            "q90": 0.021,
            "confidence": 0.9,
        }
        decision = decide_trade_v2(prediction, DecisionThresholds())
        assert decision == "BUY"

    def test_debug_mode_lowers_confidence_threshold(self):
        """Debug mode allows trades with lower confidence."""
        prediction = {
            "q50": 0.02,
            "confidence": 0.2,  # Very low
        }
        
        # Normal mode - HOLD
        normal = DecisionThresholds(min_confidence=0.65, debug_mode=False)
        assert decide_trade_v2(prediction, normal) == "HOLD"
        
        # Debug mode - BUY
        debug = DecisionThresholds(min_confidence=0.65, debug_mode=True)
        assert decide_trade_v2(prediction, debug) == "BUY"
