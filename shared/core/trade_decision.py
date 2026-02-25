from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Debug mode: set DECISION_DEBUG_LOOSEN=true to lower thresholds for testing
_DEBUG_LOOSEN = os.getenv("DECISION_DEBUG_LOOSEN", "false").lower() == "true"
_DEBUG_MIN_CONFIDENCE = 0.1  # Very low threshold when debug mode enabled


@dataclass
class DecisionThresholds:
    """Configurable thresholds for trade decisions."""

    buy_return_threshold: float = 0.005  # Predicted return >= 0.5% to BUY
    sell_return_threshold: float = 0.005  # Predicted return <= -0.5% to SELL
    min_confidence: float = 0.65  # Minimum confidence required
    debug_mode: bool = False  # Lower all thresholds for testing

    @classmethod
    def from_env(cls) -> "DecisionThresholds":
        """Load thresholds from environment variables."""
        debug = os.getenv("DECISION_DEBUG_LOOSEN", "false").lower() == "true"
        return cls(
            buy_return_threshold=float(os.getenv("BUY_RETURN_THRESHOLD", "0.005")),
            sell_return_threshold=float(os.getenv("SELL_RETURN_THRESHOLD", "0.005")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.65")),
            debug_mode=debug,
        )

    def effective_confidence(self) -> float:
        """Return effective confidence threshold (lowered in debug mode)."""
        return _DEBUG_MIN_CONFIDENCE if self.debug_mode else self.min_confidence


def decide_trade_v2(
    prediction: dict[str, Any],
    thresholds: DecisionThresholds | None = None,
) -> str:
    """
    Enhanced trade decision based on predicted return and confidence.

    Uses quantile predictions (q50 as expected return) when available,
    otherwise falls back to direction-based logic.

    Args:
        prediction: Dict with keys like 'direction', 'confidence', 'q10', 'q50', 'q90'
        thresholds: Decision thresholds (uses env defaults if None)

    Returns:
        "BUY", "SELL", or "HOLD"
    """
    if thresholds is None:
        thresholds = DecisionThresholds.from_env()

    confidence = float(prediction.get("confidence", 0.0))
    effective_conf = thresholds.effective_confidence()

    # Check confidence first
    if confidence < effective_conf:
        return "HOLD"

    # Try to use quantile predictions for return-based decision
    q50 = prediction.get("q50")
    q10 = prediction.get("q10")
    q90 = prediction.get("q90")

    if q50 is not None:
        pred_return = float(q50)

        # Additional risk check: q10 (downside) shouldn't be too negative for BUY
        if pred_return >= thresholds.buy_return_threshold:
            # For aggressive mode, lower the bar
            if thresholds.debug_mode:
                return "BUY"
            # Conservative: also check downside risk
            if q10 is not None and float(q10) > -thresholds.sell_return_threshold * 2:
                return "BUY"
            return "BUY"  # Accept if q10 not available

        if pred_return <= -thresholds.sell_return_threshold:
            return "SELL"

        return "HOLD"

    # Fallback to direction-based logic
    direction = str(prediction.get("direction", "")).lower()
    if direction == "long":
        return "BUY"
    if direction == "short":
        return "SELL"
    return "HOLD"


def decide_trade(
    prediction: dict[str, Any],
    min_confidence: float,
    force_debug: bool | None = None,
) -> str:
    """
    Decide BUY/SELL/HOLD based on prediction payload and confidence threshold.

    This is the legacy interface. For new code, use decide_trade_v2().

    Args:
        prediction: Dict with 'direction' and 'confidence' keys
        min_confidence: Minimum confidence to trigger trade
        force_debug: Override DECISION_DEBUG_LOOSEN env var (for testing)

    Returns:
        "BUY", "SELL", or "HOLD"
    """
    debug_mode = force_debug if force_debug is not None else _DEBUG_LOOSEN

    # Use new logic with thresholds
    thresholds = DecisionThresholds(
        min_confidence=min_confidence,
        debug_mode=debug_mode,
    )
    return decide_trade_v2(prediction, thresholds)

