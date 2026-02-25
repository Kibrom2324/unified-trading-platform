from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from src.models.base import BaseEvent


class TradeDecisionEvent(BaseEvent):
    decision: str
    confidence: float
    reason: str
    # Price fields propagated from PredictionEvent for downstream risk sizing
    price: float = 0.0
    prediction: float = 0.0
    # Quantile predictions (returns) for risk calculations
    horizon_30_q10: float = 0.0
    horizon_30_q50: float = 0.0
    horizon_30_q90: float = 0.0
    direction: str = "neutral"
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
