from __future__ import annotations

from src.models.base import BaseEvent


class RiskEvent(BaseEvent):
    """Risk evaluation event with institutional-grade metrics.

    Contains approval decision, risk metrics (CVaR, max loss),
    and position sizing information.
    """
    approved: bool
    reason: str
    risk_score: float
    max_position: float
    max_notional: float
    min_confidence: float
    cvar_95: float = 0.0
    max_loss: float = 0.0
    direction: str = "neutral"
