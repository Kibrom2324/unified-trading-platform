from __future__ import annotations

from src.models.base import BaseEvent


class PredictionEvent(BaseEvent):
    """TFT prediction event with quantile outputs.

    Contains quantile predictions (q10, q50, q90) for
    multiple time horizons (30s, 45s, 60s).
    """
    prediction: float
    confidence: float
    horizon_minutes: int
    model_version: str
    horizon_30_q10: float = 0.0
    horizon_30_q50: float = 0.0
    horizon_30_q90: float = 0.0
    horizon_45_q10: float = 0.0
    horizon_45_q50: float = 0.0
    horizon_45_q90: float = 0.0
    horizon_60_q10: float = 0.0
    horizon_60_q50: float = 0.0
    horizon_60_q90: float = 0.0
    direction: str = "neutral"
