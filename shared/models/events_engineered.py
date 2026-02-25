from __future__ import annotations

from src.models.base import BaseEvent


class MarketEngineeredEvent(BaseEvent):
    """Market event with computed features.

    Includes price, volume, and derived technical indicators
    computed by the FeatureEngineeringAgent.
    """
    price: float
    volume: float
    returns_1: float = 0.0
    returns_5: float = 0.0
    returns_15: float = 0.0
    volatility_5: float = 0.0
    volatility_15: float = 0.0
    sma_5: float = 0.0
    sma_20: float = 0.0
    ema_5: float = 0.0
    ema_20: float = 0.0
    rsi_14: float = 50.0
    volume_zscore: float = 0.0
    minute_of_day: int = 0
    day_of_week: int = 0
