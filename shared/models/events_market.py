from __future__ import annotations

from pydantic import Field

from src.models.base import BaseEvent


class MarketRawEvent(BaseEvent):
    open: float = Field(default=0.0)
    high: float = Field(default=0.0)
    low: float = Field(default=0.0)
    close: float = Field(default=0.0)
    price: float  # backward compat: alias for close
    volume: float = Field(default=0.0)
    vwap: float | None = Field(default=None)
    trades: int | None = Field(default=None)
