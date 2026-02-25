from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.market_data.providers import OHLCVBar
from src.models.events_market import MarketRawEvent


class TestDataIngestionAgent:
    def test_market_raw_event_with_ohlcv(self) -> None:
        event = MarketRawEvent(
            idempotency_key="AAPL:20240101120000",
            symbol="AAPL",
            source="data_ingestion",
            correlation_id=uuid4(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            price=100.5,
            volume=1000000,
            vwap=100.25,
            trades=5000,
        )
        assert event.open == 100.0
        assert event.high == 101.0
        assert event.low == 99.0
        assert event.close == 100.5
        assert event.price == 100.5
        assert event.vwap == 100.25
        assert event.trades == 5000

    def test_market_raw_event_backward_compat(self) -> None:
        """Test that old-style events with just price/volume still work."""
        event = MarketRawEvent(
            idempotency_key="AAPL:123",
            symbol="AAPL",
            source="test",
            correlation_id=uuid4(),
            price=100.5,
            volume=1000,
        )
        assert event.price == 100.5
        assert event.open == 0.0  # default
        assert event.vwap is None

    def test_ohlcv_bar_to_event(self) -> None:
        bar = OHLCVBar(
            symbol="MSFT",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            open=400.0,
            high=405.0,
            low=398.0,
            close=402.5,
            volume=2000000,
            vwap=401.0,
            trades=10000,
        )
        event = MarketRawEvent(
            idempotency_key=f"{bar.symbol}:{bar.timestamp.strftime('%Y%m%d%H%M%S')}",
            symbol=bar.symbol,
            source="data_ingestion",
            correlation_id=uuid4(),
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            price=bar.close,
            volume=bar.volume,
            vwap=bar.vwap,
            trades=bar.trades,
        )
        assert event.symbol == "MSFT"
        assert event.close == 402.5
        assert event.idempotency_key == "MSFT:20240101120000"

    def test_to_avro_dict(self) -> None:
        event = MarketRawEvent(
            idempotency_key="GOOG:123",
            symbol="GOOG",
            source="test",
            correlation_id=uuid4(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            price=151.0,
            volume=500000,
        )
        avro_dict = event.to_avro_dict()
        assert avro_dict["symbol"] == "GOOG"
        assert avro_dict["open"] == 150.0
        assert avro_dict["close"] == 151.0
        assert avro_dict["price"] == 151.0
        assert "event_id" in avro_dict
        assert "correlation_id" in avro_dict
