from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.market_data.providers import (
    AlpacaProvider,
    OHLCVBar,
    PolygonProvider,
    create_provider,
)


@pytest.fixture
def polygon_response() -> dict:
    return {
        "results": [
            {
                "t": 1704067200000,
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.5,
                "v": 1000000,
                "vw": 100.25,
                "n": 5000,
            },
            {
                "t": 1704067260000,
                "o": 100.5,
                "h": 101.5,
                "l": 100.0,
                "c": 101.0,
                "v": 1200000,
                "vw": 100.75,
                "n": 6000,
            },
        ]
    }


@pytest.fixture
def alpaca_response() -> dict:
    return {
        "bars": [
            {
                "t": "2024-01-01T00:00:00Z",
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.5,
                "v": 1000000,
                "vw": 100.25,
                "n": 5000,
            },
        ]
    }


class TestPolygonProvider:
    @pytest.mark.asyncio
    async def test_get_bars_parses_response(self, polygon_response: dict) -> None:
        provider = PolygonProvider(
            api_key="test_key",
            base_url="https://api.polygon.io",
            rate_limit_per_minute=100,
        )
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = polygon_response
            bars = await provider.get_bars(
                "AAPL",
                "1Min",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
            assert len(bars) == 2
            assert bars[0].symbol == "AAPL"
            assert bars[0].close == 100.5
            assert bars[0].volume == 1000000
            assert bars[0].vwap == 100.25
            assert bars[0].trades == 5000
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_latest_bar(self) -> None:
        provider = PolygonProvider(
            api_key="test_key",
            base_url="https://api.polygon.io",
        )
        response = {
            "results": [
                {
                    "t": 1704067200000,
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1000000,
                }
            ]
        }
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = response
            bar = await provider.get_latest_bar("AAPL")
            assert bar is not None
            assert bar.symbol == "AAPL"
            assert bar.close == 100.5
        await provider.close()

    def test_parse_timeframe(self) -> None:
        provider = PolygonProvider(api_key="test", base_url="https://api.polygon.io")
        assert provider._parse_timeframe("1Min") == (1, "minute")
        assert provider._parse_timeframe("5Min") == (5, "minute")
        assert provider._parse_timeframe("1H") == (1, "hour")
        with pytest.raises(ValueError):
            provider._parse_timeframe("invalid")


class TestAlpacaProvider:
    @pytest.mark.asyncio
    async def test_get_bars_parses_response(self, alpaca_response: dict) -> None:
        provider = AlpacaProvider(
            api_key="key:secret",
            base_url="https://data.alpaca.markets",
            rate_limit_per_minute=100,
        )
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = alpaca_response
            bars = await provider.get_bars(
                "AAPL",
                "1Min",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
            assert len(bars) == 1
            assert bars[0].symbol == "AAPL"
            assert bars[0].close == 100.5
        await provider.close()


class TestProviderFactory:
    def test_create_polygon_provider(self) -> None:
        provider = create_provider(
            "polygon",
            "test_key",
            "https://api.polygon.io",
        )
        assert isinstance(provider, PolygonProvider)

    def test_create_alpaca_provider(self) -> None:
        provider = create_provider(
            "alpaca",
            "key:secret",
            "https://data.alpaca.markets",
        )
        assert isinstance(provider, AlpacaProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown", "key", "https://example.com")


class TestOHLCVBar:
    def test_bar_creation(self) -> None:
        bar = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
        )
        assert bar.symbol == "AAPL"
        assert bar.close == 100.5
        assert bar.vwap is None
        assert bar.trades is None
