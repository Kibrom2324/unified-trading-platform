"""Tests for YFinance market data provider."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.market_data.providers import YFinanceProvider, OHLCVBar


@pytest.fixture
def provider():
    """Create YFinance provider instance."""
    return YFinanceProvider()


@pytest.fixture
def mock_df():
    """Create mock DataFrame similar to yfinance output."""
    dates = pd.date_range(start="2026-01-10", periods=3, freq="D")
    data = {
        "Open": [150.0, 151.0, 152.0],
        "High": [155.0, 156.0, 157.0],
        "Low": [149.0, 150.0, 151.0],
        "Close": [154.0, 155.0, 156.0],
        "Volume": [1000000, 1100000, 1200000],
    }
    return pd.DataFrame(data, index=dates)


@pytest.mark.asyncio
async def test_get_bars_returns_ohlcv_bars(provider: YFinanceProvider, mock_df: pd.DataFrame):
    """Test that get_bars returns list of OHLCVBar objects."""
    with patch.object(provider, "_get_yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.return_value.Ticker.return_value = mock_ticker
        
        start = datetime(2026, 1, 10, tzinfo=timezone.utc)
        end = datetime(2026, 1, 13, tzinfo=timezone.utc)
        
        bars = await provider.get_bars("AAPL", "1D", start, end)
        
        assert len(bars) == 3
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].symbol == "AAPL"
        assert bars[0].open == 150.0
        assert bars[0].close == 154.0
        assert bars[0].volume == 1000000


@pytest.mark.asyncio
async def test_get_bars_empty_df(provider: YFinanceProvider):
    """Test that get_bars handles empty DataFrame."""
    with patch.object(provider, "_get_yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.return_value.Ticker.return_value = mock_ticker
        
        start = datetime(2026, 1, 10, tzinfo=timezone.utc)
        end = datetime(2026, 1, 13, tzinfo=timezone.utc)
        
        bars = await provider.get_bars("INVALID", "1D", start, end)
        
        assert bars == []


@pytest.mark.asyncio
async def test_get_latest_bar(provider: YFinanceProvider, mock_df: pd.DataFrame):
    """Test that get_latest_bar returns the most recent bar."""
    with patch.object(provider, "_get_yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.return_value.Ticker.return_value = mock_ticker
        
        bar = await provider.get_latest_bar("AAPL")
        
        assert bar is not None
        assert bar.symbol == "AAPL"
        # Should be the last row
        assert bar.close == 156.0
        assert bar.volume == 1200000


def test_convert_timeframe(provider: YFinanceProvider):
    """Test timeframe conversion."""
    assert provider._convert_timeframe("1Min") == "1m"
    assert provider._convert_timeframe("5Min") == "5m"
    assert provider._convert_timeframe("1H") == "1h"
    assert provider._convert_timeframe("1D") == "1d"
    
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        provider._convert_timeframe("invalid")


def test_no_api_key_required():
    """Test that YFinance provider doesn't require API key."""
    # Should not raise
    provider = YFinanceProvider()
    assert provider.api_key == "none"
    
    # Also works with empty strings
    provider = YFinanceProvider(api_key="", base_url="")
    assert provider is not None
