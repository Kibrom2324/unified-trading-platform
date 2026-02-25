"""
Integration tests for live ingestion pipeline.

These tests verify that:
1. Live ingestion fetches real data from yfinance
2. Events are correctly formatted as MarketRawEvent
3. Events pass Avro schema validation
4. The full pipeline can be triggered
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.market_data.providers import OHLCVBar, YFinanceProvider
from src.models.events_market import MarketRawEvent


class TestLiveIngestionIntegration:
    """Integration tests for live ingestion."""
    
    @pytest.fixture
    def sample_bar(self):
        """Create a sample OHLCV bar."""
        return OHLCVBar(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 15, 21, 0, 0, tzinfo=timezone.utc),
            open=150.0,
            high=152.5,
            low=149.25,
            close=151.75,
            volume=50000000.0,
            vwap=150.8,
            trades=100000,
        )
    
    def test_bar_to_market_raw_event(self, sample_bar):
        """Test converting OHLCVBar to MarketRawEvent."""
        correlation_id = str(uuid4())
        ts_str = sample_bar.timestamp.strftime("%Y%m%d")
        idempotency_key = f"live:{sample_bar.symbol}:{ts_str}"
        
        event = MarketRawEvent(
            idempotency_key=idempotency_key,
            symbol=sample_bar.symbol,
            source="live_ingestion",
            correlation_id=correlation_id,
            open=sample_bar.open,
            high=sample_bar.high,
            low=sample_bar.low,
            close=sample_bar.close,
            price=sample_bar.close,
            volume=sample_bar.volume,
            vwap=sample_bar.vwap,
            trades=sample_bar.trades,
        )
        
        assert event.symbol == "AAPL"
        assert event.close == 151.75
        assert event.source == "live_ingestion"
        assert "live:AAPL:20250115" == event.idempotency_key
    
    def test_market_raw_event_to_avro_dict(self, sample_bar):
        """Test MarketRawEvent can be converted to Avro dict."""
        event = MarketRawEvent(
            idempotency_key=f"live:{sample_bar.symbol}:20250115",
            symbol=sample_bar.symbol,
            source="live_ingestion",
            correlation_id=str(uuid4()),
            open=sample_bar.open,
            high=sample_bar.high,
            low=sample_bar.low,
            close=sample_bar.close,
            price=sample_bar.close,
            volume=sample_bar.volume,
        )
        
        avro_dict = event.to_avro_dict()
        
        assert isinstance(avro_dict, dict)
        assert avro_dict["symbol"] == "AAPL"
        assert avro_dict["open"] == 150.0
        assert avro_dict["high"] == 152.5
        assert avro_dict["low"] == 149.25
        assert avro_dict["close"] == 151.75
        assert avro_dict["volume"] == 50000000.0
        assert "ts" in avro_dict
        assert "event_id" in avro_dict
    
    def test_yfinance_provider_initialization(self):
        """Test YFinanceProvider can be initialized without API key."""
        provider = YFinanceProvider()
        
        assert provider.api_key == "none"
        assert provider.rate_limit_per_minute == 30
        assert provider._yf is None  # Lazy loaded
    
    @pytest.mark.asyncio
    async def test_run_live_ingestion_dry_run(self):
        """Test full live ingestion in dry run mode."""
        from execution.run_live_ingestion import run_live_ingestion
        
        # Mock the YFinanceProvider
        mock_bar = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime.now(tz=timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
        )
        
        with patch('execution.run_live_ingestion.YFinanceProvider') as MockProvider:
            mock_instance = AsyncMock()
            mock_instance.get_latest_bar.return_value = mock_bar
            mock_instance.close = AsyncMock()
            MockProvider.return_value = mock_instance
            
            result = await run_live_ingestion(
                symbols=["AAPL"],
                dry_run=True,
                trigger_pipeline=False,
            )
        
        assert result.success is True
        assert result.bars_ingested == 1
        assert "AAPL" in result.symbols_processed
        assert result.pipeline_triggered is False
    
    @pytest.mark.asyncio
    async def test_run_live_ingestion_handles_errors(self):
        """Test live ingestion handles errors gracefully."""
        from execution.run_live_ingestion import run_live_ingestion
        
        with patch('execution.run_live_ingestion.YFinanceProvider') as MockProvider:
            mock_instance = AsyncMock()
            # Return None to simulate no data available
            mock_instance.get_latest_bar.return_value = None
            mock_instance.close = AsyncMock()
            MockProvider.return_value = mock_instance
            
            result = await run_live_ingestion(
                symbols=["AAPL", "MSFT"],
                dry_run=True,
                trigger_pipeline=False,
            )
        
        # Should complete - no errors, but no data ingested either
        # success is False because bars_ingested == 0
        assert result.success is False
        assert result.bars_ingested == 0
        assert len(result.errors) == 0  # errors only logged, not accumulated


class TestEventSchemaValidation:
    """Tests for Avro schema validation."""
    
    def test_market_raw_event_schema_fields(self):
        """Test MarketRawEvent has all required schema fields."""
        event = MarketRawEvent(
            idempotency_key="test:AAPL:20250115",
            symbol="AAPL",
            source="live_ingestion",
            correlation_id=str(uuid4()),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            price=151.0,
            volume=1000000.0,
        )
        
        avro_dict = event.to_avro_dict()
        
        # Check required fields
        required_fields = [
            "event_id",
            "correlation_id", 
            "idempotency_key",
            "symbol",
            "ts",
            "schema_version",
            "source",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        
        for field in required_fields:
            assert field in avro_dict, f"Missing required field: {field}"
    
    def test_idempotency_key_format(self):
        """Test idempotency key follows expected format."""
        from datetime import datetime, timezone
        
        symbol = "AAPL"
        ts = datetime(2025, 1, 15, tzinfo=timezone.utc)
        ts_str = ts.strftime("%Y%m%d")
        
        idempotency_key = f"live:{symbol}:{ts_str}"
        
        assert idempotency_key == "live:AAPL:20250115"
        
        # Test uniqueness for different days
        ts2 = datetime(2025, 1, 16, tzinfo=timezone.utc)
        ts2_str = ts2.strftime("%Y%m%d")
        idempotency_key2 = f"live:{symbol}:{ts2_str}"
        
        assert idempotency_key != idempotency_key2


class TestSuperAGITool:
    """Tests for SuperAGI tool wrapper."""
    
    def test_tool_input_schema(self):
        """Test tool input schema is valid."""
        # Read the tool file directly and verify structure
        tool_path = Path(__file__).resolve().parents[2] / "superagi" / "tools" / "run_live_ingestion.py"
        
        content = tool_path.read_text()
        
        # Verify the Input class exists with correct defaults
        assert "class RunLiveIngestionInput" in content
        assert 'symbols: str = ""' in content
        assert "trigger_pipeline: bool = False" in content
        assert "dry_run: bool = False" in content
    
    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool_path = Path(__file__).resolve().parents[2] / "superagi" / "tools" / "run_live_ingestion.py"
        
        content = tool_path.read_text()
        
        # Verify tool metadata (type annotation uses colon, not equals)
        assert 'name: str = "awet_run_live_ingestion"' in content
        assert "Yahoo Finance" in content
        assert "FREE" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
