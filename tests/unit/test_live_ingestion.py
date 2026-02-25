"""
Unit tests for live ingestion module.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.market_data.providers import OHLCVBar, YFinanceProvider


class TestLiveIngestionResult:
    """Tests for LiveIngestionResult class."""
    
    def test_result_initialization(self):
        """Test result object initializes correctly."""
        from execution.run_live_ingestion import LiveIngestionResult
        
        result = LiveIngestionResult()
        
        assert result.correlation_id is not None
        assert len(result.symbols_processed) == 0
        assert result.bars_ingested == 0
        assert len(result.errors) == 0
        assert result.pipeline_triggered is False
        assert result.start_time is not None
        assert result.end_time is None
    
    def test_result_success_with_bars(self):
        """Test success property when bars are ingested."""
        from execution.run_live_ingestion import LiveIngestionResult
        
        result = LiveIngestionResult()
        result.bars_ingested = 5
        result.symbols_processed = ["AAPL", "MSFT"]
        
        assert result.success is True
    
    def test_result_failure_with_errors(self):
        """Test success property when errors exist."""
        from execution.run_live_ingestion import LiveIngestionResult
        
        result = LiveIngestionResult()
        result.bars_ingested = 3
        result.errors = ["NVDA: connection timeout"]
        
        assert result.success is False
    
    def test_result_failure_no_bars(self):
        """Test success property when no bars ingested."""
        from execution.run_live_ingestion import LiveIngestionResult
        
        result = LiveIngestionResult()
        result.symbols_processed = ["AAPL"]
        
        assert result.success is False
    
    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        from execution.run_live_ingestion import LiveIngestionResult
        
        result = LiveIngestionResult()
        result.bars_ingested = 3
        result.symbols_processed = ["AAPL", "MSFT", "NVDA"]
        result.end_time = datetime.now(tz=timezone.utc)
        
        data = result.to_dict()
        
        assert "correlation_id" in data
        assert data["bars_ingested"] == 3
        assert data["symbols_processed"] == ["AAPL", "MSFT", "NVDA"]
        assert "duration_seconds" in data
        assert "success" in data


class TestFetchLatestBar:
    """Tests for fetch_latest_bar function."""
    
    @pytest.mark.asyncio
    async def test_fetch_latest_bar_success(self):
        """Test successful bar fetch."""
        from execution.run_live_ingestion import fetch_latest_bar
        import structlog
        
        mock_bar = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime.now(tz=timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
        )
        
        mock_provider = AsyncMock(spec=YFinanceProvider)
        mock_provider.get_latest_bar.return_value = mock_bar
        
        logger = structlog.get_logger()
        
        result = await fetch_latest_bar(mock_provider, "AAPL", logger)
        
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.close == 151.0
        mock_provider.get_latest_bar.assert_called_once_with("AAPL")
    
    @pytest.mark.asyncio
    async def test_fetch_latest_bar_no_data(self):
        """Test when no bar is available."""
        from execution.run_live_ingestion import fetch_latest_bar
        import structlog
        
        mock_provider = AsyncMock(spec=YFinanceProvider)
        mock_provider.get_latest_bar.return_value = None
        
        logger = structlog.get_logger()
        
        result = await fetch_latest_bar(mock_provider, "INVALID", logger)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_latest_bar_error(self):
        """Test error handling during fetch."""
        from execution.run_live_ingestion import fetch_latest_bar
        import structlog
        
        mock_provider = AsyncMock(spec=YFinanceProvider)
        mock_provider.get_latest_bar.side_effect = Exception("API error")
        
        logger = structlog.get_logger()
        
        result = await fetch_latest_bar(mock_provider, "AAPL", logger)
        
        assert result is None


class TestPublishBarToKafka:
    """Tests for publish_bar_to_kafka function."""
    
    @pytest.mark.asyncio
    async def test_publish_dry_run(self, capsys):
        """Test dry run doesn't publish to Kafka."""
        from execution.run_live_ingestion import publish_bar_to_kafka
        
        bar = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime.now(tz=timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
        )
        
        result = await publish_bar_to_kafka(
            bar=bar,
            producer=None,
            audit=None,
            correlation_id=str(uuid4()),
            dry_run=True,
        )
        
        assert result is True
        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "AAPL" in captured.out
    
    @pytest.mark.asyncio
    async def test_publish_skips_duplicate(self):
        """Test duplicate detection skips already-processed bars."""
        from execution.run_live_ingestion import publish_bar_to_kafka
        
        bar = OHLCVBar(
            symbol="AAPL",
            timestamp=datetime.now(tz=timezone.utc),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
        )
        
        mock_audit = AsyncMock()
        mock_audit.is_duplicate.return_value = True
        
        mock_producer = MagicMock()
        
        result = await publish_bar_to_kafka(
            bar=bar,
            producer=mock_producer,
            audit=mock_audit,
            correlation_id=str(uuid4()),
            dry_run=False,
        )
        
        assert result is False
        mock_producer.produce.assert_not_called()


class TestYFinanceProvider:
    """Tests for YFinanceProvider integration."""
    
    def test_provider_no_api_key_required(self):
        """Test provider works without API key."""
        provider = YFinanceProvider()
        
        # Should not raise
        assert provider.api_key == "none"
        assert provider.rate_limit_per_minute == 30
    
    def test_provider_custom_rate_limit(self):
        """Test custom rate limit configuration."""
        provider = YFinanceProvider(rate_limit_per_minute=60)
        
        assert provider.rate_limit_per_minute == 60
    
    @pytest.mark.asyncio
    async def test_convert_timeframe(self):
        """Test timeframe conversion."""
        provider = YFinanceProvider()
        
        assert provider._convert_timeframe("1D") == "1d"
        assert provider._convert_timeframe("1H") == "1h"
        assert provider._convert_timeframe("5Min") == "5m"


class TestMetrics:
    """Tests for live ingestion metrics."""
    
    def test_metrics_exist(self):
        """Test that live ingestion metrics are registered."""
        from src.monitoring.metrics import (
            LIVE_BARS_INGESTED,
            LIVE_INGESTION_LATENCY,
            LIVE_INGESTION_FAILURES,
        )
        
        # Metrics should be importable without error
        assert LIVE_BARS_INGESTED is not None
        assert LIVE_INGESTION_LATENCY is not None
        assert LIVE_INGESTION_FAILURES is not None
    
    def test_metrics_labels(self):
        """Test metrics have correct labels."""
        from src.monitoring.metrics import LIVE_BARS_INGESTED
        
        # Should have symbol and source labels
        LIVE_BARS_INGESTED.labels(symbol="AAPL", source="yfinance").inc()
        # No exception means labels are correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
