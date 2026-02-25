"""
Tests for end-to-end trade chain traceability.

Verifies that:
1. correlation_id flows from events through LLM traces
2. show_trade_chain.py returns correct data for a seeded DB row
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestCorrelationIdBinding:
    """Test that correlation_id is properly bound and flows through the system."""

    def test_set_correlation_id_stores_value(self):
        """Verify set_correlation_id stores the value in context."""
        from src.core.logging import get_correlation_id, set_correlation_id
        
        test_id = str(uuid4())
        set_correlation_id(test_id)
        
        assert get_correlation_id() == test_id
        
        # Cleanup
        set_correlation_id(None)

    def test_get_correlation_id_returns_none_when_not_set(self):
        """Verify get_correlation_id returns None when not set."""
        from src.core.logging import get_correlation_id, set_correlation_id
        
        set_correlation_id(None)
        assert get_correlation_id() is None

    @pytest.mark.asyncio
    async def test_llm_client_generates_fallback_when_no_correlation_id(self):
        """Verify LLMClient generates fallback ID and logs warning when correlation_id missing."""
        from src.core.logging import set_correlation_id
        
        # Ensure no correlation_id is set
        set_correlation_id(None)
        
        captured_corr_id = None
        captured_logs = []
        
        async def mock_trace(**kwargs):
            nonlocal captured_corr_id
            captured_corr_id = kwargs.get("correlation_id")
        
        mock_tracer = MagicMock()
        mock_tracer.trace = mock_trace
        
        # Capture log output
        mock_logger = MagicMock()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.core.config import LLMConfig
            from src.llm.client import LLMClient
            
            config = LLMConfig(
                base_url="http://localhost:11434/v1",
                model="test-model",
            )
            client = LLMClient(config, agent_name="TestAgent", tracer=mock_tracer)
            
            await client.chat([{"role": "user", "content": "test"}])
        
        # Verify a fallback ID was generated (starts with "unbound-")
        assert captured_corr_id is not None
        assert captured_corr_id.startswith("unbound-")
        
        # Cleanup
        set_correlation_id(None)

    @pytest.mark.asyncio
    async def test_llm_client_uses_correlation_id_from_context(self):
        """Verify LLMClient picks up correlation_id from context."""
        from src.core.logging import set_correlation_id
        
        test_corr_id = str(uuid4())
        set_correlation_id(test_corr_id)
        
        # Mock the tracer and httpx
        captured_corr_id = None
        
        async def mock_trace(**kwargs):
            nonlocal captured_corr_id
            captured_corr_id = kwargs.get("correlation_id")
        
        mock_tracer = MagicMock()
        mock_tracer.trace = mock_trace
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.core.config import LLMConfig
            from src.llm.client import LLMClient
            
            config = LLMConfig(
                base_url="http://localhost:11434/v1",
                model="test-model",
            )
            client = LLMClient(config, agent_name="TestAgent", tracer=mock_tracer)
            
            await client.chat([{"role": "user", "content": "test"}])
        
        # Verify correlation_id was passed to tracer
        assert captured_corr_id == test_corr_id
        
        # Cleanup
        set_correlation_id(None)


class TestShowTradeChain:
    """Tests for show_trade_chain.py script functionality."""

    @pytest.mark.asyncio
    async def test_fetch_trades_returns_list(self):
        """Test that fetch_trades returns a list of trade dicts."""
        # Import the function
        import sys
        sys.path.insert(0, "scripts")
        from show_trade_chain import fetch_trades
        
        # Mock pool with proper async context manager
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "ts": datetime.now(tz=timezone.utc),
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100,
                "intended_notional": 15000.0,
                "avg_fill_price": 150.0,
                "status": "filled",
                "alpaca_order_id": "test-order-123",
                "alpaca_status": "filled",
                "error_message": None,
                "paper_trade": True,
                "dry_run": False,
            }
        ]
        
        # Create async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_cm
        
        result = await fetch_trades(mock_pool, "test-correlation-id")
        
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["status"] == "filled"

    @pytest.mark.asyncio
    async def test_fetch_llm_traces_returns_list(self):
        """Test that fetch_llm_traces returns a list of trace dicts."""
        import sys
        sys.path.insert(0, "scripts")
        from show_trade_chain import fetch_llm_traces
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "ts": datetime.now(tz=timezone.utc),
                "agent_name": "RiskAgent",
                "model": "llama3.2:1b",
                "latency_ms": 1234.5,
                "status": "ok",
                "error_message": None,
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"content": "test response"},
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        ]
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_cm
        
        result = await fetch_llm_traces(mock_pool, "test-correlation-id")
        
        assert len(result) == 1
        assert result[0]["agent_name"] == "RiskAgent"
        assert result[0]["status"] == "ok"

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        import sys
        sys.path.insert(0, "scripts")
        from show_trade_chain import format_timestamp
        
        ts = datetime(2026, 1, 16, 12, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_timestamp(ts)
        
        assert "2026-01-16" in result
        assert "12:30:45" in result
        
        assert format_timestamp(None) == "N/A"

    def test_truncate(self):
        """Test text truncation."""
        import sys
        sys.path.insert(0, "scripts")
        from show_trade_chain import truncate
        
        # Short text unchanged
        assert truncate("hello", 10) == "hello"
        
        # Long text truncated
        long_text = "a" * 100
        result = truncate(long_text, 20)
        assert len(result) == 20
        assert result.endswith("...")
        
        # None returns empty
        assert truncate(None, 10) == ""

    @pytest.mark.asyncio
    async def test_get_correlation_id_from_alpaca_order(self):
        """Test looking up correlation_id from Alpaca order ID."""
        import sys
        sys.path.insert(0, "scripts")
        from show_trade_chain import get_correlation_id_from_alpaca_order
        
        test_corr_id = str(uuid4())
        
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"correlation_id": test_corr_id}
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_cm
        
        result = await get_correlation_id_from_alpaca_order(mock_pool, "test-order-123")
        
        assert result == test_corr_id

    @pytest.mark.asyncio
    async def test_get_correlation_id_from_alpaca_order_not_found(self):
        """Test looking up correlation_id returns None when not found."""
        import sys
        sys.path.insert(0, "scripts")
        from show_trade_chain import get_correlation_id_from_alpaca_order
        
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None
        
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_cm
        
        result = await get_correlation_id_from_alpaca_order(mock_pool, "nonexistent-order")
        
        assert result is None


class TestTradeRecordHasCorrelationId:
    """Verify TradeRecord includes correlation_id field."""

    def test_trade_record_has_correlation_id(self):
        """Verify TradeRecord dataclass has correlation_id field."""
        from src.integrations.trades_repository import TradeRecord
        
        record = TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=100,
            intended_notional=15000.0,
            avg_fill_price=150.0,
            status="filled",
            alpaca_order_id="test-123",
            alpaca_status="filled",
            error_message=None,
            correlation_id="test-corr-id",
            idempotency_key="test-idemp-key",
        )
        
        assert record.correlation_id == "test-corr-id"
        
        # Verify it's in to_dict()
        d = record.to_dict()
        assert d["correlation_id"] == "test-corr-id"
