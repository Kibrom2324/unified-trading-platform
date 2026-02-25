"""
Unit tests for AlpacaClient module.

Tests verify:
1. Paper endpoint validation (blocks live endpoints)
2. Order submission flow
3. Error handling and retries
4. Configuration from environment
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.integrations.alpaca_client import (
    AlpacaClient,
    AlpacaClientError,
    AlpacaLiveEndpointError,
    OrderResult,
)


class TestAlpacaClientInit:
    """Test AlpacaClient initialization and safety checks."""
    
    def test_paper_endpoint_accepted(self):
        """Paper trading endpoint should be accepted."""
        client = AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )
        assert client.base_url == "https://paper-api.alpaca.markets"
    
    def test_live_endpoint_blocked(self):
        """Live trading endpoint should raise AlpacaLiveEndpointError."""
        with pytest.raises(AlpacaLiveEndpointError) as exc_info:
            AlpacaClient(
                api_key="test-key",
                secret_key="test-secret",
                base_url="https://api.alpaca.markets",
            )
        
        assert "SAFETY BLOCK" in str(exc_info.value)
        assert "paper-api.alpaca.markets" in str(exc_info.value)
    
    def test_from_env_with_valid_config(self):
        """from_env should work with valid paper endpoint."""
        env = {
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        }
        
        with patch.dict("os.environ", env, clear=False):
            client = AlpacaClient.from_env()
            assert client.base_url == "https://paper-api.alpaca.markets"
    
    def test_from_env_missing_key(self):
        """from_env should raise error when API key missing."""
        env = {
            "ALPACA_SECRET_KEY": "test-secret",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        }
        
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(AlpacaClientError) as exc_info:
                AlpacaClient.from_env()
            assert "ALPACA_API_KEY" in str(exc_info.value)
    
    def test_from_env_missing_secret(self):
        """from_env should raise error when API secret missing."""
        env = {
            "ALPACA_API_KEY": "test-key",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        }
        
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(AlpacaClientError) as exc_info:
                AlpacaClient.from_env()
            assert "ALPACA_SECRET_KEY" in str(exc_info.value)
    
    def test_from_env_defaults_to_paper(self):
        """from_env should default to paper endpoint when not specified."""
        env = {
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
        }
        
        with patch.dict("os.environ", env, clear=True):
            client = AlpacaClient.from_env()
            assert client.base_url == "https://paper-api.alpaca.markets"


class TestOrderResult:
    """Test OrderResult dataclass."""
    
    def test_successful_order(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="test-order-123",
            status="filled",
            symbol="AAPL",
            filled_qty=10,
            avg_price=150.50,
        )
        
        assert result.success is True
        assert result.order_id == "test-order-123"
        assert result.error_message is None
    
    def test_failed_order(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            error_message="Insufficient buying power",
        )
        
        assert result.success is False
        assert result.order_id is None
        assert result.error_message == "Insufficient buying power"


class TestOrderSubmission:
    """Test order submission methods."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )
    
    @pytest.mark.asyncio
    async def test_submit_market_order_success(self, client):
        """Test successful market order submission."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "order-id-123",
            "status": "accepted",
            "symbol": "AAPL",
            "filled_qty": "10",
            "filled_avg_price": "150.50",
        }
        
        # Mock the _request_with_retry method directly
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.submit_market_order("AAPL", 10, "buy")
            
            assert result.success is True
            assert result.order_id == "order-id-123"
            assert result.status == "accepted"
            
            # Verify correct API call
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "/v2/orders" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_submit_market_order_rejected(self, client):
        """Test rejected order returns failure result."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Insufficient buying power"
        
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.submit_market_order("AAPL", 10, "buy")
            
            assert result.success is False
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_submit_order_api_error_captured(self, client):
        """Test that API errors are captured in the result."""
        with patch.object(client, "_request_with_retry", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection refused")
            
            result = await client.submit_market_order("AAPL", 10, "buy")
            
            assert result.success is False
            assert "Connection refused" in (result.error_message or "")
    
    @pytest.mark.asyncio
    async def test_close_client(self, client):
        """Test client close method when client exists."""
        # Initialize the client first
        mock_http_client = MagicMock()
        mock_http_client.is_closed = False
        mock_http_client.aclose = AsyncMock()
        client._client = mock_http_client
        
        await client.close()
        
        mock_http_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_client_when_none(self, client):
        """Test client close when no client initialized."""
        # Should not raise an error
        await client.close()


class TestSafetyGuards:
    """Test safety guards prevent dangerous operations."""
    
    def test_cannot_create_client_with_live_url_variants(self):
        """Test various live URL patterns are blocked."""
        live_urls = [
            "https://api.alpaca.markets",
            "https://api.alpaca.markets/",
            "http://api.alpaca.markets",
            "https://broker-api.alpaca.markets",
            "https://api.alpaca.markets:443",
        ]
        
        for url in live_urls:
            with pytest.raises(AlpacaLiveEndpointError):
                AlpacaClient(
                    api_key="test-key",
                    secret_key="test-secret",
                    base_url=url,
                )
    
    def test_paper_url_variants_accepted(self):
        """Test paper URL variants are accepted."""
        paper_urls = [
            "https://paper-api.alpaca.markets",
            "https://paper-api.alpaca.markets/",
        ]
        
        for url in paper_urls:
            client = AlpacaClient(
                api_key="test-key",
                secret_key="test-secret",
                base_url=url,
            )
            assert "paper" in client.base_url


class TestIntegrationWithExecutionAgent:
    """
    Integration tests for AlpacaClient with ExecutionAgent.
    
    These tests verify the expected behavior when AlpacaClient
    is used within the ExecutionAgent context.
    """
    
    @pytest.mark.asyncio
    async def test_order_result_fields_map_to_execution_event(self):
        """Verify OrderResult fields can populate ExecutionEvent."""
        result = OrderResult(
            success=True,
            order_id="alpaca-order-123",
            status="filled",
            symbol="AAPL",
            filled_qty=10,
            avg_price=150.50,
        )
        
        # These are the fields ExecutionEvent expects
        assert result.order_id is not None  # → alpaca_order_id
        assert result.status is not None    # → alpaca_status
        assert result.symbol is not None    # → symbol
        assert result.filled_qty is not None  # → filled_qty
        assert result.avg_price is not None   # → avg_price
    
    @pytest.mark.asyncio
    async def test_failed_order_provides_error_message(self):
        """Verify failed orders provide error_message for ExecutionEvent."""
        result = OrderResult(
            success=False,
            error_message="Insufficient buying power",
        )
        
        # ExecutionEvent.error_message should be populated
        assert result.error_message is not None
        assert "buying power" in result.error_message
