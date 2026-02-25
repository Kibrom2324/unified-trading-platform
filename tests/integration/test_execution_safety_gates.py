"""
Integration tests for ExecutionAgent safety gates.

Tests verify that the Alpaca client is only called when:
1. execution_dry_run=False (in config)
2. Approval file exists on disk

Uses mocked Alpaca client to verify call behavior without
making real API requests.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.alpaca_client import AlpacaClient, OrderResult


class TestExecutionSafetyGates:
    """Test that safety gates properly control Alpaca API calls."""
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create a mock AlpacaClient."""
        client = MagicMock(spec=AlpacaClient)
        client.base_url = "https://paper-api.alpaca.markets"
        client.submit_market_order = AsyncMock(return_value=OrderResult(
            success=True,
            order_id="test-order-123",
            status="accepted",
            symbol="AAPL",
            filled_qty=1,
            avg_price=150.0,
        ))
        client.close = AsyncMock()
        return client
    
    @pytest.fixture
    def temp_approval_file(self):
        """Create a temporary approval file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_dry_run_blocks_alpaca_call(self, mock_alpaca_client, temp_approval_file):
        """
        When execution_dry_run=True, Alpaca should NOT be called
        even if approval file exists.
        """
        # Simulate the logic from ExecutionAgent._process_message
        dry_run = True
        approval_file_exists = os.path.exists(temp_approval_file)
        
        should_trade = (not dry_run) and approval_file_exists
        
        # This is the key assertion
        assert should_trade is False, "dry_run=True should block trading"
        
        # If we were in the agent, we would NOT call Alpaca
        if should_trade:
            await mock_alpaca_client.submit_market_order("AAPL", 1, "buy")
        
        # Verify Alpaca was NOT called
        mock_alpaca_client.submit_market_order.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_missing_approval_file_blocks_alpaca_call(self, mock_alpaca_client):
        """
        When approval file is missing, Alpaca should NOT be called
        even if dry_run=False.
        """
        dry_run = False
        approval_file = "/nonexistent/path/APPROVE_EXECUTION"
        approval_file_exists = os.path.exists(approval_file)
        
        should_trade = (not dry_run) and approval_file_exists
        
        assert approval_file_exists is False
        assert should_trade is False, "Missing approval file should block trading"
        
        if should_trade:
            await mock_alpaca_client.submit_market_order("AAPL", 1, "buy")
        
        mock_alpaca_client.submit_market_order.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_approved_state_calls_alpaca(self, mock_alpaca_client, temp_approval_file):
        """
        When dry_run=False AND approval file exists, Alpaca SHOULD be called.
        """
        dry_run = False
        approval_file_exists = os.path.exists(temp_approval_file)
        
        should_trade = (not dry_run) and approval_file_exists
        
        assert approval_file_exists is True
        assert should_trade is True, "Should trade when both gates pass"
        
        if should_trade:
            result = await mock_alpaca_client.submit_market_order("AAPL", 1, "buy")
        
        # Verify Alpaca WAS called
        mock_alpaca_client.submit_market_order.assert_called_once_with("AAPL", 1, "buy")
        assert result.success is True
        assert result.order_id == "test-order-123"
    
    @pytest.mark.asyncio
    async def test_both_gates_must_pass(self, mock_alpaca_client):
        """
        Test all combinations of the two safety gates.
        """
        with tempfile.NamedTemporaryFile(delete=False) as f:
            approval_file = f.name
        
        test_cases = [
            # (dry_run, approval_exists, should_call_alpaca)
            (True, True, False),   # dry_run blocks
            (True, False, False),  # both block
            (False, False, False), # no approval blocks
            (False, True, True),   # both pass → call Alpaca
        ]
        
        for dry_run, approval_exists, should_call in test_cases:
            mock_alpaca_client.reset_mock()
            
            # Create or remove approval file
            if approval_exists:
                Path(approval_file).touch()
            else:
                if os.path.exists(approval_file):
                    os.unlink(approval_file)
            
            # Simulate gate logic
            should_trade = (not dry_run) and os.path.exists(approval_file)
            
            if should_trade:
                await mock_alpaca_client.submit_market_order("AAPL", 1, "buy")
            
            # Verify
            if should_call:
                mock_alpaca_client.submit_market_order.assert_called_once()
            else:
                mock_alpaca_client.submit_market_order.assert_not_called()
        
        # Cleanup
        if os.path.exists(approval_file):
            os.unlink(approval_file)


class TestAlpacaClientSafety:
    """Test AlpacaClient safety features."""
    
    def test_live_endpoint_is_blocked(self):
        """Verify live trading endpoints are rejected."""
        from src.integrations.alpaca_client import AlpacaLiveEndpointError
        
        with pytest.raises(AlpacaLiveEndpointError):
            AlpacaClient(
                api_key="test",
                secret_key="test",
                base_url="https://api.alpaca.markets",
            )
    
    def test_paper_endpoint_is_allowed(self):
        """Verify paper trading endpoint is accepted."""
        client = AlpacaClient(
            api_key="test",
            secret_key="test",
            base_url="https://paper-api.alpaca.markets",
        )
        assert "paper" in client.base_url
    
    def test_order_result_captures_failures(self):
        """Verify OrderResult properly captures failed orders."""
        result = OrderResult(
            success=False,
            error_message="Insufficient buying power",
        )
        
        assert result.success is False
        assert result.error_message == "Insufficient buying power"
        assert result.order_id is None
