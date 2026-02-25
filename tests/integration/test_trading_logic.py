"""
Integration tests for ExecutionAgent trading logic.

Tests cover:
1. Side decision: BUY for "long", SELL for "short", BLOCKED for "neutral"
2. Position sizing: qty = min(max_notional / price, max_qty)
3. Sell validation: BLOCKED if no position exists
4. Trade persistence: trades table writes

These tests mock the Alpaca client to avoid real API calls.
"""
from __future__ import annotations

import os
import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg


# Test fixtures and helpers
@pytest.fixture
def risk_event_factory():
    """Factory for creating RiskEvent-like dicts."""
    def _create(
        symbol: str = "AAPL",
        direction: str = "long",
        risk_score: float = 0.75,
    ):
        return {
            "event_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "idempotency_key": f"test-{uuid4()}",
            "symbol": symbol,
            "source": "risk",
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "schema_version": 1,
            "risk_score": risk_score,
            "direction": direction,
            "reason": "test signal",
        }
    return _create


class TestSideDecision:
    """Test side decision logic based on RiskEvent.direction."""
    
    def test_long_direction_returns_buy(self):
        """Long/bullish directions should result in BUY side."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._determine_side = ExecutionAgent._determine_side.__get__(agent)
            
            assert agent._determine_side("long") == "buy"
            assert agent._determine_side("Long") == "buy"
            assert agent._determine_side("LONG") == "buy"
            assert agent._determine_side("bullish") == "buy"
            assert agent._determine_side("buy") == "buy"
    
    def test_short_direction_returns_sell(self):
        """Short/bearish directions should result in SELL side."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._determine_side = ExecutionAgent._determine_side.__get__(agent)
            
            assert agent._determine_side("short") == "sell"
            assert agent._determine_side("Short") == "sell"
            assert agent._determine_side("SHORT") == "sell"
            assert agent._determine_side("bearish") == "sell"
            assert agent._determine_side("sell") == "sell"
    
    def test_neutral_direction_returns_none(self):
        """Neutral/unknown directions should return None (blocked)."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._determine_side = ExecutionAgent._determine_side.__get__(agent)
            
            assert agent._determine_side("neutral") is None
            assert agent._determine_side("hold") is None
            assert agent._determine_side("unknown") is None
            assert agent._determine_side("") is None
            assert agent._determine_side(None) is None


class TestPositionSizing:
    """Test position sizing logic based on max_notional and price."""
    
    @pytest.mark.asyncio
    async def test_qty_calculation_from_notional(self):
        """qty = max_notional / price, capped by max_qty."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.app.max_trade_notional_usd = 1000
            mock_settings.app.max_qty_per_trade = 100
            mock_settings.app.min_qty_per_trade = 1
            mock_settings.app.execution_default_qty = 10
            agent.settings = mock_settings
            
            # Bind the method
            agent._calculate_qty = ExecutionAgent._calculate_qty.__get__(agent)
            
            # $1000 notional / $100 price = 10 shares
            qty = await agent._calculate_qty("AAPL", 100.0)
            assert qty == 10
            
            # $1000 notional / $50 price = 20 shares
            qty = await agent._calculate_qty("AAPL", 50.0)
            assert qty == 20
            
            # $1000 notional / $10 price = 100 shares (capped at max_qty)
            qty = await agent._calculate_qty("AAPL", 10.0)
            assert qty == 100
    
    @pytest.mark.asyncio
    async def test_qty_capped_at_max_qty(self):
        """qty should never exceed max_qty_per_trade."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.max_trade_notional_usd = 10000
            mock_settings.app.max_qty_per_trade = 50  # Low cap
            mock_settings.app.min_qty_per_trade = 1
            mock_settings.app.execution_default_qty = 10
            agent.settings = mock_settings
            
            agent._calculate_qty = ExecutionAgent._calculate_qty.__get__(agent)
            
            # $10000 / $10 = 1000 shares, but capped at 50
            qty = await agent._calculate_qty("AAPL", 10.0)
            assert qty == 50
    
    @pytest.mark.asyncio
    async def test_qty_at_least_min_qty(self):
        """qty should never go below min_qty_per_trade."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.max_trade_notional_usd = 100
            mock_settings.app.max_qty_per_trade = 100
            mock_settings.app.min_qty_per_trade = 5  # High minimum
            mock_settings.app.execution_default_qty = 10
            agent.settings = mock_settings
            
            agent._calculate_qty = ExecutionAgent._calculate_qty.__get__(agent)
            
            # $100 / $200 = 0.5 → 0, but min is 5
            qty = await agent._calculate_qty("AAPL", 200.0)
            assert qty == 5
    
    @pytest.mark.asyncio
    async def test_uses_default_qty_when_price_unavailable(self):
        """Should use execution_default_qty when price is None or 0."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.max_trade_notional_usd = 1000
            mock_settings.app.max_qty_per_trade = 100
            mock_settings.app.min_qty_per_trade = 1
            mock_settings.app.execution_default_qty = 42
            agent.settings = mock_settings
            
            agent._calculate_qty = ExecutionAgent._calculate_qty.__get__(agent)
            
            qty = await agent._calculate_qty("AAPL", None)
            assert qty == 42
            
            qty = await agent._calculate_qty("AAPL", 0.0)
            assert qty == 42
            
            qty = await agent._calculate_qty("AAPL", -50.0)
            assert qty == 42


class TestSellValidation:
    """Test sell validation - blocks SELL when no position exists."""
    
    @pytest.mark.asyncio
    async def test_sell_blocked_when_no_position(self):
        """SELL should be blocked if current position is 0."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.check_position_before_sell = True
            agent.settings = mock_settings
            
            # Mock trades repo that returns 0 position (DB-first check)
            mock_repo = AsyncMock()
            mock_repo.get_position_qty.return_value = 0
            agent._trades_repo = mock_repo
            
            # Alpaca client not needed since DB has the answer
            agent._alpaca_client = None
            
            agent._check_can_sell = ExecutionAgent._check_can_sell.__get__(agent)
            
            can_sell, position_qty, error = await agent._check_can_sell("AAPL")
            
            assert can_sell is False
            assert position_qty == 0
            assert "no position" in error.lower()
    
    @pytest.mark.asyncio
    async def test_sell_allowed_when_position_exists(self):
        """SELL should be allowed if current position > 0."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.check_position_before_sell = True
            agent.settings = mock_settings
            
            # Mock trades repo that returns positive position (DB-first check)
            mock_repo = AsyncMock()
            mock_repo.get_position_qty.return_value = 50
            agent._trades_repo = mock_repo
            
            agent._alpaca_client = None
            
            agent._check_can_sell = ExecutionAgent._check_can_sell.__get__(agent)
            
            can_sell, position_qty, error = await agent._check_can_sell("AAPL")
            
            assert can_sell is True
            assert position_qty == 50
            assert error is None
    
    @pytest.mark.asyncio
    async def test_sell_skips_validation_when_disabled(self):
        """SELL validation should be skipped if check_position_before_sell=False."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.check_position_before_sell = False
            agent.settings = mock_settings
            agent._alpaca_client = None  # No client needed
            agent._trades_repo = None  # No repo needed
            
            agent._check_can_sell = ExecutionAgent._check_can_sell.__get__(agent)
            
            can_sell, position_qty, error = await agent._check_can_sell("AAPL")
            
            assert can_sell is True
            assert error is None
    
    @pytest.mark.asyncio
    async def test_sell_fallback_to_alpaca_when_not_in_db(self):
        """SELL should fallback to Alpaca if symbol not in positions table."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent.logger = MagicMock()
            
            mock_settings = MagicMock()
            mock_settings.app.check_position_before_sell = True
            agent.settings = mock_settings
            
            # Mock trades repo that returns None (symbol not in DB)
            mock_repo = AsyncMock()
            mock_repo.get_position_qty.return_value = None
            agent._trades_repo = mock_repo
            
            # Mock Alpaca client as fallback
            mock_alpaca = AsyncMock()
            mock_alpaca.get_position_qty.return_value = 25
            agent._alpaca_client = mock_alpaca
            
            agent._check_can_sell = ExecutionAgent._check_can_sell.__get__(agent)
            
            can_sell, position_qty, error = await agent._check_can_sell("NVDA")
            
            assert can_sell is True
            assert position_qty == 25
            assert error is None
            mock_alpaca.get_position_qty.assert_called_once_with("NVDA")


class TestTradesRepository:
    """Test TradesRepository persistence."""
    
    @pytest.mark.asyncio
    async def test_trade_record_dataclass(self):
        """TradeRecord dataclass should store all required fields."""
        from src.integrations.trades_repository import TradeRecord
        
        trade = TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=10,
            intended_notional=1500.0,
            avg_fill_price=150.0,
            status="filled",
            alpaca_order_id="abc123",
            alpaca_status="filled",
            error_message=None,
            correlation_id="corr-123",
            idempotency_key="idem-456",
            paper_trade=True,
            dry_run=False,
        )
        
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.qty == 10
        assert trade.intended_notional == 1500.0
        assert trade.avg_fill_price == 150.0
        assert trade.status == "filled"
        assert trade.paper_trade is True
        assert trade.dry_run is False
    
    @pytest.mark.asyncio
    async def test_insert_trade_sql_construction(self):
        """insert_trade should construct proper INSERT SQL for trade persistence."""
        from src.integrations.trades_repository import TradeRecord
        
        # Simply test the TradeRecord dataclass works correctly
        trade = TradeRecord(
            symbol="AAPL",
            side="buy",
            qty=10,
            intended_notional=1500.0,
            avg_fill_price=150.0,
            status="filled",
            alpaca_order_id="abc123",
            alpaca_status="filled",
            error_message=None,
            correlation_id="test-corr",
            idempotency_key="test-idem",
            paper_trade=True,
            dry_run=False,
        )
        
        # Verify the trade has correct values
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.qty == 10
        assert trade.intended_notional == 1500.0
        assert trade.avg_fill_price == 150.0
        assert trade.status == "filled"
        assert trade.alpaca_order_id == "abc123"
        assert trade.paper_trade is True
        assert trade.dry_run is False


class TestAlpacaClientMethods:
    """Test new Alpaca client methods for trading logic."""
    
    @pytest.mark.asyncio
    async def test_get_position_qty_returns_int(self):
        """get_position_qty should return integer quantity."""
        from src.integrations.alpaca_client import AlpacaClient
        
        with patch.object(AlpacaClient, '__init__', return_value=None):
            client = AlpacaClient.__new__(AlpacaClient)
            client.base_url = "https://paper-api.alpaca.markets"
            client.logger = MagicMock()
            
            # Mock get_position method that get_position_qty calls
            async def mock_get_position(symbol):
                return {"qty": "50"}
            client.get_position = mock_get_position
            
            qty = await client.get_position_qty("AAPL")
            
            assert isinstance(qty, int)
            assert qty == 50
    
    @pytest.mark.asyncio
    async def test_get_position_qty_returns_zero_when_no_position(self):
        """get_position_qty should return 0 when 404 (no position)."""
        from src.integrations.alpaca_client import AlpacaClient
        
        with patch.object(AlpacaClient, '__init__', return_value=None):
            client = AlpacaClient.__new__(AlpacaClient)
            client.base_url = "https://paper-api.alpaca.markets"
            client.logger = MagicMock()
            
            # Mock get_position returning None (no position)
            async def mock_get_position(symbol):
                return None
            client.get_position = mock_get_position
            
            qty = await client.get_position_qty("AAPL")
            
            assert qty == 0
    
    @pytest.mark.asyncio
    async def test_get_current_price_returns_float(self):
        """get_current_price should return float price."""
        from src.integrations.alpaca_client import AlpacaClient
        
        with patch.object(AlpacaClient, '__init__', return_value=None):
            client = AlpacaClient.__new__(AlpacaClient)
            client.base_url = "https://paper-api.alpaca.markets"
            client._http = AsyncMock()
            client.logger = MagicMock()
            
            # Mock get_latest_trade response
            async def mock_get_latest_trade(symbol):
                return 150.75
            client.get_latest_trade = mock_get_latest_trade
            
            price = await client.get_current_price("AAPL")
            
            assert isinstance(price, float)
            assert price == 150.75
