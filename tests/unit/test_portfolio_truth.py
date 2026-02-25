"""
Unit tests for Portfolio Truth v1 - positions, exposure caps, SELL validation.

Tests cover:
1. Position reconciliation logic
2. SELL blocked when DB qty=0
3. SELL uses DB first, Alpaca as fallback
4. Exposure caps block BUY orders
5. TradesRepository position methods
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class MockSettings:
    """Mock settings for tests."""
    
    class App:
        check_position_before_sell = True
        max_total_exposure_usd = 50000
        max_exposure_per_symbol_usd = 10000
        max_orders_per_minute = 5
        max_orders_per_symbol_per_day = 3
        max_open_orders_total = 20
        cooldown_seconds_per_symbol = 900
        execution_dry_run = True
        execution_approval_file = "/tmp/awet-paper-approved"
        execution_default_qty = 10
        max_trade_notional_usd = 1000
        max_qty_per_trade = 100
        min_qty_per_trade = 1
    
    app = App()


class TestCheckCanSell:
    """Test _check_can_sell() with DB-first logic."""
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        repo.get_position_qty = AsyncMock(return_value=None)
        return repo
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create a mock AlpacaClient."""
        client = AsyncMock()
        client.get_position_qty = AsyncMock(return_value=0)
        return client
    
    @pytest.fixture
    def execution_agent(self, mock_trades_repo, mock_alpaca_client):
        """Create ExecutionAgent with mocked dependencies."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent._alpaca_client = mock_alpaca_client
            agent.settings = MockSettings()
            agent.logger = MagicMock()
            agent._check_can_sell = ExecutionAgent._check_can_sell.__get__(agent, ExecutionAgent)
            return agent
    
    @pytest.mark.asyncio
    async def test_sell_blocked_when_db_qty_zero(self, execution_agent, mock_trades_repo):
        """SELL should be blocked when DB says qty=0."""
        mock_trades_repo.get_position_qty.return_value = 0
        
        can_sell, qty, error = await execution_agent._check_can_sell("AAPL")
        
        assert can_sell is False
        assert qty == 0
        assert "no position in DB" in error
        # Should NOT call Alpaca since DB had the answer
        execution_agent._alpaca_client.get_position_qty.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_sell_allowed_when_db_has_position(self, execution_agent, mock_trades_repo):
        """SELL should be allowed when DB has qty > 0."""
        mock_trades_repo.get_position_qty.return_value = 50
        
        can_sell, qty, error = await execution_agent._check_can_sell("AAPL")
        
        assert can_sell is True
        assert qty == 50
        assert error is None
        # Should NOT call Alpaca since DB had the answer
        execution_agent._alpaca_client.get_position_qty.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_sell_fallback_to_alpaca_when_not_in_db(
        self, execution_agent, mock_trades_repo, mock_alpaca_client
    ):
        """SELL should fallback to Alpaca API when symbol not in DB."""
        mock_trades_repo.get_position_qty.return_value = None  # Not in DB
        mock_alpaca_client.get_position_qty.return_value = 25
        
        can_sell, qty, error = await execution_agent._check_can_sell("NVDA")
        
        assert can_sell is True
        assert qty == 25
        assert error is None
        # Should call Alpaca as fallback
        mock_alpaca_client.get_position_qty.assert_called_once_with("NVDA")
    
    @pytest.mark.asyncio
    async def test_sell_blocked_when_alpaca_fallback_no_position(
        self, execution_agent, mock_trades_repo, mock_alpaca_client
    ):
        """SELL blocked when Alpaca fallback also shows no position."""
        mock_trades_repo.get_position_qty.return_value = None  # Not in DB
        mock_alpaca_client.get_position_qty.return_value = 0
        
        can_sell, qty, error = await execution_agent._check_can_sell("TSLA")
        
        assert can_sell is False
        assert "no position" in error
    
    @pytest.mark.asyncio
    async def test_sell_allowed_when_check_disabled(self, execution_agent):
        """SELL allowed without checks when check_position_before_sell=False."""
        execution_agent.settings.app.check_position_before_sell = False
        
        can_sell, qty, error = await execution_agent._check_can_sell("AAPL")
        
        assert can_sell is True
        assert error is None


class TestCheckCanBuy:
    """Test _check_can_buy() with allow_add_to_position logic."""

    @pytest.fixture
    def mock_trades_repo(self):
        repo = AsyncMock()
        repo.get_position_qty = AsyncMock(return_value=10)
        return repo

    @pytest.fixture
    def mock_alpaca_client(self):
        client = AsyncMock()
        client.get_position_qty = AsyncMock(return_value=10)
        return client

    @pytest.fixture
    def execution_agent(self, mock_trades_repo, mock_alpaca_client):
        from src.agents.execution_agent import ExecutionAgent

        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent._alpaca_client = mock_alpaca_client
            agent.settings = MockSettings()
            agent.logger = MagicMock()
            agent._check_can_buy = ExecutionAgent._check_can_buy.__get__(agent, ExecutionAgent)
            return agent

    @pytest.mark.asyncio
    async def test_buy_blocked_when_already_holding(self, execution_agent, mock_trades_repo):
        execution_agent.settings.app.allow_add_to_position = False
        mock_trades_repo.get_position_qty.return_value = 5

        can_buy, error = await execution_agent._check_can_buy("AAPL")

        assert can_buy is False
        assert "already holding" in error


class TestExposureCaps:
    """Test _check_exposure_caps() logic."""
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        repo.get_total_exposure = AsyncMock(return_value=0.0)
        repo.get_symbol_exposure = AsyncMock(return_value=0.0)
        return repo
    
    @pytest.fixture
    def execution_agent(self, mock_trades_repo):
        """Create ExecutionAgent with mocked dependencies."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent.settings = MockSettings()
            agent._check_exposure_caps = ExecutionAgent._check_exposure_caps.__get__(
                agent, ExecutionAgent
            )
            return agent
    
    @pytest.mark.asyncio
    async def test_buy_allowed_under_caps(self, execution_agent, mock_trades_repo):
        """BUY allowed when under both exposure caps."""
        mock_trades_repo.get_total_exposure.return_value = 10000
        mock_trades_repo.get_symbol_exposure.return_value = 2000
        
        passed, reason = await execution_agent._check_exposure_caps("AAPL", "buy", 1000)
        
        assert passed is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_buy_blocked_exceeds_total_cap(self, execution_agent, mock_trades_repo):
        """BUY blocked when would exceed total exposure cap."""
        mock_trades_repo.get_total_exposure.return_value = 49500  # Near $50k cap
        mock_trades_repo.get_symbol_exposure.return_value = 0
        
        passed, reason = await execution_agent._check_exposure_caps("AAPL", "buy", 1000)
        
        assert passed is False
        assert "exposure_cap" in reason
        assert "50500" in reason  # Would be $50.5k
    
    @pytest.mark.asyncio
    async def test_buy_blocked_exceeds_symbol_cap(self, execution_agent, mock_trades_repo):
        """BUY blocked when would exceed per-symbol exposure cap."""
        mock_trades_repo.get_total_exposure.return_value = 20000
        mock_trades_repo.get_symbol_exposure.return_value = 9500  # Near $10k cap
        
        passed, reason = await execution_agent._check_exposure_caps("AAPL", "buy", 1000)
        
        assert passed is False
        assert "symbol_exposure_cap" in reason
        assert "AAPL" in reason
    
    @pytest.mark.asyncio
    async def test_sell_always_passes_exposure_check(self, execution_agent):
        """SELL orders should always pass exposure checks (they reduce exposure)."""
        passed, reason = await execution_agent._check_exposure_caps("AAPL", "sell", 5000)
        
        assert passed is True
        assert reason is None


class TestTradesRepositoryPositionMethods:
    """Test TradesRepository position query methods."""
    
    @pytest.mark.asyncio
    async def test_get_position_qty_no_pool(self):
        """get_position_qty returns None when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        qty = await repo.get_position_qty("AAPL")
        assert qty is None
    
    @pytest.mark.asyncio
    async def test_get_total_exposure_no_pool(self):
        """get_total_exposure returns 0 when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        exposure = await repo.get_total_exposure()
        assert exposure == 0.0
    
    @pytest.mark.asyncio
    async def test_get_symbol_exposure_no_pool(self):
        """get_symbol_exposure returns 0 when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        exposure = await repo.get_symbol_exposure("AAPL")
        assert exposure == 0.0


class TestReconcilePositions:
    """Test reconcile_positions.py helper functions."""
    
    @pytest.mark.asyncio
    async def test_fetch_alpaca_positions_success(self):
        """fetch_alpaca_positions returns positions list."""
        from scripts.reconcile_positions import fetch_alpaca_positions
        
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": "10", "market_value": "1500.00"},
            {"symbol": "MSFT", "qty": "5", "market_value": "2000.00"},
        ]
        
        positions = await fetch_alpaca_positions(mock_client)
        
        assert len(positions) == 2
        assert positions[0]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_fetch_alpaca_positions_error(self):
        """fetch_alpaca_positions returns empty list on error."""
        from scripts.reconcile_positions import fetch_alpaca_positions
        
        mock_client = AsyncMock()
        mock_client.get_positions.side_effect = Exception("API error")
        
        positions = await fetch_alpaca_positions(mock_client)
        
        assert positions == []

    @pytest.mark.asyncio
    async def test_upsert_positions_updates_existing(self):
        """upsert_positions updates existing rows when symbol exists."""
        from scripts.reconcile_positions import upsert_positions

        conn = AsyncMock()
        conn.fetchval.return_value = "AAPL"

        positions = [
            {
                "symbol": "AAPL",
                "qty": "10",
                "avg_entry_price": "150",
                "market_value": "1500",
                "unrealized_pl": "50",
                "cost_basis": "1450",
            }
        ]

        inserted, updated = await upsert_positions(conn, positions, dry_run=False)

        assert inserted == 0
        assert updated == 1
        conn.execute.assert_called()


class TestExposureCapDefaults:
    """Test that exposure cap defaults work correctly."""
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        repo.get_total_exposure = AsyncMock(return_value=0.0)
        repo.get_symbol_exposure = AsyncMock(return_value=0.0)
        return repo
    
    @pytest.mark.asyncio
    async def test_missing_config_uses_defaults(self, mock_trades_repo):
        """Missing config attributes should use hardcoded defaults."""
        from src.agents.execution_agent import ExecutionAgent
        
        # Create settings without exposure config
        class MinimalSettings:
            class App:
                pass
            app = App()
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent.settings = MinimalSettings()
            agent._check_exposure_caps = ExecutionAgent._check_exposure_caps.__get__(
                agent, ExecutionAgent
            )
            
            # Should not raise, uses getattr defaults ($50k total, $10k per symbol)
            passed, reason = await agent._check_exposure_caps("AAPL", "buy", 1000)
            assert passed is True
