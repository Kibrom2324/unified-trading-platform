"""
Unit tests for Exit Logic v1 - automatic position closing.

Tests cover:
1. Take profit triggers sell
2. Stop loss triggers sell
3. Time exit triggers sell
4. No position -> blocked
5. Safety gates (dry_run, approval file)
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class MockSettings:
    """Mock settings for tests."""
    
    class App:
        enable_exit_logic = True
        take_profit_pct = 1.0
        stop_loss_pct = 0.5
        max_holding_minutes = 60
        exit_check_interval_seconds = 60
        execution_dry_run = False
        execution_approval_file = "/tmp/test-approval"
    
    class Logging:
        level = "DEBUG"
    
    app = App()
    logging = Logging()


class TestExitConditions:
    """Test _check_exit_condition() logic."""
    
    @pytest.fixture
    def exit_agent(self):
        """Create ExitAgent with mocked dependencies."""
        from src.agents.exit_agent import ExitAgent
        
        with patch.object(ExitAgent, '__init__', return_value=None):
            agent = ExitAgent.__new__(ExitAgent)
            agent.take_profit_pct = 1.0
            agent.stop_loss_pct = 0.5
            agent.max_holding_minutes = 60
            agent._check_exit_condition = ExitAgent._check_exit_condition.__get__(
                agent, ExitAgent
            )
            return agent
    
    def test_take_profit_triggers_exit(self, exit_agent):
        """Exit should trigger when PnL% >= take_profit_pct."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=1.5, holding_minutes=10
        )
        
        assert should_exit is True
        assert reason == "take_profit"
    
    def test_take_profit_exact_threshold(self, exit_agent):
        """Exit should trigger at exactly take_profit_pct."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=1.0, holding_minutes=10
        )
        
        assert should_exit is True
        assert reason == "take_profit"
    
    def test_stop_loss_triggers_exit(self, exit_agent):
        """Exit should trigger when PnL% <= -stop_loss_pct."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=-0.8, holding_minutes=10
        )
        
        assert should_exit is True
        assert reason == "stop_loss"
    
    def test_stop_loss_exact_threshold(self, exit_agent):
        """Exit should trigger at exactly -stop_loss_pct."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=-0.5, holding_minutes=10
        )
        
        assert should_exit is True
        assert reason == "stop_loss"
    
    def test_time_exit_triggers_exit(self, exit_agent):
        """Exit should trigger when holding_minutes >= max_holding_minutes."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=0.2, holding_minutes=65
        )
        
        assert should_exit is True
        assert reason == "time_exit"
    
    def test_time_exit_exact_threshold(self, exit_agent):
        """Exit should trigger at exactly max_holding_minutes."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=0.2, holding_minutes=60
        )
        
        assert should_exit is True
        assert reason == "time_exit"
    
    def test_no_exit_within_bounds(self, exit_agent):
        """No exit when PnL% and time are within bounds."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=0.5, holding_minutes=30
        )
        
        assert should_exit is False
        assert reason is None
    
    def test_take_profit_priority_over_time(self, exit_agent):
        """Take profit should trigger first even if time also exceeded."""
        should_exit, reason = exit_agent._check_exit_condition(
            pnl_pct=2.0, holding_minutes=120
        )
        
        assert should_exit is True
        assert reason == "take_profit"


class TestPnLCalculation:
    """Test _calculate_pnl_pct() logic."""
    
    @pytest.fixture
    def exit_agent(self):
        """Create ExitAgent with mocked dependencies."""
        from src.agents.exit_agent import ExitAgent
        
        with patch.object(ExitAgent, '__init__', return_value=None):
            agent = ExitAgent.__new__(ExitAgent)
            agent._calculate_pnl_pct = ExitAgent._calculate_pnl_pct.__get__(
                agent, ExitAgent
            )
            return agent
    
    def test_positive_pnl(self, exit_agent):
        """Calculate positive PnL correctly."""
        pnl = exit_agent._calculate_pnl_pct(100.0, 101.0)
        assert pnl == pytest.approx(1.0, rel=0.01)
    
    def test_negative_pnl(self, exit_agent):
        """Calculate negative PnL correctly."""
        pnl = exit_agent._calculate_pnl_pct(100.0, 99.0)
        assert pnl == pytest.approx(-1.0, rel=0.01)
    
    def test_zero_entry_price(self, exit_agent):
        """Return 0 when entry price is 0."""
        pnl = exit_agent._calculate_pnl_pct(0.0, 100.0)
        assert pnl == 0.0
    
    def test_negative_entry_price(self, exit_agent):
        """Return 0 when entry price is negative."""
        pnl = exit_agent._calculate_pnl_pct(-100.0, 100.0)
        assert pnl == 0.0


class TestSafetyGates:
    """Test _check_safety_gates() respects dry_run and approval file."""
    
    @pytest.fixture
    def exit_agent(self):
        """Create ExitAgent with mocked dependencies."""
        from src.agents.exit_agent import ExitAgent
        
        with patch.object(ExitAgent, '__init__', return_value=None):
            agent = ExitAgent.__new__(ExitAgent)
            agent.settings = MockSettings()
            agent._check_safety_gates = ExitAgent._check_safety_gates.__get__(
                agent, ExitAgent
            )
            return agent
    
    def test_blocked_when_dry_run(self, exit_agent):
        """Should be blocked when dry_run=True."""
        exit_agent.settings.app.execution_dry_run = True
        
        allowed, error = exit_agent._check_safety_gates()
        
        assert allowed is False
        assert "dry_run" in error
    
    def test_blocked_when_approval_missing(self, exit_agent):
        """Should be blocked when approval file doesn't exist."""
        exit_agent.settings.app.execution_dry_run = False
        exit_agent.settings.app.execution_approval_file = "/nonexistent/file"
        
        allowed, error = exit_agent._check_safety_gates()
        
        assert allowed is False
        assert "approval" in error.lower()
    
    def test_allowed_when_all_gates_pass(self, exit_agent):
        """Should be allowed when dry_run=False and approval file exists."""
        import tempfile
        import os
        
        exit_agent.settings.app.execution_dry_run = False
        
        # Create a temp approval file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            exit_agent.settings.app.execution_approval_file = f.name
        
        try:
            allowed, error = exit_agent._check_safety_gates()
            assert allowed is True
            assert error is None
        finally:
            os.unlink(exit_agent.settings.app.execution_approval_file)


class TestSubmitExit:
    """Test _submit_exit() flow."""
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create a mock AlpacaClient."""
        from src.integrations.alpaca_client import OrderResult
        
        client = AsyncMock()
        client.submit_market_order.return_value = OrderResult(
            success=True,
            order_id="test-order-123",
            status="filled",
            filled_qty=10,
            avg_price=150.0,
        )
        return client
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        repo.insert_trade = AsyncMock(return_value=1)
        return repo
    
    @pytest.fixture
    def exit_agent(self, mock_alpaca_client, mock_trades_repo):
        """Create ExitAgent with mocked dependencies."""
        from src.agents.exit_agent import ExitAgent
        
        with patch.object(ExitAgent, '__init__', return_value=None):
            agent = ExitAgent.__new__(ExitAgent)
            agent.settings = MockSettings()
            agent.settings.app.execution_dry_run = False
            agent.logger = MagicMock()
            agent._alpaca_client = mock_alpaca_client
            agent._trades_repo = mock_trades_repo
            agent._check_safety_gates = ExitAgent._check_safety_gates.__get__(agent, ExitAgent)
            agent._submit_exit = ExitAgent._submit_exit.__get__(agent, ExitAgent)
            return agent
    
    @pytest.mark.asyncio
    async def test_exit_submits_sell_order(self, exit_agent, mock_alpaca_client):
        """Exit should submit a SELL order to Alpaca."""
        import tempfile
        import os
        
        # Create approval file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            exit_agent.settings.app.execution_approval_file = f.name
        
        try:
            success = await exit_agent._submit_exit(
                symbol="AAPL",
                qty=10,
                reason="take_profit",
                pnl_pct=1.5,
                current_price=150.0,
            )
            
            assert success is True
            mock_alpaca_client.submit_market_order.assert_called_once_with(
                symbol="AAPL",
                qty=10,
                side="sell",
            )
        finally:
            os.unlink(exit_agent.settings.app.execution_approval_file)
    
    @pytest.mark.asyncio
    async def test_exit_blocked_by_dry_run(self, exit_agent, mock_alpaca_client, mock_trades_repo):
        """Exit should be blocked when dry_run=True."""
        exit_agent.settings.app.execution_dry_run = True
        
        success = await exit_agent._submit_exit(
            symbol="AAPL",
            qty=10,
            reason="stop_loss",
            pnl_pct=-1.0,
            current_price=145.0,
        )
        
        assert success is False
        mock_alpaca_client.submit_market_order.assert_not_called()
        # But trade should still be persisted as blocked
        mock_trades_repo.insert_trade.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exit_persists_trade_record(self, exit_agent, mock_trades_repo):
        """Exit should persist trade to database."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            exit_agent.settings.app.execution_approval_file = f.name
        
        try:
            await exit_agent._submit_exit(
                symbol="MSFT",
                qty=5,
                reason="time_exit",
                pnl_pct=0.3,
                current_price=400.0,
            )
            
            mock_trades_repo.insert_trade.assert_called_once()
            trade = mock_trades_repo.insert_trade.call_args[0][0]
            assert trade.symbol == "MSFT"
            assert trade.side == "sell"
            assert "time_exit" in trade.error_message
            assert trade.paper_trade is True
        finally:
            os.unlink(exit_agent.settings.app.execution_approval_file)


class TestCheckPositions:
    """Test check_positions() integration."""
    
    @pytest.fixture
    def exit_agent(self):
        """Create ExitAgent with mocked dependencies."""
        from src.agents.exit_agent import ExitAgent
        
        with patch.object(ExitAgent, '__init__', return_value=None):
            agent = ExitAgent.__new__(ExitAgent)
            agent.settings = MockSettings()
            agent.enabled = True
            agent.take_profit_pct = 1.0
            agent.stop_loss_pct = 0.5
            agent.max_holding_minutes = 60
            agent.logger = MagicMock()
            
            # Mock dependencies
            agent._get_positions = AsyncMock(return_value=[])
            agent._get_current_price = AsyncMock(return_value=150.0)
            agent._get_entry_time = AsyncMock(return_value=datetime.now(tz=timezone.utc) - timedelta(minutes=30))
            agent._submit_exit = AsyncMock(return_value=True)
            
            # Bind methods
            agent._calculate_pnl_pct = ExitAgent._calculate_pnl_pct.__get__(agent, ExitAgent)
            agent._check_exit_condition = ExitAgent._check_exit_condition.__get__(agent, ExitAgent)
            agent.check_positions = ExitAgent.check_positions.__get__(agent, ExitAgent)
            
            return agent
    
    @pytest.mark.asyncio
    async def test_check_positions_disabled(self, exit_agent):
        """Should return early when exit logic disabled."""
        exit_agent.enabled = False
        
        result = await exit_agent.check_positions()
        
        assert result["enabled"] is False
        assert result["checked"] == 0
    
    @pytest.mark.asyncio
    async def test_check_positions_no_positions(self, exit_agent):
        """Should handle empty positions list."""
        exit_agent._get_positions.return_value = []
        
        result = await exit_agent.check_positions()
        
        assert result["checked"] == 0
        assert result["exits_triggered"] == 0
    
    @pytest.mark.asyncio
    async def test_check_positions_triggers_take_profit(self, exit_agent):
        """Should trigger exit on take profit."""
        exit_agent._get_positions.return_value = [{
            "symbol": "AAPL",
            "qty": 10,
            "avg_entry_price": 100.0,
            "market_value": 1020.0,
            "updated_at": datetime.now(tz=timezone.utc) - timedelta(minutes=30),
            "source": "db",
        }]
        exit_agent._get_current_price.return_value = 102.0  # 2% profit
        
        result = await exit_agent.check_positions()
        
        assert result["checked"] == 1
        assert result["exits_triggered"] == 1
        exit_agent._submit_exit.assert_called_once()
        call_args = exit_agent._submit_exit.call_args
        assert call_args[1]["reason"] == "take_profit"
    
    @pytest.mark.asyncio
    async def test_check_positions_triggers_stop_loss(self, exit_agent):
        """Should trigger exit on stop loss."""
        exit_agent._get_positions.return_value = [{
            "symbol": "TSLA",
            "qty": 5,
            "avg_entry_price": 200.0,
            "market_value": 990.0,
            "updated_at": datetime.now(tz=timezone.utc) - timedelta(minutes=10),
            "source": "db",
        }]
        exit_agent._get_current_price.return_value = 198.0  # -1% loss
        
        result = await exit_agent.check_positions()
        
        assert result["exits_triggered"] == 1
        call_args = exit_agent._submit_exit.call_args
        assert call_args[1]["reason"] == "stop_loss"
    
    @pytest.mark.asyncio
    async def test_check_positions_triggers_time_exit(self, exit_agent):
        """Should trigger exit on max holding time."""
        exit_agent._get_positions.return_value = [{
            "symbol": "NVDA",
            "qty": 3,
            "avg_entry_price": 500.0,
            "market_value": 1502.0,
            "updated_at": datetime.now(tz=timezone.utc) - timedelta(minutes=65),
            "source": "db",
        }]
        exit_agent._get_current_price.return_value = 500.5  # minimal profit
        exit_agent._get_entry_time.return_value = datetime.now(tz=timezone.utc) - timedelta(minutes=65)
        
        result = await exit_agent.check_positions()
        
        assert result["exits_triggered"] == 1
        call_args = exit_agent._submit_exit.call_args
        assert call_args[1]["reason"] == "time_exit"
