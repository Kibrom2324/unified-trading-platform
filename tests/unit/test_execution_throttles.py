"""
Unit tests for ExecutionAgent throttle guardrails.

Tests cover:
1. Duplicate idempotency key detection
2. Rate limiting (max_orders_per_minute)
3. Per-symbol daily limit (max_orders_per_symbol_per_day)
4. Max open orders check
5. Cooldown per symbol
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class MockSettings:
    """Mock settings for throttle configuration."""
    
    class App:
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
        check_position_before_sell = True
    
    app = App()


class TestThrottleChecks:
    """Test _check_throttles() method logic."""
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        repo.check_idempotency_exists = AsyncMock(return_value=False)
        repo.count_orders_in_window = AsyncMock(return_value=0)
        repo.count_orders_for_symbol_today = AsyncMock(return_value=0)
        repo.count_open_orders = AsyncMock(return_value=0)
        repo.get_last_trade_time = AsyncMock(return_value=None)
        return repo
    
    @pytest.fixture
    def execution_agent(self, mock_trades_repo):
        """Create ExecutionAgent with mocked dependencies."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent.settings = MockSettings()
            agent._check_throttles = ExecutionAgent._check_throttles.__get__(agent, ExecutionAgent)
            return agent
    
    @pytest.mark.asyncio
    async def test_all_throttles_pass(self, execution_agent):
        """All throttle checks pass - should return (True, None)."""
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-1")
        
        assert passed is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_duplicate_idempotency_key_blocked(self, execution_agent, mock_trades_repo):
        """Duplicate idempotency key should block."""
        mock_trades_repo.check_idempotency_exists.return_value = True
        
        passed, reason = await execution_agent._check_throttles("AAPL", "duplicate-key")
        
        assert passed is False
        assert reason == "duplicate_idempotency_key"
        mock_trades_repo.check_idempotency_exists.assert_called_once_with("duplicate-key")
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, execution_agent, mock_trades_repo):
        """Rate limit (max_orders_per_minute) exceeded should block."""
        mock_trades_repo.count_orders_in_window.return_value = 5  # At limit
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-2")
        
        assert passed is False
        assert "rate_limit" in reason
        assert "5/5" in reason
    
    @pytest.mark.asyncio
    async def test_rate_limit_under_limit(self, execution_agent, mock_trades_repo):
        """Under rate limit should pass."""
        mock_trades_repo.count_orders_in_window.return_value = 4  # Under limit
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-3")
        
        assert passed is True
    
    @pytest.mark.asyncio
    async def test_max_symbol_orders_exceeded(self, execution_agent, mock_trades_repo):
        """Max orders per symbol per day exceeded should block."""
        mock_trades_repo.count_orders_for_symbol_today.return_value = 3  # At limit
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-4")
        
        assert passed is False
        assert "max_symbol_orders" in reason
        assert "3/3" in reason
        assert "AAPL" in reason
    
    @pytest.mark.asyncio
    async def test_max_symbol_orders_under_limit(self, execution_agent, mock_trades_repo):
        """Under max symbol orders should pass."""
        mock_trades_repo.count_orders_for_symbol_today.return_value = 2
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-5")
        
        assert passed is True
    
    @pytest.mark.asyncio
    async def test_max_open_orders_exceeded(self, execution_agent, mock_trades_repo):
        """Max open orders exceeded should block."""
        mock_trades_repo.count_open_orders.return_value = 20  # At limit
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-6")
        
        assert passed is False
        assert "max_open_orders" in reason
        assert "20/20" in reason
    
    @pytest.mark.asyncio
    async def test_max_open_orders_under_limit(self, execution_agent, mock_trades_repo):
        """Under max open orders should pass."""
        mock_trades_repo.count_open_orders.return_value = 19
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-7")
        
        assert passed is True
    
    @pytest.mark.asyncio
    async def test_cooldown_active(self, execution_agent, mock_trades_repo):
        """Symbol cooldown active should block."""
        # Last trade was 300 seconds ago (cooldown is 900)
        recent_trade_time = datetime.now(tz=timezone.utc) - timedelta(seconds=300)
        mock_trades_repo.get_last_trade_time.return_value = recent_trade_time
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-8")
        
        assert passed is False
        assert "cooldown" in reason
        assert "AAPL" in reason
    
    @pytest.mark.asyncio
    async def test_cooldown_expired(self, execution_agent, mock_trades_repo):
        """Symbol cooldown expired should pass."""
        # Last trade was 1000 seconds ago (cooldown is 900)
        old_trade_time = datetime.now(tz=timezone.utc) - timedelta(seconds=1000)
        mock_trades_repo.get_last_trade_time.return_value = old_trade_time
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key-9")
        
        assert passed is True
    
    @pytest.mark.asyncio
    async def test_no_previous_trades_no_cooldown(self, execution_agent, mock_trades_repo):
        """No previous trades means no cooldown."""
        mock_trades_repo.get_last_trade_time.return_value = None
        
        passed, reason = await execution_agent._check_throttles("NVDA", "test-key-10")
        
        assert passed is True


class TestThrottlePriority:
    """Test that throttle checks are evaluated in correct order."""
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        return repo
    
    @pytest.fixture
    def execution_agent(self, mock_trades_repo):
        """Create ExecutionAgent with mocked dependencies."""
        from src.agents.execution_agent import ExecutionAgent
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent.settings = MockSettings()
            agent._check_throttles = ExecutionAgent._check_throttles.__get__(agent, ExecutionAgent)
            return agent
    
    @pytest.mark.asyncio
    async def test_duplicate_check_first(self, execution_agent, mock_trades_repo):
        """Duplicate check should happen first (short-circuit)."""
        mock_trades_repo.check_idempotency_exists.return_value = True
        mock_trades_repo.count_orders_in_window.return_value = 999  # Would also fail
        
        passed, reason = await execution_agent._check_throttles("AAPL", "dup-key")
        
        assert passed is False
        assert reason == "duplicate_idempotency_key"
        # Other methods should not be called
        mock_trades_repo.count_orders_in_window.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_rate_limit_before_symbol_limit(self, execution_agent, mock_trades_repo):
        """Rate limit check should come before per-symbol limit."""
        mock_trades_repo.check_idempotency_exists.return_value = False
        mock_trades_repo.count_orders_in_window.return_value = 10  # Exceeds limit
        mock_trades_repo.count_orders_for_symbol_today.return_value = 999  # Would also fail
        
        passed, reason = await execution_agent._check_throttles("AAPL", "test-key")
        
        assert passed is False
        assert "rate_limit" in reason
        # Per-symbol check should not be called
        mock_trades_repo.count_orders_for_symbol_today.assert_not_called()


class TestTradesRepositoryThrottleMethods:
    """Test TradesRepository throttle query methods."""
    
    @pytest.mark.asyncio
    async def test_count_orders_in_window_no_pool(self):
        """count_orders_in_window returns 0 when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        count = await repo.count_orders_in_window(1)
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_count_orders_for_symbol_today_no_pool(self):
        """count_orders_for_symbol_today returns 0 when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        count = await repo.count_orders_for_symbol_today("AAPL")
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_count_open_orders_no_pool(self):
        """count_open_orders returns 0 when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        count = await repo.count_open_orders()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_last_trade_time_no_pool(self):
        """get_last_trade_time returns None when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        ts = await repo.get_last_trade_time("AAPL")
        assert ts is None
    
    @pytest.mark.asyncio
    async def test_check_idempotency_exists_no_pool(self):
        """check_idempotency_exists returns False when no pool."""
        from src.integrations.trades_repository import TradesRepository
        
        repo = TradesRepository("postgresql://test:test@localhost/test")
        repo._pool = None
        
        exists = await repo.check_idempotency_exists("some-key")
        assert exists is False


class TestThrottleConfigDefaults:
    """Test that throttle config defaults work correctly."""
    
    @pytest.fixture
    def mock_trades_repo(self):
        """Create a mock TradesRepository."""
        repo = AsyncMock()
        repo.check_idempotency_exists = AsyncMock(return_value=False)
        repo.count_orders_in_window = AsyncMock(return_value=0)
        repo.count_orders_for_symbol_today = AsyncMock(return_value=0)
        repo.count_open_orders = AsyncMock(return_value=0)
        repo.get_last_trade_time = AsyncMock(return_value=None)
        return repo
    
    @pytest.mark.asyncio
    async def test_missing_config_uses_defaults(self, mock_trades_repo):
        """Missing config attributes should use hardcoded defaults."""
        from src.agents.execution_agent import ExecutionAgent
        
        # Create settings without throttle attributes
        class MinimalSettings:
            class App:
                pass
            app = App()
        
        with patch.object(ExecutionAgent, '__init__', return_value=None):
            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._trades_repo = mock_trades_repo
            agent.settings = MinimalSettings()
            agent._check_throttles = ExecutionAgent._check_throttles.__get__(agent, ExecutionAgent)
            
            # Should not raise, uses getattr defaults
            passed, reason = await agent._check_throttles("AAPL", "test-key")
            assert passed is True
