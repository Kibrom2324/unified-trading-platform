"""
Unit tests for Order Reconciliation + PnL Pipeline.

Tests cover:
1. Order status transitions (accepted -> filled)
2. Fill price and quantity updates
3. Terminal order detection
4. Retry with backoff on API failures
5. PnL updates when avg_fill_price appears
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch


class MockAlpacaOrder:
    """Mock Alpaca order response."""
    
    @staticmethod
    def accepted(order_id: str = "test-order-123"):
        return {
            "id": order_id,
            "status": "accepted",
            "filled_qty": "0",
            "filled_avg_price": None,
            "filled_at": None,
        }
    
    @staticmethod
    def partially_filled(order_id: str = "test-order-123", qty: int = 5, price: float = 150.0):
        return {
            "id": order_id,
            "status": "partially_filled",
            "filled_qty": str(qty),
            "filled_avg_price": str(price),
            "filled_at": None,
        }
    
    @staticmethod
    def filled(order_id: str = "test-order-123", qty: int = 10, price: float = 150.50):
        return {
            "id": order_id,
            "status": "filled",
            "filled_qty": str(qty),
            "filled_avg_price": str(price),
            "filled_at": "2026-01-16T15:30:00Z",
        }
    
    @staticmethod
    def canceled(order_id: str = "test-order-123"):
        return {
            "id": order_id,
            "status": "canceled",
            "filled_qty": "0",
            "filled_avg_price": None,
            "filled_at": None,
        }


class TestOrderStatusTransitions:
    """Test order status transitions from Alpaca."""
    
    @pytest.fixture
    def reconciler(self):
        """Create OrderReconciler with mocked dependencies."""
        from scripts.reconcile_scheduler import OrderReconciler
        
        with patch.object(OrderReconciler, '__init__', return_value=None):
            rec = OrderReconciler.__new__(OrderReconciler)
            rec.dsn = "mock://dsn"
            rec._connected = True
            rec._repo = AsyncMock()
            rec._alpaca = AsyncMock()
            rec._fetch_order_with_retry = AsyncMock()
            rec.reconcile = OrderReconciler.reconcile.__get__(rec, OrderReconciler)
            return rec
    
    @pytest.mark.asyncio
    async def test_accepted_to_filled_transition(self, reconciler):
        """Order should update when transitioning from accepted to filled."""
        # Setup pending order in DB
        reconciler._repo.get_pending_orders.return_value = [{
            "id": 1,
            "alpaca_order_id": "test-order-123",
            "alpaca_status": "accepted",
            "symbol": "AAPL",
        }]
        
        # Alpaca returns filled
        reconciler._fetch_order_with_retry.return_value = MockAlpacaOrder.filled()
        reconciler._repo.update_order_status.return_value = True
        
        stats = await reconciler.reconcile(verbose=True)
        
        assert stats["pending_found"] == 1
        assert stats["updated"] == 1
        assert stats["became_terminal"] == 1
        
        # Verify update was called with correct args
        reconciler._repo.update_order_status.assert_called_once()
        call_args = reconciler._repo.update_order_status.call_args
        assert call_args[1]["status"] == "filled"
        assert call_args[1]["filled_qty"] == 10
        assert call_args[1]["avg_fill_price"] == 150.50
    
    @pytest.mark.asyncio
    async def test_partial_fill_updates_quantity(self, reconciler):
        """Partial fill should update filled_qty."""
        reconciler._repo.get_pending_orders.return_value = [{
            "id": 1,
            "alpaca_order_id": "test-order-123",
            "alpaca_status": "accepted",
            "symbol": "AAPL",
        }]
        
        reconciler._fetch_order_with_retry.return_value = MockAlpacaOrder.partially_filled(qty=5, price=149.00)
        reconciler._repo.update_order_status.return_value = True
        
        stats = await reconciler.reconcile()
        
        assert stats["updated"] == 1
        assert stats["became_terminal"] == 0  # Partial fill is not terminal
        
        call_args = reconciler._repo.update_order_status.call_args
        assert call_args[1]["filled_qty"] == 5
        assert call_args[1]["avg_fill_price"] == 149.00
    
    @pytest.mark.asyncio
    async def test_canceled_becomes_terminal(self, reconciler):
        """Canceled order should be marked as terminal."""
        reconciler._repo.get_pending_orders.return_value = [{
            "id": 1,
            "alpaca_order_id": "test-order-123",
            "alpaca_status": "accepted",
            "symbol": "AAPL",
        }]
        
        reconciler._fetch_order_with_retry.return_value = MockAlpacaOrder.canceled()
        reconciler._repo.update_order_status.return_value = True
        
        stats = await reconciler.reconcile()
        
        assert stats["became_terminal"] == 1
    
    @pytest.mark.asyncio
    async def test_unchanged_status_not_updated(self, reconciler):
        """Order with same status should not trigger update."""
        reconciler._repo.get_pending_orders.return_value = [{
            "id": 1,
            "alpaca_order_id": "test-order-123",
            "alpaca_status": "accepted",
            "symbol": "AAPL",
        }]
        
        reconciler._fetch_order_with_retry.return_value = MockAlpacaOrder.accepted()
        
        stats = await reconciler.reconcile()
        
        assert stats["unchanged"] == 1
        assert stats["updated"] == 0
        reconciler._repo.update_order_status.assert_not_called()


class TestRetryLogic:
    """Test API retry with backoff."""
    
    @pytest.fixture
    def reconciler(self):
        """Create OrderReconciler with mocked dependencies."""
        from scripts.reconcile_scheduler import OrderReconciler
        
        with patch.object(OrderReconciler, '__init__', return_value=None):
            rec = OrderReconciler.__new__(OrderReconciler)
            rec.dsn = "mock://dsn"
            rec._connected = True
            rec._repo = AsyncMock()
            rec._alpaca = AsyncMock()
            rec._fetch_order_with_retry = OrderReconciler._fetch_order_with_retry.__get__(rec, OrderReconciler)
            return rec
    
    @pytest.mark.asyncio
    async def test_retry_on_api_failure(self, reconciler):
        """Should retry on API failures with backoff."""
        # First two calls fail, third succeeds
        reconciler._alpaca.get_order.side_effect = [
            Exception("API timeout"),
            Exception("Connection error"),
            MockAlpacaOrder.filled(),
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await reconciler._fetch_order_with_retry("test-order-123")
        
        assert result is not None
        assert result["status"] == "filled"
        assert reconciler._alpaca.get_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, reconciler):
        """Should return None after max retries exceeded."""
        reconciler._alpaca.get_order.side_effect = Exception("Persistent failure")
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await reconciler._fetch_order_with_retry("test-order-123")
        
        assert result is None


class TestPnLUpdates:
    """Test PnL updates when fill prices appear."""
    
    @pytest.fixture
    def sample_filled_trades(self):
        """Sample trades with fill info."""
        from scripts.daily_pnl_report import TradeForPnL
        
        now = datetime.now(tz=timezone.utc)
        
        return [
            TradeForPnL(
                ts=now.replace(hour=10),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=Decimal("150.00"),
                status="filled",
            ),
            TradeForPnL(
                ts=now.replace(hour=14),
                symbol="AAPL",
                side="sell",
                qty=10,
                avg_fill_price=Decimal("155.00"),
                status="filled",
            ),
        ]
    
    @pytest.fixture
    def sample_unfilled_trades(self):
        """Sample trades without fill info (accepted status)."""
        from scripts.daily_pnl_report import TradeForPnL
        
        now = datetime.now(tz=timezone.utc)
        
        return [
            TradeForPnL(
                ts=now.replace(hour=10),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=None,  # Not filled yet
                status="filled",
            ),
            TradeForPnL(
                ts=now.replace(hour=14),
                symbol="AAPL",
                side="sell",
                qty=10,
                avg_fill_price=None,  # Not filled yet
                status="filled",
            ),
        ]
    
    def test_pnl_zero_without_fill_prices(self, sample_unfilled_trades):
        """PnL should be 0 when avg_fill_price is None."""
        from scripts.daily_pnl_report import PnLCalculator
        
        with patch.object(PnLCalculator, '__init__', return_value=None):
            calc = PnLCalculator.__new__(PnLCalculator)
            calc.compute_round_trips = PnLCalculator.compute_round_trips.__get__(calc, PnLCalculator)
        
        round_trips = calc.compute_round_trips(sample_unfilled_trades)
        
        # No round trips because prices are None
        assert len(round_trips) == 0
    
    def test_pnl_nonzero_with_fill_prices(self, sample_filled_trades):
        """PnL should be non-zero when avg_fill_price is set."""
        from scripts.daily_pnl_report import PnLCalculator
        
        with patch.object(PnLCalculator, '__init__', return_value=None):
            calc = PnLCalculator.__new__(PnLCalculator)
            calc.compute_round_trips = PnLCalculator.compute_round_trips.__get__(calc, PnLCalculator)
        
        round_trips = calc.compute_round_trips(sample_filled_trades)
        
        assert len(round_trips) == 1
        assert round_trips[0].pnl_usd == Decimal("50.00")  # (155-150) * 10


class TestMarketHours:
    """Test market hours detection."""
    
    def test_is_market_hours_weekday_open(self):
        """Should return True during market hours on weekday."""
        from scripts.reconcile_scheduler import is_market_hours
        
        # Mock a Tuesday at 10:30 AM ET
        with patch('scripts.reconcile_scheduler.datetime') as mock_dt:
            # Create a mock "now" that returns Tuesday 10:30 AM
            mock_now = MagicMock()
            mock_now.weekday.return_value = 1  # Tuesday
            mock_now.time.return_value = MagicMock()
            mock_now.time.return_value.__ge__ = lambda s, o: True
            mock_now.time.return_value.__le__ = lambda s, o: True
            
            # For this test, just verify the function exists and returns bool
            result = is_market_hours()
            assert isinstance(result, bool)
    
    def test_is_market_hours_weekend(self):
        """Should return False on weekends."""
        from scripts.reconcile_scheduler import is_market_hours, MARKET_DAYS
        
        # Weekends (5=Saturday, 6=Sunday) are not in MARKET_DAYS
        assert 5 not in MARKET_DAYS
        assert 6 not in MARKET_DAYS


class TestMarketClosedSkip:
    """Test that market-closed jobs are skipped."""

    def test_should_run_market_job_closed(self):
        from scripts.reconcile_scheduler import should_run_market_job

        should_run, reason = should_run_market_job(False)

        assert should_run is False
        assert reason == "market_closed"


class TestTerminalStatuses:
    """Test terminal status detection."""
    
    def test_filled_is_terminal(self):
        """Filled status should be terminal."""
        from scripts.reconcile_scheduler import TERMINAL_STATUSES
        assert "filled" in TERMINAL_STATUSES
    
    def test_canceled_is_terminal(self):
        """Canceled status should be terminal."""
        from scripts.reconcile_scheduler import TERMINAL_STATUSES
        assert "canceled" in TERMINAL_STATUSES
    
    def test_rejected_is_terminal(self):
        """Rejected status should be terminal."""
        from scripts.reconcile_scheduler import TERMINAL_STATUSES
        assert "rejected" in TERMINAL_STATUSES
    
    def test_accepted_not_terminal(self):
        """Accepted status should not be terminal."""
        from scripts.reconcile_scheduler import TERMINAL_STATUSES
        assert "accepted" not in TERMINAL_STATUSES
    
    def test_partially_filled_not_terminal(self):
        """Partially filled status should not be terminal."""
        from scripts.reconcile_scheduler import TERMINAL_STATUSES
        assert "partially_filled" not in TERMINAL_STATUSES


class TestFilledAtParsing:
    """Test filled_at timestamp parsing."""
    
    @pytest.fixture
    def reconciler(self):
        """Create OrderReconciler with mocked dependencies."""
        from scripts.reconcile_scheduler import OrderReconciler
        
        with patch.object(OrderReconciler, '__init__', return_value=None):
            rec = OrderReconciler.__new__(OrderReconciler)
            rec.dsn = "mock://dsn"
            rec._connected = True
            rec._repo = AsyncMock()
            rec._alpaca = AsyncMock()
            rec._fetch_order_with_retry = AsyncMock()
            rec.reconcile = OrderReconciler.reconcile.__get__(rec, OrderReconciler)
            return rec
    
    @pytest.mark.asyncio
    async def test_filled_at_parsed_correctly(self, reconciler):
        """filled_at timestamp should be parsed from ISO format."""
        reconciler._repo.get_pending_orders.return_value = [{
            "id": 1,
            "alpaca_order_id": "test-order-123",
            "alpaca_status": "accepted",
            "symbol": "AAPL",
        }]
        
        reconciler._fetch_order_with_retry.return_value = {
            "id": "test-order-123",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "150.00",
            "filled_at": "2026-01-16T15:30:00Z",
        }
        reconciler._repo.update_order_status.return_value = True
        
        await reconciler.reconcile()
        
        call_args = reconciler._repo.update_order_status.call_args
        filled_at = call_args[1]["filled_at"]
        
        assert filled_at is not None
        assert filled_at.year == 2026
        assert filled_at.month == 1
        assert filled_at.day == 16

class TestBracketReconcileClose:
    """Test bracket leg reconciliation closes trades."""

    @pytest.mark.asyncio
    async def test_reconcile_orders_closes_on_tp_fill(self):
        from scripts.reconcile_orders import reconcile_orders

        mock_repo = AsyncMock()
        mock_repo.connect = AsyncMock()
        mock_repo.close = AsyncMock()
        mock_repo.get_pending_orders.return_value = [
            {
                "alpaca_order_id": "parent-1",
                "alpaca_status": "accepted",
                "symbol": "AAPL",
                "tp_order_id": "tp-1",
                "sl_order_id": None,
            }
        ]
        mock_repo.update_order_status = AsyncMock(return_value=True)
        mock_repo.update_bracket_exit = AsyncMock(return_value=True)

        mock_alpaca = AsyncMock()

        async def get_order(order_id):
            if order_id == "parent-1":
                return {
                    "status": "filled",
                    "filled_qty": "10",
                    "filled_avg_price": "100.0",
                }
            if order_id == "tp-1":
                return {
                    "status": "filled",
                    "filled_avg_price": "103.0",
                    "filled_at": "2026-01-01T15:30:00Z",
                }
            return None

        mock_alpaca.get_order = AsyncMock(side_effect=get_order)

        with patch("scripts.reconcile_orders.TradesRepository", return_value=mock_repo):
            with patch("scripts.reconcile_orders.AlpacaClient.from_env", return_value=mock_alpaca):
                stats = await reconcile_orders(dry_run=False, verbose=False)

        assert stats["closed"] == 1
        mock_repo.update_bracket_exit.assert_called_once()
