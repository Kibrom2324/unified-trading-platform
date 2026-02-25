"""Unit tests for backtester risk controls."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.backtester_agent import BacktestRequest, BacktesterService


class TestBacktestRequest:
    """Tests for BacktestRequest model."""

    def test_default_risk_parameters(self):
        """Test default risk control values."""
        request = BacktestRequest(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
        )
        assert request.stop_loss_pct == 0.02  # 2% stop-loss
        assert request.take_profit_pct == 0.05  # 5% take-profit
        assert request.cooldown_bars == 3
        assert request.max_total_exposure_pct == 0.5  # 50% max exposure
        assert request.slippage_bps == 5.0
        assert request.fee_per_trade == 1.0

    def test_custom_risk_parameters(self):
        """Test custom risk control values."""
        request = BacktestRequest(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            stop_loss_pct=0.01,
            take_profit_pct=0.03,
            cooldown_bars=5,
            max_total_exposure_pct=0.3,
            slippage_bps=10.0,
            fee_per_trade=2.0,
        )
        assert request.stop_loss_pct == 0.01
        assert request.take_profit_pct == 0.03
        assert request.cooldown_bars == 5
        assert request.max_total_exposure_pct == 0.3
        assert request.slippage_bps == 10.0
        assert request.fee_per_trade == 2.0

    def test_initial_cash_default(self):
        """Test initial cash default value."""
        request = BacktestRequest(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
        )
        assert request.initial_cash == 10000.0

    def test_max_position_percentage(self):
        """Test max position percentage default."""
        request = BacktestRequest(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
        )
        assert request.max_pos_pct == 0.1  # 10% per position


class TestTransactionCosts:
    """Tests for transaction cost calculations."""

    def test_slippage_increases_buy_price(self):
        """Slippage should increase effective buy price."""
        slippage_bps = 10.0
        base_price = 100.0
        fill_price = base_price * (1 + slippage_bps / 10000)
        assert fill_price == 100.10  # 10 bps = 0.1%

    def test_slippage_decreases_sell_price(self):
        """Slippage should decrease effective sell price."""
        slippage_bps = 10.0
        base_price = 100.0
        fill_price = base_price * (1 - slippage_bps / 10000)
        assert fill_price == 99.90  # 10 bps = 0.1%

    def test_fee_reduces_proceeds(self):
        """Fixed fee should reduce net proceeds."""
        qty = 10.0
        fill_price = 100.0
        fee = 1.0
        gross_proceeds = qty * fill_price
        net_proceeds = gross_proceeds - fee
        assert gross_proceeds == 1000.0
        assert net_proceeds == 999.0


class TestStopLossTakeProfit:
    """Tests for stop-loss and take-profit logic."""

    def test_stop_loss_trigger_calculation(self):
        """Stop-loss should trigger when return <= -stop_loss_pct."""
        entry_price = 100.0
        stop_loss_pct = 0.02  # 2%
        
        # Price at exactly stop-loss level
        current_price = entry_price * (1 - stop_loss_pct)
        current_return = (current_price - entry_price) / entry_price
        assert current_return == pytest.approx(-0.02, rel=1e-6)
        assert current_return <= -stop_loss_pct  # Should trigger

        # Price above stop-loss level (no trigger)
        current_price = entry_price * (1 - stop_loss_pct + 0.01)
        current_return = (current_price - entry_price) / entry_price
        assert current_return > -stop_loss_pct  # Should not trigger

    def test_take_profit_trigger_calculation(self):
        """Take-profit should trigger when return >= take_profit_pct."""
        entry_price = 100.0
        take_profit_pct = 0.05  # 5%
        
        # Price at exactly take-profit level
        current_price = entry_price * (1 + take_profit_pct)
        current_return = (current_price - entry_price) / entry_price
        assert current_return == pytest.approx(0.05, rel=1e-6)
        assert current_return >= take_profit_pct  # Should trigger

        # Price below take-profit level (no trigger)
        current_price = entry_price * (1 + take_profit_pct - 0.01)
        current_return = (current_price - entry_price) / entry_price
        assert current_return < take_profit_pct  # Should not trigger


class TestCooldownLogic:
    """Tests for cooldown enforcement."""

    def test_cooldown_decrement(self):
        """Cooldown should decrement each bar."""
        cooldowns = {"AAPL": 3}
        
        # Bar 1
        cooldowns["AAPL"] -= 1
        assert cooldowns["AAPL"] == 2
        
        # Bar 2
        cooldowns["AAPL"] -= 1
        assert cooldowns["AAPL"] == 1
        
        # Bar 3 - should be removed
        cooldowns["AAPL"] -= 1
        if cooldowns["AAPL"] <= 0:
            del cooldowns["AAPL"]
        assert "AAPL" not in cooldowns

    def test_cooldown_blocks_new_trades(self):
        """Symbol in cooldown should block new trade entries."""
        cooldowns = {"AAPL": 2}
        symbol = "AAPL"
        
        # Should skip if in cooldown
        should_skip = symbol in cooldowns
        assert should_skip is True
        
        # Different symbol not blocked
        other_symbol = "GOOGL"
        should_skip_other = other_symbol in cooldowns
        assert should_skip_other is False


class TestMaxExposure:
    """Tests for max exposure enforcement."""

    def test_exposure_limit_blocks_new_positions(self):
        """Max exposure limit should prevent new positions."""
        cash = 50000.0
        max_total_exposure_pct = 0.5  # 50%
        
        # Current positions
        positions = {
            "AAPL": {"qty": 100, "entry_price": 150.0},
            "GOOGL": {"qty": 20, "entry_price": 140.0},
        }
        last_prices = {"AAPL": 155.0, "GOOGL": 145.0}
        
        # Calculate current exposure
        current_exposure = sum(
            pos["qty"] * last_prices.get(sym, pos["entry_price"])
            for sym, pos in positions.items()
        )
        assert current_exposure == 100 * 155.0 + 20 * 145.0  # 15500 + 2900 = 18400
        
        current_equity = cash + current_exposure
        assert current_equity == 50000 + 18400  # 68400
        
        exposure_ratio = current_exposure / current_equity
        assert exposure_ratio == pytest.approx(18400 / 68400, rel=1e-6)
        
        # Should block if at or above limit
        should_block = exposure_ratio >= max_total_exposure_pct
        assert should_block is False  # 26.9% < 50%

        # If exposure were higher (still under 50%)
        high_exposure = 40000.0
        high_equity = cash + high_exposure
        high_ratio = high_exposure / high_equity
        assert high_ratio < max_total_exposure_pct  # 44.4% < 50%, still ok

        # Very high exposure (over 50%)
        very_high_exposure = 60000.0
        very_high_equity = cash + very_high_exposure
        very_high_ratio = very_high_exposure / very_high_equity
        assert very_high_ratio > max_total_exposure_pct  # 54.5% > 50%, should block


class TestPnLCalculation:
    """Tests for P&L calculation with costs."""

    def test_pnl_includes_fees(self):
        """P&L should subtract fees."""
        entry_price = 100.0
        exit_price = 110.0
        qty = 10.0
        fee = 1.0
        
        # Entry cost includes fee
        entry_cost = qty * entry_price + fee
        assert entry_cost == 1001.0
        
        # Exit proceeds minus fee
        exit_proceeds = qty * exit_price - fee
        assert exit_proceeds == 1099.0
        
        # Total P&L
        pnl = exit_proceeds - entry_cost
        assert pnl == 98.0  # Not 100, due to 2x fees

    def test_pnl_with_slippage(self):
        """P&L should account for slippage."""
        base_entry = 100.0
        base_exit = 110.0
        slippage_bps = 10.0
        qty = 10.0
        
        # Actual fill prices
        fill_entry = base_entry * (1 + slippage_bps / 10000)
        fill_exit = base_exit * (1 - slippage_bps / 10000)
        
        assert fill_entry == 100.10
        assert fill_exit == 109.89
        
        # P&L with slippage
        pnl = (fill_exit - fill_entry) * qty
        expected_pnl = (109.89 - 100.10) * 10
        assert pnl == pytest.approx(expected_pnl, rel=1e-6)
        assert pnl < 100  # Less than without slippage


class TestReturnCalculation:
    """Tests for return percentage calculation."""

    def test_return_pct_calculation(self):
        """Return percentage should be (exit - entry) / entry."""
        entry_price = 100.0
        exit_price = 110.0
        
        return_pct = (exit_price - entry_price) / entry_price
        assert return_pct == 0.10  # 10% return

    def test_negative_return(self):
        """Negative return for losing trade."""
        entry_price = 100.0
        exit_price = 95.0
        
        return_pct = (exit_price - entry_price) / entry_price
        assert return_pct == -0.05  # -5% return
