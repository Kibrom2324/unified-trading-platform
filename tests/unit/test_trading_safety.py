"""
Unit tests for TradingSafetyLayer.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.core.trading_safety import (
    SafetyCheckResult,
    TradingLimits,
    TradingSafetyLayer,
    is_trading_enabled,
)


class TestTradingLimits:
    """Test TradingLimits configuration."""

    def test_default_values(self):
        limits = TradingLimits()
        assert limits.trading_enabled is False
        assert limits.max_notional_per_trade == 1000.0
        assert limits.max_qty_per_trade == 100
        assert limits.max_trades_per_day == 20

    def test_from_env(self):
        with patch.dict(os.environ, {
            "TRADING_ENABLED": "true",
            "MAX_NOTIONAL_PER_TRADE": "5000",
            "MAX_QTY_PER_TRADE": "50",
            "MAX_TRADES_PER_DAY": "10",
        }):
            limits = TradingLimits.from_env()
            assert limits.trading_enabled is True
            assert limits.max_notional_per_trade == 5000.0
            assert limits.max_qty_per_trade == 50
            assert limits.max_trades_per_day == 10


class TestKillSwitch:
    """Test kill switch functionality."""

    def test_kill_switch_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TRADING_ENABLED", None)
            layer = TradingSafetyLayer()
            result = layer.check_kill_switch()
            assert result.passed is False
            assert "kill switch" in result.reason.lower()

    def test_kill_switch_enabled(self):
        with patch.dict(os.environ, {"TRADING_ENABLED": "true"}):
            layer = TradingSafetyLayer()
            result = layer.check_kill_switch()
            assert result.passed is True

    def test_kill_switch_case_insensitive(self):
        with patch.dict(os.environ, {"TRADING_ENABLED": "TRUE"}):
            layer = TradingSafetyLayer()
            result = layer.check_kill_switch()
            assert result.passed is True

    def test_is_trading_enabled_helper(self):
        with patch.dict(os.environ, {"TRADING_ENABLED": "false"}):
            assert is_trading_enabled() is False
        with patch.dict(os.environ, {"TRADING_ENABLED": "true"}):
            assert is_trading_enabled() is True


class TestNotionalLimits:
    """Test per-trade notional limits."""

    def test_notional_within_limit(self):
        limits = TradingLimits(max_notional_per_trade=1000.0)
        layer = TradingSafetyLayer(limits=limits)
        result = layer.check_notional_limit(500.0, "AAPL")
        assert result.passed is True

    def test_notional_exceeds_limit(self):
        limits = TradingLimits(max_notional_per_trade=1000.0)
        layer = TradingSafetyLayer(limits=limits)
        result = layer.check_notional_limit(1500.0, "AAPL")
        assert result.passed is False
        assert "1500" in result.reason
        assert "1000" in result.reason

    def test_notional_at_limit(self):
        limits = TradingLimits(max_notional_per_trade=1000.0)
        layer = TradingSafetyLayer(limits=limits)
        result = layer.check_notional_limit(1000.0, "AAPL")
        assert result.passed is True


class TestQuantityLimits:
    """Test per-trade quantity limits."""

    def test_qty_within_limit(self):
        limits = TradingLimits(max_qty_per_trade=100)
        layer = TradingSafetyLayer(limits=limits)
        result = layer.check_quantity_limit(50, "AAPL")
        assert result.passed is True

    def test_qty_exceeds_limit(self):
        limits = TradingLimits(max_qty_per_trade=100)
        layer = TradingSafetyLayer(limits=limits)
        result = layer.check_quantity_limit(150, "AAPL")
        assert result.passed is False
        assert "150" in result.reason

    def test_qty_at_limit(self):
        limits = TradingLimits(max_qty_per_trade=100)
        layer = TradingSafetyLayer(limits=limits)
        result = layer.check_quantity_limit(100, "AAPL")
        assert result.passed is True


class TestDailyLimits:
    """Test daily trade count and notional limits."""

    def test_daily_trade_count_within_limit(self):
        limits = TradingLimits(max_trades_per_day=20)
        layer = TradingSafetyLayer(limits=limits)
        layer._daily_trade_count = 10
        layer._last_reset_date = datetime.now(tz=timezone.utc)
        result = layer.check_daily_trade_count()
        assert result.passed is True

    def test_daily_trade_count_at_limit(self):
        limits = TradingLimits(max_trades_per_day=20)
        layer = TradingSafetyLayer(limits=limits)
        layer._daily_trade_count = 20
        layer._last_reset_date = datetime.now(tz=timezone.utc)
        result = layer.check_daily_trade_count()
        assert result.passed is False

    def test_daily_counters_reset_on_new_day(self):
        limits = TradingLimits(max_trades_per_day=20)
        layer = TradingSafetyLayer(limits=limits)
        layer._daily_trade_count = 20
        layer._daily_notional = 50000.0
        layer._last_reset_date = datetime.now(tz=timezone.utc) - timedelta(days=1)

        result = layer.check_daily_trade_count()
        assert result.passed is True
        assert layer._daily_trade_count == 0

    def test_daily_notional_within_limit(self):
        limits = TradingLimits(max_notional_per_day=10000.0)
        layer = TradingSafetyLayer(limits=limits)
        layer._daily_notional = 5000.0
        layer._last_reset_date = datetime.now(tz=timezone.utc)
        result = layer.check_daily_notional(2000.0)
        assert result.passed is True

    def test_daily_notional_exceeds_limit(self):
        limits = TradingLimits(max_notional_per_day=10000.0)
        layer = TradingSafetyLayer(limits=limits)
        layer._daily_notional = 9000.0
        layer._last_reset_date = datetime.now(tz=timezone.utc)
        result = layer.check_daily_notional(2000.0)
        assert result.passed is False


class TestPositionLimits:
    """Test position sizing limits."""

    @pytest.mark.asyncio
    async def test_position_within_limit(self):
        limits = TradingLimits(max_position_pct=0.05)
        layer = TradingSafetyLayer(limits=limits)
        # 5% of 100k = 5k
        result = await layer.check_position_limit("AAPL", 4000.0, portfolio_value=100000.0)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_position_exceeds_limit(self):
        limits = TradingLimits(max_position_pct=0.05)
        layer = TradingSafetyLayer(limits=limits)
        result = await layer.check_position_limit("AAPL", 6000.0, portfolio_value=100000.0)
        assert result.passed is False
        assert "5%" in result.reason


class TestExposureLimits:
    """Test total and per-symbol exposure limits."""

    @pytest.mark.asyncio
    async def test_total_exposure_within_limit(self):
        limits = TradingLimits(max_total_exposure=50000.0)
        layer = TradingSafetyLayer(limits=limits)
        result = await layer.check_total_exposure(5000.0)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_total_exposure_with_repo(self):
        limits = TradingLimits(max_total_exposure=50000.0)
        mock_repo = AsyncMock()
        mock_repo.get_total_exposure.return_value = 45000.0
        layer = TradingSafetyLayer(limits=limits, trades_repo=mock_repo)
        result = await layer.check_total_exposure(10000.0)
        assert result.passed is False  # 45k + 10k = 55k > 50k

    @pytest.mark.asyncio
    async def test_symbol_exposure_within_limit(self):
        limits = TradingLimits(max_exposure_per_symbol=10000.0)
        layer = TradingSafetyLayer(limits=limits)
        result = await layer.check_symbol_exposure("AAPL", 5000.0)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_symbol_exposure_with_repo(self):
        limits = TradingLimits(max_exposure_per_symbol=10000.0)
        mock_repo = AsyncMock()
        mock_repo.get_symbol_exposure.return_value = 8000.0
        layer = TradingSafetyLayer(limits=limits, trades_repo=mock_repo)
        result = await layer.check_symbol_exposure("AAPL", 5000.0)
        assert result.passed is False  # 8k + 5k = 13k > 10k


class TestCooldown:
    """Test per-symbol cooldown."""

    @pytest.mark.asyncio
    async def test_cooldown_no_repo(self):
        limits = TradingLimits(cooldown_seconds=300)
        layer = TradingSafetyLayer(limits=limits)
        result = await layer.check_cooldown("AAPL")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cooldown_not_in_period(self):
        limits = TradingLimits(cooldown_seconds=300)
        mock_repo = AsyncMock()
        mock_repo.get_last_trade_time.return_value = (
            datetime.now(tz=timezone.utc) - timedelta(seconds=600)
        )
        layer = TradingSafetyLayer(limits=limits, trades_repo=mock_repo)
        result = await layer.check_cooldown("AAPL")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cooldown_in_period(self):
        limits = TradingLimits(cooldown_seconds=300)
        mock_repo = AsyncMock()
        mock_repo.get_last_trade_time.return_value = (
            datetime.now(tz=timezone.utc) - timedelta(seconds=60)
        )
        layer = TradingSafetyLayer(limits=limits, trades_repo=mock_repo)
        result = await layer.check_cooldown("AAPL")
        assert result.passed is False
        assert "cooldown" in result.reason.lower()


class TestCheckTrade:
    """Test full trade validation."""

    @pytest.mark.asyncio
    async def test_check_trade_passes_all(self):
        limits = TradingLimits(
            max_notional_per_trade=10000.0,
            max_qty_per_trade=100,
            max_trades_per_day=20,
            max_notional_per_day=50000.0,
            max_position_pct=0.10,
            max_total_exposure=100000.0,
            max_exposure_per_symbol=20000.0,
        )
        with patch.dict(os.environ, {"TRADING_ENABLED": "true"}):
            layer = TradingSafetyLayer(limits=limits)
            passed, results = await layer.check_trade(
                symbol="AAPL",
                side="buy",
                qty=10,
                price=150.0,
                portfolio_value=100000.0,
            )
            assert passed is True
            assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_check_trade_fails_kill_switch(self):
        limits = TradingLimits()
        with patch.dict(os.environ, {"TRADING_ENABLED": "false"}):
            layer = TradingSafetyLayer(limits=limits)
            passed, results = await layer.check_trade(
                symbol="AAPL",
                side="buy",
                qty=10,
                price=150.0,
            )
            assert passed is False
            failed = [r for r in results if not r.passed]
            assert any("kill_switch" in r.check_name for r in failed)

    @pytest.mark.asyncio
    async def test_check_trade_fails_notional(self):
        limits = TradingLimits(max_notional_per_trade=1000.0)
        with patch.dict(os.environ, {"TRADING_ENABLED": "true"}):
            layer = TradingSafetyLayer(limits=limits)
            passed, results = await layer.check_trade(
                symbol="AAPL",
                side="buy",
                qty=100,
                price=150.0,  # 15000 notional
            )
            assert passed is False
            failed = [r for r in results if not r.passed]
            assert any("notional_limit" in r.check_name for r in failed)

    @pytest.mark.asyncio
    async def test_check_trade_sell_skips_exposure_checks(self):
        """SELL orders should not check exposure limits (they reduce exposure)."""
        limits = TradingLimits(
            max_total_exposure=1000.0,  # Very low limit
            max_notional_per_trade=50000.0,
        )
        with patch.dict(os.environ, {"TRADING_ENABLED": "true"}):
            layer = TradingSafetyLayer(limits=limits)
            passed, results = await layer.check_trade(
                symbol="AAPL",
                side="sell",
                qty=100,
                price=150.0,  # 15000 notional - would fail exposure if buy
            )
            # Should pass because SELL doesn't check exposure
            failed = [r for r in results if not r.passed]
            assert not any("total_exposure" in r.check_name for r in failed)


class TestRecordTrade:
    """Test trade recording for daily limits."""

    def test_record_trade_increments_counters(self):
        layer = TradingSafetyLayer()
        layer._last_reset_date = datetime.now(tz=timezone.utc)
        layer.record_trade(1000.0)
        assert layer._daily_trade_count == 1
        assert layer._daily_notional == 1000.0

        layer.record_trade(500.0)
        assert layer._daily_trade_count == 2
        assert layer._daily_notional == 1500.0
