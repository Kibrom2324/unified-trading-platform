"""
Unit tests for Performance Reporting v1 - Daily PnL Report.

Tests cover:
1. PnL calculation (round-trip matching)
2. Win rate calculation
3. Max drawdown calculation
4. Best/worst symbol identification
5. Edge cases (no trades, partial fills, breakeven)
"""
from __future__ import annotations

import pytest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch


# Test fixtures
@pytest.fixture
def sample_trades():
    """Sample trades for testing."""
    from scripts.daily_pnl_report import TradeForPnL
    
    base_ts = datetime.now(tz=timezone.utc)
    
    return [
        # AAPL: Buy 10 @ 150, Sell 10 @ 153 = +$30 profit
        TradeForPnL(
            ts=base_ts - timedelta(hours=5),
            symbol="AAPL",
            side="buy",
            qty=10,
            avg_fill_price=Decimal("150.00"),
            status="filled",
        ),
        TradeForPnL(
            ts=base_ts - timedelta(hours=4),
            symbol="AAPL",
            side="sell",
            qty=10,
            avg_fill_price=Decimal("153.00"),
            status="filled",
        ),
        # MSFT: Buy 5 @ 400, Sell 5 @ 395 = -$25 loss
        TradeForPnL(
            ts=base_ts - timedelta(hours=3),
            symbol="MSFT",
            side="buy",
            qty=5,
            avg_fill_price=Decimal("400.00"),
            status="filled",
        ),
        TradeForPnL(
            ts=base_ts - timedelta(hours=2),
            symbol="MSFT",
            side="sell",
            qty=5,
            avg_fill_price=Decimal("395.00"),
            status="filled",
        ),
        # TSLA: Buy 3 @ 200, Sell 3 @ 200 = $0 breakeven
        TradeForPnL(
            ts=base_ts - timedelta(hours=1),
            symbol="TSLA",
            side="buy",
            qty=3,
            avg_fill_price=Decimal("200.00"),
            status="filled",
        ),
        TradeForPnL(
            ts=base_ts,
            symbol="TSLA",
            side="sell",
            qty=3,
            avg_fill_price=Decimal("200.00"),
            status="filled",
        ),
    ]


@pytest.fixture
def calculator():
    """Create PnLCalculator with mocked dependencies."""
    from scripts.daily_pnl_report import PnLCalculator
    
    with patch.object(PnLCalculator, '__init__', return_value=None):
        calc = PnLCalculator.__new__(PnLCalculator)
        calc.dsn = "mock://dsn"
        calc._pool = None
        calc.compute_round_trips = PnLCalculator.compute_round_trips.__get__(calc, PnLCalculator)
        calc.compute_max_drawdown = PnLCalculator.compute_max_drawdown.__get__(calc, PnLCalculator)
        return calc


class TestRoundTripMatching:
    """Test round-trip trade matching logic."""
    
    def test_simple_buy_sell_round_trip(self, calculator, sample_trades):
        """Match buy-sell pairs correctly."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        # Should have 3 round trips
        assert len(round_trips) == 3
        
        # Find AAPL round trip
        aapl_rt = next(rt for rt in round_trips if rt.symbol == "AAPL")
        assert aapl_rt.entry_price == Decimal("150.00")
        assert aapl_rt.exit_price == Decimal("153.00")
        assert aapl_rt.pnl_usd == Decimal("30.00")
        assert aapl_rt.is_win is True
    
    def test_loss_round_trip(self, calculator, sample_trades):
        """Identify losing trades correctly."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        msft_rt = next(rt for rt in round_trips if rt.symbol == "MSFT")
        assert msft_rt.pnl_usd == Decimal("-25.00")
        assert msft_rt.is_loss is True
        assert msft_rt.is_win is False
    
    def test_breakeven_round_trip(self, calculator, sample_trades):
        """Handle breakeven trades correctly."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        tsla_rt = next(rt for rt in round_trips if rt.symbol == "TSLA")
        assert tsla_rt.pnl_usd == Decimal("0.00")
        assert tsla_rt.is_win is False
        assert tsla_rt.is_loss is False
    
    def test_empty_trades_list(self, calculator):
        """Handle empty trades list."""
        round_trips = calculator.compute_round_trips([])
        assert round_trips == []
    
    def test_unmatched_buy_no_round_trip(self, calculator):
        """Unmatched buy should not create round trip."""
        from scripts.daily_pnl_report import TradeForPnL
        
        trades = [
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=Decimal("150.00"),
                status="filled",
            ),
        ]
        
        round_trips = calculator.compute_round_trips(trades)
        assert round_trips == []


class TestWinRateCalculation:
    """Test win rate calculation."""
    
    def test_win_rate_with_mixed_trades(self, calculator, sample_trades):
        """Calculate win rate correctly with wins, losses, breakeven."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        wins = [rt for rt in round_trips if rt.is_win]
        losses = [rt for rt in round_trips if rt.is_loss]
        
        assert len(wins) == 1  # AAPL
        assert len(losses) == 1  # MSFT
        # TSLA breakeven counts as neither win nor loss
        
        total_rt = len(wins) + len(losses)
        win_rate = (len(wins) * 100) / total_rt if total_rt > 0 else 0
        assert win_rate == 50.0
    
    def test_win_rate_all_wins(self, calculator):
        """Win rate should be 100% when all trades win."""
        from scripts.daily_pnl_report import TradeForPnL
        
        trades = [
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc) - timedelta(hours=1),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=Decimal("100.00"),
                status="filled",
            ),
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc),
                symbol="AAPL",
                side="sell",
                qty=10,
                avg_fill_price=Decimal("110.00"),
                status="filled",
            ),
        ]
        
        round_trips = calculator.compute_round_trips(trades)
        wins = [rt for rt in round_trips if rt.is_win]
        
        assert len(wins) == 1
        assert wins[0].pnl_usd == Decimal("100.00")
    
    def test_win_rate_all_losses(self, calculator):
        """Win rate should be 0% when all trades lose."""
        from scripts.daily_pnl_report import TradeForPnL
        
        trades = [
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc) - timedelta(hours=1),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=Decimal("100.00"),
                status="filled",
            ),
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc),
                symbol="AAPL",
                side="sell",
                qty=10,
                avg_fill_price=Decimal("90.00"),
                status="filled",
            ),
        ]
        
        round_trips = calculator.compute_round_trips(trades)
        losses = [rt for rt in round_trips if rt.is_loss]
        
        assert len(losses) == 1
        assert losses[0].pnl_usd == Decimal("-100.00")


class TestMaxDrawdown:
    """Test max drawdown calculation."""
    
    def test_max_drawdown_with_losses(self, calculator, sample_trades):
        """Calculate max drawdown from round trips."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        max_dd = calculator.compute_max_drawdown(round_trips)
        
        # Max drawdown should be positive (magnitude of worst drop)
        assert max_dd >= Decimal("0")
    
    def test_max_drawdown_no_trades(self, calculator):
        """Max drawdown should be 0 with no trades."""
        max_dd = calculator.compute_max_drawdown([])
        assert max_dd == Decimal("0")
    
    def test_max_drawdown_all_wins(self, calculator):
        """Max drawdown should be 0 when all trades are wins."""
        from scripts.daily_pnl_report import RoundTrip
        
        round_trips = [
            RoundTrip(
                symbol="AAPL",
                entry_side="buy",
                entry_price=Decimal("100"),
                entry_qty=10,
                exit_price=Decimal("110"),
                exit_qty=10,
                pnl_usd=Decimal("100"),
            ),
            RoundTrip(
                symbol="MSFT",
                entry_side="buy",
                entry_price=Decimal("200"),
                entry_qty=5,
                exit_price=Decimal("210"),
                exit_qty=5,
                pnl_usd=Decimal("50"),
            ),
        ]
        
        max_dd = calculator.compute_max_drawdown(round_trips)
        # With only wins, drawdown is 0
        assert max_dd == Decimal("0")


class TestBestWorstSymbol:
    """Test best/worst symbol identification."""
    
    def test_best_symbol_identification(self, sample_trades, calculator):
        """Identify best performing symbol."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        symbol_pnl: dict[str, Decimal] = {}
        for rt in round_trips:
            if rt.symbol not in symbol_pnl:
                symbol_pnl[rt.symbol] = Decimal("0")
            symbol_pnl[rt.symbol] += rt.pnl_usd
        
        best = max(symbol_pnl, key=symbol_pnl.get)
        assert best == "AAPL"
        assert symbol_pnl[best] == Decimal("30.00")
    
    def test_worst_symbol_identification(self, sample_trades, calculator):
        """Identify worst performing symbol."""
        round_trips = calculator.compute_round_trips(sample_trades)
        
        symbol_pnl: dict[str, Decimal] = {}
        for rt in round_trips:
            if rt.symbol not in symbol_pnl:
                symbol_pnl[rt.symbol] = Decimal("0")
            symbol_pnl[rt.symbol] += rt.pnl_usd
        
        worst = min(symbol_pnl, key=symbol_pnl.get)
        assert worst == "MSFT"
        assert symbol_pnl[worst] == Decimal("-25.00")


class TestDailySummary:
    """Test full daily summary computation."""
    
    @pytest.fixture
    def mock_calculator(self, sample_trades):
        """Create calculator with mocked DB."""
        from scripts.daily_pnl_report import PnLCalculator
        
        with patch.object(PnLCalculator, '__init__', return_value=None):
            calc = PnLCalculator.__new__(PnLCalculator)
            calc.dsn = "mock://dsn"
            calc._pool = AsyncMock()
            
            # Bind methods
            calc.get_trades_for_date = AsyncMock(return_value=sample_trades)
            calc.get_unrealized_pnl = AsyncMock(return_value=Decimal("50.00"))
            calc.compute_round_trips = PnLCalculator.compute_round_trips.__get__(calc, PnLCalculator)
            calc.compute_max_drawdown = PnLCalculator.compute_max_drawdown.__get__(calc, PnLCalculator)
            calc.compute_daily_summary = PnLCalculator.compute_daily_summary.__get__(calc, PnLCalculator)
            
            return calc
    
    @pytest.mark.asyncio
    async def test_daily_summary_totals(self, mock_calculator):
        """Verify daily summary totals are correct."""
        summary = await mock_calculator.compute_daily_summary(date.today())
        
        assert summary.num_trades == 6
        assert summary.num_buys == 3
        assert summary.num_sells == 3
        
        # Realized: +30 (AAPL) - 25 (MSFT) + 0 (TSLA) = +5
        assert summary.realized_pnl_usd == Decimal("5.00")
        
        # Unrealized from mock
        assert summary.unrealized_pnl_usd == Decimal("50.00")
        
        # Total
        assert summary.total_pnl_usd == Decimal("55.00")
    
    @pytest.mark.asyncio
    async def test_daily_summary_win_rate(self, mock_calculator):
        """Verify win rate calculation in summary."""
        summary = await mock_calculator.compute_daily_summary(date.today())
        
        # 1 win, 1 loss = 50%
        assert summary.num_wins == 1
        assert summary.num_losses == 1
        assert summary.win_rate == Decimal("50")
    
    @pytest.mark.asyncio
    async def test_daily_summary_best_worst(self, mock_calculator):
        """Verify best/worst symbol in summary."""
        summary = await mock_calculator.compute_daily_summary(date.today())
        
        assert summary.best_symbol == "AAPL"
        assert summary.best_symbol_pnl == Decimal("30.00")
        assert summary.worst_symbol == "MSFT"
        assert summary.worst_symbol_pnl == Decimal("-25.00")
    
    @pytest.mark.asyncio
    async def test_daily_summary_symbols_traded(self, mock_calculator):
        """Verify symbols traded list."""
        summary = await mock_calculator.compute_daily_summary(date.today())
        
        assert set(summary.symbols_traded) == {"AAPL", "MSFT", "TSLA"}


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_trades_with_no_fill_price(self, calculator):
        """Handle trades without fill price."""
        from scripts.daily_pnl_report import TradeForPnL
        
        trades = [
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=None,  # No fill price
                status="filled",
            ),
        ]
        
        round_trips = calculator.compute_round_trips(trades)
        assert round_trips == []
    
    def test_partial_fill_matching(self, calculator):
        """Handle partial fills with different quantities."""
        from scripts.daily_pnl_report import TradeForPnL
        
        trades = [
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc) - timedelta(hours=2),
                symbol="AAPL",
                side="buy",
                qty=10,
                avg_fill_price=Decimal("100.00"),
                status="filled",
            ),
            TradeForPnL(
                ts=datetime.now(tz=timezone.utc) - timedelta(hours=1),
                symbol="AAPL",
                side="sell",
                qty=5,  # Partial sell
                avg_fill_price=Decimal("105.00"),
                status="filled",
            ),
        ]
        
        round_trips = calculator.compute_round_trips(trades)
        
        # Should match 5 shares
        assert len(round_trips) == 1
        assert round_trips[0].exit_qty == 5
        assert round_trips[0].pnl_usd == Decimal("25.00")  # 5 * $5


class TestFormatting:
    """Test output formatting functions."""
    
    def test_format_usd_positive(self):
        """Format positive USD with green color."""
        from scripts.daily_pnl_report import format_usd
        
        result = format_usd(Decimal("100.50"))
        assert "+$100.50" in result
    
    def test_format_usd_negative(self):
        """Format negative USD with red color."""
        from scripts.daily_pnl_report import format_usd
        
        result = format_usd(Decimal("-50.25"))
        assert "$-50.25" in result or "-$50.25" in result
    
    def test_format_usd_zero(self):
        """Format zero USD without color."""
        from scripts.daily_pnl_report import format_usd
        
        result = format_usd(Decimal("0"))
        assert "$0.00" in result
    
    def test_format_pct_high(self):
        """Format high percentage with green color."""
        from scripts.daily_pnl_report import format_pct
        
        result = format_pct(Decimal("75.5"))
        assert "75.5%" in result
    
    def test_format_pct_low(self):
        """Format low percentage with red color."""
        from scripts.daily_pnl_report import format_pct
        
        result = format_pct(Decimal("25.0"))
        assert "25.0%" in result
