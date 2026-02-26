#!/usr/bin/env python3
"""
Daily PnL Report - Performance tracking and reporting.

This script:
1. Computes realized PnL from completed round-trip trades
2. Computes unrealized PnL from current positions
3. Calculates win rate, average win/loss, max drawdown
4. Persists daily summary to daily_pnl_summary table
5. Displays reports for today and last N days

Usage:
    python scripts/daily_pnl_report.py              # Today's report
    python scripts/daily_pnl_report.py --days 7     # Last 7 days
    python scripts/daily_pnl_report.py --save       # Save to DB
    python scripts/daily_pnl_report.py --watch      # Refresh every 60s
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import asyncpg
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = structlog.get_logger(__name__)


@dataclass
class DailyPnLSummary:
    """Daily PnL summary data."""
    report_date: date
    realized_pnl_usd: Decimal
    unrealized_pnl_usd: Decimal
    total_pnl_usd: Decimal
    num_trades: int
    num_buys: int
    num_sells: int
    num_wins: int
    num_losses: int
    win_rate: Decimal
    avg_win_usd: Decimal
    avg_loss_usd: Decimal
    max_drawdown_usd: Decimal
    best_symbol: str | None
    best_symbol_pnl: Decimal | None
    worst_symbol: str | None
    worst_symbol_pnl: Decimal | None
    total_volume_usd: Decimal
    symbols_traded: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display/JSON."""
        return {
            "report_date": str(self.report_date),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "unrealized_pnl_usd": float(self.unrealized_pnl_usd),
            "total_pnl_usd": float(self.total_pnl_usd),
            "num_trades": self.num_trades,
            "num_buys": self.num_buys,
            "num_sells": self.num_sells,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "win_rate": float(self.win_rate),
            "avg_win_usd": float(self.avg_win_usd),
            "avg_loss_usd": float(self.avg_loss_usd),
            "max_drawdown_usd": float(self.max_drawdown_usd),
            "best_symbol": self.best_symbol,
            "best_symbol_pnl": float(self.best_symbol_pnl) if self.best_symbol_pnl else None,
            "worst_symbol": self.worst_symbol,
            "worst_symbol_pnl": float(self.worst_symbol_pnl) if self.worst_symbol_pnl else None,
            "total_volume_usd": float(self.total_volume_usd),
            "symbols_traded": self.symbols_traded,
        }


@dataclass
class TradeForPnL:
    """Trade record for PnL calculation."""
    ts: datetime
    symbol: str
    side: str
    qty: int
    avg_fill_price: Decimal | None
    status: str


@dataclass
class RoundTrip:
    """A completed round-trip trade (buy then sell or vice versa)."""
    symbol: str
    entry_side: str
    entry_price: Decimal
    entry_qty: int
    exit_price: Decimal
    exit_qty: int
    pnl_usd: Decimal
    
    @property
    def is_win(self) -> bool:
        return self.pnl_usd > 0
    
    @property
    def is_loss(self) -> bool:
        return self.pnl_usd < 0


class PnLCalculator:
    """Calculate PnL from trades and positions."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None
    
    async def connect(self) -> None:
        """Connect to database."""
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        logger.info("pnl_calculator_connected")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def get_trades_for_date(self, target_date: date) -> list[TradeForPnL]:
        """Get all filled trades for a specific date."""
        if not self._pool:
            return []
        
        start_ts = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        end_ts = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, symbol, side, qty, avg_fill_price, status
                FROM trades
                WHERE ts >= $1 AND ts < $2
                  AND status = 'filled'
                ORDER BY ts ASC
                """,
                start_ts,
                end_ts,
            )
            
            return [
                TradeForPnL(
                    ts=row["ts"],
                    symbol=row["symbol"],
                    side=row["side"],
                    qty=row["qty"],
                    avg_fill_price=row["avg_fill_price"],
                    status=row["status"],
                )
                for row in rows
            ]
    
    async def get_unrealized_pnl(self) -> Decimal:
        """Get total unrealized PnL from positions table."""
        if not self._pool:
            return Decimal("0")
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COALESCE(SUM(unrealized_pl), 0)
                FROM positions
                WHERE qty > 0
                """
            )
            return Decimal(str(result)) if result else Decimal("0")
    
    async def get_positions_by_symbol(self) -> dict[str, dict]:
        """Get current positions grouped by symbol."""
        if not self._pool:
            return {}
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT symbol, qty, avg_entry_price, market_value, unrealized_pl
                FROM positions
                WHERE qty > 0
                """
            )
            return {
                row["symbol"]: {
                    "qty": row["qty"],
                    "avg_entry_price": row["avg_entry_price"],
                    "market_value": row["market_value"],
                    "unrealized_pl": row["unrealized_pl"],
                }
                for row in rows
            }
    
    def compute_round_trips(self, trades: list[TradeForPnL]) -> list[RoundTrip]:
        """
        Compute completed round-trip trades for PnL calculation.
        
        Uses FIFO matching: first buy matches first sell for same symbol.
        """
        # Group trades by symbol
        by_symbol: dict[str, list[TradeForPnL]] = {}
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)
        
        round_trips = []
        
        for symbol, symbol_trades in by_symbol.items():
            buys = [t for t in symbol_trades if t.side == "buy" and t.avg_fill_price]
            sells = [t for t in symbol_trades if t.side == "sell" and t.avg_fill_price]
            
            # FIFO matching
            buy_idx = 0
            sell_idx = 0
            
            while buy_idx < len(buys) and sell_idx < len(sells):
                buy = buys[buy_idx]
                sell = sells[sell_idx]
                
                matched_qty = min(buy.qty, sell.qty)
                
                if matched_qty > 0 and buy.avg_fill_price and sell.avg_fill_price:
                    pnl = (sell.avg_fill_price - buy.avg_fill_price) * matched_qty
                    
                    round_trips.append(RoundTrip(
                        symbol=symbol,
                        entry_side="buy",
                        entry_price=buy.avg_fill_price,
                        entry_qty=matched_qty,
                        exit_price=sell.avg_fill_price,
                        exit_qty=matched_qty,
                        pnl_usd=pnl,
                    ))
                
                # Advance indices (simplified FIFO)
                if buy.qty <= sell.qty:
                    buy_idx += 1
                if sell.qty <= buy.qty:
                    sell_idx += 1
        
        return round_trips
    
    def compute_max_drawdown(self, round_trips: list[RoundTrip]) -> Decimal:
        """
        Compute max drawdown from round-trip sequence.
        
        Simple implementation: tracks peak equity and max drop from peak.
        """
        if not round_trips:
            return Decimal("0")
        
        equity = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")
        
        for rt in sorted(round_trips, key=lambda x: x.pnl_usd, reverse=True):
            equity += rt.pnl_usd
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    async def compute_daily_summary(self, target_date: date) -> DailyPnLSummary:
        """Compute full PnL summary for a specific date."""
        trades = await self.get_trades_for_date(target_date)
        unrealized_pnl = await self.get_unrealized_pnl()
        
        # Count trades
        num_buys = sum(1 for t in trades if t.side == "buy")
        num_sells = sum(1 for t in trades if t.side == "sell")
        num_trades = len(trades)
        
        # Get unique symbols
        symbols_traded = list(set(t.symbol for t in trades))
        
        # Compute round trips
        round_trips = self.compute_round_trips(trades)
        
        # Realized PnL
        realized_pnl = sum((rt.pnl_usd for rt in round_trips), Decimal("0"))
        
        # Win/loss stats
        wins = [rt for rt in round_trips if rt.is_win]
        losses = [rt for rt in round_trips if rt.is_loss]
        num_wins = len(wins)
        num_losses = len(losses)
        
        # Win rate
        total_rt = num_wins + num_losses
        win_rate = Decimal(num_wins * 100) / Decimal(total_rt) if total_rt > 0 else Decimal("0")
        
        # Avg win/loss
        avg_win = sum((w.pnl_usd for w in wins), Decimal("0")) / Decimal(len(wins)) if wins else Decimal("0")
        avg_loss = sum((l.pnl_usd for l in losses), Decimal("0")) / Decimal(len(losses)) if losses else Decimal("0")
        
        # Max drawdown
        max_drawdown = self.compute_max_drawdown(round_trips)
        
        # Total volume
        total_volume = Decimal("0")
        for t in trades:
            if t.avg_fill_price:
                total_volume += t.avg_fill_price * t.qty
        
        # Best/worst symbol
        symbol_pnl: dict[str, Decimal] = {}
        for rt in round_trips:
            if rt.symbol not in symbol_pnl:
                symbol_pnl[rt.symbol] = Decimal("0")
            symbol_pnl[rt.symbol] += rt.pnl_usd
        
        best_symbol = None
        best_symbol_pnl = None
        worst_symbol = None
        worst_symbol_pnl = None
        
        if symbol_pnl:
            best_symbol = max(symbol_pnl, key=symbol_pnl.get)
            best_symbol_pnl = symbol_pnl[best_symbol]
            worst_symbol = min(symbol_pnl, key=symbol_pnl.get)
            worst_symbol_pnl = symbol_pnl[worst_symbol]
        
        return DailyPnLSummary(
            report_date=target_date,
            realized_pnl_usd=realized_pnl,
            unrealized_pnl_usd=unrealized_pnl,
            total_pnl_usd=realized_pnl + unrealized_pnl,
            num_trades=num_trades,
            num_buys=num_buys,
            num_sells=num_sells,
            num_wins=num_wins,
            num_losses=num_losses,
            win_rate=win_rate,
            avg_win_usd=avg_win,
            avg_loss_usd=avg_loss,
            max_drawdown_usd=max_drawdown,
            best_symbol=best_symbol,
            best_symbol_pnl=best_symbol_pnl,
            worst_symbol=worst_symbol,
            worst_symbol_pnl=worst_symbol_pnl,
            total_volume_usd=total_volume,
            symbols_traded=symbols_traded,
        )
    
    async def save_summary(self, summary: DailyPnLSummary) -> bool:
        """Save or update daily summary to database."""
        if not self._pool:
            return False
        
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO daily_pnl_summary (
                        report_date, realized_pnl_usd, unrealized_pnl_usd, total_pnl_usd,
                        num_trades, num_buys, num_sells, num_wins, num_losses,
                        win_rate, avg_win_usd, avg_loss_usd, max_drawdown_usd,
                        best_symbol, best_symbol_pnl, worst_symbol, worst_symbol_pnl,
                        total_volume_usd, symbols_traded, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, now())
                    ON CONFLICT (report_date) DO UPDATE SET
                        realized_pnl_usd = EXCLUDED.realized_pnl_usd,
                        unrealized_pnl_usd = EXCLUDED.unrealized_pnl_usd,
                        total_pnl_usd = EXCLUDED.total_pnl_usd,
                        num_trades = EXCLUDED.num_trades,
                        num_buys = EXCLUDED.num_buys,
                        num_sells = EXCLUDED.num_sells,
                        num_wins = EXCLUDED.num_wins,
                        num_losses = EXCLUDED.num_losses,
                        win_rate = EXCLUDED.win_rate,
                        avg_win_usd = EXCLUDED.avg_win_usd,
                        avg_loss_usd = EXCLUDED.avg_loss_usd,
                        max_drawdown_usd = EXCLUDED.max_drawdown_usd,
                        best_symbol = EXCLUDED.best_symbol,
                        best_symbol_pnl = EXCLUDED.best_symbol_pnl,
                        worst_symbol = EXCLUDED.worst_symbol,
                        worst_symbol_pnl = EXCLUDED.worst_symbol_pnl,
                        total_volume_usd = EXCLUDED.total_volume_usd,
                        symbols_traded = EXCLUDED.symbols_traded,
                        updated_at = now()
                    """,
                    summary.report_date,
                    summary.realized_pnl_usd,
                    summary.unrealized_pnl_usd,
                    summary.total_pnl_usd,
                    summary.num_trades,
                    summary.num_buys,
                    summary.num_sells,
                    summary.num_wins,
                    summary.num_losses,
                    summary.win_rate,
                    summary.avg_win_usd,
                    summary.avg_loss_usd,
                    summary.max_drawdown_usd,
                    summary.best_symbol,
                    summary.best_symbol_pnl,
                    summary.worst_symbol,
                    summary.worst_symbol_pnl,
                    summary.total_volume_usd,
                    json.dumps(summary.symbols_traded),
                )
                
                logger.info(
                    "PNL_DAILY_SUMMARY",
                    report_date=str(summary.report_date),
                    realized_pnl=float(summary.realized_pnl_usd),
                    total_pnl=float(summary.total_pnl_usd),
                    win_rate=float(summary.win_rate),
                )
                
                return True
                
            except Exception as e:
                logger.exception("save_summary_error", error=str(e))
                return False
    
    async def get_historical_summaries(self, days: int = 7) -> list[DailyPnLSummary]:
        """Get last N days of summaries from DB."""
        if not self._pool:
            return []
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM daily_pnl_summary
                WHERE report_date >= $1
                ORDER BY report_date DESC
                """,
                date.today() - timedelta(days=days),
            )
            
            return [
                DailyPnLSummary(
                    report_date=row["report_date"],
                    realized_pnl_usd=row["realized_pnl_usd"],
                    unrealized_pnl_usd=row["unrealized_pnl_usd"],
                    total_pnl_usd=row["total_pnl_usd"],
                    num_trades=row["num_trades"],
                    num_buys=row["num_buys"],
                    num_sells=row["num_sells"],
                    num_wins=row["num_wins"],
                    num_losses=row["num_losses"],
                    win_rate=row["win_rate"],
                    avg_win_usd=row["avg_win_usd"],
                    avg_loss_usd=row["avg_loss_usd"],
                    max_drawdown_usd=row["max_drawdown_usd"],
                    best_symbol=row["best_symbol"],
                    best_symbol_pnl=row["best_symbol_pnl"],
                    worst_symbol=row["worst_symbol"],
                    worst_symbol_pnl=row["worst_symbol_pnl"],
                    total_volume_usd=row["total_volume_usd"],
                    symbols_traded=json.loads(row["symbols_traded"]) if row["symbols_traded"] else [],
                )
                for row in rows
            ]


def format_usd(amount: Decimal | float) -> str:
    """Format amount as USD with color indication."""
    val = float(amount)
    if val > 0:
        return f"\033[32m+${val:,.2f}\033[0m"  # Green
    elif val < 0:
        return f"\033[31m${val:,.2f}\033[0m"   # Red
    else:
        return f"${val:,.2f}"


def format_pct(pct: Decimal | float) -> str:
    """Format percentage with color indication."""
    val = float(pct)
    if val > 50:
        return f"\033[32m{val:.1f}%\033[0m"  # Green
    elif val < 50:
        return f"\033[31m{val:.1f}%\033[0m"   # Red
    else:
        return f"{val:.1f}%"


def print_summary(summary: DailyPnLSummary) -> None:
    """Pretty-print a daily summary."""
    print(f"\n{'=' * 60}")
    print(f"  📊 Daily PnL Report: {summary.report_date}")
    print(f"{'=' * 60}")
    
    print(f"\n  💰 P&L Summary")
    print(f"  {'─' * 40}")
    print(f"  Realized PnL:    {format_usd(summary.realized_pnl_usd)}")
    print(f"  Unrealized PnL:  {format_usd(summary.unrealized_pnl_usd)}")
    print(f"  Total PnL:       {format_usd(summary.total_pnl_usd)}")
    
    print(f"\n  📈 Trade Stats")
    print(f"  {'─' * 40}")
    print(f"  Total Trades:    {summary.num_trades}")
    print(f"  Buys:            {summary.num_buys}")
    print(f"  Sells:           {summary.num_sells}")
    print(f"  Volume:          ${float(summary.total_volume_usd):,.2f}")
    
    print(f"\n  🎯 Performance")
    print(f"  {'─' * 40}")
    print(f"  Win Rate:        {format_pct(summary.win_rate)}")
    print(f"  Wins/Losses:     {summary.num_wins}/{summary.num_losses}")
    print(f"  Avg Win:         {format_usd(summary.avg_win_usd)}")
    print(f"  Avg Loss:        {format_usd(summary.avg_loss_usd)}")
    print(f"  Max Drawdown:    {format_usd(summary.max_drawdown_usd)}")
    
    if summary.best_symbol or summary.worst_symbol:
        print(f"\n  🏆 By Symbol")
        print(f"  {'─' * 40}")
        if summary.best_symbol:
            print(f"  Best:            {summary.best_symbol} ({format_usd(summary.best_symbol_pnl)})")
        if summary.worst_symbol:
            print(f"  Worst:           {summary.worst_symbol} ({format_usd(summary.worst_symbol_pnl)})")
    
    if summary.symbols_traded:
        print(f"\n  📋 Symbols Traded: {', '.join(sorted(summary.symbols_traded))}")
    
    print(f"\n{'=' * 60}\n")


def print_weekly_table(summaries: list[DailyPnLSummary]) -> None:
    """Print a table of weekly summaries."""
    if not summaries:
        print("\n  No summaries found for the period.\n")
        return
    
    print(f"\n{'=' * 80}")
    print(f"  📊 Weekly PnL Summary (Last {len(summaries)} days)")
    print(f"{'=' * 80}")
    
    # Header
    print(f"\n  {'Date':<12} {'Realized':>12} {'Unrealized':>12} {'Total':>12} {'Win%':>8} {'Trades':>8}")
    print(f"  {'─' * 70}")
    
    total_realized = Decimal("0")
    total_unrealized = Decimal("0")
    total_trades = 0
    
    for s in summaries:
        total_realized += s.realized_pnl_usd
        total_trades += s.num_trades
        
        r_pnl = float(s.realized_pnl_usd)
        u_pnl = float(s.unrealized_pnl_usd)
        t_pnl = float(s.total_pnl_usd)
        
        print(f"  {str(s.report_date):<12} "
              f"{'${:>10,.2f}'.format(r_pnl):>12} "
              f"{'${:>10,.2f}'.format(u_pnl):>12} "
              f"{'${:>10,.2f}'.format(t_pnl):>12} "
              f"{float(s.win_rate):>7.1f}% "
              f"{s.num_trades:>8}")
    
    # Footer
    print(f"  {'─' * 70}")
    print(f"  {'TOTAL':<12} "
          f"{'${:>10,.2f}'.format(float(total_realized)):>12} "
          f"{'':>12} "
          f"{'':>12} "
          f"{'':>8} "
          f"{total_trades:>8}")
    
    print(f"\n{'=' * 80}\n")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Daily PnL Report")
    parser.add_argument("--days", type=int, default=1, help="Number of days to report (default: 1)")
    parser.add_argument("--save", action="store_true", help="Save summary to database")
    parser.add_argument("--watch", action="store_true", help="Watch mode: refresh every 60s")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    dsn = os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet"
    )
    calculator = PnLCalculator(dsn)
    
    try:
        await calculator.connect()
        
        while True:
            if args.days == 1:
                # Today's report
                summary = await calculator.compute_daily_summary(date.today())
                
                if args.save:
                    saved = await calculator.save_summary(summary)
                    if saved:
                        print(f"✅ Saved summary for {summary.report_date}")
                
                if args.json:
                    print(json.dumps(summary.to_dict(), indent=2))
                else:
                    print_summary(summary)
            else:
                # Multi-day report
                summaries = []
                for i in range(args.days):
                    target_date = date.today() - timedelta(days=i)
                    summary = await calculator.compute_daily_summary(target_date)
                    summaries.append(summary)
                    
                    if args.save:
                        await calculator.save_summary(summary)
                
                if args.json:
                    print(json.dumps([s.to_dict() for s in summaries], indent=2))
                else:
                    print_weekly_table(summaries)
            
            if not args.watch:
                break
            
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Refreshing in 60s... (Ctrl+C to stop)")
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        await calculator.close()


if __name__ == "__main__":
    asyncio.run(main())
