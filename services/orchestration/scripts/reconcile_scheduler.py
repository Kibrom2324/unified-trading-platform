#!/usr/bin/env python3
"""
Order Reconciliation Scheduler - Automatic fill tracking during market hours.

This service runs continuously and:
1. During market hours (9:30 AM - 4:00 PM ET, Mon-Fri):
   - Polls Alpaca every 60 seconds for order updates
   - Updates trades table with fill info (filled_qty, avg_fill_price, filled_at)
   - Stops polling terminal orders (filled/canceled/rejected)

2. After market close (4:00 PM ET):
   - Runs final reconciliation
   - Computes daily PnL summary
   - Saves to daily_pnl_summary table

Usage:
    python scripts/reconcile_scheduler.py           # Run scheduler
    python scripts/reconcile_scheduler.py --once    # One-time reconcile
    python scripts/reconcile_scheduler.py --eod     # Force EOD job

Environment Variables:
    DATABASE_URL: PostgreSQL connection string
    ALPACA_API_KEY: Alpaca API key
    ALPACA_API_SECRET: Alpaca API secret
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus_client import start_http_server

from src.integrations.alpaca_client import AlpacaClient
from src.integrations.trades_repository import TradesRepository
from src.monitoring.metrics import (
    REGISTRY,
    RECONCILE_ERRORS,
    PENDING_ORDERS,
    OPEN_POSITIONS,
    PORTFOLIO_VALUE,
)
from scripts.reconcile_positions import reconcile_once as reconcile_positions_once
from scripts.reconcile_positions import get_portfolio_summary

logger = structlog.get_logger(__name__)

# Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
MARKET_DAYS = {0, 1, 2, 3, 4}  # Monday-Friday

# Terminal order statuses that don't need further reconciliation
TERMINAL_STATUSES = {"filled", "canceled", "rejected", "expired", "replaced"}

# Reconciliation interval in seconds
RECONCILE_INTERVAL = 60
POSITIONS_INTERVAL = 120
PORTFOLIO_SUMMARY_TIME = time(16, 10)
OPS_METRICS_PORT = int(os.getenv("OPS_METRICS_PORT", "9108"))

# Retry settings for API failures
MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]  # seconds


def _build_dsn() -> str:
    """Build database DSN from environment."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet"
    )


def is_market_hours() -> bool:
    """Check if current time is during US market hours."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            now_et = datetime.now(et)
        except ImportError:
            # Last resort: assume ET is UTC-5 (WRONG during EDT!)
            now_et = datetime.now(timezone(timedelta(hours=-5)))
    
    # Check day of week
    if now_et.weekday() not in MARKET_DAYS:
        return False
    
    # Check time
    current_time = now_et.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


async def get_market_open_status(alpaca: AlpacaClient | None) -> tuple[bool, str]:
    """Check market open status using Alpaca clock if available."""
    if alpaca:
        try:
            clock = await alpaca.get_clock()
            return (bool(clock.get("is_open")), "alpaca_clock")
        except Exception as e:
            logger.warning("alpaca_clock_failed", error=str(e))

    return (is_market_hours(), "local_time")


def should_run_market_job(is_open: bool) -> tuple[bool, str]:
    """Return whether to run a market-hours job and the reason if skipped."""
    if not is_open:
        return (False, "market_closed")
    return (True, "ok")


async def _connect_db():
    import asyncpg
    return await asyncpg.connect(_build_dsn())


def is_after_market_close() -> bool:
    """Check if current time is after market close (for EOD job)."""
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)
    except ImportError:
        now_et = datetime.now(timezone(timedelta(hours=-5)))
    
    if now_et.weekday() not in MARKET_DAYS:
        return False
    
    current_time = now_et.time()
    eod_window_end = time(17, 0)  # 5 PM ET
    return MARKET_CLOSE < current_time <= eod_window_end


class OrderReconciler:
    """Reconcile pending orders with Alpaca API."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._repo: TradesRepository | None = None
        self._alpaca: AlpacaClient | None = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Initialize connections."""
        try:
            self._repo = TradesRepository(self.dsn)
            await self._repo.connect()
            
            self._alpaca = AlpacaClient.from_env()
            self._connected = True
            
            logger.info("reconciler_connected")
            return True
            
        except Exception as e:
            logger.error("reconciler_connect_failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close connections."""
        if self._repo:
            await self._repo.close()
        self._connected = False
    
    async def _fetch_order_with_retry(self, order_id: str) -> dict[str, Any] | None:
        """Fetch order from Alpaca with retry and backoff."""
        for attempt in range(MAX_RETRIES):
            try:
                order = await self._alpaca.get_order(order_id)
                return order
            except Exception as e:
                wait_time = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(
                    "alpaca_api_retry",
                    order_id=order_id,
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)
        
        logger.error(
            "alpaca_api_failed",
            order_id=order_id,
            message="Max retries exceeded",
        )
        return None
    
    async def reconcile(self, verbose: bool = False) -> dict[str, int]:
        """
        Reconcile all pending orders.
        
        Returns:
            Stats dict with counts
        """
        stats = {
            "pending_found": 0,
            "updated": 0,
            "became_terminal": 0,
            "unchanged": 0,
            "errors": 0,
            "closed": 0,
        }
        
        if not self._connected:
            logger.error("reconciler_not_connected")
            return stats
        
        # Get pending orders
        pending = await self._repo.get_pending_orders()
        stats["pending_found"] = len(pending)
        if not pending:
            if verbose:
                logger.debug("no_pending_orders")
            return stats
        
        logger.info("reconcile_starting", pending_count=len(pending))
        
        for trade in pending:
            order_id = trade["alpaca_order_id"]
            current_status = trade["alpaca_status"]
            symbol = trade["symbol"]
            
            # Fetch order from Alpaca
            order = await self._fetch_order_with_retry(order_id)
            
            if order is None:
                stats["errors"] += 1
                continue
            
            new_status = order.get("status", "unknown")
            filled_qty = int(order.get("filled_qty", 0) or 0)
            avg_price_str = order.get("filled_avg_price")
            avg_price = float(avg_price_str) if avg_price_str else None
            
            # Parse filled_at timestamp
            filled_at = None
            if order.get("filled_at"):
                try:
                    filled_at = datetime.fromisoformat(
                        order["filled_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            
            # Check if status changed
            if new_status == current_status and filled_qty == 0:
                stats["unchanged"] += 1
                if verbose:
                    logger.debug(
                        "order_unchanged",
                        order_id=order_id,
                        symbol=symbol,
                        status=new_status,
                    )
                continue
            
            # Update DB
            updated = await self._repo.update_order_status(
                alpaca_order_id=order_id,
                status=new_status,
                filled_qty=filled_qty if filled_qty > 0 else None,
                avg_fill_price=avg_price,
                filled_at=filled_at,
            )
            
            if updated:
                stats["updated"] += 1
                
                # Track terminal transitions
                if new_status in TERMINAL_STATUSES:
                    stats["became_terminal"] += 1
                
                logger.info(
                    "order_reconciled",
                    order_id=order_id,
                    symbol=symbol,
                    old_status=current_status,
                    new_status=new_status,
                    filled_qty=filled_qty,
                    avg_price=avg_price,
                    is_terminal=new_status in TERMINAL_STATUSES,
                )

            if new_status == "filled":
                for leg_id in [trade.get("tp_order_id"), trade.get("sl_order_id")]:
                    if not leg_id:
                        continue
                    leg = await self._fetch_order_with_retry(leg_id)
                    if not leg:
                        continue
                    leg_status = leg.get("status", "unknown")
                    if leg_status == "filled":
                        leg_price = leg.get("filled_avg_price")
                        leg_price_val = float(leg_price) if leg_price else None
                        leg_filled_at = None
                        if leg.get("filled_at"):
                            try:
                                leg_filled_at = datetime.fromisoformat(
                                    leg["filled_at"].replace("Z", "+00:00")
                                )
                            except (ValueError, AttributeError):
                                leg_filled_at = None
                        closed = await self._repo.update_bracket_exit(
                            exit_order_id=leg_id,
                            status="filled",
                            exit_fill_price=leg_price_val,
                            exit_filled_at=leg_filled_at,
                        )
                        if closed:
                            stats["closed"] += 1
                            logger.info(
                                "trade_closed",
                                symbol=symbol,
                                exit_order_id=leg_id,
                                exit_price=leg_price_val,
                            )
        
        return stats


async def run_reconcile_once(verbose: bool = False) -> dict[str, int]:
    """Run a single reconciliation cycle."""
    reconciler = OrderReconciler(_build_dsn())
    
    if not await reconciler.connect():
        return {"errors": 1}
    
    try:
        stats = await reconciler.reconcile(verbose=verbose)
        return stats
    finally:
        await reconciler.close()


async def run_reconcile_loop(interval: int = RECONCILE_INTERVAL) -> None:
    """Run continuous reconciliation loop during market hours."""
    reconciler = OrderReconciler(_build_dsn())
    
    if not await reconciler.connect():
        logger.error("Failed to connect, exiting")
        return
    
    logger.info(
        "reconcile_loop_started",
        interval=interval,
        market_hours=f"{MARKET_OPEN}-{MARKET_CLOSE} ET Mon-Fri",
    )

    start_http_server(OPS_METRICS_PORT, registry=REGISTRY)
    logger.info("ops_metrics_server_started", port=OPS_METRICS_PORT)

    last_positions_run: datetime | None = None
    last_portfolio_summary_date: date | None = None
    
    try:
        while True:
            now = datetime.now(timezone.utc)
            is_open, source = await get_market_open_status(reconciler._alpaca)
            force_run = os.getenv("FORCE_RUN_WHEN_MARKET_CLOSED") == "1"
            if not is_open and force_run:
                logger.info(
                    "force_run_market_closed",
                    force_run_market_closed=1,
                    source=source,
                )
                should_run, reason = (True, "forced_market_closed")
            else:
                should_run, reason = should_run_market_job(is_open)
            
            if should_run:
                stats = await reconciler.reconcile(verbose=True)
                logger.info(
                    "RECONCILE_ORDERS",
                    updated_count=stats.get("updated", 0),
                    pending_count=stats.get("pending_found", 0),
                    errors=stats.get("errors", 0),
                )
                PENDING_ORDERS.set(stats.get("pending_found", 0))
                if stats.get("errors", 0):
                    RECONCILE_ERRORS.labels(type="orders").inc(stats.get("errors", 0))
                
                if not last_positions_run or (now - last_positions_run).total_seconds() >= POSITIONS_INTERVAL:
                    positions_ok = await reconcile_positions_once(dry_run=False)
                    summary_conn = await _connect_db()
                    summary = await get_portfolio_summary(summary_conn)
                    await summary_conn.close()
                    OPEN_POSITIONS.set(int(summary.get("position_count", 0) or 0))
                    PORTFOLIO_VALUE.set(float(summary.get("total_market_value", 0) or 0))
                    if not positions_ok:
                        RECONCILE_ERRORS.labels(type="positions").inc()
                    last_positions_run = now
            else:
                logger.info(
                    "skipped_market_closed",
                    reason=reason,
                    source=source,
                )
            
            local_now = datetime.now().astimezone()
            if (
                local_now.time() >= PORTFOLIO_SUMMARY_TIME
                and last_portfolio_summary_date != local_now.date()
            ):
                summary_conn = await _connect_db()
                summary = await get_portfolio_summary(summary_conn)
                await summary_conn.close()
                logger.info(
                    "DAILY_PORTFOLIO_SUMMARY",
                    positions_count=int(summary.get("position_count", 0) or 0),
                    total_market_value=float(summary.get("total_market_value", 0) or 0),
                    total_unrealized_pl=float(summary.get("total_unrealized_pl", 0) or 0),
                    at_time=str(PORTFOLIO_SUMMARY_TIME),
                )
                last_portfolio_summary_date = local_now.date()
            
            await asyncio.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("reconcile_loop_stopped")
    finally:
        await reconciler.close()


async def run_eod_job() -> bool:
    """
    Run end-of-day reconciliation and PnL computation.
    
    Returns:
        True if successful
    """
    logger.info("EOD_JOB_START", date=str(date.today()))
    
    # Step 1: Final reconciliation
    print("📋 Step 1: Final order reconciliation...")
    stats = await run_reconcile_once(verbose=True)
    
    print(f"   Pending: {stats.get('pending_found', 0)}")
    print(f"   Updated: {stats.get('updated', 0)}")
    print(f"   Terminal: {stats.get('became_terminal', 0)}")
    
    # Step 2: Reconcile positions
    print("📊 Step 2: Reconciling positions...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/reconcile_positions.py", "--once"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode == 0:
            print("   ✅ Positions reconciled")
        else:
            print(f"   ⚠️ Position reconciliation failed: {result.stderr[:100]}")
    except Exception as e:
        print(f"   ⚠️ Position reconciliation error: {e}")
    
    # Step 3: Compute and save PnL
    print("💰 Step 3: Computing daily PnL...")
    try:
        from scripts.daily_pnl_report import PnLCalculator
        
        dsn = _build_dsn()
        calculator = PnLCalculator(dsn)
        await calculator.connect()
        
        try:
            summary = await calculator.compute_daily_summary(date.today())
            saved = await calculator.save_summary(summary)
            
            if saved:
                print(f"   ✅ PnL summary saved for {date.today()}")
                print(f"\n{'=' * 50}")
                print(f"  📊 End-of-Day Summary: {date.today()}")
                print(f"{'=' * 50}")
                print(f"  Realized PnL:   ${float(summary.realized_pnl_usd):>12,.2f}")
                print(f"  Unrealized PnL: ${float(summary.unrealized_pnl_usd):>12,.2f}")
                print(f"  Total PnL:      ${float(summary.total_pnl_usd):>12,.2f}")
                print(f"  Trades:         {summary.num_trades:>12}")
                print(f"  Win Rate:       {float(summary.win_rate):>11.1f}%")
                print(f"{'=' * 50}\n")
                
                logger.info(
                    "EOD_JOB_COMPLETE",
                    date=str(date.today()),
                    realized_pnl=float(summary.realized_pnl_usd),
                    total_pnl=float(summary.total_pnl_usd),
                    num_trades=summary.num_trades,
                    win_rate=float(summary.win_rate),
                )
                return True
            else:
                print("   ❌ Failed to save PnL summary")
                return False
        finally:
            await calculator.close()
            
    except Exception as e:
        logger.exception("EOD_PNL_ERROR", error=str(e))
        print(f"   ❌ PnL computation error: {e}")
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Order Reconciliation Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single reconciliation and exit",
    )
    parser.add_argument(
        "--eod",
        action="store_true",
        help="Run end-of-day job (reconcile + PnL)",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuous reconciliation during market hours (alias for default mode)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=RECONCILE_INTERVAL,
        help=f"Reconciliation interval in seconds (default: {RECONCILE_INTERVAL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Configure logging (centralized — keeps correlation_id + contextvars)
    from src.core.logging import configure_logging
    configure_logging("INFO", console=True)
    
    if args.eod:
        success = await run_eod_job()
        return 0 if success else 1
    
    elif args.once:
        stats = await run_reconcile_once(verbose=args.verbose)
        
        print()
        print("=" * 50)
        print("ORDER RECONCILIATION COMPLETE")
        print("=" * 50)
        print(f"  Pending Found:    {stats.get('pending_found', 0)}")
        print(f"  Updated:          {stats.get('updated', 0)}")
        print(f"  Became Terminal:  {stats.get('became_terminal', 0)}")
        print(f"  Unchanged:        {stats.get('unchanged', 0)}")
        print(f"  Errors:           {stats.get('errors', 0)}")
        print()
        
        return 1 if stats.get("errors", 0) > 0 else 0
    
    else:
        # Run continuous loop
        await run_reconcile_loop(interval=args.interval)
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Stopped")
        sys.exit(0)
