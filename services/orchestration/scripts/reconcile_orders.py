#!/usr/bin/env python3
"""
reconcile_orders.py - Update trade records with final Alpaca order status.

This script polls Alpaca for order status updates and reconciles the local
trades table with the actual fill information. This ensures the DB is the
"source of truth" for order lifecycle.

Designed to run:
- Periodically (e.g., every 5 minutes via cron)
- On-demand after a trading session
- As a background task

Usage:
    python scripts/reconcile_orders.py           # One-time reconciliation
    python scripts/reconcile_orders.py --watch   # Continuous polling
    python scripts/reconcile_orders.py --dry-run # Show what would be updated

Example cron entry (every 5 minutes during market hours):
    */5 9-16 * * 1-5 cd /home/kironix/Awet && .venv/bin/python scripts/reconcile_orders.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrations.alpaca_client import AlpacaClient
from src.integrations.trades_repository import TradesRepository

logger = structlog.get_logger(__name__)


def _build_dsn() -> str:
    """Build database DSN from environment."""
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5433')}/"
        f"{os.getenv('POSTGRES_DB', 'awet')}"
    )


async def reconcile_orders(
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Reconcile pending orders with Alpaca.
    
    Args:
        dry_run: If True, don't update DB, just log what would happen
        verbose: If True, log each order check
        
    Returns:
        Stats dict with counts of updated, unchanged, errors
    """
    stats = {
        "checked": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
        "closed": 0,
    }
    
    # Initialize clients
    repo = TradesRepository(_build_dsn())
    await repo.connect()
    
    try:
        alpaca = AlpacaClient.from_env()
    except Exception as e:
        logger.error("alpaca_client_init_failed", error=str(e))
        await repo.close()
        return stats
    
    try:
        # Get pending orders from DB
        pending = await repo.get_pending_orders()
        
        if not pending:
            if verbose:
                logger.info("no_pending_orders")
            return stats
        
        logger.info("reconcile_starting", pending_count=len(pending))
        
        for trade in pending:
            order_id = trade["alpaca_order_id"]
            current_status = trade["alpaca_status"]
            stats["checked"] += 1
            
            try:
                # Fetch current status from Alpaca
                order = await alpaca.get_order(order_id)
                
                if order is None:
                    if verbose:
                        logger.warning(
                            "order_not_found",
                            alpaca_order_id=order_id,
                            symbol=trade["symbol"],
                        )
                    stats["errors"] += 1
                    continue
                
                new_status = order.get("status", "unknown")
                filled_qty = int(order.get("filled_qty", 0) or 0)
                avg_price_str = order.get("filled_avg_price")
                avg_price = float(avg_price_str) if avg_price_str else None
                
                # Check if status changed
                if new_status == current_status:
                    if verbose:
                        logger.debug(
                            "order_unchanged",
                            alpaca_order_id=order_id,
                            status=new_status,
                        )
                    stats["unchanged"] += 1
                    continue
                
                # Status changed - update DB
                if dry_run:
                    logger.info(
                        "would_update",
                        alpaca_order_id=order_id,
                        symbol=trade["symbol"],
                        old_status=current_status,
                        new_status=new_status,
                        filled_qty=filled_qty,
                        avg_price=avg_price,
                    )
                else:
                    updated = await repo.update_order_status(
                        alpaca_order_id=order_id,
                        status=new_status,
                        filled_qty=filled_qty if filled_qty > 0 else None,
                        avg_fill_price=avg_price,
                    )
                    
                    if updated:
                        logger.info(
                            "order_reconciled",
                            alpaca_order_id=order_id,
                            symbol=trade["symbol"],
                            old_status=current_status,
                            new_status=new_status,
                            filled_qty=filled_qty,
                            avg_price=avg_price,
                        )

                # Bracket leg reconciliation
                if new_status == "filled" and not dry_run:
                    tp_order_id = trade.get("tp_order_id")
                    sl_order_id = trade.get("sl_order_id")
                    for leg_id in [tp_order_id, sl_order_id]:
                        if not leg_id:
                            continue
                        leg = await alpaca.get_order(leg_id)
                        if not leg:
                            continue
                        leg_status = leg.get("status", "unknown")
                        if leg_status == "filled":
                            leg_price = leg.get("filled_avg_price")
                            leg_price_val = float(leg_price) if leg_price else None
                            filled_at = None
                            if leg.get("filled_at"):
                                try:
                                    filled_at = datetime.fromisoformat(
                                        leg["filled_at"].replace("Z", "+00:00")
                                    )
                                except (ValueError, AttributeError):
                                    filled_at = None
                            closed = await repo.update_bracket_exit(
                                exit_order_id=leg_id,
                                status="filled",
                                exit_fill_price=leg_price_val,
                                exit_filled_at=filled_at,
                            )
                            if closed:
                                stats["closed"] += 1
                                logger.info(
                                    "trade_closed",
                                    symbol=trade["symbol"],
                                    exit_order_id=leg_id,
                                    exit_price=leg_price_val,
                                )
                
                stats["updated"] += 1
                
            except Exception as e:
                logger.exception(
                    "order_reconcile_error",
                    alpaca_order_id=order_id,
                    error=str(e),
                )
                stats["errors"] += 1
        
        return stats
        
    finally:
        await repo.close()


async def watch_orders(interval: int = 60, dry_run: bool = False) -> None:
    """
    Continuously poll for order updates.
    
    Args:
        interval: Seconds between polls
        dry_run: If True, don't update DB
    """
    logger.info("watch_mode_started", interval=interval, dry_run=dry_run)
    
    while True:
        try:
            stats = await reconcile_orders(dry_run=dry_run, verbose=True)
            logger.info(
                "reconcile_cycle_complete",
                **stats,
                next_run_in=interval,
            )
        except Exception as e:
            logger.exception("reconcile_cycle_error", error=str(e))
        
        await asyncio.sleep(interval)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile Alpaca order status with local trades table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll for updates (default: one-time run)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between polls in watch mode (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without changing DB",
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
    
    if args.watch:
        await watch_orders(interval=args.interval, dry_run=args.dry_run)
    else:
        stats = await reconcile_orders(dry_run=args.dry_run, verbose=args.verbose)
        
        print()
        print("=" * 50)
        print("ORDER RECONCILIATION COMPLETE")
        print("=" * 50)
        print(f"  Checked:   {stats['checked']}")
        print(f"  Updated:   {stats['updated']}")
        print(f"  Closed:    {stats['closed']}")
        print(f"  Unchanged: {stats['unchanged']}")
        print(f"  Errors:    {stats['errors']}")
        print()
        
        if stats["errors"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
