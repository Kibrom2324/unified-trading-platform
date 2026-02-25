#!/usr/bin/env python3
"""
End-of-Day PnL Job - Runs at market close to compute and persist daily summary.

This script:
1. Reconciles orders with Alpaca (ensures trades table is up to date)
2. Reconciles positions (ensures positions table is up to date)
3. Computes daily PnL summary
4. Saves to daily_pnl_summary table
5. Logs PNL_DAILY_SUMMARY event

Designed to be run via cron at 23:59 ET or after market close.

Usage:
    python scripts/eod_pnl_job.py
    
    # Via cron (example):
    # 59 23 * * 1-5 /path/to/venv/bin/python /path/to/scripts/eod_pnl_job.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import date, datetime, timezone

import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_pnl_report import PnLCalculator

logger = structlog.get_logger(__name__)


async def run_reconcile_orders() -> bool:
    """Run order reconciliation script."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/reconcile_orders.py", "--once"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode == 0:
            logger.info("eod_reconcile_orders_ok")
            return True
        else:
            logger.warning("eod_reconcile_orders_failed", stderr=result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.warning("eod_reconcile_orders_timeout")
        return False
    except FileNotFoundError:
        # Script doesn't exist yet - skip
        logger.debug("eod_reconcile_orders_skipped", reason="script_not_found")
        return True
    except Exception as e:
        logger.exception("eod_reconcile_orders_error", error=str(e))
        return False


async def run_reconcile_positions() -> bool:
    """Run position reconciliation script."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/reconcile_positions.py", "--once"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode == 0:
            logger.info("eod_reconcile_positions_ok")
            return True
        else:
            logger.warning("eod_reconcile_positions_failed", stderr=result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.warning("eod_reconcile_positions_timeout")
        return False
    except FileNotFoundError:
        logger.debug("eod_reconcile_positions_skipped", reason="script_not_found")
        return True
    except Exception as e:
        logger.exception("eod_reconcile_positions_error", error=str(e))
        return False


async def main() -> int:
    """Run end-of-day PnL job."""
    start_time = datetime.now(tz=timezone.utc)
    logger.info("EOD_PNL_JOB_START", date=str(date.today()))
    
    # Step 1: Reconcile orders
    print("📋 Step 1: Reconciling orders...")
    await run_reconcile_orders()
    
    # Step 2: Reconcile positions
    print("📊 Step 2: Reconciling positions...")
    await run_reconcile_positions()
    
    # Step 3: Compute and save PnL summary
    print("💰 Step 3: Computing daily PnL summary...")
    dsn = os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet"
    )
    calculator = PnLCalculator(dsn)
    
    try:
        await calculator.connect()
        
        summary = await calculator.compute_daily_summary(date.today())
        saved = await calculator.save_summary(summary)
        
        if saved:
            print(f"✅ Saved daily PnL summary for {date.today()}")
            logger.info(
                "EOD_PNL_JOB_COMPLETE",
                date=str(date.today()),
                realized_pnl=float(summary.realized_pnl_usd),
                unrealized_pnl=float(summary.unrealized_pnl_usd),
                total_pnl=float(summary.total_pnl_usd),
                num_trades=summary.num_trades,
                win_rate=float(summary.win_rate),
                duration_seconds=(datetime.now(tz=timezone.utc) - start_time).total_seconds(),
            )
            
            # Print summary
            print(f"\n{'=' * 50}")
            print(f"  📊 End-of-Day Summary: {date.today()}")
            print(f"{'=' * 50}")
            print(f"  Realized PnL:   ${float(summary.realized_pnl_usd):>12,.2f}")
            print(f"  Unrealized PnL: ${float(summary.unrealized_pnl_usd):>12,.2f}")
            print(f"  Total PnL:      ${float(summary.total_pnl_usd):>12,.2f}")
            print(f"  Trades:         {summary.num_trades:>12}")
            print(f"  Win Rate:       {float(summary.win_rate):>11.1f}%")
            print(f"{'=' * 50}\n")
            
            return 0
        else:
            print("❌ Failed to save daily PnL summary")
            return 1
            
    except Exception as e:
        logger.exception("EOD_PNL_JOB_ERROR", error=str(e))
        print(f"❌ Error: {e}")
        return 1
    finally:
        await calculator.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
