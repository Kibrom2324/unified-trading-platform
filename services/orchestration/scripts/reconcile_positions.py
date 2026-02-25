#!/usr/bin/env python3
"""
Reconcile Alpaca positions with local positions table.

Usage:
    python scripts/reconcile_positions.py           # One-shot reconcile
    python scripts/reconcile_positions.py --watch   # Run every 60s
    python scripts/reconcile_positions.py --dry-run # Show what would change

This script:
1. Fetches all open positions from Alpaca Paper API
2. Upserts them into the positions table
3. Zeros out any symbols no longer held
4. Prints a portfolio summary
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import asyncpg

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrations.alpaca_client import AlpacaClient, AlpacaClientError
import structlog

logger = structlog.get_logger(__name__)


def get_dsn() -> str:
    """Get database connection string."""
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5433')}/"
        f"{os.getenv('POSTGRES_DB', 'awet')}"
    )


async def fetch_alpaca_positions(client: AlpacaClient) -> list[dict[str, Any]]:
    """Fetch all open positions from Alpaca."""
    try:
        return await client.get_positions()
    except Exception as e:
        print(f"❌ Error fetching Alpaca positions: {e}")
        return []


async def upsert_positions(
    conn: asyncpg.Connection,
    positions: list[dict[str, Any]],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Upsert positions into database.
    
    Returns:
        Tuple of (inserted_count, updated_count)
    """
    inserted = 0
    updated = 0
    
    for pos in positions:
        symbol = pos.get("symbol", "").upper()
        if not symbol:
            continue
        
        qty = Decimal(str(pos.get("qty", 0)))
        avg_entry_price = Decimal(str(pos.get("avg_entry_price", 0)))
        market_value = Decimal(str(pos.get("market_value", 0)))
        unrealized_pl = Decimal(str(pos.get("unrealized_pl", 0)))
        cost_basis = Decimal(str(pos.get("cost_basis", 0)))
        
        if dry_run:
            print(f"  [DRY-RUN] Would upsert {symbol}: qty={qty}, mv=${market_value:.2f}")
            continue
        
        # Check if exists
        existing = await conn.fetchval(
            "SELECT symbol FROM positions WHERE symbol = $1",
            symbol,
        )
        
        if existing:
            await conn.execute(
                """
                UPDATE positions SET
                    qty = $2,
                    avg_entry_price = $3,
                    market_value = $4,
                    unrealized_pl = $5,
                    cost_basis = $6,
                    updated_at = NOW(),
                    raw = $7
                WHERE symbol = $1
                """,
                symbol,
                qty,
                avg_entry_price,
                market_value,
                unrealized_pl,
                cost_basis,
                pos,
            )
            updated += 1
        else:
            await conn.execute(
                """
                INSERT INTO positions (symbol, qty, avg_entry_price, market_value, unrealized_pl, cost_basis, raw)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                symbol,
                qty,
                avg_entry_price,
                market_value,
                unrealized_pl,
                cost_basis,
                pos,
            )
            inserted += 1
    
    return inserted, updated


async def zero_stale_positions(
    conn: asyncpg.Connection,
    current_symbols: set[str],
    dry_run: bool = False,
) -> int:
    """
    Zero out positions for symbols no longer held.
    
    Returns:
        Count of zeroed positions
    """
    # Get all symbols currently in DB with qty > 0
    rows = await conn.fetch(
        "SELECT symbol FROM positions WHERE qty > 0"
    )
    db_symbols = {row["symbol"] for row in rows}
    
    # Find stale symbols (in DB but not in Alpaca)
    stale_symbols = db_symbols - current_symbols
    
    if not stale_symbols:
        return 0
    
    if dry_run:
        for symbol in stale_symbols:
            print(f"  [DRY-RUN] Would zero out {symbol}")
        return len(stale_symbols)
    
    for symbol in stale_symbols:
        await conn.execute(
            """
            UPDATE positions SET
                qty = 0,
                market_value = 0,
                unrealized_pl = 0,
                updated_at = NOW()
            WHERE symbol = $1
            """,
            symbol,
        )
        print(f"  ⚠️  Zeroed out {symbol} (no longer held)")
    
    return len(stale_symbols)


async def get_portfolio_summary(conn: asyncpg.Connection) -> dict[str, Any]:
    """Get portfolio summary from positions table."""
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE qty > 0) as position_count,
            COALESCE(SUM(market_value) FILTER (WHERE qty > 0), 0) as total_market_value,
            COALESCE(SUM(unrealized_pl), 0) as total_unrealized_pl,
            COALESCE(SUM(cost_basis) FILTER (WHERE qty > 0), 0) as total_cost_basis
        FROM positions
        """
    )
    return dict(row) if row else {}


async def print_portfolio(conn: asyncpg.Connection) -> None:
    """Print formatted portfolio summary."""
    summary = await get_portfolio_summary(conn)
    
    print("\n" + "=" * 70)
    print("📊 Portfolio Summary")
    print("=" * 70)
    
    positions = await conn.fetch(
        """
        SELECT symbol, qty, avg_entry_price, market_value, unrealized_pl
        FROM positions
        WHERE qty > 0
        ORDER BY market_value DESC
        """
    )
    
    if not positions:
        print("  No open positions")
    else:
        print(f"  {'SYMBOL':<8} {'QTY':>10} {'AVG PRICE':>12} {'MKT VALUE':>14} {'P/L':>12}")
        print("-" * 70)
        
        for pos in positions:
            pl_str = f"${pos['unrealized_pl']:,.2f}" if pos['unrealized_pl'] else "$0.00"
            pl_emoji = "🟢" if (pos['unrealized_pl'] or 0) >= 0 else "🔴"
            print(
                f"  {pos['symbol']:<8} "
                f"{float(pos['qty']):>10.2f} "
                f"${float(pos['avg_entry_price'] or 0):>10.2f} "
                f"${float(pos['market_value'] or 0):>12.2f} "
                f"{pl_emoji} {pl_str:>10}"
            )
    
    print("-" * 70)
    total_mv = float(summary.get("total_market_value", 0))
    total_pl = float(summary.get("total_unrealized_pl", 0))
    total_cost = float(summary.get("total_cost_basis", 0))
    pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
    print(f"  Positions: {summary.get('position_count', 0)}")
    print(f"  Total Market Value: ${total_mv:,.2f}")
    print(f"  Total Cost Basis:   ${total_cost:,.2f}")
    print(f"  Total Unrealized P/L: ${total_pl:,.2f} ({pl_pct:+.2f}%)")
    print("=" * 70)


async def reconcile_once(dry_run: bool = False) -> bool:
    """
    Run one reconciliation cycle.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n🔄 Reconciling positions at {datetime.now(tz=timezone.utc).isoformat()}")
    
    # Connect to Alpaca
    try:
        client = AlpacaClient.from_env()
    except AlpacaClientError as e:
        print(f"❌ Alpaca client error: {e}")
        return False
    
    # Connect to database
    try:
        conn = await asyncpg.connect(get_dsn())
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        await client.close()
        return False
    
    try:
        # Fetch Alpaca positions
        positions = await fetch_alpaca_positions(client)
        print(f"  📥 Fetched {len(positions)} positions from Alpaca")
        
        current_symbols = {p.get("symbol", "").upper() for p in positions}
        
        # Upsert positions
        inserted, updated = await upsert_positions(conn, positions, dry_run)
        if not dry_run:
            print(f"  ✅ Inserted: {inserted}, Updated: {updated}")
        
        # Zero stale positions
        zeroed = await zero_stale_positions(conn, current_symbols, dry_run)
        if zeroed > 0 and not dry_run:
            print(f"  ⚠️  Zeroed: {zeroed} stale positions")
        
        # Print summary
        await print_portfolio(conn)

        summary = await get_portfolio_summary(conn)
        logger.info(
            "RECONCILE_POSITIONS",
            positions_count=int(summary.get("position_count", 0) or 0),
            total_market_value=float(summary.get("total_market_value", 0) or 0),
            total_unrealized_pl=float(summary.get("total_unrealized_pl", 0) or 0),
        )
        
        return True
        
    finally:
        await conn.close()
        await client.close()


async def watch_loop(interval: int = 60) -> None:
    """Run reconciliation in a loop."""
    print(f"👀 Watching positions (reconcile every {interval}s, Ctrl+C to stop)")
    
    while True:
        try:
            await reconcile_once()
        except Exception as e:
            print(f"❌ Error in reconciliation: {e}")
        
        await asyncio.sleep(interval)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Alpaca positions with database")
    parser.add_argument("--once", action="store_true", help="Run single reconciliation and exit")
    parser.add_argument("--watch", action="store_true", help="Run continuously every 60s")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    parser.add_argument("--portfolio", action="store_true", help="Just show portfolio summary")
    args = parser.parse_args()
    
    if args.portfolio:
        # Just show portfolio from DB
        try:
            conn = await asyncpg.connect(get_dsn())
            await print_portfolio(conn)
            await conn.close()
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        return
    
    if args.watch:
        try:
            await watch_loop(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Stopped watching")
    else:
        success = await reconcile_once(args.dry_run)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
