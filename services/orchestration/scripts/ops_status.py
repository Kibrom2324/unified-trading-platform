#!/usr/bin/env python3
"""
Ops status snapshot: pending orders, positions count, exposure, last PnL summary.
"""
from __future__ import annotations

import asyncio
import os

import asyncpg


def _build_dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet",
    )


async def main() -> None:
    conn = await asyncpg.connect(_build_dsn())
    try:
        pending = await conn.fetchval("SELECT COUNT(*) FROM trades WHERE status = 'pending'")
        positions_count = await conn.fetchval("SELECT COUNT(*) FROM positions WHERE qty > 0")
        total_exposure = await conn.fetchval(
            "SELECT COALESCE(SUM(market_value), 0) FROM positions WHERE qty > 0"
        )
        per_symbol = await conn.fetch(
            """
            SELECT symbol, market_value
            FROM positions
            WHERE qty > 0
            ORDER BY market_value DESC
            LIMIT 5
            """
        )
        last_summary = await conn.fetchval(
            "SELECT MAX(updated_at) FROM daily_pnl_summary"
        )

        print("\n=== OPS STATUS ===")
        print(f"Pending orders:     {pending}")
        print(f"Open positions:     {positions_count}")
        print(f"Total exposure USD: ${float(total_exposure or 0):,.2f}")
        if per_symbol:
            print("Top exposures:")
            for row in per_symbol:
                print(f"  {row['symbol']}: ${float(row['market_value'] or 0):,.2f}")
        print(f"Last PnL summary:   {last_summary or 'n/a'}")
        print("==================\n")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
