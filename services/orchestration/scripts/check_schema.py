#!/usr/bin/env python3
"""Check trades schema for bracket order columns."""
from __future__ import annotations

import asyncio
import os

import asyncpg


def _build_dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet",
    )


async def main() -> int:
    expected = {
        "order_class",
        "tp_order_id",
        "sl_order_id",
        "exit_fill_price",
        "exit_filled_at",
        "alpaca_raw",
    }
    conn = await asyncpg.connect(_build_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'trades'
            """
        )
        cols = {r["column_name"] for r in rows}
        missing = expected - cols
        if missing:
            print(f"❌ Missing columns: {', '.join(sorted(missing))}")
            return 1
        print("✅ trades schema has bracket columns")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
