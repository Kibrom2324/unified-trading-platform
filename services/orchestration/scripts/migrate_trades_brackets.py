#!/usr/bin/env python3
"""Idempotent migration for bracket order fields on trades table."""
from __future__ import annotations

import asyncio
import os

import asyncpg


def _build_dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet",
    )


async def _column_exists(conn: asyncpg.Connection, column: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'trades' AND column_name = $1
            )
            """,
            column,
        )
    )


async def _index_exists(conn: asyncpg.Connection, index_name: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'public' AND indexname = $1
            )
            """,
            index_name,
        )
    )


async def _find_status_check(conn: asyncpg.Connection) -> str | None:
    row = await conn.fetchrow(
        """
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'trades'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) ILIKE '%status%'
        LIMIT 1
        """
    )
    return row["conname"] if row else None


async def migrate() -> int:
    conn = await asyncpg.connect(_build_dsn())
    changed = []
    try:
        columns = [
            ("order_class", "TEXT"),
            ("tp_order_id", "TEXT"),
            ("sl_order_id", "TEXT"),
            ("exit_fill_price", "NUMERIC(18,6)"),
            ("exit_filled_at", "TIMESTAMPTZ"),
            ("alpaca_raw", "JSONB"),
        ]

        for name, col_type in columns:
            if not await _column_exists(conn, name):
                await conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {col_type}")
                changed.append(f"added column {name}")

        # Update status constraint to include 'closed'
        constraint_name = await _find_status_check(conn)
        if constraint_name:
            await conn.execute(f"ALTER TABLE trades DROP CONSTRAINT {constraint_name}")
            changed.append(f"dropped constraint {constraint_name}")
        await conn.execute(
            """
            ALTER TABLE trades
            ADD CONSTRAINT trades_status_check
            CHECK (status IN ('filled','blocked','rejected','pending','closed'))
            """
        )
        changed.append("added constraint trades_status_check")

        # Indexes (idempotent)
        if not await _index_exists(conn, "trades_alpaca_order_idx"):
            await conn.execute("CREATE INDEX trades_alpaca_order_idx ON trades (alpaca_order_id)")
            changed.append("added index trades_alpaca_order_idx")
        if not await _index_exists(conn, "trades_status_idx"):
            await conn.execute("CREATE INDEX trades_status_idx ON trades (status)")
            changed.append("added index trades_status_idx")
        if not await _index_exists(conn, "trades_correlation_idx"):
            await conn.execute("CREATE INDEX trades_correlation_idx ON trades (correlation_id)")
            changed.append("added index trades_correlation_idx")

        print("✅ Migration complete")
        if changed:
            print("Changes:")
            for item in changed:
                print(f"  - {item}")
        else:
            print("No changes needed")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(migrate()))
