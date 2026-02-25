#!/usr/bin/env python3
"""Database inventory: list all tables, row counts, and hypertables."""
from __future__ import annotations

import os
import sys

import psycopg2


def main() -> int:
    user = os.getenv("POSTGRES_USER", "awet")
    password = os.getenv("POSTGRES_PASSWORD", "awet")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "awet")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
    except Exception as e:
        print(f"❌ Connection failed: {e}", file=sys.stderr)
        return 1

    # Get all tables in public schema
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]

    print("=" * 70)
    print("AWET DATABASE INVENTORY")
    print("=" * 70)
    print(f"  Host:     {host}:{port}")
    print(f"  Database: {db}")
    print(f"  Schema:   public")
    print()
    print(f"  Total tables: {len(tables)}")
    print()

    if not tables:
        print("  (no tables found)")
    else:
        print(f"{'#':<4} {'Table':<40} {'Rows':>12} {'Latest Timestamp':<26}")
        print("-" * 70)

        for i, table in enumerate(tables, 1):
            # Row count
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                row_count = cur.fetchone()[0]
            except Exception:
                row_count = "?"

            # Latest timestamp (try ts, created_at, updated_at)
            latest_ts = None
            for ts_col in ("ts", "created_at", "updated_at"):
                try:
                    cur.execute(f'SELECT MAX("{ts_col}") FROM "{table}"')
                    val = cur.fetchone()[0]
                    if val is not None:
                        latest_ts = str(val)[:26]
                        break
                except Exception:
                    continue

            if latest_ts is None:
                latest_ts = "-"

            row_str = f"{row_count:,}" if isinstance(row_count, int) else row_count
            print(f"{i:<4} {table:<40} {row_str:>12} {latest_ts:<26}")

    print()

    # Hypertables (TimescaleDB)
    print("=" * 70)
    print("TIMESCALEDB HYPERTABLES")
    print("=" * 70)

    try:
        cur.execute("""
            SELECT hypertable_name, num_chunks
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'public'
            ORDER BY hypertable_name
        """)
        hypertables = cur.fetchall()

        if not hypertables:
            print("  (no hypertables found)")
        else:
            print(f"{'#':<4} {'Hypertable':<40} {'Chunks':>10}")
            print("-" * 70)
            for i, (name, chunks) in enumerate(hypertables, 1):
                print(f"{i:<4} {name:<40} {chunks:>10}")
    except Exception:
        print("  (TimescaleDB extension not available or no hypertables)")

    print()
    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
