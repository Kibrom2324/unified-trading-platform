#!/usr/bin/env python3
"""
Smoke test for market data ingestion.

Ingests 3 days of AAPL data and verifies rows exist.

Usage:
    python scripts/smoke_ingest.py
    python scripts/smoke_ingest.py --source yfinance
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


async def verify_rows() -> int:
    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )
    conn = await asyncpg.connect(dsn)
    count = await conn.fetchval(
        "SELECT COUNT(*) FROM market_raw_day WHERE ticker = 'AAPL'"
    )
    await conn.close()
    return count


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test: ingest market data")
    parser.add_argument(
        "--source",
        choices=["synthetic", "yfinance"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SMOKE TEST: Market Data Ingestion")
    print("=" * 60)

    # Run ingestion
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "ingest_market_data.py"),
        "--symbols", "AAPL",
        "--start", "2024-01-01",
        "--end", "2024-01-03",
        "--source", args.source,
    ]

    print(f"\n[1/2] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"❌ Ingestion failed (exit {result.returncode})")
        print(result.stderr)
        return 1

    print(result.stdout)

    # Verify rows
    print("\n[2/2] Verifying rows in market_raw_day...")
    try:
        count = asyncio.run(verify_rows())
    except Exception as exc:
        print(f"❌ DB verification failed: {exc}")
        return 1

    if count >= 1:
        print(f"✅ SMOKE_INGEST_PASS: {count} AAPL rows in market_raw_day")
        return 0
    else:
        print(f"❌ SMOKE_INGEST_FAIL: No AAPL rows found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
