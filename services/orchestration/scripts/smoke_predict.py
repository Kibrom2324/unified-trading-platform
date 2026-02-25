#!/usr/bin/env python3
"""
Smoke test for predictions.

Generates predictions for AAPL and verifies rows exist.

Usage:
    python scripts/smoke_predict.py
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
        "SELECT COUNT(*) FROM predictions_tft WHERE ticker = 'AAPL'"
    )
    await conn.close()
    return count


async def ensure_market_data() -> int:
    """Ensure we have market data."""
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
    print("=" * 60)
    print("SMOKE TEST: Predictions")
    print("=" * 60)

    # Check if we have market data
    print("\n[1/3] Checking market data exists...")
    try:
        market_count = asyncio.run(ensure_market_data())
    except Exception as exc:
        print(f"❌ DB check failed: {exc}")
        return 1

    if market_count < 1:
        print("⚠️  No market data found. Running ingest first...")
        ingest_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ingest_market_data.py"),
            "--symbols", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-10",
            "--source", "synthetic",
        ]
        result = subprocess.run(ingest_cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"❌ Ingest failed: {result.stderr}")
            return 1
        print("   ✅ Ingested synthetic data")
    else:
        print(f"   ✅ Found {market_count} market rows")

    # Run prediction (using synthetic for smoke test)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "predict.py"),
        "--symbols", "AAPL",
        "--start", "2024-01-01",
        "--end", "2024-01-10",
        "--use-synthetic",
    ]

    print(f"\n[2/3] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"❌ Prediction failed (exit {result.returncode})")
        print(result.stderr)
        return 1

    print(result.stdout)

    # Verify rows
    print("\n[3/3] Verifying rows in predictions_tft...")
    try:
        count = asyncio.run(verify_rows())
    except Exception as exc:
        print(f"❌ DB verification failed: {exc}")
        return 1

    if count >= 1:
        print(f"✅ SMOKE_PREDICT_PASS: {count} AAPL rows in predictions_tft")
        return 0
    else:
        print(f"❌ SMOKE_PREDICT_FAIL: No AAPL rows found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
