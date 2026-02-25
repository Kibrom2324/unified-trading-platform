#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _build_dsn() -> str:
    return (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


async def _run() -> int:
    start = datetime(2024, 1, 3, tzinfo=timezone.utc).date()
    end = datetime(2024, 1, 10, tzinfo=timezone.utc).date()

    ingest_cmd = [
        PYTHON,
        "scripts/ingest_market_data.py",
        "--symbols",
        "AAPL",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
    ]
    ingest = _run_cmd(ingest_cmd)
    if ingest.returncode != 0:
        print(ingest.stdout)
        print(ingest.stderr)
        return ingest.returncode

    pool = await asyncpg.create_pool(dsn=_build_dsn(), min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM market_raw_day
                WHERE ticker = $1 AND ts BETWEEN $2 AND $3
                """,
                "AAPL",
                datetime(2024, 1, 3, tzinfo=timezone.utc),
                datetime(2024, 1, 10, 23, 59, 59, tzinfo=timezone.utc),
            )
            if not count or count == 0:
                print("No market_raw_day rows after ingest")
                return 1
    finally:
        await pool.close()

    backtest_cmd = [
        PYTHON,
        "scripts/run_backtest.py",
        "--symbols",
        "AAPL",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--initial-cash",
        "2000",
    ]
    result = _run_cmd(backtest_cmd)
    output = result.stdout + result.stderr
    if "Final equity:" not in output or "Trades:" not in output or "backtest_prices_loaded" not in output:
        print(output)
        return 1

    print("OK")
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
