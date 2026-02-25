#!/usr/bin/env python3
"""Run the AWET end-to-end demo and verify DB outputs."""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from typing import Callable, Tuple

import asyncpg


def _run_demo() -> None:
    result = subprocess.run(["make", "demo"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "E2E demo failed.\nSTDOUT:\n"
            f"{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )


async def _fetch_counts(database_url: str) -> Tuple[int, int]:
    conn = await asyncpg.connect(dsn=database_url)
    try:
        paper_trades = await conn.fetchval("SELECT COUNT(*) FROM paper_trades;")
        audit_events = await conn.fetchval("SELECT COUNT(*) FROM audit_events;")
        return int(paper_trades or 0), int(audit_events or 0)
    finally:
        await conn.close()


def run_e2e_demo(
    demo_runner: Callable[[], None] = _run_demo,
    count_fetcher: Callable[[str], asyncio.Future] = _fetch_counts,
) -> Tuple[int, int]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for DB checks")

    demo_runner()
    return asyncio.run(count_fetcher(database_url))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AWET E2E demo and DB checks")
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip running the demo (for testing only)",
    )
    args = parser.parse_args()

    try:
        if args.skip_demo:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise RuntimeError("DATABASE_URL is required for DB checks")
            counts = asyncio.run(_fetch_counts(database_url))
        else:
            counts = run_e2e_demo()
        paper_trades, audit_events = counts
        if paper_trades > 0 and audit_events > 0:
            print(f"E2E demo OK: {paper_trades} paper trades, {audit_events} audit events.")
        else:
            print(f"E2E demo FAILED: {paper_trades} paper trades, {audit_events} audit events.")
    except Exception as exc:
        print(f"E2E demo FAILED: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
