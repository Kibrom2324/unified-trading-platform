#!/usr/bin/env python3
"""
Production-grade E2E test runner for AWET trading pipeline.

Validates the full pipeline: preflight → ingest → backtest → DB verify.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import asyncpg

REQUIRED_SERVICES = {"timescaledb", "kafka", "redis"}


@dataclass
class E2EConfig:
    symbols: str = "AAPL"
    start: str = "2024-01-01"
    end: str = "2024-01-03"
    initial_cash: float = 2000.0
    timeout_seconds: int = 120
    no_ingest: bool = False
    no_db_verify: bool = False
    verbose: bool = False


@dataclass
class E2EResult:
    passed: bool = False
    stage_failed: str | None = None
    services_ok: list[str] = field(default_factory=list)
    ingested_rows: int = 0
    prediction_rows: int = 0
    feature_rows: int = 0
    decision_rows: int = 0
    risk_rows: int = 0
    execution_rows: int = 0
    prices_loaded: int = 0
    trades: int = 0
    final_equity: float = 0.0
    run_id: str | None = None
    last_output: str = ""
    error_message: str = ""


def _run(
    cmd: list[str],
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def _parse_services_json(output: str) -> dict[str, str]:
    """Parse docker compose ps --format json output."""
    services: dict[str, str] = {}
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            service = data.get("Service", "")
            state = data.get("State", "unknown")
            if service:
                services[service] = state
        except json.JSONDecodeError:
            continue
    return services


def _build_dsn() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'awet')}"
    )


def _extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _last_n_lines(text: str, n: int = 40) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _print_verbose(output: str, verbose: bool) -> None:
    if verbose and output.strip():
        print("--- OUTPUT ---")
        print(output)
        print("--------------")


def _preflight_services(config: E2EConfig, result: E2EResult) -> bool:
    """Check required docker services are running."""
    proc = _run(["docker", "compose", "ps", "--format", "json"], config.timeout_seconds)
    if proc.returncode != 0:
        result.stage_failed = "preflight"
        result.error_message = f"docker compose ps failed: {proc.stderr}"
        result.last_output = proc.stderr
        return False

    services = _parse_services_json(proc.stdout)
    missing = []
    for svc in REQUIRED_SERVICES:
        state = services.get(svc, "")
        if state != "running":
            missing.append(f"{svc} ({state or 'not found'})")

    if missing:
        result.stage_failed = "preflight"
        result.error_message = f"Services not running: {', '.join(missing)}"
        return False

    # Check TimescaleDB connectivity
    try:
        conn_check = asyncio.run(_db_check_connection())
        if not conn_check:
            result.stage_failed = "preflight"
            result.error_message = "TimescaleDB not accepting connections"
            return False
    except Exception as exc:
        result.stage_failed = "preflight"
        result.error_message = f"TimescaleDB connection failed: {exc}"
        return False

    result.services_ok = sorted(REQUIRED_SERVICES)
    return True


async def _db_check_connection() -> bool:
    """Simple SELECT 1 to verify DB is accepting connections."""
    dsn = _build_dsn()
    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=10)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


def _ingest_stage(config: E2EConfig, result: E2EResult) -> bool:
    """Run market data ingest."""
    cmd = [
        sys.executable,
        "scripts/ingest_market_data.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "ingest"
        result.error_message = f"Ingest timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "ingest"
        result.error_message = f"Ingest failed (exit {proc.returncode})"
        return False

    # Parse rows - try JSON logs first, then plain text
    rows = _extract_int(r'"market_rows":\s*(\d+)', output)
    if rows is None:
        rows = _extract_int(r"Inserted (\d+) market rows", output)
    if rows is None or rows <= 0:
        result.stage_failed = "ingest"
        result.error_message = "Ingest produced no market rows"
        return False

    preds = _extract_int(r'"prediction_rows":\s*(\d+)', output)
    if preds is None:
        preds = _extract_int(r"Inserted (\d+) prediction rows", output) or 0

    result.ingested_rows = rows
    result.prediction_rows = preds
    return True


def _feature_stage(config: E2EConfig, result: E2EResult) -> bool:
    cmd = [
        sys.executable,
        "scripts/feature_engineer.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--source", "day",
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "features"
        result.error_message = f"Feature engineering timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "features"
        result.error_message = f"Feature engineering failed (exit {proc.returncode})"
        return False

    rows = _extract_int(r"Inserted (\d+) feature rows", output)
    if rows is None:
        rows = _extract_int(r'"rows":\s*(\d+)', output) or 0
    result.feature_rows = rows
    return True


def _predict_stage(config: E2EConfig, result: E2EResult) -> bool:
    cmd = [
        sys.executable,
        "scripts/predict.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--use-synthetic",
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "predict"
        result.error_message = f"Prediction timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "predict"
        result.error_message = f"Prediction failed (exit {proc.returncode})"
        return False

    preds = _extract_int(r"Inserted (\d+) prediction rows", output)
    if preds is None:
        preds = _extract_int(r'"rows":\s*(\d+)', output) or 0
    result.prediction_rows = preds
    return True


def _decision_stage(config: E2EConfig, result: E2EResult) -> bool:
    cmd = [
        sys.executable,
        "scripts/decide_trades.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "decisions"
        result.error_message = f"Trade decisions timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "decisions"
        result.error_message = f"Trade decisions failed (exit {proc.returncode})"
        return False

    rows = _extract_int(r"Inserted (\d+) trade decisions", output) or 0
    result.decision_rows = rows
    return True


def _risk_stage(config: E2EConfig, result: E2EResult) -> bool:
    cmd = [
        sys.executable,
        "scripts/run_risk.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "risk"
        result.error_message = f"Risk checks timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "risk"
        result.error_message = f"Risk checks failed (exit {proc.returncode})"
        return False

    rows = _extract_int(r"Inserted (\d+) risk decisions", output) or 0
    result.risk_rows = rows
    return True


def _execute_stage(config: E2EConfig, result: E2EResult) -> bool:
    cmd = [
        sys.executable,
        "scripts/execute_paper.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]
    env = dict(os.environ)
    env["TRADING_ENABLED"] = "true"
    try:
        proc = _run(cmd, config.timeout_seconds, env=env)
    except subprocess.TimeoutExpired:
        result.stage_failed = "execution"
        result.error_message = f"Execution timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "execution"
        result.error_message = f"Execution failed (exit {proc.returncode})"
        return False

    rows = _extract_int(r"Executed (\d+) paper trades", output) or 0
    result.execution_rows = rows
    return True


def _promote_stage(config: E2EConfig, result: E2EResult) -> bool:
    cmd = [
        sys.executable,
        "scripts/promote_model.py",
        "--dry-run",
        "--json",
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "promote"
        result.error_message = f"Promotion timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "promote"
        result.error_message = f"Promotion failed (exit {proc.returncode})"
        return False
    return True


def _backtest_stage(config: E2EConfig, result: E2EResult) -> bool:
    """Run backtest."""
    cmd = [
        sys.executable,
        "scripts/run_backtest.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--initial-cash", str(int(config.initial_cash)),
    ]
    try:
        proc = _run(cmd, config.timeout_seconds)
    except subprocess.TimeoutExpired:
        result.stage_failed = "backtest"
        result.error_message = f"Backtest timed out after {config.timeout_seconds}s"
        return False

    output = proc.stdout + proc.stderr
    result.last_output = output
    _print_verbose(output, config.verbose)

    if proc.returncode != 0:
        result.stage_failed = "backtest"
        result.error_message = f"Backtest failed (exit {proc.returncode})"
        return False

    # Extract metrics - try multiple patterns
    prices_loaded = _extract_int(r"backtest_prices_loaded.*rows=(\d+)", output)
    if prices_loaded is None:
        prices_loaded = _extract_int(r'"prices_loaded":\s*(\d+)', output)
    if prices_loaded is None or prices_loaded <= 0:
        result.stage_failed = "backtest"
        result.error_message = "No prices loaded. Check market_raw_day/minute tables have data."
        return False

    trades = _extract_int(r"Trades:\s*(\d+)", output)
    if trades is None:
        trades = _extract_int(r'"num_trades":\s*(\d+)', output)
    if trades is None:
        result.stage_failed = "backtest"
        result.error_message = "Could not parse trade count from output."
        return False
    if trades < 1:
        result.stage_failed = "backtest"
        result.error_message = "No trades executed. Check predictions_tft table has signals."
        return False

    final_equity = _extract_float(r"Final equity:\s*([0-9.]+)", output)
    if final_equity is None:
        final_equity = _extract_float(r'"final_equity":\s*([0-9.]+)', output)
    if final_equity is None:
        result.stage_failed = "backtest"
        result.error_message = "Could not parse final equity from output."
        return False

    result.prices_loaded = prices_loaded
    result.trades = trades
    result.final_equity = final_equity
    return True


async def _db_verify_run(config: E2EConfig) -> str | None:
    """Verify backtest run exists in DB."""
    dsn = _build_dsn()
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, timeout=10)
    try:
        async with pool.acquire() as conn:
            symbols_list = [s.strip() for s in config.symbols.split(",")]
            row = await conn.fetchrow(
                """
                SELECT run_id
                FROM backtest_runs
                WHERE params->'symbols' @> $1::jsonb
                  AND params->>'start' = $2
                  AND params->>'end' = $3
                ORDER BY created_at DESC
                LIMIT 1
                """,
                json.dumps(symbols_list),
                config.start,
                config.end,
            )
            return str(row["run_id"]) if row else None
    finally:
        await pool.close()


async def _db_table_counts(symbols: list[str], start: str, end: str) -> dict[str, int]:
    dsn = _build_dsn()
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, timeout=10)
    try:
        async with pool.acquire() as conn:
            counts = {}
            counts["market_raw_day"] = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM market_raw_day
                    WHERE ticker = ANY($1) AND ts >= $2 AND ts <= $3
                    """,
                    symbols,
                    start,
                    end,
                )
            )
            counts["features_tft"] = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM features_tft
                    WHERE ticker = ANY($1) AND ts >= $2 AND ts <= $3
                    """,
                    symbols,
                    start,
                    end,
                )
            )
            counts["predictions_tft"] = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM predictions_tft
                    WHERE ticker = ANY($1) AND ts >= $2 AND ts <= $3
                    """,
                    symbols,
                    start,
                    end,
                )
            )
            counts["trade_decisions"] = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM trade_decisions
                    WHERE ticker = ANY($1) AND ts >= $2 AND ts <= $3
                    """,
                    symbols,
                    start,
                    end,
                )
            )
            counts["risk_decisions"] = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM risk_decisions
                    WHERE ticker = ANY($1) AND ts >= $2 AND ts <= $3
                    """,
                    symbols,
                    start,
                    end,
                )
            )
            counts["trades"] = int(
                await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM trades
                    WHERE symbol = ANY($1) AND ts >= $2 AND ts <= $3
                    """,
                    symbols,
                    start,
                    end,
                )
            )
            return counts
    finally:
        await pool.close()


def _db_verify_stage(config: E2EConfig, result: E2EResult) -> bool:
    """Verify backtest run recorded in database."""
    try:
        run_id = asyncio.run(_db_verify_run(config))
        if not run_id:
            result.stage_failed = "db_verify"
            result.error_message = f"No backtest_runs row found for {config.symbols} / {config.start}-{config.end}"
            return False
        result.run_id = run_id
        counts = asyncio.run(
            _db_table_counts([s.strip() for s in config.symbols.split(",")], config.start, config.end)
        )
        for table, count in counts.items():
            if count <= 0:
                result.stage_failed = "db_verify"
                result.error_message = f"No rows found in {table}"
                return False
        return True
    except Exception as exc:
        result.stage_failed = "db_verify"
        result.error_message = f"DB verify failed: {exc}"
        return False


def _print_summary(result: E2EResult) -> None:
    """Print structured summary block."""
    print()
    print("=" * 60)
    if result.passed:
        print("  ✅ E2E TEST PASSED")
    else:
        print(f"  ❌ E2E TEST FAILED at stage: {result.stage_failed}")
    print("=" * 60)
    print()

    if result.services_ok:
        print(f"Services OK:     {', '.join(result.services_ok)}")

    if result.ingested_rows > 0:
        print(f"Ingested rows:   {result.ingested_rows} (predictions: {result.prediction_rows})")

    if result.feature_rows > 0:
        print(f"Feature rows:    {result.feature_rows}")

    if result.decision_rows > 0:
        print(f"Decisions:       {result.decision_rows}")

    if result.risk_rows > 0:
        print(f"Risk decisions:  {result.risk_rows}")

    if result.execution_rows > 0:
        print(f"Executions:      {result.execution_rows}")

    if result.prices_loaded > 0:
        print(f"Prices loaded:   {result.prices_loaded}")
        print(f"Trades:          {result.trades}")
        print(f"Final equity:    {result.final_equity:.2f}")

    if result.run_id:
        print(f"Run ID:          {result.run_id}")

    if result.error_message:
        print()
        print(f"Error: {result.error_message}")
        if result.last_output:
            print()
            print("--- Last 40 lines of output ---")
            print(_last_n_lines(result.last_output, 40))
            print("-------------------------------")

    print()


def parse_args() -> E2EConfig:
    parser = argparse.ArgumentParser(
        description="Production-grade E2E test for AWET trading pipeline"
    )
    parser.add_argument(
        "--symbols", default="AAPL",
        help="Comma-separated symbols (default: AAPL)"
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01)"
    )
    parser.add_argument(
        "--end", default="2024-01-03",
        help="End date YYYY-MM-DD (default: 2024-01-03)"
    )
    parser.add_argument(
        "--initial-cash", type=float, default=2000.0,
        help="Initial cash for backtest (default: 2000)"
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=120,
        help="Timeout per stage in seconds (default: 120)"
    )
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Skip ingest stage (assume data exists)"
    )
    parser.add_argument(
        "--no-db-verify", action="store_true",
        help="Skip DB verification stage"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full subprocess output"
    )
    args = parser.parse_args()

    return E2EConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        timeout_seconds=args.timeout_seconds,
        no_ingest=args.no_ingest,
        no_db_verify=args.no_db_verify,
        verbose=args.verbose,
    )


def main() -> int:
    config = parse_args()
    result = E2EResult()

    # Stage 1: Preflight
    print("[1/10] Preflight: checking services...")
    if not _preflight_services(config, result):
        _print_summary(result)
        return 1

    # Stage 2: Ingest
    if config.no_ingest:
        print("[2/10] Ingest: SKIPPED (--no-ingest)")
    else:
        print(f"[2/10] Ingest: {config.symbols} from {config.start} to {config.end}...")
        if not _ingest_stage(config, result):
            _print_summary(result)
            return 1

    # Stage 3: Feature Engineering
    print(f"[3/10] Features: {config.symbols} from {config.start} to {config.end}...")
    if not _feature_stage(config, result):
        _print_summary(result)
        return 1

    # Stage 4: Prediction
    print(f"[4/10] Predict: {config.symbols} from {config.start} to {config.end}...")
    if not _predict_stage(config, result):
        _print_summary(result)
        return 1

    # Stage 5: Trade Decisions
    print(f"[5/10] Decisions: {config.symbols} from {config.start} to {config.end}...")
    if not _decision_stage(config, result):
        _print_summary(result)
        return 1

    # Stage 6: Risk Checks
    print(f"[6/10] Risk: {config.symbols} from {config.start} to {config.end}...")
    if not _risk_stage(config, result):
        _print_summary(result)
        return 1

    # Stage 7: Execution (paper)
    print(f"[7/10] Execute: {config.symbols} from {config.start} to {config.end}...")
    if not _execute_stage(config, result):
        _print_summary(result)
        return 1

    # Stage 8: Backtest
    print(f"[8/10] Backtest: initial_cash={config.initial_cash}...")
    if not _backtest_stage(config, result):
        _print_summary(result)
        return 1

    # Stage 9: DB Verify
    if config.no_db_verify:
        print("[9/10] DB Verify: SKIPPED (--no-db-verify)")
    else:
        print("[9/10] DB Verify: checking tables...")
        if not _db_verify_stage(config, result):
            _print_summary(result)
            return 1

    # Stage 10: Promotion (dry-run)
    print("[10/10] Promotion: dry-run...")
    if not _promote_stage(config, result):
        _print_summary(result)
        return 1

    result.passed = True
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
