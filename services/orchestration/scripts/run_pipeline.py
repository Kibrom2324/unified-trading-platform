#!/usr/bin/env python3
"""
Single-command pipeline orchestrator for AWET.

Runs the full pipeline: docker up → preflight → ingest → train → backtest → paper trade.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SERVICES = {"timescaledb", "kafka", "redis"}


@dataclass
class PipelineConfig:
    symbols: str = "AAPL"
    start: str = "2024-01-01"
    end: str = "2024-01-03"
    initial_cash: float = 10000.0
    train: bool = False
    paper: bool = False
    promote: bool = False
    with_reddit: bool = False
    source: str = "synthetic"
    timeout_seconds: int = 300
    verbose: bool = False


@dataclass
class StageResult:
    name: str
    passed: bool
    duration_seconds: float = 0.0
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    passed: bool = False
    stages: list[StageResult] = field(default_factory=list)
    error_stage: str | None = None
    error_message: str = ""
    last_output: str = ""


def _run(
    cmd: list[str],
    timeout: int = 300,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        cwd=cwd or REPO_ROOT,
        env=env,
    )


def _build_dsn() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'awet')}"
    )


def _parse_services_json(output: str) -> dict[str, str]:
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


async def _db_check_connection() -> bool:
    dsn = _build_dsn()
    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=10)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


def _print_verbose(output: str, verbose: bool) -> None:
    if verbose and output.strip():
        print("--- OUTPUT ---")
        print(output)
        print("--------------")


def _last_n_lines(text: str, n: int = 40) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Docker Compose Up
# ─────────────────────────────────────────────────────────────────────────────

def stage_docker_up(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    proc = _run(["docker", "compose", "up", "-d"], timeout=120, cwd=REPO_ROOT)
    duration = (datetime.now() - start).total_seconds()

    output = proc.stdout + proc.stderr

    # docker compose may return non-zero for non-critical failures (e.g., port conflicts on optional services)
    # Check if critical services failed
    critical_failed = False
    for svc in REQUIRED_SERVICES:
        if f"Container {svc}" in output and "Error" in output.split(f"Container {svc}")[0][-200:]:
            critical_failed = True
            break

    if proc.returncode != 0 and critical_failed:
        return StageResult(
            name="docker_up",
            passed=False,
            duration_seconds=duration,
            message=f"docker compose up failed for critical service: {output[:200]}",
        )

    _print_verbose(output, config.verbose)

    # Warn about non-critical failures but continue
    if proc.returncode != 0:
        return StageResult(
            name="docker_up",
            passed=True,
            duration_seconds=duration,
            message="Docker started (some optional services may have failed)",
        )

    return StageResult(
        name="docker_up",
        passed=True,
        duration_seconds=duration,
        message="Docker containers started",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Preflight Checks
# ─────────────────────────────────────────────────────────────────────────────

def stage_preflight(config: PipelineConfig) -> StageResult:
    start = datetime.now()

    # Check docker services
    proc = _run(["docker", "compose", "ps", "--format", "json"], timeout=30)
    if proc.returncode != 0:
        return StageResult(
            name="preflight",
            passed=False,
            duration_seconds=(datetime.now() - start).total_seconds(),
            message=f"docker compose ps failed: {proc.stderr}",
        )

    services = _parse_services_json(proc.stdout)
    missing = []
    for svc in REQUIRED_SERVICES:
        state = services.get(svc, "")
        if state != "running":
            missing.append(f"{svc} ({state or 'not found'})")

    if missing:
        return StageResult(
            name="preflight",
            passed=False,
            duration_seconds=(datetime.now() - start).total_seconds(),
            message=f"Services not running: {', '.join(missing)}",
        )

    # Check DB connection
    try:
        ok = asyncio.run(_db_check_connection())
        if not ok:
            return StageResult(
                name="preflight",
                passed=False,
                duration_seconds=(datetime.now() - start).total_seconds(),
                message="TimescaleDB not accepting connections",
            )
    except Exception as exc:
        return StageResult(
            name="preflight",
            passed=False,
            duration_seconds=(datetime.now() - start).total_seconds(),
            message=f"DB connection failed: {exc}",
        )

    duration = (datetime.now() - start).total_seconds()
    return StageResult(
        name="preflight",
        passed=True,
        duration_seconds=duration,
        message=f"Services OK: {', '.join(sorted(REQUIRED_SERVICES))}",
        metrics={"services": list(REQUIRED_SERVICES)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Ingest Market Data
# ─────────────────────────────────────────────────────────────────────────────

def stage_ingest(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/ingest_market_data.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--source", config.source,
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="ingest",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Ingest timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="ingest",
            passed=False,
            duration_seconds=duration,
            message=f"Ingest failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    # Extract metrics
    import re
    market_rows = 0
    pred_rows = 0
    match = re.search(r"Inserted (\d+) market rows", output)
    if match:
        market_rows = int(match.group(1))
    match = re.search(r"Inserted (\d+) prediction rows", output)
    if match:
        pred_rows = int(match.group(1))

    if market_rows == 0:
        return StageResult(
            name="ingest",
            passed=False,
            duration_seconds=duration,
            message="Ingest produced no market rows",
        )

    return StageResult(
        name="ingest",
        passed=True,
        duration_seconds=duration,
        message=f"Ingested {market_rows} market rows, {pred_rows} predictions",
        metrics={"market_rows": market_rows, "prediction_rows": pred_rows},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Reddit Ingest (optional)
# ─────────────────────────────────────────────────────────────────────────────

def stage_reddit_ingest(config: PipelineConfig) -> StageResult:
    """Ingest Reddit data via PRAW API."""
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/ingest_reddit_july_2025.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--write-db",
        "--update-features",
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="reddit_ingest",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Reddit ingest timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        # Reddit ingest failure is non-fatal - continue pipeline
        return StageResult(
            name="reddit_ingest",
            passed=True,
            duration_seconds=duration,
            message=f"Reddit ingest failed (non-fatal): {_last_n_lines(output, 5)}",
        )

    return StageResult(
        name="reddit_ingest",
        passed=True,
        duration_seconds=duration,
        message="Reddit data ingested",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def stage_feature_engineer(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/feature_engineer.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--source", "day",
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="features",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Feature engineering timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="features",
            passed=False,
            duration_seconds=duration,
            message=f"Feature engineering failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    return StageResult(
        name="features",
        passed=True,
        duration_seconds=duration,
        message="Features built",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Train Model (optional)
# ─────────────────────────────────────────────────────────────────────────────

def stage_train(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/train_tft_baseline.py",
        "--max-epochs", "10",
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="train",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Training timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="train",
            passed=False,
            duration_seconds=duration,
            message=f"Training failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    # Extract metrics
    import re
    val_loss = None
    match = re.search(r"val_loss[=:\s]+([0-9.e-]+)", output)
    if match:
        val_loss = float(match.group(1))

    return StageResult(
        name="train",
        passed=True,
        duration_seconds=duration,
        message=f"Training completed (val_loss={val_loss})",
        metrics={"val_loss": val_loss},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Predict
# ─────────────────────────────────────────────────────────────────────────────

def stage_predict(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/predict.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]
    if not config.train:
        cmd.append("--use-synthetic")

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="predict",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Prediction timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="predict",
            passed=False,
            duration_seconds=duration,
            message=f"Prediction failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    return StageResult(
        name="predict",
        passed=True,
        duration_seconds=duration,
        message="Predictions generated",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Trade Decisions
# ─────────────────────────────────────────────────────────────────────────────

def stage_decisions(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/decide_trades.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="decisions",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Trade decisions timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="decisions",
            passed=False,
            duration_seconds=duration,
            message=f"Trade decisions failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    return StageResult(
        name="decisions",
        passed=True,
        duration_seconds=duration,
        message="Trade decisions generated",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Risk Checks
# ─────────────────────────────────────────────────────────────────────────────

def stage_risk(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/run_risk.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="risk",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Risk checks timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="risk",
            passed=False,
            duration_seconds=duration,
            message=f"Risk checks failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    return StageResult(
        name="risk",
        passed=True,
        duration_seconds=duration,
        message="Risk decisions generated",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Execution (paper)
# ─────────────────────────────────────────────────────────────────────────────

def stage_execute(config: PipelineConfig) -> StageResult:
    start = datetime.now()
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
        proc = _run(cmd, timeout=config.timeout_seconds, env=env)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="execute",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Execution timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="execute",
            passed=False,
            duration_seconds=duration,
            message=f"Execution failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    return StageResult(
        name="execute",
        passed=True,
        duration_seconds=duration,
        message="Paper execution completed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Backtest
# ─────────────────────────────────────────────────────────────────────────────

def stage_backtest(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/run_backtest.py",
        "--symbols", config.symbols,
        "--start", config.start,
        "--end", config.end,
        "--initial-cash", str(int(config.initial_cash)),
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="backtest",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Backtest timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="backtest",
            passed=False,
            duration_seconds=duration,
            message=f"Backtest failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    # Extract metrics
    import re
    trades = 0
    final_equity = 0.0
    match = re.search(r"Trades:\s*(\d+)", output)
    if match:
        trades = int(match.group(1))
    match = re.search(r"Final equity:\s*([0-9.]+)", output)
    if match:
        final_equity = float(match.group(1))

    if trades < 1:
        return StageResult(
            name="backtest",
            passed=False,
            duration_seconds=duration,
            message="Backtest produced no trades",
        )

    return StageResult(
        name="backtest",
        passed=True,
        duration_seconds=duration,
        message=f"Backtest: {trades} trades, final equity ${final_equity:.2f}",
        metrics={"trades": trades, "final_equity": final_equity},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Verify DB
# ─────────────────────────────────────────────────────────────────────────────

async def _verify_backtest_run(config: PipelineConfig) -> str | None:
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


def stage_db_verify(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    try:
        run_id = asyncio.run(_verify_backtest_run(config))
        duration = (datetime.now() - start).total_seconds()
        if not run_id:
            return StageResult(
                name="db_verify",
                passed=False,
                duration_seconds=duration,
                message=f"No backtest_runs row found for {config.symbols}",
            )
        return StageResult(
            name="db_verify",
            passed=True,
            duration_seconds=duration,
            message=f"Backtest run recorded: {run_id}",
            metrics={"run_id": run_id},
        )
    except Exception as exc:
        return StageResult(
            name="db_verify",
            passed=False,
            duration_seconds=(datetime.now() - start).total_seconds(),
            message=f"DB verify failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Paper Trade Demo (optional)
# ─────────────────────────────────────────────────────────────────────────────

def stage_paper_trade(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/paper_trade_smoke.py",
        "--symbols", config.symbols,
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="paper_trade",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Paper trade timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    # Paper trade demo may fail if Alpaca not configured - treat as warning
    if proc.returncode != 0:
        return StageResult(
            name="paper_trade",
            passed=True,  # Non-blocking stage
            duration_seconds=duration,
            message=f"Paper trade demo failed (optional): {_last_n_lines(output, 5)}",
        )

    return StageResult(
        name="paper_trade",
        passed=True,
        duration_seconds=duration,
        message="Paper trade demo completed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Promote Model (optional)
# ─────────────────────────────────────────────────────────────────────────────

def stage_promote(config: PipelineConfig) -> StageResult:
    start = datetime.now()
    cmd = [
        sys.executable,
        "scripts/promote_model.py",
        "--dry-run",
        "--json",
    ]

    try:
        proc = _run(cmd, timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        return StageResult(
            name="promote",
            passed=False,
            duration_seconds=config.timeout_seconds,
            message=f"Promotion timed out after {config.timeout_seconds}s",
        )

    output = proc.stdout + proc.stderr
    _print_verbose(output, config.verbose)
    duration = (datetime.now() - start).total_seconds()

    if proc.returncode != 0:
        return StageResult(
            name="promote",
            passed=False,
            duration_seconds=duration,
            message=f"Promotion failed (exit {proc.returncode}): {_last_n_lines(output, 10)}",
        )

    return StageResult(
        name="promote",
        passed=True,
        duration_seconds=duration,
        message="Promotion gate evaluated",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(result: PipelineResult, config: PipelineConfig) -> None:
    print()
    print("=" * 70)
    if result.passed:
        print("  ✅ PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print(f"  ❌ PIPELINE FAILED at stage: {result.error_stage}")
    print("=" * 70)
    print()

    print(f"{'Stage':<15} {'Status':<8} {'Duration':<10} Message")
    print("-" * 70)
    for stage in result.stages:
        status = "✓ PASS" if stage.passed else "✗ FAIL"
        print(f"{stage.name:<15} {status:<8} {stage.duration_seconds:>6.1f}s    {stage.message[:40]}")

    if result.error_message:
        print()
        print(f"Error: {result.error_message}")
        if result.last_output:
            print()
            print("--- Last output ---")
            print(_last_n_lines(result.last_output, 20))

    # Print metrics summary
    print()
    print("Metrics Summary:")
    for stage in result.stages:
        if stage.metrics:
            for k, v in stage.metrics.items():
                print(f"  {stage.name}.{k}: {v}")
    print()


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Single-command AWET pipeline orchestrator"
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
        "--initial-cash", type=float, default=10000.0,
        help="Initial cash for backtest (default: 10000)"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Run model training stage"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Run paper trading demo stage"
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="Run model promotion gate (dry-run)"
    )
    parser.add_argument(
        "--with-reddit", action="store_true",
        help="Run Reddit ingest stage before feature engineering"
    )
    parser.add_argument(
        "--source", choices=["synthetic", "yfinance", "alpaca", "polygon"],
        default="synthetic",
        help="Data source for ingest (default: synthetic)"
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=300,
        help="Timeout per stage in seconds (default: 300)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full subprocess output"
    )
    args = parser.parse_args()

    return PipelineConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        train=args.train,
        paper=args.paper,
        promote=args.promote,
        with_reddit=args.with_reddit,
        source=args.source,
        timeout_seconds=args.timeout_seconds,
        verbose=args.verbose,
    )


def main() -> int:
    config = parse_args()
    result = PipelineResult()

    stages = [
        ("docker_up", stage_docker_up),
        ("preflight", stage_preflight),
        ("ingest", stage_ingest),
    ]

    if config.with_reddit:
        stages.append(("reddit_ingest", stage_reddit_ingest))

    stages.append(("features", stage_feature_engineer))

    if config.train:
        stages.append(("train", stage_train))

    stages.append(("predict", stage_predict))
    stages.append(("decisions", stage_decisions))
    stages.append(("risk", stage_risk))

    if config.paper:
        stages.append(("execute", stage_execute))

    stages.append(("backtest", stage_backtest))
    stages.append(("db_verify", stage_db_verify))

    if config.promote:
        stages.append(("promote", stage_promote))

    for name, stage_fn in stages:
        print(f"[{len(result.stages) + 1}/{len(stages)}] Running {name}...")
        stage_result = stage_fn(config)
        result.stages.append(stage_result)

        if not stage_result.passed:
            result.error_stage = name
            result.error_message = stage_result.message
            _print_summary(result, config)
            return 1

    result.passed = True
    _print_summary(result, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
