#!/usr/bin/env python3
"""
Pipeline Scheduler for AWET.

Schedules pipeline runs:
- Nightly: ingest + train + backtest + promote
- Market hours: infer + decide + risk-check + paper trade

Uses APScheduler for in-process scheduling.
Can also be run as a one-shot for cron/systemd.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import structlog

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = structlog.get_logger("scheduler")

# Check if APScheduler is available
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    logger.warning("apscheduler_not_installed", message="pip install apscheduler for daemon mode")


def _run_script(script: str, args: list[str] | None = None) -> tuple[int, str]:
    """Run a Python script and return (exit_code, output)."""
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)

    logger.info("running_script", script=script, args=args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
            cwd=REPO_ROOT,
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        logger.error("script_timeout", script=script)
        return 1, "Timeout"
    except Exception as exc:
        logger.error("script_error", script=script, error=str(exc))
        return 1, str(exc)


def job_nightly_pipeline() -> None:
    """Nightly job: ingest + train + backtest + promote."""
    logger.info("job_started", job="nightly_pipeline")
    start = datetime.now(tz=timezone.utc)

    symbols = os.getenv("AWET_SYMBOLS", "AAPL,MSFT,GOOGL")
    source = os.getenv("AWET_DATA_SOURCE", "yfinance")

    # Run pipeline with training
    code, output = _run_script(
        "scripts/run_pipeline.py",
        [
            "--symbols", symbols,
            "--train",
            "--source", source,
            "--start", "2023-01-01",
            "--end", "2024-12-31",
        ],
    )

    if code != 0:
        logger.error("nightly_pipeline_failed", exit_code=code, output=output[-500:])
        return

    # Run promotion check
    code, output = _run_script("scripts/promote_model.py", ["--json"])
    if code != 0:
        logger.warning("model_promotion_rejected", output=output[-500:])
    else:
        logger.info("model_promoted")

    duration = (datetime.now(tz=timezone.utc) - start).total_seconds()
    logger.info("job_completed", job="nightly_pipeline", duration_seconds=duration)


def job_morning_health_check() -> None:
    """Morning job: health check + smoke test."""
    logger.info("job_started", job="morning_health_check")

    # Run E2E test as smoke test
    code, output = _run_script("scripts/e2e_test.py", ["--no-ingest"])
    if code != 0:
        logger.error("morning_health_check_failed", output=output[-500:])
        # Could trigger alert here
    else:
        logger.info("morning_health_check_passed")


def job_market_hours_infer() -> None:
    """Market hours job: run paper trade cycle."""
    logger.info("job_started", job="market_hours_infer")

    # Check if trading is enabled
    if os.getenv("TRADING_ENABLED", "false").lower() != "true":
        logger.info("trading_disabled_skipping")
        return

    symbols = os.getenv("AWET_SYMBOLS", "AAPL")
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    for script in (
        "scripts/predict.py",
        "scripts/decide_trades.py",
        "scripts/run_risk.py",
        "scripts/execute_paper.py",
    ):
        args = ["--symbols", symbols, "--start", today, "--end", today]
        if script.endswith("predict.py"):
            args.append("--use-synthetic")
        if script.endswith("execute_paper.py"):
            os.environ["TRADING_ENABLED"] = "true"
        code, output = _run_script(script, args)
        if code != 0:
            logger.warning("paper_trade_cycle_failed", script=script, output=output[-300:])
            return

    logger.info("paper_trade_cycle_completed")


def job_daily_ingest() -> None:
    """Daily ingest job: fetch latest market data."""
    logger.info("job_started", job="daily_ingest")

    symbols = os.getenv("AWET_SYMBOLS", "AAPL,MSFT,GOOGL")
    source = os.getenv("AWET_DATA_SOURCE", "yfinance")

    # Get yesterday's date for daily data
    from datetime import timedelta
    yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    code, output = _run_script(
        "scripts/ingest_market_data.py",
        [
            "--symbols", symbols,
            "--start", yesterday,
            "--end", today,
            "--source", source,
        ],
    )

    if code != 0:
        logger.error("daily_ingest_failed", output=output[-500:])
    else:
        logger.info("daily_ingest_completed")


async def run_daemon(args: argparse.Namespace) -> None:
    """Run scheduler as a daemon with APScheduler."""
    if not HAS_APSCHEDULER:
        print("ERROR: APScheduler not installed. Run: pip install apscheduler", file=sys.stderr)
        sys.exit(1)

    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    # Nightly at 11 PM ET: ingest + train + backtest + promote
    scheduler.add_job(
        job_nightly_pipeline,
        CronTrigger(hour=23, minute=0),
        id="nightly_pipeline",
        name="Nightly Pipeline (ingest+train+backtest+promote)",
    )

    # Morning at 8 AM ET: health check
    scheduler.add_job(
        job_morning_health_check,
        CronTrigger(hour=8, minute=0),
        id="morning_health_check",
        name="Morning Health Check",
    )

    # Daily at 6 PM ET: ingest latest data
    scheduler.add_job(
        job_daily_ingest,
        CronTrigger(hour=18, minute=0),
        id="daily_ingest",
        name="Daily Market Data Ingest",
    )

    # Market hours (9:30 AM - 4 PM ET): every 15 minutes
    if args.paper:
        scheduler.add_job(
            job_market_hours_infer,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*/15",
            ),
            id="market_hours_infer",
            name="Market Hours Inference",
        )
        logger.info("market_hours_job_enabled")

    scheduler.start()
    logger.info(
        "scheduler_started",
        jobs=[j.id for j in scheduler.get_jobs()],
    )

    # Handle shutdown
    stop_event = asyncio.Event()

    def shutdown_handler(signum, frame):
        logger.info("shutdown_signal_received")
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    await stop_event.wait()
    scheduler.shutdown()
    logger.info("scheduler_stopped")


def run_once(args: argparse.Namespace) -> int:
    """Run a single job and exit."""
    job_map = {
        "nightly": job_nightly_pipeline,
        "morning": job_morning_health_check,
        "infer": job_market_hours_infer,
        "ingest": job_daily_ingest,
    }

    if args.job not in job_map:
        print(f"Unknown job: {args.job}. Available: {list(job_map.keys())}", file=sys.stderr)
        return 1

    job_map[args.job]()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWET Pipeline Scheduler")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Daemon mode
    daemon = subparsers.add_parser("daemon", help="Run as daemon with APScheduler")
    daemon.add_argument(
        "--paper", action="store_true",
        help="Enable market hours paper trading job"
    )

    # One-shot mode
    once = subparsers.add_parser("run", help="Run a single job and exit")
    once.add_argument(
        "job",
        choices=["nightly", "morning", "infer", "ingest"],
        help="Job to run",
    )

    # List jobs
    subparsers.add_parser("list", help="List scheduled jobs")

    return parser.parse_args()


def main() -> int:
    from src.core.logging import configure_logging
    configure_logging()

    args = parse_args()

    if args.command == "daemon":
        asyncio.run(run_daemon(args))
        return 0
    elif args.command == "run":
        return run_once(args)
    elif args.command == "list":
        print("Available jobs:")
        print("  nightly  - Nightly pipeline: ingest + train + backtest + promote")
        print("  morning  - Morning health check")
        print("  infer    - Market hours inference cycle")
        print("  ingest   - Daily market data ingest")
        print()
        print("Daemon schedules (US/Eastern):")
        print("  23:00    - Nightly pipeline")
        print("  08:00    - Morning health check")
        print("  18:00    - Daily ingest")
        print("  09:30-16:00 Mon-Fri every 15min - Market hours (if --paper)")
        return 0
    else:
        print("Usage: python schedule_pipeline.py {daemon|run|list} [options]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
