#!/usr/bin/env python3
"""
Live Daily Ingestion Script

Fetches the LATEST daily OHLCV bars from Yahoo Finance (FREE) and publishes
to Kafka, then optionally triggers the full trading pipeline.

This is designed for:
- Daily trading workflows (run once per day after market close)
- On-demand ingestion when you need fresh data
- SuperAGI orchestration via awet_run_live_ingestion tool

Usage:
    # Fetch latest bars for configured symbols
    python -m execution.run_live_ingestion
    
    # Fetch and trigger full pipeline
    python -m execution.run_live_ingestion --trigger-pipeline
    
    # Custom symbols
    python -m execution.run_live_ingestion --symbols AAPL,MSFT,NVDA
    
    # Dry run (print only)
    python -m execution.run_live_ingestion --dry-run
    
Environment:
    MARKET_DATA_PROVIDER=yfinance  # Default, uses free Yahoo Finance
"""

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog

from src.core.config import load_settings
from src.core.logging import configure_logging, set_correlation_id
from src.market_data.providers import YFinanceProvider, OHLCVBar
from src.models.events_market import MarketRawEvent
from src.monitoring.metrics import (
    LIVE_BARS_INGESTED, 
    LIVE_INGESTION_LATENCY, 
    LIVE_INGESTION_FAILURES,
)
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import MARKET_RAW
from src.audit.trail_logger import AuditTrailLogger

SCHEMA_PATH = "src/schemas/market_raw.avsc"

logger = structlog.get_logger(__name__)


class LiveIngestionResult:
    """Result of a live ingestion run."""
    def __init__(self):
        self.correlation_id: str = str(uuid4())
        self.symbols_processed: list[str] = []
        self.bars_ingested: int = 0
        self.errors: list[str] = []
        self.pipeline_triggered: bool = False
        self.start_time: datetime = datetime.now(tz=timezone.utc)
        self.end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.bars_ingested > 0
    
    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "symbols_processed": self.symbols_processed,
            "bars_ingested": self.bars_ingested,
            "errors": self.errors,
            "pipeline_triggered": self.pipeline_triggered,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
        }


async def fetch_latest_bar(
    provider: YFinanceProvider,
    symbol: str,
    logger: structlog.BoundLogger,
) -> Optional[OHLCVBar]:
    """Fetch the most recent bar for a symbol."""
    try:
        bar = await provider.get_latest_bar(symbol)
        if bar:
            logger.info("fetched_latest_bar", symbol=symbol, close=bar.close, ts=bar.timestamp.isoformat())
        else:
            logger.warning("no_latest_bar", symbol=symbol)
        return bar
    except Exception as e:
        logger.error("fetch_latest_bar_error", symbol=symbol, error=str(e))
        return None


async def publish_bar_to_kafka(
    bar: OHLCVBar,
    producer: AvroProducer,
    audit: AuditTrailLogger,
    correlation_id: str,
    dry_run: bool = False,
) -> bool:
    """Publish a single bar to Kafka and audit trail."""
    set_correlation_id(correlation_id)
    
    ts_str = bar.timestamp.strftime("%Y%m%d")
    idempotency_key = f"live:{bar.symbol}:{ts_str}"
    
    # Check for duplicates
    if not dry_run and await audit.is_duplicate(MARKET_RAW, idempotency_key):
        logger.info("skipping_duplicate", symbol=bar.symbol, idempotency_key=idempotency_key)
        return False
    
    event = MarketRawEvent(
        idempotency_key=idempotency_key,
        symbol=bar.symbol,
        source="live_ingestion",
        correlation_id=correlation_id,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        price=bar.close,
        volume=bar.volume,
        vwap=bar.vwap,
        trades=bar.trades,
    )
    
    if dry_run:
        print(f"[DRY-RUN] Would publish: {bar.symbol} | {bar.timestamp.isoformat()} | "
              f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:.0f}")
        return True
    
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            schema_str = handle.read()
        payload = event.to_avro_dict()
        producer.produce(MARKET_RAW, schema_str, payload, key=bar.symbol)
        await audit.write_event(MARKET_RAW, payload)
        logger.info("published_live_bar", symbol=bar.symbol, close=bar.close)
        return True
    except Exception as e:
        logger.error("publish_error", symbol=bar.symbol, error=str(e))
        return False


def trigger_pipeline_agents(correlation_id: str, timeout: int = 120) -> bool:
    """
    Trigger the downstream pipeline agents after ingestion.
    
    The pipeline flow:
    market.raw → feature_engineering → market.engineered
    market.engineered → prediction → predictions.tft
    predictions.tft → risk → risk.approved/rejected
    risk.approved → execution → execution.completed/blocked
    
    This runs the demo script which orchestrates all agents.
    """
    logger.info("triggering_pipeline", correlation_id=correlation_id)
    
    try:
        # Use the existing demo script which runs all agents
        result = subprocess.run(
            [sys.executable, "-m", "execution.demo", "--timeout", str(timeout)],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        
        if result.returncode == 0:
            logger.info("pipeline_completed", correlation_id=correlation_id)
            return True
        else:
            logger.error("pipeline_failed", 
                        correlation_id=correlation_id, 
                        stderr=result.stderr[-500:] if result.stderr else "")
            return False
    except subprocess.TimeoutExpired:
        logger.error("pipeline_timeout", correlation_id=correlation_id, timeout=timeout)
        return False
    except Exception as e:
        logger.error("pipeline_error", correlation_id=correlation_id, error=str(e))
        return False


async def run_live_ingestion(
    symbols: list[str],
    dry_run: bool = False,
    trigger_pipeline: bool = False,
    pipeline_timeout: int = 120,
) -> LiveIngestionResult:
    """
    Main live ingestion function.
    
    1. Fetches latest bars for all symbols
    2. Publishes to Kafka (market.raw topic)
    3. Optionally triggers full pipeline
    
    Returns LiveIngestionResult with details.
    """
    result = LiveIngestionResult()
    correlation_id = result.correlation_id
    set_correlation_id(correlation_id)
    
    configure_logging()
    logger.info("live_ingestion_start", 
                correlation_id=correlation_id,
                symbols=symbols, 
                dry_run=dry_run,
                trigger_pipeline=trigger_pipeline)
    
    settings = load_settings()
    provider = YFinanceProvider(rate_limit_per_minute=settings.market_data.yfinance.rate_limit_per_minute)
    
    producer = None
    audit = None
    
    if not dry_run:
        producer = AvroProducer(settings.kafka)
        audit = AuditTrailLogger(settings)
        await audit.connect()
    
    start_time = datetime.now(tz=timezone.utc)
    
    try:
        for symbol in symbols:
            try:
                bar = await fetch_latest_bar(provider, symbol, logger)
                if bar:
                    if dry_run or (producer and audit):
                        published = await publish_bar_to_kafka(
                            bar, producer, audit, correlation_id, dry_run
                        )
                        if published:
                            result.bars_ingested += 1
                            LIVE_BARS_INGESTED.labels(symbol=symbol, source="yfinance").inc()
                    result.symbols_processed.append(symbol)
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                result.errors.append(error_msg)
                LIVE_INGESTION_FAILURES.labels(symbol=symbol, source="yfinance").inc()
                logger.error("symbol_ingestion_error", symbol=symbol, error=str(e))
        
        result.end_time = datetime.now(tz=timezone.utc)
        LIVE_INGESTION_LATENCY.labels(source="yfinance").observe(result.duration_seconds)
        
        # Trigger pipeline if requested and we have data
        if trigger_pipeline and result.bars_ingested > 0 and not dry_run:
            result.pipeline_triggered = trigger_pipeline_agents(correlation_id, pipeline_timeout)
        
        logger.info("live_ingestion_complete",
                   correlation_id=correlation_id,
                   bars_ingested=result.bars_ingested,
                   symbols=result.symbols_processed,
                   duration_seconds=result.duration_seconds,
                   success=result.success)
        
    finally:
        await provider.close()
        if audit:
            await audit.close()
    
    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Live Daily Ingestion - Fetch latest OHLCV and publish to Kafka"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print bars without publishing to Kafka",
    )
    parser.add_argument(
        "--trigger-pipeline",
        action="store_true",
        help="Trigger full pipeline after ingestion (feature eng → prediction → risk → execution)",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for pipeline execution (default: 120)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    
    args = parser.parse_args()
    
    # Get symbols from args or config
    settings = load_settings()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = settings.app.symbols
    
    result = await run_live_ingestion(
        symbols=symbols,
        dry_run=args.dry_run,
        trigger_pipeline=args.trigger_pipeline,
        pipeline_timeout=args.pipeline_timeout,
    )
    
    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Live Ingestion Result")
        print(f"{'='*60}")
        print(f"Correlation ID: {result.correlation_id}")
        print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Symbols: {', '.join(result.symbols_processed)}")
        print(f"Bars Ingested: {result.bars_ingested}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        if result.pipeline_triggered:
            print(f"Pipeline: ✅ Triggered")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for err in result.errors:
                print(f"  - {err}")
        print(f"{'='*60}\n")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
