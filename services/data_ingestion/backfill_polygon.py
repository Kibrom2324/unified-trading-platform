#!/usr/bin/env python3
"""Backfill Polygon historical data into Kafka.

Reads .csv.gz files from local Polygon data dumps and publishes
MarketRawEvent messages to Kafka topic market.raw.

Supports resumable backfill via checkpoint table in TimescaleDB.

Usage:
    python execution/backfill_polygon.py --data-dir /path/to/data
    python execution/backfill_polygon.py --data-dir /path/to/data --symbols AAPL,MSFT --dry-run
    python execution/backfill_polygon.py --data-dir /path/to/data --resume  # Resume from last checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka.serializing_producer import SerializingProducer

from src.backfill.checkpoint import BackfillCheckpoint
from src.backfill.polygon_loader import PolygonBar, PolygonCSVLoader
from src.core.config import load_settings
from src.core.logging import configure_logging, set_correlation_id
from src.models.events_market import MarketRawEvent
from src.monitoring.metrics import EVENTS_PROCESSED
from src.streaming.topics import MARKET_RAW

logger = structlog.get_logger("backfill_polygon")

RAW_SCHEMA_PATH = "src/schemas/market_raw.avsc"

_delivered = 0
_failed = 0

def _on_delivery(err, msg):
    global _delivered, _failed
    if err:
        _failed += 1
        print(f"Delivery FAILED: {err}")
    else:
        _delivered += 1


def bar_to_event(bar: PolygonBar, correlation_id: str) -> MarketRawEvent:
    """Convert Polygon bar to MarketRawEvent."""
    """Convert Polygon bar to MarketRawEvent."""
    return MarketRawEvent(
        idempotency_key=bar.idempotency_key(),
        symbol=bar.ticker,
        source="polygon_backfill",
        correlation_id=uuid.UUID(correlation_id),
        price=bar.close,
        volume=bar.volume,
        ts=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        vwap=bar.vwap if bar.vwap else (bar.open + bar.high + bar.low + bar.close) / 4,
        trades=bar.transactions,
    )


async def backfill_polygon(
    data_dir: str,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    batch_size: int = 1000,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, int]:
    """Backfill Polygon data to Kafka.

    Args:
        data_dir: Path to Polygon data directory (containing month subdirs)
        symbols: Filter to these symbols, or None for all
        start_date: Filter files starting from this date
        end_date: Filter files ending at this date
        batch_size: Records per batch
        dry_run: If True, validate but don't publish
        resume: If True, skip files already completed in checkpoint table

    Returns:
        Stats dict with processed/errors counts
    """
    configure_logging()
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    logger.info(
        "backfill_start",
        data_dir=data_dir,
        symbols=symbols,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
        dry_run=dry_run,
        resume=resume,
        correlation_id=correlation_id,
    )

    # Initialize checkpoint manager
    checkpoint = BackfillCheckpoint(source="polygon") if not dry_run else None

    # Load schema
    with open(RAW_SCHEMA_PATH, "r", encoding="utf-8") as f:
        raw_schema = f.read()

    # Initialize producer (only if not dry run)
    producer = None
    if not dry_run:
        settings = load_settings()
        # Use direct SerializingProducer instead of wrapper
        schema_registry = SchemaRegistryClient({"url": settings.kafka.schema_registry_url})
        value_serializer = AvroSerializer(schema_registry, raw_schema)
        producer = SerializingProducer({
            "bootstrap.servers": ",".join(settings.kafka.bootstrap_servers),
            "key.serializer": StringSerializer("utf_8"),
            "value.serializer": value_serializer,
        })

    # Initialize loader
    loader = PolygonCSVLoader(
        data_dir=data_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    processed = 0
    errors = 0
    batch_num = 0
    skipped_files = 0
    current_file: str | None = None
    file_record_count = 0

    for batch in loader.load_batched(batch_size):
        batch_num += 1
        batch_start = datetime.now(tz=timezone.utc)

        for bar in batch:
            # Track file changes for checkpointing
            bar_file = getattr(bar, '_source_file', None)
            if bar_file and bar_file != current_file:
                # Save checkpoint for previous file
                if current_file and checkpoint and file_record_count > 0:
                    checkpoint.mark_completed(current_file, file_record_count)

                # Check if new file should be skipped
                if resume and checkpoint and checkpoint.is_file_completed(bar_file):
                    skipped_files += 1
                    logger.debug("skipping_completed_file", filename=bar_file)
                    continue

                current_file = bar_file
                file_record_count = 0

            try:
                event = bar_to_event(bar, correlation_id)

                if not dry_run and producer:
                    payload = event.to_avro_dict()
                    producer.produce(
                        topic=MARKET_RAW,
                        key=event.symbol,
                        value=payload,
                        on_delivery=_on_delivery,
                    )

                processed += 1
                file_record_count += 1

            except Exception as e:
                errors += 1
                logger.error("backfill_error", error=str(e), symbol=bar.ticker)
                import traceback
                traceback.print_exc()

        # Poll to trigger delivery reports and clear queue
        if producer:
            producer.poll(0.1)  # Quick poll to process callbacks

        # Update checkpoint progress periodically
        if checkpoint and current_file and batch_num % 10 == 0:
            checkpoint.update_progress(current_file, file_record_count, len(batch))

        EVENTS_PROCESSED.labels(agent="backfill_polygon", event_type=MARKET_RAW).inc(len(batch))

        batch_duration = (datetime.now(tz=timezone.utc) - batch_start).total_seconds()
        if batch_num % 10 == 0:
            logger.info(
                "backfill_progress",
                batch=batch_num,
                processed=processed,
                errors=errors,
                skipped_files=skipped_files,
                batch_duration_s=f"{batch_duration:.2f}",
            )

    # Mark final file as completed
    if checkpoint and current_file and file_record_count > 0:
        checkpoint.mark_completed(current_file, file_record_count)

    # Final flush to deliver all remaining messages
    global _delivered, _failed
    if producer is not None:
        queue_len = len(producer)
        logger.info("flushing_kafka", remaining=queue_len)
        remaining = producer.flush(30.0)
        logger.info("flushed_kafka", remaining=remaining, delivered=_delivered, failed=_failed)

    # Log checkpoint stats
    if checkpoint:
        checkpoint_stats = checkpoint.get_stats()
        logger.info("checkpoint_stats", **checkpoint_stats)
        checkpoint.close()

    stats = {
        "processed": processed,
        "errors": errors,
        "batches": batch_num,
        "skipped_files": skipped_files,
        "correlation_id": correlation_id,
    }
    logger.info("backfill_complete", **stats)
    return stats


def parse_date(s: str) -> datetime:
    """Parse date string to datetime."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Polygon data to Kafka")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to Polygon data directory (e.g., /home/kironix/train/poligon/Minute Aggregates)",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to filter (default: all)",
    )
    parser.add_argument(
        "--start-date",
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for Kafka publishing (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without publishing to Kafka",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip already completed files)",
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None

    stats = asyncio.run(
        backfill_polygon(
            data_dir=args.data_dir,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    )

    print(f"\n✅ Backfill complete: {stats['processed']} records, {stats['errors']} errors, {stats['skipped_files']} files skipped")
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
