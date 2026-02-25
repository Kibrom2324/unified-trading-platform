"""Backfill ingestion service for historical data.

Orchestrates loading from Polygon and Reddit historical files
and publishing to Kafka topics.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.audit.trail_logger import AuditTrailLogger
from src.backfill.polygon_loader import PolygonBar, PolygonCSVLoader
from src.core.config import load_settings
from src.core.logging import get_correlation_id, set_correlation_id
from src.models.events_market import MarketRawEvent
from src.monitoring.metrics import EVENTS_PROCESSED, EVENT_LATENCY
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import MARKET_RAW

logger = structlog.get_logger("backfill_service")

RAW_SCHEMA_PATH = "src/schemas/market_raw.avsc"


class BackfillService:
    """Service for backfilling historical data to Kafka."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings or load_settings()
        self.producer = AvroProducer(self.settings.kafka)
        self.audit = AuditTrailLogger(self.settings)
        with open(RAW_SCHEMA_PATH, "r", encoding="utf-8") as f:
            self._raw_schema = f.read()
        self._processed = 0
        self._duplicates = 0
        self._errors = 0

    async def connect(self) -> None:
        """Initialize connections."""
        await self.audit.connect()

    async def close(self) -> None:
        """Close connections."""
        await self.audit.close()

    def _bar_to_event(self, bar: PolygonBar, correlation_id: str) -> MarketRawEvent:
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
        self,
        aggregate_type: str = "minute",
        batch_size: int = 1000,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Backfill Polygon historical data.

        Args:
            aggregate_type: "minute" or "day"
            batch_size: Records per batch
            dry_run: If True, validate but don't publish

        Returns:
            Stats dict with processed/duplicates/errors counts
        """
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        logger.info(
            "backfill_start",
            aggregate_type=aggregate_type,
            correlation_id=correlation_id,
        )
        data_config = getattr(self.settings, "data", None)
        data_root = getattr(data_config, "data_root", None) if data_config else None
        if not data_root:
            import os
            data_root = os.getenv("DATA_ROOT", "/home/kironix/train")
        
        # Build full path to data directory
        if aggregate_type == "minute":
            data_dir = f"{data_root}/poligon/Minute Aggregates"
        elif aggregate_type == "day":
            data_dir = f"{data_root}/poligon/Day Aggregates"
        else:
            raise ValueError(f"Unknown aggregate_type: {aggregate_type}")
        
        symbols = self.settings.app.symbols
        start_date = None
        end_date = None
        if data_config:
            if hasattr(data_config, "start_date") and data_config.start_date:
                start_date = datetime.fromisoformat(data_config.start_date)
            if hasattr(data_config, "end_date") and data_config.end_date:
                end_date = datetime.fromisoformat(data_config.end_date)
        loader = PolygonCSVLoader(
            data_dir=data_dir,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        self._processed = 0
        self._duplicates = 0
        self._errors = 0
        batch_num = 0

        for batch in loader.load_batched(batch_size):
            batch_num += 1
            start_ts = datetime.now(tz=timezone.utc)
            for bar in batch:
                try:
                    event = self._bar_to_event(bar, correlation_id)
                    if await self.audit.is_duplicate(MARKET_RAW, event.idempotency_key):
                        self._duplicates += 1
                        continue
                    if not dry_run:
                        payload = event.to_avro_dict()
                        self.producer.produce(
                            MARKET_RAW,
                            self._raw_schema,
                            payload,
                            key=event.symbol,
                        )
                        await self.audit.write_event(MARKET_RAW, payload)
                    self._processed += 1
                except Exception as e:
                    self._errors += 1
                    logger.error("backfill_error", error=str(e), symbol=bar.ticker)
            duration = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
            EVENTS_PROCESSED.labels(agent="backfill", event_type=MARKET_RAW).inc(len(batch))
            EVENT_LATENCY.labels(agent="backfill", event_type=MARKET_RAW).observe(duration)
            if batch_num % 10 == 0:
                logger.info(
                    "backfill_progress",
                    batch=batch_num,
                    processed=self._processed,
                    duplicates=self._duplicates,
                    errors=self._errors,
                )

        stats = {
            "processed": self._processed,
            "duplicates": self._duplicates,
            "errors": self._errors,
            "aggregate_type": aggregate_type,
        }
        logger.info("backfill_complete", **stats)
        return stats


async def run_backfill(
    aggregate_type: str = "minute",
    batch_size: int = 1000,
    dry_run: bool = False,
) -> dict[str, int]:
    """CLI entrypoint for backfill."""
    service = BackfillService()
    await service.connect()
    try:
        return await service.backfill_polygon(
            aggregate_type=aggregate_type,
            batch_size=batch_size,
            dry_run=dry_run,
        )
    finally:
        await service.close()


def main() -> None:
    """CLI main."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill historical data")
    parser.add_argument(
        "--type",
        choices=["minute", "day"],
        default="minute",
        help="Aggregate type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without publishing",
    )
    args = parser.parse_args()
    stats = asyncio.run(
        run_backfill(
            aggregate_type=args.type,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )
    print(f"Backfill complete: {stats}")


if __name__ == "__main__":
    main()
