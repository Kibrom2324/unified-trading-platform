#!/usr/bin/env python3
"""Backfill Reddit historical data into Kafka.

Reads .zst compressed files from local Reddit data dumps and publishes
SocialRawEvent messages to Kafka topic social.raw.

Supports subdirectory structure (e.g., submissions/wallstreetbets/*.zst)
and resumable backfill via checkpoint table.

Usage:
    python execution/backfill_reddit.py --submissions-dir /path/to/submissions
    python execution/backfill_reddit.py --submissions-dir /path/to/submissions --dry-run
    python execution/backfill_reddit.py --submissions-dir /path/to/submissions --resume
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backfill.checkpoint import BackfillCheckpoint
from src.core.config import load_settings
from src.core.logging import configure_logging, set_correlation_id
from src.monitoring.metrics import EVENTS_PROCESSED
from src.streaming.kafka_producer import AvroProducer

logger = structlog.get_logger("backfill_reddit")

# Finance-related subreddits
FINANCE_SUBREDDITS = {
    "wallstreetbets", "stocks", "investing", "options", "stockmarket",
    "robinhood", "pennystocks", "daytrading", "algotrading", "finance",
    "cryptocurrency", "bitcoin", "ethereum", "superstonk", "thetagang",
}

# Ticker extraction pattern
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b')


@dataclass
class SocialRawEvent:
    """Social media raw event."""
    event_id: str
    correlation_id: str
    idempotency_key: str
    platform: str
    post_id: str
    subreddit: str
    author: str
    title: str
    body: str
    score: int
    num_comments: int
    created_utc: datetime
    tickers: list[str]
    schema_version: int = 1
    source: str = "reddit_backfill"

    def to_avro_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "platform": self.platform,
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "author": self.author,
            "title": self.title,
            "body": self.body[:10000] if self.body else "",  # Truncate for safety
            "score": self.score,
            "num_comments": self.num_comments,
            "ts": self.created_utc.isoformat(),
            "tickers": self.tickers,
            "schema_version": self.schema_version,
            "source": self.source,
        }


def extract_tickers(text: str) -> list[str]:
    """Extract potential ticker symbols from text."""
    if not text:
        return []
    matches = TICKER_PATTERN.findall(text)
    tickers = set()
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and len(ticker) >= 2:
            # Filter common words
            if ticker not in {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                             "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS",
                             "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY",
                             "WHO", "BOY", "DID", "GET", "LET", "PUT", "SAY", "TOO",
                             "USE", "CEO", "CFO", "IPO", "ETF", "USD", "USA"}:
                tickers.add(ticker)
    return list(tickers)[:10]  # Limit to 10 tickers


def generate_idempotency_key(platform: str, post_id: str) -> str:
    """Generate deterministic idempotency key."""
    raw = f"{platform}:{post_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def load_zst_file(filepath: Path) -> Iterator[dict]:
    """Load zstandard compressed JSON lines file."""
    try:
        import zstandard as zstd
    except ImportError:
        logger.error("zstandard_not_installed", hint="pip install zstandard")
        return

    logger.info("loading_file", path=str(filepath))

    with open(filepath, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            import io
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def parse_submission(data: dict, correlation_id: str) -> SocialRawEvent | None:
    """Parse Reddit submission to SocialRawEvent."""
    subreddit = data.get("subreddit", "").lower()
    if subreddit not in FINANCE_SUBREDDITS:
        return None

    post_id = data.get("id", "")
    if not post_id:
        return None

    title = data.get("title", "")
    body = data.get("selftext", "")
    tickers = extract_tickers(f"{title} {body}")

    created_utc = data.get("created_utc", 0)
    if isinstance(created_utc, (int, float)):
        ts = datetime.fromtimestamp(created_utc, tz=timezone.utc)
    else:
        ts = datetime.now(tz=timezone.utc)

    return SocialRawEvent(
        event_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        idempotency_key=generate_idempotency_key("reddit", post_id),
        platform="reddit",
        post_id=post_id,
        subreddit=subreddit,
        author=data.get("author", "[deleted]"),
        title=title,
        body=body,
        score=data.get("score", 0),
        num_comments=data.get("num_comments", 0),
        created_utc=ts,
        tickers=tickers,
    )


SOCIAL_RAW_SCHEMA = """{
  "type": "record",
  "name": "SocialRawEvent",
  "namespace": "com.awet.social",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "correlation_id", "type": "string"},
    {"name": "idempotency_key", "type": "string"},
    {"name": "platform", "type": "string"},
    {"name": "post_id", "type": "string"},
    {"name": "subreddit", "type": "string"},
    {"name": "author", "type": "string"},
    {"name": "title", "type": "string"},
    {"name": "body", "type": "string"},
    {"name": "score", "type": "int"},
    {"name": "num_comments", "type": "int"},
    {"name": "ts", "type": "string"},
    {"name": "tickers", "type": {"type": "array", "items": "string"}},
    {"name": "schema_version", "type": "int"},
    {"name": "source", "type": "string"}
  ]
}"""


async def backfill_reddit(
    submissions_dir: str,
    batch_size: int = 500,
    max_records: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, int]:
    """Backfill Reddit data to Kafka.

    Args:
        submissions_dir: Path to Reddit submissions directory
        batch_size: Records per batch
        max_records: Max records to process (for testing)
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
        submissions_dir=submissions_dir,
        dry_run=dry_run,
        resume=resume,
        correlation_id=correlation_id,
    )

    # Initialize checkpoint manager
    checkpoint = BackfillCheckpoint(source="reddit") if not dry_run else None

    # Initialize producer (only if not dry run)
    producer = None
    if not dry_run:
        settings = load_settings()
        producer = AvroProducer(settings.kafka)

    submissions_path = Path(submissions_dir)
    if not submissions_path.exists():
        logger.error("directory_not_found", path=submissions_dir)
        return {"processed": 0, "errors": 1}

    processed = 0
    skipped = 0
    errors = 0
    skipped_files = 0
    batch: list[SocialRawEvent] = []

    # Find all .zst files (including in subdirectories)
    zst_files = list(submissions_path.glob("**/*.zst"))
    logger.info("found_files", count=len(zst_files))

    for zst_file in zst_files:
        file_key = str(zst_file.relative_to(submissions_path))

        # Check if file already completed
        if resume and checkpoint and checkpoint.is_file_completed(file_key):
            skipped_files += 1
            logger.debug("skipping_completed_file", filename=file_key)
            continue

        file_record_count = 0

        for record in load_zst_file(zst_file):
            if max_records and processed >= max_records:
                break

            try:
                event = parse_submission(record, correlation_id)
                if event is None:
                    skipped += 1
                    continue

                batch.append(event)
                file_record_count += 1

                if len(batch) >= batch_size:
                    # Publish batch
                    if producer:
                        for evt in batch:
                            producer.produce(
                                "social.raw",
                                SOCIAL_RAW_SCHEMA,
                                evt.to_avro_dict(),
                                key=evt.subreddit,
                            )
                        producer.flush()

                    processed += len(batch)
                    EVENTS_PROCESSED.labels(agent="backfill_reddit", event_type="social.raw").inc(len(batch))
                    logger.info("batch_published", processed=processed, skipped=skipped)
                    batch = []

            except Exception as e:
                errors += 1
                logger.error("parse_error", error=str(e))

        # Mark file as completed
        if checkpoint and file_record_count > 0:
            checkpoint.mark_completed(file_key, file_record_count)

        if max_records and processed >= max_records:
            break

    # Publish remaining batch
    if batch:
        if producer:
            for evt in batch:
                producer.produce(
                    "social.raw",
                    SOCIAL_RAW_SCHEMA,
                    evt.to_avro_dict(),
                    key=evt.subreddit,
                )
            producer.flush()
        processed += len(batch)

    # Log checkpoint stats
    if checkpoint:
        checkpoint_stats = checkpoint.get_stats()
        logger.info("checkpoint_stats", **checkpoint_stats)
        checkpoint.close()

    stats = {
        "processed": processed,
        "skipped": skipped,
        "skipped_files": skipped_files,
        "errors": errors,
        "correlation_id": correlation_id,
    }
    logger.info("backfill_complete", **stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Reddit data to Kafka")
    parser.add_argument(
        "--submissions-dir",
        required=True,
        help="Path to Reddit submissions directory (can contain subdirectories)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Kafka publishing (default: 500)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Max records to process (for testing)",
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

    stats = asyncio.run(
        backfill_reddit(
            submissions_dir=args.submissions_dir,
            batch_size=args.batch_size,
            max_records=args.max_records,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    )

    print(f"\n✅ Backfill complete: {stats['processed']} records, {stats['skipped']} skipped, {stats['skipped_files']} files skipped, {stats['errors']} errors")
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
