#!/usr/bin/env python3
"""
Stream-ingest Reddit .zst JSONL dumps into TimescaleDB/Postgres.

Supports comments (RC_*) and submissions (RS_*). Filters to a target month
(UTC) and extracts ticker mentions based on config/universe.csv or TRAIN_SYMBOLS.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import structlog
import zstandard as zstd

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
UNIVERSE_FILE = PROJECT_ROOT / "config" / "universe.csv"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://awet:awet@localhost:5433/awet")

REDDIT_MENTIONS_TABLE = "reddit_mentions"
CHECKPOINT_TABLE = "backfill_checkpoints"

TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|\(([A-Z]{1,5})\)|\b([A-Z]{2,5})\b")
EXCLUDED_WORDS = {
    "I", "A", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AM", "AN", "AS", "DD", "EV", "FA", "TA", "AI", "CEO", "CFO", "CTO",
    "IPO", "ETF", "GDP", "SEC", "FBI", "CIA", "USA", "UK", "EU", "NYC",
    "LA", "SF", "TX", "CA", "NY", "FL", "PM", "AM", "EST", "PST", "UTC",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAS",
    "HER", "WAS", "ONE", "OUR", "OUT", "HIS", "HAS", "HAD", "GET", "GOT",
    "NOW", "NEW", "OLD", "BIG", "TOP", "LOW", "ATH", "ATL", "IMO", "TBH",
    "LOL", "OMG", "WTF", "FYI", "BTW", "YOLO", "FOMO", "HODL", "TLDR",
    "EDIT", "POST", "LINK", "PUTS", "CALL", "LONG", "SHORT", "HOLD",
    "SELL", "MOON", "CASH", "GAIN", "LOSS", "PLAY", "MOVE", "WEEK",
    "TODAY", "STOCK", "SHARE", "PRICE", "VALUE", "MONEY", "TRADE",
}


def load_universe() -> set[str]:
    env_symbols = os.getenv("TRAIN_SYMBOLS")
    if env_symbols:
        return {s.strip().upper() for s in env_symbols.split(",") if s.strip()}

    if not UNIVERSE_FILE.exists():
        return set()

    symbols = set()
    with open(UNIVERSE_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbols.add(row["ticker"].strip().upper())
    return symbols


def extract_tickers(text: str, valid_tickers: set[str]) -> list[str]:
    if not text:
        return []
    matches = TICKER_PATTERN.findall(text)
    tickers = set()
    for match in matches:
        ticker = match[0] or match[1] or match[2]
        if not ticker:
            continue
        if ticker in EXCLUDED_WORDS:
            continue
        if ticker in valid_tickers:
            tickers.add(ticker)
    return list(tickers)


def month_bounds(month: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def guess_source(file_path: Path) -> str:
    name = file_path.name
    if name.startswith("RC_") or "comments" in file_path.parts:
        return "comments"
    return "submissions"


def extract_text(post: dict, source: str) -> tuple[str, str]:
    if source == "comments":
        title = ""
        body = post.get("body") or ""
        return title, body
    title = post.get("title") or ""
    body = post.get("selftext") or ""
    return title, body


def normalize_row(post: dict, source: str, ticker: str) -> tuple:
    created_utc = post.get("created_utc")
    created_ts = datetime.fromtimestamp(created_utc, tz=timezone.utc) if created_utc else None

    title, body = extract_text(post, source)
    return (
        ticker,
        created_ts,
        source,
        post.get("subreddit"),
        post.get("author"),
        int(post.get("score") or 0),
        int(post.get("num_comments") or 0),
        post.get("id") or post.get("name"),
        post.get("permalink"),
        title[:500],
        body[:4000],
        created_ts,
        json.dumps(post),
    )


async def ensure_schema(conn: asyncpg.Connection) -> None:
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REDDIT_MENTIONS_TABLE} (
            ticker TEXT NOT NULL,
            ts TIMESTAMPTZ NOT NULL,
            source TEXT NOT NULL,
            subreddit TEXT,
            author TEXT,
            score INT DEFAULT 0,
            num_comments INT DEFAULT 0,
            post_id TEXT NOT NULL,
            permalink TEXT,
            title TEXT,
            body TEXT,
            created_utc TIMESTAMPTZ,
            raw_json JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (post_id, source, ticker)
        )
        """
    )
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {REDDIT_MENTIONS_TABLE}_ticker_idx ON {REDDIT_MENTIONS_TABLE} (ticker, ts DESC)"
    )
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {REDDIT_MENTIONS_TABLE}_ts_idx ON {REDDIT_MENTIONS_TABLE} (ts DESC)"
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reddit_daily_mentions (
            day DATE NOT NULL,
            ticker TEXT NOT NULL,
            mentions INT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (day, ticker)
        )
        """
    )


def _checkpoint_id(file_path: Path) -> tuple[str, str]:
    return "reddit_zst", file_path.name


async def load_checkpoint(conn: asyncpg.Connection, file_path: Path) -> int:
    source, filename = _checkpoint_id(file_path)
    row = await conn.fetchrow(
        f"""
        SELECT last_offset FROM {CHECKPOINT_TABLE}
        WHERE source=$1 AND filename=$2
        """,
        source,
        filename,
    )
    return int(row["last_offset"]) if row else 0


async def update_checkpoint(conn: asyncpg.Connection, file_path: Path, offset: int, processed: int) -> None:
    source, filename = _checkpoint_id(file_path)
    await conn.execute(
        f"""
        INSERT INTO {CHECKPOINT_TABLE} (source, filename, last_offset, records_processed)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (source, filename) DO UPDATE
        SET last_offset = EXCLUDED.last_offset,
            records_processed = EXCLUDED.records_processed,
            updated_at = now()
        """,
        source,
        filename,
        offset,
        processed,
    )


async def mark_checkpoint_complete(conn: asyncpg.Connection, file_path: Path, offset: int, processed: int) -> None:
    source, filename = _checkpoint_id(file_path)
    await conn.execute(
        f"""
        INSERT INTO {CHECKPOINT_TABLE} (source, filename, last_offset, records_processed, completed_at)
        VALUES ($1, $2, $3, $4, now())
        ON CONFLICT (source, filename) DO UPDATE
        SET last_offset = EXCLUDED.last_offset,
            records_processed = EXCLUDED.records_processed,
            completed_at = now(),
            updated_at = now()
        """,
        source,
        filename,
        offset,
        processed,
    )


async def ingest_file(
    conn: asyncpg.Connection,
    file_path: Path,
    valid_tickers: set[str],
    month: str,
    batch_size: int,
) -> dict:
    stats = {
        "file": file_path.name,
        "processed": 0,
        "mentions": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    start_ts, end_ts = month_bounds(month)
    source = guess_source(file_path)
    checkpoint = await load_checkpoint(conn, file_path)

    logger.info("ingest_start", file=file_path.name, source=source, checkpoint=checkpoint)

    batch = []
    last_report = time.perf_counter()
    last_count = 0

    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            import io
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_stream:
                stats["processed"] += 1

                if stats["processed"] <= checkpoint:
                    stats["skipped"] += 1
                    continue

                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    stats["errors"] += 1
                    continue

                created_utc = post.get("created_utc")
                if not created_utc:
                    continue

                created_ts = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                if created_ts < start_ts or created_ts >= end_ts:
                    continue

                title, body = extract_text(post, source)
                tickers = extract_tickers(f"{title} {body}", valid_tickers)
                if not tickers:
                    continue

                for ticker in tickers:
                    row = normalize_row(post, source, ticker)
                    batch.append(row)
                    stats["mentions"] += 1

                if len(batch) >= batch_size:
                    await conn.executemany(
                        f"""
                        INSERT INTO {REDDIT_MENTIONS_TABLE} (
                            ticker, ts, source, subreddit, author, score, num_comments,
                            post_id, permalink, title, body, created_utc, raw_json
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7,
                            $8, $9, $10, $11, $12, $13
                        )
                        ON CONFLICT (post_id, source, ticker) DO NOTHING
                        """,
                        batch,
                    )
                    stats["inserted"] += len(batch)
                    await update_checkpoint(conn, file_path, stats["processed"], stats["processed"])
                    batch = []

                if stats["processed"] % 100000 == 0:
                    now = time.perf_counter()
                    rate = (stats["processed"] - last_count) / max(0.001, now - last_report)
                    logger.info(
                        "ingest_progress",
                        file=file_path.name,
                        processed=stats["processed"],
                        mentions=stats["mentions"],
                        inserted=stats["inserted"],
                        rate=round(rate, 1),
                    )
                    last_report = now
                    last_count = stats["processed"]

    if batch:
        await conn.executemany(
            f"""
            INSERT INTO {REDDIT_MENTIONS_TABLE} (
                ticker, ts, source, subreddit, author, score, num_comments,
                post_id, permalink, title, body, created_utc, raw_json
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12, $13
            )
            ON CONFLICT (post_id, source, ticker) DO NOTHING
            """,
            batch,
        )
        stats["inserted"] += len(batch)
        await update_checkpoint(conn, file_path, stats["processed"], stats["processed"])

    await mark_checkpoint_complete(conn, file_path, stats["processed"], stats["processed"])
    return stats


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Reddit .zst dumps")
    parser.add_argument("--file", action="append", required=True, help="Path to .zst file")
    parser.add_argument("--month", default="2025-07", help="Month to ingest (YYYY-MM)")
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    files = [Path(f) for f in args.file]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return 1

    universe = load_universe()
    if not universe:
        print("❌ No universe tickers found. Set TRAIN_SYMBOLS or create config/universe.csv")
        return 1

    print(f"Universe loaded: {len(universe)} tickers")

    conn = await asyncpg.connect(DATABASE_URL)
    await ensure_schema(conn)

    totals = Counter()
    for file_path in files:
        stats = await ingest_file(conn, file_path, universe, args.month, args.batch_size)
        print(
            f"✅ {stats['file']}: processed={stats['processed']:,} "
            f"mentions={stats['mentions']:,} inserted={stats['inserted']:,} "
            f"skipped={stats['skipped']:,} errors={stats['errors']:,}"
        )
        totals.update(stats)

    month_start, month_end = month_bounds(args.month)
    await conn.execute(
        """
        INSERT INTO reddit_daily_mentions (day, ticker, mentions)
        SELECT ts::date AS day, ticker, COUNT(*) AS mentions
        FROM reddit_mentions
        WHERE ts >= $1 AND ts < $2
        GROUP BY 1, 2
        ON CONFLICT (day, ticker) DO UPDATE
        SET mentions = EXCLUDED.mentions
        """,
        month_start,
        month_end,
    )
    print("✅ Aggregated reddit_daily_mentions for month", args.month)

    await conn.close()

    print(
        f"\nTOTAL: processed={totals['processed']:,} mentions={totals['mentions']:,} "
        f"inserted={totals['inserted']:,} skipped={totals['skipped']:,} errors={totals['errors']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
