"""Reddit historical data loader for social sentiment analysis.

Reads .zst (zstandard) compressed JSON files from Reddit data dumps
and emits SocialRawEvent to Kafka.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import structlog

logger = structlog.get_logger("reddit_loader")

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard_not_installed", hint="pip install zstandard")


@dataclass
class RedditPost:
    """Parsed Reddit submission or comment."""
    id: str
    author: str
    subreddit: str
    title: str | None
    body: str
    score: int
    created_utc: datetime
    post_type: str
    url: str | None = None
    parent_id: str | None = None

    def idempotency_key(self, source: str = "reddit_backfill") -> str:
        """Generate stable idempotency key."""
        raw = f"{source}:{self.post_type}:{self.id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def extract_tickers(self, known_tickers: set[str] | None = None) -> list[str]:
        """Extract potential stock tickers from text."""
        text = f"{self.title or ''} {self.body}"
        pattern = r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b"
        matches = re.findall(pattern, text)
        tickers = set()
        for m in matches:
            ticker = m[0] or m[1]
            if known_tickers is None or ticker in known_tickers:
                tickers.add(ticker)
        return list(tickers)


class RedditZstLoader:
    """Load Reddit historical data from zstandard compressed files."""

    FINANCE_SUBREDDITS = {
        "wallstreetbets", "stocks", "investing", "options",
        "stockmarket", "pennystocks", "robinhood", "thetagang",
        "daytrading", "cryptocurrency", "bitcoin", "algotrading",
        "securityanalysis", "valueinvesting", "dividends",
    }

    def __init__(
        self,
        data_root: str,
        content_type: str = "submissions",
        subreddits: set[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        finance_only: bool = True,
    ) -> None:
        if not ZSTD_AVAILABLE:
            raise RuntimeError("zstandard library required: pip install zstandard")
        self.data_root = Path(data_root)
        self.content_type = content_type
        self.subreddits = subreddits or (self.FINANCE_SUBREDDITS if finance_only else None)
        self.start_date = start_date
        self.end_date = end_date
        if content_type == "submissions":
            self.data_dir = self.data_root / "reddit" / "submissions"
        elif content_type == "comments":
            self.data_dir = self.data_root / "reddit" / "comments"
        else:
            raise ValueError(f"Unknown content_type: {content_type}")

    def discover_files(self) -> list[Path]:
        """Find all .zst files in data directory."""
        files = []
        if not self.data_dir.exists():
            logger.warning("data_dir_not_found", path=str(self.data_dir))
            return files
        for zst_file in sorted(self.data_dir.glob("*.zst")):
            file_date = self._parse_file_date(zst_file.name)
            if file_date:
                if self.start_date and file_date < self.start_date:
                    continue
                if self.end_date and file_date > self.end_date:
                    continue
            files.append(zst_file)
        logger.info("discovered_files", count=len(files), dir=str(self.data_dir))
        return files

    def _parse_file_date(self, filename: str) -> datetime | None:
        """Parse date from filename like RS_2024-07.zst."""
        try:
            match = re.search(r"(\d{4})-(\d{2})", filename)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                return datetime(year, month, 1, tzinfo=timezone.utc)
        except ValueError:
            pass
        return None

    def load_file(self, filepath: Path) -> Iterator[RedditPost]:
        """Load and parse a single zstandard compressed file."""
        logger.info("loading_file", path=str(filepath))
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = reader.read().decode("utf-8", errors="ignore")
                for line in text_stream.split("\n"):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        post = self._parse_post(data)
                        if post is None:
                            continue
                        if self.subreddits and post.subreddit.lower() not in self.subreddits:
                            continue
                        if self.start_date and post.created_utc < self.start_date:
                            continue
                        if self.end_date and post.created_utc > self.end_date:
                            continue
                        yield post
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        continue

    def _parse_post(self, data: dict) -> RedditPost | None:
        """Parse JSON object into RedditPost."""
        try:
            created_utc = datetime.fromtimestamp(
                data.get("created_utc", 0), tz=timezone.utc
            )
            if "title" in data:
                return RedditPost(
                    id=data["id"],
                    author=data.get("author", "[deleted]"),
                    subreddit=data.get("subreddit", "unknown"),
                    title=data.get("title"),
                    body=data.get("selftext", ""),
                    score=data.get("score", 0),
                    created_utc=created_utc,
                    post_type="submission",
                    url=data.get("url"),
                )
            else:
                return RedditPost(
                    id=data["id"],
                    author=data.get("author", "[deleted]"),
                    subreddit=data.get("subreddit", "unknown"),
                    title=None,
                    body=data.get("body", ""),
                    score=data.get("score", 0),
                    created_utc=created_utc,
                    post_type="comment",
                    parent_id=data.get("parent_id"),
                )
        except (KeyError, TypeError):
            return None

    def load_all(self) -> Iterator[RedditPost]:
        """Load all files and yield posts."""
        files = self.discover_files()
        total_posts = 0
        for filepath in files:
            for post in self.load_file(filepath):
                total_posts += 1
                yield post
        logger.info("load_complete", total_posts=total_posts)

    def load_batched(self, batch_size: int = 100) -> Iterator[list[RedditPost]]:
        """Load posts in batches for efficient processing."""
        batch: list[RedditPost] = []
        for post in self.load_all():
            batch.append(post)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
