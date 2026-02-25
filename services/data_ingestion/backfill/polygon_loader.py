"""Polygon historical data loader for backfill ingestion.

Reads .csv.gz files from local Polygon data dumps and emits
MarketRawEvent to Kafka with proper idempotency keys.
"""

from __future__ import annotations

import gzip
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import structlog

logger = structlog.get_logger("polygon_loader")


@dataclass
class PolygonBar:
    """Parsed Polygon OHLCV bar."""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    transactions: int
    timestamp: datetime

    def idempotency_key(self, source: str = "polygon_backfill") -> str:
        """Generate stable idempotency key."""
        raw = f"{source}:{self.ticker}:{self.timestamp.isoformat()}:bar"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class PolygonCSVLoader:
    """Load Polygon historical data from gzipped CSV files."""

    def __init__(
        self,
        data_dir: str,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        """
        Initialize the loader.
        
        Args:
            data_dir: Direct path to the folder containing month subdirs,
                e.g., "/home/kironix/train/poligon/Minute Aggregates"
            symbols: Filter to these symbols, or None for all
            start_date: Filter files starting from this date
            end_date: Filter files ending at this date
        """
        self.data_dir = Path(data_dir)
        self.symbols = set(symbols) if symbols else None
        self.start_date = start_date
        self.end_date = end_date

    def discover_files(self) -> list[Path]:
        """Find all .csv.gz files in data directory."""
        files = []
        if not self.data_dir.exists():
            logger.warning("data_dir_not_found", path=str(self.data_dir))
            return files
        for month_dir in sorted(self.data_dir.iterdir()):
            if month_dir.is_dir():
                for csv_file in sorted(month_dir.glob("*.csv.gz")):
                    file_date = self._parse_file_date(csv_file.name)
                    if file_date:
                        if self.start_date and file_date < self.start_date.date():
                            continue
                        if self.end_date and file_date > self.end_date.date():
                            continue
                        files.append(csv_file)
        logger.info("discovered_files", count=len(files), dir=str(self.data_dir))
        return files

    def _parse_file_date(self, filename: str) -> datetime.date | None:
        """Parse date from filename like 2025-07-01.csv.gz."""
        try:
            date_str = filename.replace(".csv.gz", "")
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None

    def _parse_timestamp(self, ns_timestamp: str) -> datetime:
        """Parse nanosecond timestamp to datetime."""
        ns = int(ns_timestamp)
        seconds = ns / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    def load_file(self, filepath: Path) -> Iterator[PolygonBar]:
        """Load and parse a single gzipped CSV file."""
        logger.info("loading_file", path=str(filepath))
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            header = f.readline().strip()
            columns = header.split(",")
            col_idx = {col: i for i, col in enumerate(columns)}
            for line_num, line in enumerate(f, start=2):
                try:
                    parts = line.strip().split(",")
                    ticker = parts[col_idx["ticker"]]
                    if self.symbols and ticker not in self.symbols:
                        continue
                    ts = self._parse_timestamp(parts[col_idx["window_start"]])
                    yield PolygonBar(
                        ticker=ticker,
                        open=float(parts[col_idx["open"]]),
                        high=float(parts[col_idx["high"]]),
                        low=float(parts[col_idx["low"]]),
                        close=float(parts[col_idx["close"]]),
                        volume=float(parts[col_idx["volume"]]),
                        vwap=0.0,
                        transactions=int(parts[col_idx["transactions"]]),
                        timestamp=ts,
                    )
                except (ValueError, IndexError, KeyError) as e:
                    logger.warning(
                        "parse_error",
                        file=str(filepath),
                        line=line_num,
                        error=str(e),
                    )
                    continue

    def load_all(self) -> Iterator[PolygonBar]:
        """Load all files and yield bars."""
        files = self.discover_files()
        total_bars = 0
        for filepath in files:
            for bar in self.load_file(filepath):
                total_bars += 1
                yield bar
        logger.info("load_complete", total_bars=total_bars)

    def load_batched(self, batch_size: int = 1000) -> Iterator[list[PolygonBar]]:
        """Load bars in batches for efficient Kafka publishing."""
        batch: list[PolygonBar] = []
        for bar in self.load_all():
            batch.append(bar)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
