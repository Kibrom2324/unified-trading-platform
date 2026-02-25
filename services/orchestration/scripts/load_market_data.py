#!/usr/bin/env python3
"""
Load Polygon market data from gzipped CSV files into TimescaleDB.

Data location: /home/kironix/train/poligon/
  - Minute Aggregates/July/*.csv.gz
  - Day Aggregates/July/*.csv.gz

CSV columns: ticker, volume, open, close, high, low, window_start, transactions
  - window_start is nanoseconds since epoch

Target tables: market_raw_minute, market_raw_day

SINGLE SOURCE OF TRUTH: config/universe.csv
  - If universe.csv exists, tickers are read from it by default
  - Override with --symbols or --all-symbols flags

Usage:
    python scripts/load_market_data.py                     # Load from universe.csv
    python scripts/load_market_data.py --minute-only       # Only minute bars
    python scripts/load_market_data.py --day-only          # Only daily bars
    python scripts/load_market_data.py --symbols AAPL,MSFT # Override with specific symbols
    python scripts/load_market_data.py --all-symbols       # Load ALL symbols (no filter)
    python scripts/load_market_data.py --dry-run           # Preview without inserting
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

# Default data paths
DATA_ROOT = Path("/home/kironix/train/poligon")
MINUTE_DIR = DATA_ROOT / "Minute Aggregates" / "July"
DAY_DIR = DATA_ROOT / "Day Aggregates" / "July"

# Universe file (SINGLE SOURCE OF TRUTH)
PROJECT_ROOT = Path(__file__).parent.parent
UNIVERSE_FILE = PROJECT_ROOT / "config" / "universe.csv"

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://awet:awet@localhost:5433/awet"
)

# Batch size for inserts
BATCH_SIZE = 5000

# Parallel loading settings
MAX_WORKERS = 4  # Number of parallel file loaders


def load_universe_symbols() -> set[str] | None:
    """Load symbols from universe.csv if it exists."""
    if not UNIVERSE_FILE.exists():
        return None
    
    symbols = set()
    with open(UNIVERSE_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ticker" in row:
                symbols.add(row["ticker"])
    
    return symbols if symbols else None


def parse_polygon_csv(
    file_path: Path,
    symbol_filter: set[str] | None = None,
) -> Iterator[tuple]:
    """Parse a gzipped Polygon CSV file and yield rows.
    
    CSV columns: ticker, volume, open, close, high, low, window_start, transactions
    
    Yields:
        Tuple of (ticker, ts, open, high, low, close, volume, transactions, vwap, idempotency_key)
    """
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        # Skip header
        header = next(f)
        expected_header = "ticker,volume,open,close,high,low,window_start,transactions"
        if header.strip() != expected_header:
            logger.warning(
                "unexpected_header",
                file=str(file_path),
                header=header.strip(),
                expected=expected_header,
            )
        
        for line_num, line in enumerate(f, start=2):
            try:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                
                ticker = parts[0]
                
                # Skip if not in filter
                if symbol_filter and ticker not in symbol_filter:
                    continue
                
                volume = float(parts[1])
                open_price = float(parts[2])
                close_price = float(parts[3])
                high_price = float(parts[4])
                low_price = float(parts[5])
                
                # window_start is nanoseconds since epoch
                ns_timestamp = int(parts[6])
                ts = datetime.fromtimestamp(ns_timestamp / 1_000_000_000, tz=timezone.utc)
                
                transactions = int(parts[7])
                
                # No VWAP in this data format
                vwap = None
                
                # Create idempotency key
                idempotency_key = f"{ticker}_{ts.isoformat()}"
                
                yield (
                    ticker,
                    ts,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    transactions,
                    vwap,
                    idempotency_key,
                )
                
            except (ValueError, IndexError) as e:
                logger.warning(
                    "parse_error",
                    file=str(file_path),
                    line=line_num,
                    error=str(e),
                )
                continue


async def load_to_table(
    conn: asyncpg.Connection,
    table: str,
    rows: list[tuple],
    dry_run: bool = False,
) -> int:
    """Insert rows into the specified table using COPY for performance."""
    if not rows:
        return 0
    
    if dry_run:
        return len(rows)
    
    # Use copy_records_to_table for high performance
    try:
        await conn.copy_records_to_table(
            table,
            records=rows,
            columns=[
                "ticker", "ts", "open", "high", "low", "close",
                "volume", "transactions", "vwap", "idempotency_key",
            ],
        )
        return len(rows)
    except asyncpg.UniqueViolationError:
        # Fall back to individual inserts with ON CONFLICT
        inserted = 0
        for row in rows:
            try:
                await conn.execute(
                    f"""
                    INSERT INTO {table} (ticker, ts, open, high, low, close, volume, transactions, vwap, idempotency_key)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (ts, ticker) DO NOTHING
                    """,
                    *row,
                )
                inserted += 1
            except asyncpg.UniqueViolationError:
                pass
        return inserted


async def load_single_file(
    file_path: Path,
    table: str,
    symbol_filter: set[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Load a single CSV file into the specified table."""
    start = time.perf_counter()
    file_rows = 0
    symbols = set()

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        batch: list[tuple] = []
        for row in parse_polygon_csv(file_path, symbol_filter):
            batch.append(row)
            symbols.add(row[0])  # ticker

            if len(batch) >= BATCH_SIZE:
                inserted = await load_to_table(conn, table, batch, dry_run)
                file_rows += inserted
                batch = []

        if batch:
            inserted = await load_to_table(conn, table, batch, dry_run)
            file_rows += inserted

    finally:
        await conn.close()

    duration = time.perf_counter() - start
    return {
        "file": file_path.name,
        "rows": file_rows,
        "symbols": symbols,
        "duration": duration,
    }


async def load_files(
    source_dir: Path,
    table: str,
    symbol_filter: set[str] | None = None,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
) -> dict[str, int]:
    """Load all CSV files from a directory into a table (parallel)."""
    stats = {"files": 0, "rows": 0, "symbols": set()}

    if not source_dir.exists():
        logger.warning("directory_not_found", path=str(source_dir))
        return stats

    csv_files = sorted(source_dir.glob("*.csv.gz"))
    if not csv_files:
        logger.warning("no_csv_files", path=str(source_dir))
        return stats

    print(f"  Found {len(csv_files)} files. Loading with {max_workers} workers...")
    sem = asyncio.Semaphore(max_workers)

    async def run_file(file_path: Path) -> dict:
        async with sem:
            logger.info("loading_file", file=file_path.name, table=table)
            return await load_single_file(file_path, table, symbol_filter, dry_run)

    tasks = [run_file(f) for f in csv_files]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        stats["files"] += 1
        stats["rows"] += result["rows"]
        stats["symbols"].update(result["symbols"])
        logger.info(
            "file_loaded",
            file=result["file"],
            rows=result["rows"],
            seconds=f"{result['duration']:.2f}",
            dry_run=dry_run,
        )

    return stats


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load Polygon market data into TimescaleDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--minute-only",
        action="store_true",
        help="Only load minute bars",
    )
    parser.add_argument(
        "--day-only",
        action="store_true",
        help="Only load daily bars",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides universe.csv)",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Load ALL symbols (ignore universe.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without inserting data",
    )
    parser.add_argument(
        "--minute-dir",
        type=str,
        default=str(MINUTE_DIR),
        help=f"Path to minute aggregates (default: {MINUTE_DIR})",
    )
    parser.add_argument(
        "--day-dir",
        type=str,
        default=str(DAY_DIR),
        help=f"Path to day aggregates (default: {DAY_DIR})",
    )
    
    args = parser.parse_args()
    
    # Determine symbol filter
    # Priority: --symbols > --all-symbols > universe.csv
    symbol_filter = None
    symbol_source = "ALL (no filter)"
    
    if args.symbols:
        symbol_filter = set(s.strip().upper() for s in args.symbols.split(","))
        symbol_source = f"CLI --symbols ({len(symbol_filter)} tickers)"
    elif args.all_symbols:
        symbol_filter = None
        symbol_source = "ALL (--all-symbols flag)"
    else:
        # Try to load from universe.csv
        universe_symbols = load_universe_symbols()
        if universe_symbols:
            symbol_filter = universe_symbols
            symbol_source = f"universe.csv ({len(symbol_filter)} tickers)"
        else:
            print("❌ ERROR: No universe.csv found.")
            print("   Run: make universe N=500 DAYS=180 MODE=daily")
            print("   Or pass --symbols / --all-symbols explicitly.")
            return 1
    
    # Configure logging (centralized — keeps correlation_id + contextvars)
    from src.core.logging import configure_logging
    configure_logging("INFO", console=True)
    
    print()
    print("=" * 60)
    print("POLYGON MARKET DATA LOADER")
    print("=" * 60)
    print(f"  Minute dir:    {args.minute_dir}")
    print(f"  Day dir:       {args.day_dir}")
    print(f"  Symbol source: {symbol_source}")
    if symbol_filter and len(symbol_filter) <= 20:
        print(f"  Symbols:       {', '.join(sorted(symbol_filter))}")
    elif symbol_filter:
        sample = list(sorted(symbol_filter))[:10]
        print(f"  Symbols:       {', '.join(sample)}... (+{len(symbol_filter)-10} more)")
    print(f"  Dry run:       {args.dry_run}")
    print()
    
    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        total_stats = {"minute": {}, "day": {}}
        
        # Load minute data
        if not args.day_only:
            print("Loading minute aggregates...")
            start_minute = time.perf_counter()
            total_stats["minute"] = await load_files(
                Path(args.minute_dir),
                "market_raw_minute",
                symbol_filter,
                args.dry_run,
            )
            total_stats["minute"]["seconds"] = time.perf_counter() - start_minute
        
        # Load day data
        if not args.minute_only:
            print("\nLoading day aggregates...")
            start_day = time.perf_counter()
            total_stats["day"] = await load_files(
                Path(args.day_dir),
                "market_raw_day",
                symbol_filter,
                args.dry_run,
            )
            total_stats["day"]["seconds"] = time.perf_counter() - start_day
        
        # Print summary
        print()
        print("=" * 60)
        print("LOAD COMPLETE" + (" (DRY RUN)" if args.dry_run else ""))
        print("=" * 60)
        
        if total_stats["minute"]:
            m = total_stats["minute"]
            print(f"  Minute bars:")
            print(f"    Files:   {m.get('files', 0)}")
            print(f"    Rows:    {m.get('rows', 0):,}")
            print(f"    Symbols: {len(m.get('symbols', set()))}")
            if "seconds" in m:
                print(f"    Time:    {m['seconds']:.1f}s")
        
        if total_stats["day"]:
            d = total_stats["day"]
            print(f"  Day bars:")
            print(f"    Files:   {d.get('files', 0)}")
            print(f"    Rows:    {d.get('rows', 0):,}")
            print(f"    Symbols: {len(d.get('symbols', set()))}")
            if "seconds" in d:
                print(f"    Time:    {d['seconds']:.1f}s")
        
        print()
        
        # Verify data in DB
        if not args.dry_run:
            minute_count = await conn.fetchval("SELECT COUNT(*) FROM market_raw_minute")
            day_count = await conn.fetchval("SELECT COUNT(*) FROM market_raw_day")
            print(f"  Database totals:")
            print(f"    market_raw_minute: {minute_count:,} rows")
            print(f"    market_raw_day:    {day_count:,} rows")
            print()
        
        return 0
        
    finally:
        await conn.close()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
