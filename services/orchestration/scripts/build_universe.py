#!/usr/bin/env python3
"""
Build trading universe based on liquidity and data coverage.

Reads from: market_raw_minute or market_raw_day (TimescaleDB)
Outputs:    config/universe.csv (SINGLE SOURCE OF TRUTH)

Liquidity score: AVG(close * volume) over the lookback period
Coverage filter: Minimum % of trading days with data

Filters:
  - Coverage >= 90% (configurable)
  - Avg price >= $5
  - Avg volume >= 200,000 (for daily data) or 10,000 (for minute data)
  - Excludes tickers containing '.' (classes like BRK.A)
  - Excludes tickers ending with 'WS' (warrants)

Usage:
    python scripts/build_universe.py                           # Top 100, 30 days, minute
    python scripts/build_universe.py --top 500 --days 60 --mode daily
    python scripts/build_universe.py --top 1000 --days 90 --mode daily
    python scripts/build_universe.py --min-coverage 0.9         # 90% day coverage
    python scripts/build_universe.py --mode auto                # fallback to daily if minute sparse
    python scripts/build_universe.py --dry-run                  # Preview only
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://awet:awet@localhost:5433/awet"
)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "config" / "universe.csv"

# Default parameters - optimized for scale
DEFAULT_TOP_N = 500
DEFAULT_DAYS = 180
DEFAULT_MIN_COVERAGE = 0.90  # 90% of trading days
DEFAULT_MIN_AVG_VOLUME = 200_000  # Minimum average daily volume
DEFAULT_MIN_PRICE = 5.0  # Minimum average price

# Ticker exclusion patterns
EXCLUDED_PATTERNS = [
    re.compile(r'\.'),      # Contains dot (e.g., BRK.A, BRK.B)
    re.compile(r'WS$'),     # Ends with WS (warrants)
    re.compile(r'\.U$'),    # Unit securities
    re.compile(r'\.W$'),    # Warrant securities
    re.compile(r'\.R$'),    # Rights
    re.compile(r'-UN$'),    # Units
    re.compile(r'-WT$'),    # Warrants
]


def is_valid_ticker(ticker: str) -> bool:
    """Check if ticker passes exclusion filters."""
    if not ticker or len(ticker) > 5:
        return False
    for pattern in EXCLUDED_PATTERNS:
        if pattern.search(ticker):
            return False
    return True


async def build_universe(
    top_n: int = DEFAULT_TOP_N,
    days: int = DEFAULT_DAYS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_avg_volume: float = DEFAULT_MIN_AVG_VOLUME,
    min_price: float = DEFAULT_MIN_PRICE,
    mode: str = "daily",
    dry_run: bool = False,
) -> list[dict]:
    """Build universe from market data.
    
    Args:
        top_n: Number of top symbols to select
        days: Lookback period in days
        min_coverage: Minimum fraction of trading days with data
        min_avg_volume: Minimum average daily volume (auto-detect if None)
        min_price: Minimum average price
        use_daily: Use daily data instead of minute data
        dry_run: Preview only, don't write file
        
    Returns:
        List of selected symbols with metrics
    """
    conn = await asyncpg.connect(DATABASE_URL)
    
    # Track rejection reasons for summary
    rejections = {
        "low_coverage": 0,
        "low_volume": 0,
        "low_price": 0,
        "excluded_pattern": 0,
    }
    rejection_samples: dict[str, list[str]] = {k: [] for k in rejections}
    
    def normalize_mode(value: str) -> str:
        value = (value or "").lower().strip()
        return value if value in {"minute", "daily", "auto"} else "minute"

    async def print_table_ranges() -> None:
        for table in ("market_raw_minute", "market_raw_day"):
            stats = await conn.fetchrow(
                f"""
                SELECT COUNT(*) AS rows,
                       MIN(ts::date) AS min_day,
                       MAX(ts::date) AS max_day
                FROM {table}
                """
            )
            if stats and stats["rows"]:
                print(f"  {table}: {stats['min_day']} -> {stats['max_day']} ({stats['rows']:,} rows)")
            else:
                print(f"  {table}: no data")

    async def run_for_table(table: str) -> tuple[list, dict]:
        # Get date range
        date_range = await conn.fetchrow(f"""
            SELECT 
                MIN(ts::date) as min_date,
                MAX(ts::date) as max_date,
                COUNT(DISTINCT ts::date) as trading_days
            FROM {table}
        """)

        if not date_range or date_range["trading_days"] == 0:
            return [], {"date_range": date_range, "table": table}

        expected_days = min(days, date_range["trading_days"])
        min_required_days = int(expected_days * min_coverage)

        query = f"""
            WITH date_bounds AS (
                SELECT 
                    MAX(ts::date) as max_date,
                    MAX(ts::date) - INTERVAL '{days} days' as min_date
                FROM {table}
            ),
            ticker_stats AS (
                SELECT 
                    ticker,
                    COUNT(DISTINCT ts::date) as days_with_data,
                    AVG(close) as avg_price,
                    AVG(volume) as avg_volume,
                    AVG(close * volume) as avg_dollar_volume,
                    MIN(ts) as first_seen,
                    MAX(ts) as last_seen
                FROM {table}, date_bounds
                WHERE ts::date >= date_bounds.min_date
                GROUP BY ticker
            )
            SELECT 
                ticker,
                days_with_data,
                avg_price,
                avg_volume,
                avg_dollar_volume,
                first_seen,
                last_seen,
                days_with_data::float / {expected_days} as coverage
            FROM ticker_stats
            ORDER BY avg_dollar_volume DESC
        """

        all_rows = await conn.fetch(query)
        meta = {
            "date_range": date_range,
            "table": table,
            "expected_days": expected_days,
            "min_required_days": min_required_days,
        }
        return all_rows, meta

    try:
        mode = normalize_mode(mode)
        tables_to_try = ["market_raw_minute"] if mode == "minute" else ["market_raw_day"]
        if mode == "auto":
            tables_to_try = ["market_raw_minute", "market_raw_day"]

        rows = []
        total_candidates = 0
        chosen_table = None
        date_range = None
        expected_days = None
        min_required_days = None

        for table in tables_to_try:
            all_rows, meta = await run_for_table(table)
            date_range = meta.get("date_range")
            expected_days = meta.get("expected_days")
            min_required_days = meta.get("min_required_days")

            if not date_range or date_range["trading_days"] == 0:
                continue

            # Filter candidates and track rejections
            total_candidates = len(all_rows)
            filtered_rows = []
            for row in all_rows:
                ticker = row["ticker"]

                if not is_valid_ticker(ticker):
                    rejections["excluded_pattern"] += 1
                    if len(rejection_samples["excluded_pattern"]) < 10:
                        rejection_samples["excluded_pattern"].append(ticker)
                    continue

                if row["days_with_data"] < min_required_days:
                    rejections["low_coverage"] += 1
                    if len(rejection_samples["low_coverage"]) < 10:
                        rejection_samples["low_coverage"].append(ticker)
                    continue

                if row["avg_volume"] < min_avg_volume:
                    rejections["low_volume"] += 1
                    if len(rejection_samples["low_volume"]) < 10:
                        rejection_samples["low_volume"].append(ticker)
                    continue

                if row["avg_price"] < min_price:
                    rejections["low_price"] += 1
                    if len(rejection_samples["low_price"]) < 10:
                        rejection_samples["low_price"].append(ticker)
                    continue

                filtered_rows.append(row)

            if filtered_rows:
                rows = filtered_rows
                chosen_table = table
                break

            # Auto-mode fallback if minute bars are too sparse
            if mode == "auto":
                continue

        if not date_range or date_range["trading_days"] == 0:
            print("❌ No data found in market tables")
            return []

        print("Data ranges by table:")
        await print_table_ranges()
        print()
        print(f"Selected table range: {date_range['min_date']} to {date_range['max_date']}")
        print(f"Trading days: {date_range['trading_days']}")

        print(f"Lookback: {days} days, Min coverage: {min_coverage*100:.0f}%")
        print(f"Minimum required days: {min_required_days}")
        print(f"Min avg volume: {min_avg_volume:,.0f}, Min avg price: ${min_price:.2f}")
        print(f"Mode: {mode} (selected: {chosen_table or 'none'})")
        print()

        if not rows:
            print("❌ No symbols meet the criteria")
            print()
            print("Rejection summary (top reasons):")
            for reason, count in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    samples = ", ".join(rejection_samples[reason])
                    sample_text = f" (e.g., {samples})" if samples else ""
                    print(f"  {reason}: {count}{sample_text}")
            print()
            if mode != "daily":
                print("Suggestion: try MODE=daily or lower --min-volume / --min-coverage")
            else:
                print("Suggestion: lower --min-volume / --min-coverage")
            return []
        
        # Build results from filtered candidates
        results = []
        for i, row in enumerate(rows[:top_n]):
            results.append({
                "rank": i + 1,
                "ticker": row["ticker"],
                "avg_dollar_volume": float(row["avg_dollar_volume"]),
                "avg_price": float(row["avg_price"]),
                "avg_volume": float(row["avg_volume"]),
                "days_with_data": row["days_with_data"],
                "coverage": float(row["coverage"]),
            })
        
        # Print summary
        print("=" * 70)
        print(f"UNIVERSE SELECTION: Top {len(results)} by Liquidity")
        print("=" * 70)
        print()
        print(f"{'Rank':<6}{'Ticker':<8}{'Avg $Vol':>15}{'Avg Price':>12}{'Coverage':>10}")
        print("-" * 70)
        
        for r in results[:20]:  # Show top 20
            print(f"{r['rank']:<6}{r['ticker']:<8}{r['avg_dollar_volume']:>15,.0f}"
                  f"{r['avg_price']:>12,.2f}{r['coverage']*100:>9.1f}%")
        
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")
        
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Total candidates in database:  {total_candidates}")
        print(f"  Rejected (pattern exclusion):  {rejections['excluded_pattern']}")
        print(f"  Rejected (low coverage):       {rejections['low_coverage']}")
        print(f"  Rejected (low volume):         {rejections['low_volume']}")
        print(f"  Rejected (low price):          {rejections['low_price']}")
        print(f"  Passed all filters:            {len(rows)}")
        print(f"  Selected for universe:         {len(results)}")
        if results:
            print(f"  First 20 tickers:              {', '.join([r['ticker'] for r in results[:20]])}")
            print("  Top 10 by liquidity:")
            for r in results[:10]:
                print(f"    {r['ticker']}: ${r['avg_dollar_volume']:,.0f}")
        
        if not dry_run:
            # Write CSV
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "rank", "ticker", "avg_dollar_volume", "avg_price",
                    "avg_volume", "days_with_data", "coverage"
                ])
                writer.writeheader()
                writer.writerows(results)
            
            print()
            print(f"✅ Universe saved to: {OUTPUT_FILE}")
            
            # Also print as comma-separated for Makefile
            tickers = ",".join([r["ticker"] for r in results])
            print()
            print("For Makefile TRAIN_SYMBOLS:")
            print(f"  {tickers[:200]}{'...' if len(tickers) > 200 else ''}")
        else:
            print()
            print("🔍 DRY RUN - no file written")
        
        return results
        
    finally:
        await conn.close()


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build trading universe by liquidity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top symbols to select (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Lookback period in days (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=f"Minimum data coverage 0-1 (default: {DEFAULT_MIN_COVERAGE})",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_MIN_AVG_VOLUME,
        help=f"Minimum avg volume (default: {DEFAULT_MIN_AVG_VOLUME:,.0f})",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=DEFAULT_MIN_PRICE,
        help=f"Minimum average price (default: {DEFAULT_MIN_PRICE})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="daily",
        help="Data mode: minute, daily, or auto (default: daily)",
    )
    parser.add_argument(
        "--use-daily",
        action="store_true",
        help="Use daily data (deprecated: use --mode daily)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, don't write file",
    )
    
    args = parser.parse_args()
    
    # Volume threshold
    min_volume = args.min_volume
    
    # Configure logging (centralized — keeps correlation_id + contextvars)
    from src.core.logging import configure_logging
    configure_logging("INFO", console=True)
    
    print()
    print("=" * 70)
    print("AWET UNIVERSE BUILDER")
    print("=" * 70)
    print(f"  Top N:        {args.top}")
    print(f"  Lookback:     {args.days} days")
    print(f"  Min coverage: {args.min_coverage*100:.0f}%")
    print(f"  Min volume:   {min_volume:,.0f}")
    print(f"  Min price:    ${args.min_price:.2f}")
    mode = "daily" if args.use_daily else args.mode
    print(f"  Mode:        {mode}")
    print()
    
    results = await build_universe(
        top_n=args.top,
        days=args.days,
        min_coverage=args.min_coverage,
        min_avg_volume=min_volume,
        min_price=args.min_price,
        mode=mode,
        dry_run=args.dry_run,
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
