#!/usr/bin/env python3
"""
Build TFT features from market_raw_minute data.

Reads from: market_raw_minute (TimescaleDB)
Writes to:  features_tft (TimescaleDB)

Features computed:
- price, open, high, low, close, volume
- returns_1, returns_5, returns_15 (log returns)
- target_return (future return for prediction target)
- volatility_5, volatility_15 (rolling std of returns)
- sma_5, sma_20 (simple moving average)
- ema_5, ema_20 (exponential moving average)
- rsi_14 (relative strength index)
- volume_zscore (volume normalized by rolling mean/std)
- minute_of_day, hour_of_day, day_of_week (temporal features)
- reddit_mentions (daily reddit mention count, optional)

Usage:
    python scripts/build_features.py                   # Build for all symbols
    python scripts/build_features.py --symbols AAPL,MSFT
    python scripts/build_features.py --horizon 30      # 30-minute prediction target
    python scripts/build_features.py --validate        # Run validation checks
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://awet:awet@localhost:5433/awet"
)

REQUIRE_REDDIT = os.getenv("REQUIRE_REDDIT", "0") == "1"
MIN_REDDIT_COVERAGE_PCT = float(os.getenv("MIN_REDDIT_COVERAGE_PCT", "0.05"))
REDDIT_FEATURE_COLUMNS = [
    "reddit_mentions_count",
    "reddit_sentiment_mean",
    "reddit_sentiment_weighted",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
]

PROJECT_ROOT = Path(__file__).parent.parent
UNIVERSE_FILE = PROJECT_ROOT / "config" / "universe.csv"

# Feature parameters
DEFAULT_HORIZON = 30  # minutes ahead for target
LOOKBACK_WINDOW = 120  # minimum rows needed for features
BATCH_SIZE = 10000


def load_universe_symbols() -> list[str] | None:
    """Load symbols from universe.csv if it exists."""
    if not UNIVERSE_FILE.exists():
        return None

    symbols = []
    with open(UNIVERSE_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ticker" in row:
                symbols.append(row["ticker"])

    return symbols if symbols else None


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Use exponential moving average for gains/losses
    alpha = 1 / period
    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)
    
    avg_gains[0] = gains[0]
    avg_losses[0] = losses[0]
    
    for i in range(1, len(gains)):
        avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]
    
    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average."""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema


def compute_features(
    rows: list[tuple],
    horizon: int = DEFAULT_HORIZON,
    reddit_by_date: dict | None = None,
) -> list[tuple]:
    """Compute features for a single symbol's time series.
    
    Args:
        rows: List of (ticker, ts, open, high, low, close, volume) tuples
        horizon: Minutes ahead for target return
        
    Returns:
        List of feature tuples ready for DB insert
    """
    if len(rows) < LOOKBACK_WINDOW:
        return []
    
    # Extract arrays
    ticker = rows[0][0]
    timestamps = np.array([r[1] for r in rows])
    opens = np.array([float(r[2]) for r in rows])
    highs = np.array([float(r[3]) for r in rows])
    lows = np.array([float(r[4]) for r in rows])
    closes = np.array([float(r[5]) for r in rows])
    volumes = np.array([float(r[6]) for r in rows])
    
    n = len(rows)
    features = []
    
    # Compute derived arrays
    log_returns_1 = np.zeros(n)
    log_returns_1[1:] = np.log(closes[1:] / closes[:-1])
    
    log_returns_5 = np.zeros(n)
    log_returns_5[5:] = np.log(closes[5:] / closes[:-5])
    
    log_returns_15 = np.zeros(n)
    log_returns_15[15:] = np.log(closes[15:] / closes[:-15])
    
    # Target return (future)
    target_returns = np.zeros(n)
    target_returns[:-horizon] = np.log(closes[horizon:] / closes[:-horizon])
    
    # Rolling volatility
    vol_5 = np.zeros(n)
    vol_15 = np.zeros(n)
    for i in range(5, n):
        vol_5[i] = np.std(log_returns_1[i-4:i+1])
    for i in range(15, n):
        vol_15[i] = np.std(log_returns_1[i-14:i+1])
    
    # Moving averages
    sma_5 = np.convolve(closes, np.ones(5)/5, mode='same')
    sma_20 = np.convolve(closes, np.ones(20)/20, mode='same')
    
    ema_5 = compute_ema(closes, 5)
    ema_20 = compute_ema(closes, 20)
    
    # RSI
    rsi_14 = compute_rsi(closes, 14)
    
    # Volume z-score (rolling 20-period)
    vol_zscore = np.zeros(n)
    for i in range(20, n):
        window = volumes[i-19:i+1]
        mean_vol = np.mean(window)
        std_vol = np.std(window)
        if std_vol > 0:
            vol_zscore[i] = (volumes[i] - mean_vol) / std_vol
    
    # Build feature rows (skip first LOOKBACK_WINDOW rows)
    for i in range(LOOKBACK_WINDOW, n - horizon):
        ts = timestamps[i]
        
        # Temporal features
        minute_of_day = ts.hour * 60 + ts.minute
        hour_of_day = ts.hour
        day_of_week = ts.weekday()
        
        # Reddit daily signals (optional)
        reddit_mentions_count = 0
        reddit_sentiment_mean = 0.0
        reddit_sentiment_weighted = 0.0
        reddit_positive_ratio = 0.0
        reddit_negative_ratio = 0.0
        if reddit_by_date is not None:
            reddit_row = reddit_by_date.get(ts.date(), {})
            reddit_mentions_count = int(reddit_row.get("reddit_mentions_count", 0))
            reddit_sentiment_mean = float(reddit_row.get("reddit_sentiment_mean", 0.0))
            reddit_sentiment_weighted = float(reddit_row.get("reddit_sentiment_weighted", 0.0))
            reddit_positive_ratio = float(reddit_row.get("reddit_positive_ratio", 0.0))
            reddit_negative_ratio = float(reddit_row.get("reddit_negative_ratio", 0.0))

        # Create idempotency key
        idempotency_key = f"feat_{ticker}_{ts.isoformat()}"
        
        # Split: first 80% train, last 20% val
        split_idx = int(0.8 * (n - LOOKBACK_WINDOW - horizon))
        split = "train" if (i - LOOKBACK_WINDOW) < split_idx else "val"
        
        # Time index for TFT
        time_idx = i - LOOKBACK_WINDOW
        
        feature_row = (
            ticker,                          # ticker
            ts,                              # ts
            time_idx,                        # time_idx
            split,                           # split
            float(closes[i]),                # price
            float(opens[i]),                 # open
            float(highs[i]),                 # high
            float(lows[i]),                  # low
            float(closes[i]),                # close
            float(volumes[i]),               # volume
            float(log_returns_1[i]),         # returns_1
            float(log_returns_5[i]),         # returns_5
            float(log_returns_15[i]),        # returns_15
            float(target_returns[i]),        # target_return
            float(vol_5[i]),                 # volatility_5
            float(vol_15[i]),                # volatility_15
            float(sma_5[i]),                 # sma_5
            float(sma_20[i]),                # sma_20
            float(ema_5[i]),                 # ema_5
            float(ema_20[i]),                # ema_20
            float(rsi_14[i]),                # rsi_14
            float(vol_zscore[i]),            # volume_zscore
            minute_of_day,                   # minute_of_day
            hour_of_day,                     # hour_of_day
            day_of_week,                     # day_of_week
            reddit_mentions_count,           # reddit_mentions_count
            reddit_sentiment_mean,           # reddit_sentiment_mean
            reddit_sentiment_weighted,       # reddit_sentiment_weighted
            reddit_positive_ratio,           # reddit_positive_ratio
            reddit_negative_ratio,           # reddit_negative_ratio
            idempotency_key,                 # idempotency_key
        )
        features.append(feature_row)
    
    return features


async def table_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    """Check if a table exists in the public schema."""
    exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = $1
        )
        """,
        table_name,
    )
    return bool(exists)


async def column_exists(conn: asyncpg.Connection, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
        )
        """,
        table_name,
        column_name,
    )
    return bool(exists)


async def print_data_lineage_summary(conn: asyncpg.Connection) -> None:
    """Print explicit data lineage summary for market + reddit tables."""
    market_stats = await conn.fetchrow(
        """
        SELECT COUNT(*) AS rows,
               MIN(ts) AS min_ts,
               MAX(ts) AS max_ts
        FROM market_raw_minute
        """
    )

    reddit_exists = await table_exists(conn, "reddit_daily_mentions")
    reddit_stats = None
    if reddit_exists:
        reddit_stats = await conn.fetchrow(
            """
            SELECT COUNT(*) AS rows,
                   MIN(day) AS min_day,
                   MAX(day) AS max_day
            FROM reddit_daily_mentions
            """
        )

    print("Data lineage summary:")
    print("  Market table:   market_raw_minute")
    print("    Columns:      ticker, ts, open, high, low, close, volume")
    print(f"    Rows:         {market_stats['rows']:,}"
          if market_stats and market_stats["rows"] is not None else "    Rows:         0")
    print(f"    Time range:   {market_stats['min_ts']} -> {market_stats['max_ts']}")

    if reddit_exists:
        print("  Reddit table:   reddit_daily_mentions")
        print("    Columns:      ticker, day, mentions/mentions_count, sentiment_mean, sentiment_weighted, positive_ratio, negative_ratio")
        print(f"    Rows:         {reddit_stats['rows']:,}"
              if reddit_stats and reddit_stats["rows"] is not None else "    Rows:         0")
        print(f"    Date range:   {reddit_stats['min_day']} -> {reddit_stats['max_day']}")
    else:
        print("  Reddit table:   reddit_daily_mentions (MISSING)")

    if reddit_exists:
        overlap = await conn.fetchrow(
            """
            WITH market_days AS (
                SELECT DISTINCT ticker, ts::date AS day
                FROM market_raw_minute
            ),
            reddit_days AS (
                SELECT DISTINCT ticker, day
                FROM reddit_daily_mentions
            )
            SELECT
                (SELECT COUNT(*) FROM market_days) AS market_days,
                (SELECT COUNT(*) FROM reddit_days) AS reddit_days,
                (SELECT COUNT(DISTINCT ticker) FROM market_days) AS market_tickers,
                (SELECT COUNT(DISTINCT ticker) FROM reddit_days) AS reddit_tickers,
                (SELECT COUNT(DISTINCT market_days.ticker) FROM market_days
                 JOIN reddit_days USING (ticker)) AS overlap_tickers,
                (SELECT COUNT(*) FROM market_days
                 JOIN reddit_days USING (ticker, day)) AS overlap_days
            """
        )

        market_days = int(overlap["market_days"] or 0)
        overlap_days = int(overlap["overlap_days"] or 0)
        coverage = (overlap_days / market_days) if market_days else 0.0

        print("  Join alignment:")
        print("    Join keys:    market_raw_minute.ticker + date(ts) = reddit_daily_mentions.ticker + day")
        print("    Join SQL:")
        print("      SELECT m.ticker, m.ts, r.mentions_count, r.sentiment_mean, r.sentiment_weighted,")
        print("             r.positive_ratio, r.negative_ratio")
        print("      FROM market_raw_minute m")
        print("      LEFT JOIN reddit_daily_mentions r")
        print("        ON r.ticker = m.ticker AND r.day = DATE(m.ts)")
        print(f"    Overlap days: {overlap_days:,}")
        print(f"    Overlap tickers: {overlap['overlap_tickers']}")
        print(f"    Coverage:     {coverage:.2%}")

        if overlap_days == 0:
            if REQUIRE_REDDIT:
                raise RuntimeError("REDDIT_NOT_USED: no overlap between market and reddit")
            print("  REDDIT NOT USED: no overlap between market and reddit")
        elif coverage < MIN_REDDIT_COVERAGE_PCT:
            print(f"  ⚠️  LOW REDDIT COVERAGE: {coverage:.2%} < {MIN_REDDIT_COVERAGE_PCT:.0%}")

    print()


async def load_reddit_mentions_by_date(
    conn: asyncpg.Connection,
    symbol: str,
) -> dict:
    """Load daily reddit signals for a symbol (by day)."""
    has_mentions = await column_exists(conn, "reddit_daily_mentions", "mentions")
    has_mentions_count = await column_exists(conn, "reddit_daily_mentions", "mentions_count")
    has_sentiment_mean = await column_exists(conn, "reddit_daily_mentions", "sentiment_mean")
    has_sentiment_weighted = await column_exists(conn, "reddit_daily_mentions", "sentiment_weighted")
    has_positive_ratio = await column_exists(conn, "reddit_daily_mentions", "positive_ratio")
    has_negative_ratio = await column_exists(conn, "reddit_daily_mentions", "negative_ratio")

    select_parts = ["day"]
    if has_mentions_count:
        select_parts.append("mentions_count")
    elif has_mentions:
        select_parts.append("mentions")
    if has_sentiment_mean:
        select_parts.append("sentiment_mean")
    if has_sentiment_weighted:
        select_parts.append("sentiment_weighted")
    if has_positive_ratio:
        select_parts.append("positive_ratio")
    if has_negative_ratio:
        select_parts.append("negative_ratio")

    query = f"""
        SELECT {', '.join(select_parts)}
        FROM reddit_daily_mentions
        WHERE ticker = $1
    """
    rows = await conn.fetch(query, symbol)

    by_date: dict = {}
    for r in rows:
        mentions = r.get("mentions_count")
        if mentions is None:
            mentions = r.get("mentions")
        by_date[r["day"]] = {
            "reddit_mentions_count": int(mentions or 0),
            "reddit_sentiment_mean": float(r.get("sentiment_mean") or 0.0),
            "reddit_sentiment_weighted": float(r.get("sentiment_weighted") or 0.0),
            "reddit_positive_ratio": float(r.get("positive_ratio") or 0.0),
            "reddit_negative_ratio": float(r.get("negative_ratio") or 0.0),
        }
    return by_date


async def build_features_for_symbol(
    conn: asyncpg.Connection,
    symbol: str,
    horizon: int = DEFAULT_HORIZON,
) -> int:
    """Build features for a single symbol."""
    # Fetch raw data ordered by timestamp
    rows = await conn.fetch(
        """
        SELECT ticker, ts, open, high, low, close, volume
        FROM market_raw_minute
        WHERE ticker = $1
        ORDER BY ts ASC
        """,
        symbol,
    )
    
    if not rows:
        return 0
    
    logger.info("computing_features", symbol=symbol, raw_rows=len(rows))
    
    # Load reddit daily signals if available
    reddit_by_date = None
    if await table_exists(conn, "reddit_daily_mentions"):
        reddit_by_date = await load_reddit_mentions_by_date(conn, symbol)

    # Guard: ensure features_tft has required reddit_* columns
    missing_cols = []
    for col in REDDIT_FEATURE_COLUMNS:
        if not await column_exists(conn, "features_tft", col):
            missing_cols.append(col)
    if missing_cols:
        raise RuntimeError(
            "features_tft missing required reddit columns: " + ", ".join(missing_cols)
        )

    # Compute features
    features = compute_features(list(rows), horizon, reddit_by_date)
    
    if not features:
        logger.warning("no_features", symbol=symbol, reason="insufficient data")
        return 0
    
    # Insert in batches
    inserted = 0
    for i in range(0, len(features), BATCH_SIZE):
        batch = features[i:i + BATCH_SIZE]
        
        # Use executemany with ON CONFLICT
        await conn.executemany(
            """
            INSERT INTO features_tft (
                ticker, ts, time_idx, split, price, open, high, low, close, volume,
                returns_1, returns_5, returns_15, target_return,
                volatility_5, volatility_15, sma_5, sma_20, ema_5, ema_20,
                rsi_14, volume_zscore, minute_of_day, hour_of_day, day_of_week,
                reddit_mentions_count, reddit_sentiment_mean, reddit_sentiment_weighted,
                reddit_positive_ratio, reddit_negative_ratio, idempotency_key
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31
            )
            ON CONFLICT (ts, ticker) DO NOTHING
            """,
            batch,
        )
        inserted += len(batch)
    
    logger.info("features_inserted", symbol=symbol, count=inserted)
    return inserted


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build TFT features from market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides universe.csv)",
        default=None,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Prediction horizon in minutes (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks after building",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Process all symbols in market_raw_minute (slow!)",
    )
    
    args = parser.parse_args()
    
    # Configure logging (centralized — keeps correlation_id + contextvars)
    from src.core.logging import configure_logging
    configure_logging("INFO", console=True)
    
    print()
    print("=" * 60)
    print("TFT FEATURE BUILDER")
    print("=" * 60)
    print(f"  Horizon:    {args.horizon} minutes")
    print(f"  Lookback:   {LOOKBACK_WINDOW} rows minimum")
    print()
    
    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        await print_data_lineage_summary(conn)

        # Ensure reddit_mentions column exists (minimal v1)
        await conn.execute(
            "ALTER TABLE features_tft ADD COLUMN IF NOT EXISTS reddit_mentions INTEGER DEFAULT 0"
        )

        # Get symbols
        if args.all_symbols:
            symbols = await conn.fetch(
                "SELECT DISTINCT ticker FROM market_raw_minute ORDER BY ticker"
            )
            symbols = [r["ticker"] for r in symbols]
            print(f"  Processing ALL {len(symbols)} symbols...")
        else:
            if args.symbols:
                symbols = [s.strip().upper() for s in args.symbols.split(",")]
                print(f"  Symbols:    {symbols}")
            else:
                universe_symbols = load_universe_symbols()
                if universe_symbols:
                    symbols = universe_symbols
                    print(f"  Symbols:    {len(symbols)} from universe.csv")
                else:
                    print("❌ ERROR: No universe.csv found.")
                    print("   Run: make universe N=500 DAYS=180 MODE=daily")
                    print("   Or pass --symbols / --all-symbols explicitly.")
                    return 1
        
        print()
        
        # Check if we have raw data
        total_raw = await conn.fetchval("SELECT COUNT(*) FROM market_raw_minute")
        if total_raw == 0:
            print("❌ ERROR: No data in market_raw_minute!")
            print("   Run: python scripts/load_market_data.py first")
            return 1
        
        print(f"  Raw data:   {total_raw:,} rows in market_raw_minute")
        print()
        
        # Build features for each symbol
        total_features = 0
        for symbol in symbols:
            count = await build_features_for_symbol(conn, symbol, args.horizon)
            total_features += count
        
        # Summary
        print()
        print("=" * 60)
        print("FEATURE BUILD COMPLETE")
        print("=" * 60)
        print(f"  Total features: {total_features:,}")

        # Reddit coverage proof
        reddit_exists = await table_exists(conn, "reddit_daily_mentions")
        if not reddit_exists:
            print("  Reddit table:   reddit_daily_mentions (MISSING)")
            if REQUIRE_REDDIT:
                print("❌ REDDIT NOT USED: table missing")
                return 1
            print("  REDDIT NOT USED: table missing")
        else:
            reddit_rows = await conn.fetchval(
                "SELECT COUNT(*) FROM reddit_daily_mentions WHERE ticker = ANY($1)",
                symbols,
            )
            total_feat_rows = await conn.fetchval(
                "SELECT COUNT(*) FROM features_tft WHERE ticker = ANY($1)",
                symbols,
            )
            reddit_nonzero = await conn.fetchval(
                """
                SELECT COUNT(*) FROM features_tft
                WHERE ticker = ANY($1) AND reddit_mentions > 0
                """,
                symbols,
            )
            coverage = (reddit_nonzero / total_feat_rows) if total_feat_rows else 0.0

            print("  Reddit tables used: reddit_daily_mentions")
            print(f"  Reddit rows loaded: {reddit_rows:,}")
            print(f"  Reddit feature columns: {', '.join(REDDIT_FEATURE_COLUMNS)}")
            print(f"  Reddit non-null coverage: {coverage:.2%}")

            if coverage < MIN_REDDIT_COVERAGE_PCT:
                print("  ⚠️  WARNING: REDDIT NOT CONTRIBUTING")
                if REQUIRE_REDDIT:
                    print("❌ REDDIT NOT USED: coverage below threshold")
                    return 1
        
        # Validation
        if args.validate or True:  # Always validate
            print()
            print("Validation:")
            
            # Check for NaNs
            nan_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM features_tft
                WHERE returns_1 IS NULL OR volatility_5 IS NULL OR rsi_14 IS NULL
                """
            )
            if nan_count > 0:
                print(f"  ⚠️  Rows with NULL values: {nan_count}")
            else:
                print("  ✅ No NULL values in key columns")
            
            # Check train/val split
            train_count = await conn.fetchval(
                "SELECT COUNT(*) FROM features_tft WHERE split = 'train'"
            )
            val_count = await conn.fetchval(
                "SELECT COUNT(*) FROM features_tft WHERE split = 'val'"
            )
            print(f"  ✅ Train rows: {train_count:,}")
            print(f"  ✅ Val rows:   {val_count:,}")
            
            # Check symbols
            symbol_counts = await conn.fetch(
                """
                SELECT ticker, COUNT(*) as cnt
                FROM features_tft
                GROUP BY ticker
                ORDER BY cnt DESC
                """
            )
            print(f"  ✅ Symbols with features: {len(symbol_counts)}")
            for row in symbol_counts[:5]:
                print(f"     {row['ticker']}: {row['cnt']:,} rows")
        
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
