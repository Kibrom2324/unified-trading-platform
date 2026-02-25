#!/usr/bin/env python3
"""
Feature engineering wrapper script.

Reads from: market_raw_day or market_raw_minute (TimescaleDB)
Writes to:  features_tft (TimescaleDB)

This is a thin wrapper around build_features.py for CLI consistency.

Features computed:
- Price/volume features
- Technical indicators (SMA, EMA, RSI, volatility)
- Calendar features (is_weekend, day_of_month, week_of_year, month_of_year, is_month_end)
- Reddit sentiment with lag features (lag1, lag3, lag5)
- Market benchmark context (SPY/QQQ returns/volatility)

Usage:
    python scripts/feature_engineer.py --symbols AAPL --start 2024-01-01 --end 2024-01-31
    python scripts/feature_engineer.py --symbols AAPL,MSFT --source day
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import asyncpg
import numpy as np
import structlog

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logging import configure_logging

logger = structlog.get_logger("feature_engineer")

# Market benchmark symbols for context
MARKET_BENCHMARKS = ["SPY", "QQQ"]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.astimezone(timezone.utc)


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    alpha = 1 / period
    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)
    avg_gains[0] = gains[0]
    avg_losses[0] = losses[0]

    for i in range(1, len(gains)):
        avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
        avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]

    # Safe division avoiding divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    return 100 - (100 / (1 + rs))


def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average."""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_calendar_features(ts: datetime) -> Dict[str, int]:
    """Compute calendar features from timestamp."""
    day_of_week = ts.weekday()
    day_of_month = ts.day
    week_of_year = ts.isocalendar()[1]
    month_of_year = ts.month
    
    # is_weekend: Saturday=5, Sunday=6
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # is_month_end: last 3 days of month
    _, last_day = calendar.monthrange(ts.year, ts.month)
    is_month_end = 1 if day_of_month >= last_day - 2 else 0
    
    return {
        "is_weekend": is_weekend,
        "day_of_month": day_of_month,
        "week_of_year": week_of_year,
        "month_of_year": month_of_year,
        "is_month_end": is_month_end,
    }


async def load_reddit_by_date(
    conn: asyncpg.Connection,
    symbol: str,
) -> Dict[datetime, Dict[str, float]]:
    """Load Reddit daily signals for a symbol, keyed by date."""
    # Check if table exists
    table_exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'reddit_daily_mentions'
        )
        """
    )
    if not table_exists:
        return {}
    
    rows = await conn.fetch(
        """
        SELECT day,
               COALESCE(mentions_count, 0) as mentions_count,
               COALESCE(sentiment_mean, 0.0) as sentiment_mean,
               COALESCE(sentiment_weighted, 0.0) as sentiment_weighted,
               COALESCE(positive_ratio, 0.0) as positive_ratio,
               COALESCE(negative_ratio, 0.0) as negative_ratio
        FROM reddit_daily_mentions
        WHERE ticker = $1
        ORDER BY day
        """,
        symbol,
    )
    
    result = {}
    for r in rows:
        result[r["day"]] = {
            "mentions_count": int(r["mentions_count"] or 0),
            "sentiment_mean": float(r["sentiment_mean"] or 0.0),
            "sentiment_weighted": float(r["sentiment_weighted"] or 0.0),
            "positive_ratio": float(r["positive_ratio"] or 0.0),
            "negative_ratio": float(r["negative_ratio"] or 0.0),
        }
    return result


async def load_market_benchmark(
    conn: asyncpg.Connection,
    start: datetime,
    end: datetime,
    source: str = "day",
) -> Dict[datetime, Dict[str, float]]:
    """Load market benchmark (SPY) data for context features."""
    table = "market_raw_day" if source == "day" else "market_raw_minute"
    
    # Try SPY first, then QQQ
    for benchmark in MARKET_BENCHMARKS:
        rows = await conn.fetch(
            f"""
            SELECT ts, close
            FROM {table}
            WHERE ticker = $1 AND ts >= $2 AND ts <= $3
            ORDER BY ts
            """,
            benchmark,
            start,
            end,
        )
        if len(rows) > 5:
            break
    else:
        # No benchmark data available
        return {}
    
    # Compute returns and volatility
    closes = np.array([float(r["close"]) for r in rows])
    timestamps = [r["ts"] for r in rows]
    n = len(rows)
    
    # Safe log ratio computation
    def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator != 0)
            return np.log(np.clip(ratio, 1e-10, None))
    
    returns_1 = np.zeros(n)
    returns_1[1:] = _safe_log_ratio(closes[1:], closes[:-1])
    
    volatility_5 = np.zeros(n)
    for i in range(5, n):
        volatility_5[i] = np.std(returns_1[i - 5 : i])
    
    result = {}
    for i, ts in enumerate(timestamps):
        # Normalize to date for day-level matching
        key = ts.date() if hasattr(ts, 'date') else ts
        result[key] = {
            "market_return_1": float(returns_1[i]),
            "market_volatility_5": float(volatility_5[i]),
        }
    return result


async def build_features(
    conn: asyncpg.Connection,
    symbols: list[str],
    start: datetime,
    end: datetime,
    source: str = "day",
    horizon: int = 1,
) -> int:
    """Build features for given symbols and date range."""
    table = "market_raw_day" if source == "day" else "market_raw_minute"
    inserted = 0

    # Load market benchmark data (SPY/QQQ) once for all symbols
    market_benchmark = await load_market_benchmark(conn, start, end, source)
    logger.info("market_benchmark_loaded", entries=len(market_benchmark))

    for symbol in symbols:
        # Fetch market data
        rows = await conn.fetch(
            f"""
            SELECT ticker, ts, open, high, low, close, volume
            FROM {table}
            WHERE ticker = $1 AND ts >= $2 AND ts <= $3
            ORDER BY ts
            """,
            symbol,
            start,
            end,
        )

        if len(rows) < 5:
            logger.warning("insufficient_data", symbol=symbol, rows=len(rows))
            continue

        # Load Reddit data for this symbol
        reddit_by_date = await load_reddit_by_date(conn, symbol)
        logger.debug("reddit_data_loaded", symbol=symbol, days=len(reddit_by_date))

        # Convert to arrays
        timestamps = [r["ts"] for r in rows]
        opens = np.array([float(r["open"]) for r in rows])
        highs = np.array([float(r["high"]) for r in rows])
        lows = np.array([float(r["low"]) for r in rows])
        closes = np.array([float(r["close"]) for r in rows])
        volumes = np.array([float(r["volume"]) for r in rows])

        n = len(rows)

        # Helper for safe log ratio computation (avoids divide-by-zero warnings)
        def safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            """Compute log(numerator/denominator) safely, returning 0 where denominator is 0."""
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator != 0)
                return np.log(np.clip(ratio, 1e-10, None))

        # Compute features
        returns_1 = np.zeros(n)
        returns_1[1:] = safe_log_ratio(closes[1:], closes[:-1])

        returns_5 = np.zeros(n)
        if n > 5:
            returns_5[5:] = safe_log_ratio(closes[5:], closes[:-5])

        sma_5 = np.convolve(closes, np.ones(5) / 5, mode="same")
        sma_20 = np.convolve(closes, np.ones(min(20, n)) / min(20, n), mode="same")
        ema_5 = compute_ema(closes, 5)
        ema_20 = compute_ema(closes, 20)
        rsi_14 = compute_rsi(closes, 14)

        vol_mean = np.mean(volumes)
        vol_std = np.std(volumes) + 1e-8
        volume_zscore = (volumes - vol_mean) / vol_std

        volatility_5 = np.zeros(n)
        for i in range(5, n):
            volatility_5[i] = np.std(returns_1[i - 5 : i])

        # Build Reddit time-series arrays with lag computation
        # NOTE: Lags are computed at the same granularity as features_tft (day or minute)
        # For day-level data: lag1 = 1 day ago, lag3 = 3 days ago, lag5 = 5 days ago
        # For minute-level data: lag1 = 1 bar ago (interpret accordingly)
        # Missing values are filled with 0.0 (safe default for model)
        reddit_mentions = np.zeros(n)
        reddit_sentiment_mean = np.zeros(n)
        reddit_sentiment_weighted = np.zeros(n)
        reddit_positive_ratio = np.zeros(n)
        reddit_negative_ratio = np.zeros(n)

        for i, ts in enumerate(timestamps):
            day = ts.date() if hasattr(ts, 'date') else ts
            if day in reddit_by_date:
                rd = reddit_by_date[day]
                reddit_mentions[i] = rd["mentions_count"]
                reddit_sentiment_mean[i] = rd["sentiment_mean"]
                reddit_sentiment_weighted[i] = rd["sentiment_weighted"]
                reddit_positive_ratio[i] = rd["positive_ratio"]
                reddit_negative_ratio[i] = rd["negative_ratio"]

        # Compute lag features (lag 1, 3, 5 bars)
        # Data is already sorted by ts (ORDER BY ts in query), so lag is correct
        def lag_array(arr: np.ndarray, lag: int) -> np.ndarray:
            """Shift array by lag positions. result[i] = arr[i-lag]. Fills with 0."""
            result = np.zeros_like(arr)
            if lag < len(arr):
                result[lag:] = arr[:-lag]
            return result

        mentions_lag1 = lag_array(reddit_mentions, 1)
        mentions_lag3 = lag_array(reddit_mentions, 3)
        mentions_lag5 = lag_array(reddit_mentions, 5)
        sentiment_mean_lag1 = lag_array(reddit_sentiment_mean, 1)
        sentiment_mean_lag3 = lag_array(reddit_sentiment_mean, 3)
        sentiment_mean_lag5 = lag_array(reddit_sentiment_mean, 5)
        sentiment_weighted_lag1 = lag_array(reddit_sentiment_weighted, 1)
        sentiment_weighted_lag3 = lag_array(reddit_sentiment_weighted, 3)
        sentiment_weighted_lag5 = lag_array(reddit_sentiment_weighted, 5)
        positive_ratio_lag1 = lag_array(reddit_positive_ratio, 1)
        positive_ratio_lag3 = lag_array(reddit_positive_ratio, 3)
        positive_ratio_lag5 = lag_array(reddit_positive_ratio, 5)
        negative_ratio_lag1 = lag_array(reddit_negative_ratio, 1)
        negative_ratio_lag3 = lag_array(reddit_negative_ratio, 3)
        negative_ratio_lag5 = lag_array(reddit_negative_ratio, 5)

        # Target return (future)
        target_return = np.zeros(n)
        if n > horizon:
            target_return[:-horizon] = safe_log_ratio(closes[horizon:], closes[:-horizon])

        # Insert features
        for i in range(n):
            ts = timestamps[i]
            idempotency_key = f"{symbol}:{ts.isoformat()}:features"

            # Compute calendar features
            cal = compute_calendar_features(ts)

            # Get market benchmark data for this timestamp
            day_key = ts.date() if hasattr(ts, 'date') else ts
            mkt = market_benchmark.get(day_key, {"market_return_1": 0.0, "market_volatility_5": 0.0})

            await conn.execute(
                """
                INSERT INTO features_tft (
                    ticker, ts, time_idx, split, price, open, high, low, close, volume,
                    returns_1, returns_5, returns_15, target_return,
                    volatility_5, volatility_15,
                    sma_5, sma_20, ema_5, ema_20, rsi_14, volume_zscore,
                    minute_of_day, hour_of_day, day_of_week,
                    reddit_mentions_count, reddit_sentiment_mean, reddit_sentiment_weighted,
                    reddit_positive_ratio, reddit_negative_ratio,
                    is_weekend, day_of_month, week_of_year, month_of_year, is_month_end,
                    reddit_mentions_lag1, reddit_mentions_lag3, reddit_mentions_lag5,
                    reddit_sentiment_mean_lag1, reddit_sentiment_mean_lag3, reddit_sentiment_mean_lag5,
                    reddit_sentiment_weighted_lag1, reddit_sentiment_weighted_lag3, reddit_sentiment_weighted_lag5,
                    reddit_positive_ratio_lag1, reddit_positive_ratio_lag3, reddit_positive_ratio_lag5,
                    reddit_negative_ratio_lag1, reddit_negative_ratio_lag3, reddit_negative_ratio_lag5,
                    market_return_1, market_volatility_5,
                    idempotency_key
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22,
                    $23, $24, $25, $26, $27, $28, $29, $30,
                    $31, $32, $33, $34, $35,
                    $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50,
                    $51, $52, $53
                )
                ON CONFLICT (ts, ticker) DO UPDATE SET
                    is_weekend = EXCLUDED.is_weekend,
                    day_of_month = EXCLUDED.day_of_month,
                    week_of_year = EXCLUDED.week_of_year,
                    month_of_year = EXCLUDED.month_of_year,
                    is_month_end = EXCLUDED.is_month_end,
                    reddit_mentions_count = EXCLUDED.reddit_mentions_count,
                    reddit_sentiment_mean = EXCLUDED.reddit_sentiment_mean,
                    reddit_sentiment_weighted = EXCLUDED.reddit_sentiment_weighted,
                    reddit_positive_ratio = EXCLUDED.reddit_positive_ratio,
                    reddit_negative_ratio = EXCLUDED.reddit_negative_ratio,
                    reddit_mentions_lag1 = EXCLUDED.reddit_mentions_lag1,
                    reddit_mentions_lag3 = EXCLUDED.reddit_mentions_lag3,
                    reddit_mentions_lag5 = EXCLUDED.reddit_mentions_lag5,
                    reddit_sentiment_mean_lag1 = EXCLUDED.reddit_sentiment_mean_lag1,
                    reddit_sentiment_mean_lag3 = EXCLUDED.reddit_sentiment_mean_lag3,
                    reddit_sentiment_mean_lag5 = EXCLUDED.reddit_sentiment_mean_lag5,
                    reddit_sentiment_weighted_lag1 = EXCLUDED.reddit_sentiment_weighted_lag1,
                    reddit_sentiment_weighted_lag3 = EXCLUDED.reddit_sentiment_weighted_lag3,
                    reddit_sentiment_weighted_lag5 = EXCLUDED.reddit_sentiment_weighted_lag5,
                    reddit_positive_ratio_lag1 = EXCLUDED.reddit_positive_ratio_lag1,
                    reddit_positive_ratio_lag3 = EXCLUDED.reddit_positive_ratio_lag3,
                    reddit_positive_ratio_lag5 = EXCLUDED.reddit_positive_ratio_lag5,
                    reddit_negative_ratio_lag1 = EXCLUDED.reddit_negative_ratio_lag1,
                    reddit_negative_ratio_lag3 = EXCLUDED.reddit_negative_ratio_lag3,
                    reddit_negative_ratio_lag5 = EXCLUDED.reddit_negative_ratio_lag5,
                    market_return_1 = EXCLUDED.market_return_1,
                    market_volatility_5 = EXCLUDED.market_volatility_5
                """,
                symbol,                            # $1
                ts,                                # $2
                i,                                 # $3
                "train" if i < n * 0.8 else "val", # $4
                float(closes[i]),                  # $5 price
                float(opens[i]),                   # $6
                float(highs[i]),                   # $7
                float(lows[i]),                    # $8
                float(closes[i]),                  # $9
                float(volumes[i]),                 # $10
                float(returns_1[i]),               # $11
                float(returns_5[i]) if n > 5 else 0.0,  # $12
                0.0,                               # $13 returns_15 placeholder
                float(target_return[i]),           # $14
                float(volatility_5[i]),            # $15
                0.0,                               # $16 volatility_15 placeholder
                float(sma_5[i]),                   # $17
                float(sma_20[i]),                  # $18
                float(ema_5[i]),                   # $19
                float(ema_20[i]),                  # $20
                float(rsi_14[i]),                  # $21
                float(volume_zscore[i]),           # $22
                ts.hour * 60 + ts.minute,          # $23 minute_of_day
                ts.hour,                           # $24 hour_of_day
                ts.weekday(),                      # $25 day_of_week
                float(reddit_mentions[i]),         # $26
                float(reddit_sentiment_mean[i]),   # $27
                float(reddit_sentiment_weighted[i]),  # $28
                float(reddit_positive_ratio[i]),   # $29
                float(reddit_negative_ratio[i]),   # $30
                cal["is_weekend"],                 # $31
                cal["day_of_month"],               # $32
                cal["week_of_year"],               # $33
                cal["month_of_year"],              # $34
                cal["is_month_end"],               # $35
                float(mentions_lag1[i]),           # $36
                float(mentions_lag3[i]),           # $37
                float(mentions_lag5[i]),           # $38
                float(sentiment_mean_lag1[i]),     # $39
                float(sentiment_mean_lag3[i]),     # $40
                float(sentiment_mean_lag5[i]),     # $41
                float(sentiment_weighted_lag1[i]), # $42
                float(sentiment_weighted_lag3[i]), # $43
                float(sentiment_weighted_lag5[i]), # $44
                float(positive_ratio_lag1[i]),     # $45
                float(positive_ratio_lag3[i]),     # $46
                float(positive_ratio_lag5[i]),     # $47
                float(negative_ratio_lag1[i]),     # $48
                float(negative_ratio_lag3[i]),     # $49
                float(negative_ratio_lag5[i]),     # $50
                float(mkt["market_return_1"]),     # $51
                float(mkt["market_volatility_5"]), # $52
                idempotency_key,                   # $53
            )
            inserted += 1

        logger.info("features_built", symbol=symbol, rows=n)

    return inserted


async def main() -> int:
    parser = argparse.ArgumentParser(description="Build features from market data")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--source",
        choices=["day", "minute"],
        default="day",
        help="Data source table (default: day)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in bars (default: 1)",
    )
    args = parser.parse_args()

    configure_logging()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = _parse_date(args.start)
    end = _parse_date(args.end, end=True)

    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)

    try:
        async with pool.acquire() as conn:
            inserted = await build_features(
                conn,
                symbols,
                start,
                end,
                source=args.source,
                horizon=args.horizon,
            )

            logger.info(
                "feature_engineering_complete",
                symbols=symbols,
                start=str(start),
                end=str(end),
                rows=inserted,
            )
            print(f"Inserted {inserted} feature rows")

    finally:
        await pool.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
