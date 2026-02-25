#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg
import structlog

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logging import configure_logging
from src.market_data.providers import YFinanceProvider

logger = structlog.get_logger("ingest_market_data")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
    return dt.astimezone(timezone.utc)


@dataclass
class _SyntheticBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    vwap: float | None


def _generate_synthetic_bars(symbol: str, start: datetime, end: datetime) -> list[_SyntheticBar]:
    bars: list[_SyntheticBar] = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    price = 100.0
    while current <= end:
        if current.weekday() < 5:
            drift = 0.001
            close = price * (1 + drift)
            high = max(price, close) * 1.01
            low = min(price, close) * 0.99
            bars.append(
                _SyntheticBar(
                    symbol=symbol,
                    timestamp=current,
                    open=price,
                    high=high,
                    low=low,
                    close=close,
                    volume=1_000_000,
                    trades=1000,
                    vwap=(price + close) / 2,
                )
            )
            price = close
        current += timedelta(days=1)
    return bars


async def _ensure_tables(conn: asyncpg.Connection, table: str) -> None:
    for name in (table, "predictions_tft"):
        exists = await conn.fetchval("SELECT to_regclass($1)", name)
        if exists is None:
            raise RuntimeError(f"Missing required table: {name}")


async def _insert_market_rows(conn: asyncpg.Connection, rows: list[tuple], table: str) -> int:
    if not rows:
        return 0
    inserted = 0
    for row in rows:
        await conn.execute(
            f"""
            INSERT INTO {table} (
                ticker, ts, open, high, low, close, volume, transactions, vwap, idempotency_key
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT (ts, ticker) DO NOTHING
            """,
            *row,
        )
        inserted += 1
    return inserted


async def _insert_prediction_rows(conn: asyncpg.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    inserted = 0
    for row in rows:
        await conn.execute(
            """
            INSERT INTO predictions_tft (
                event_id, correlation_id, idempotency_key, ticker, ts,
                horizon_minutes, direction, confidence, q10, q50, q90, model_version
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (ts, idempotency_key) DO NOTHING
            """,
            *row,
        )
        inserted += 1
    return inserted


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest market data into TimescaleDB")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. AAPL,MSFT)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--skip-predictions", action="store_true", help="Only load market data")
    parser.add_argument(
        "--source",
        choices=["synthetic", "yfinance", "alpaca", "polygon"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    parser.add_argument(
        "--interval",
        choices=["day", "minute"],
        default="day",
        help="Bar interval (default: day)",
    )
    # Keep --provider as alias for backward compatibility
    parser.add_argument(
        "--provider",
        choices=["synthetic", "yfinance"],
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # --provider overrides --source for backward compatibility
    source = args.provider if args.provider else args.source

    configure_logging()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = _parse_date(args.start)
    end = _parse_date(args.end, end=True)

    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    table = "market_raw_day" if args.interval == "day" else "market_raw_minute"

    # Initialize provider based on source
    provider = None
    if source == "yfinance":
        try:
            provider = YFinanceProvider()
            logger.info("using_yfinance_provider")
        except Exception as exc:
            logger.error("yfinance_init_failed", error=str(exc))
            print("ERROR: yfinance not available. Install with: pip install yfinance", file=sys.stderr)
            return 1
    elif source in ("alpaca", "polygon"):
        logger.warning("provider_not_implemented", source=source)
        print(f"WARNING: {source} provider not implemented, falling back to synthetic", file=sys.stderr)
        source = "synthetic"

    try:
        async with pool.acquire() as conn:
            await _ensure_tables(conn, table)

            total_bars = 0
            total_preds = 0
            for symbol in symbols:
                bars = []
                if provider:
                    try:
                        bars = await provider.get_bars(symbol, "1D", start, end)
                    except Exception as exc:
                        logger.warning("yfinance_fetch_failed", symbol=symbol, error=str(exc))
                if not bars:
                    bars = _generate_synthetic_bars(symbol, start, end)
                    logger.info("using_synthetic_bars", symbol=symbol, rows=len(bars))
                market_rows = []
                pred_rows = []
                for bar in bars:
                    idempotency_key = f"{bar.symbol}:{bar.timestamp.isoformat()}"
                    market_rows.append(
                        (
                            bar.symbol,
                            bar.timestamp,
                            float(bar.open),
                            float(bar.high),
                            float(bar.low),
                            float(bar.close),
                            float(bar.volume),
                            int(bar.trades or 0),
                            float(bar.vwap) if bar.vwap is not None else None,
                            idempotency_key,
                        )
                    )
                    if not args.skip_predictions:
                        direction = "long" if bar.close >= bar.open else "short"
                        pred_rows.append(
                            (
                                uuid4(),
                                uuid4(),
                                f"pred:{idempotency_key}",
                                bar.symbol,
                                bar.timestamp,
                                1440,
                                direction,
                                0.7,
                                -0.01,
                                0.0,
                                0.01,
                                "ingest_heuristic",
                            )
                        )
                inserted = await _insert_market_rows(conn, market_rows, table)
                total_bars += inserted
                if pred_rows:
                    total_preds += await _insert_prediction_rows(conn, pred_rows)
                row_count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE ticker = $1 AND ts BETWEEN $2 AND $3
                    """,
                    symbol,
                    start,
                    end,
                )
                min_ts = await conn.fetchval(
                    f"""
                    SELECT MIN(ts) FROM {table}
                    WHERE ticker = $1 AND ts BETWEEN $2 AND $3
                    """,
                    symbol,
                    start,
                    end,
                )
                max_ts = await conn.fetchval(
                    f"""
                    SELECT MAX(ts) FROM {table}
                    WHERE ticker = $1 AND ts BETWEEN $2 AND $3
                    """,
                    symbol,
                    start,
                    end,
                )
                logger.info(
                    "ingest_symbol_complete",
                    symbol=symbol,
                    bars=inserted,
                    rows=int(row_count or 0),
                    min_ts=str(min_ts) if min_ts else None,
                    max_ts=str(max_ts) if max_ts else None,
                )

            print(f"Inserted {total_bars} market rows")
            if not args.skip_predictions:
                print(f"Inserted {total_preds} prediction rows")
    finally:
        if provider:
            await provider.close()
        await pool.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
