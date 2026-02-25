#!/usr/bin/env python3
"""
Generate trade decisions (BUY/SELL/HOLD) from predictions.

Reads from: predictions_tft
Writes to:  trade_decisions + audit_events

Usage:
    python scripts/decide_trades.py --symbols AAPL --start 2024-01-01 --end 2024-01-31
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

from src.core.config import load_settings
from src.core.trade_decision import DecisionThresholds, decide_trade_v2


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.astimezone(timezone.utc)


def _decision_reason(decision: str) -> str:
    if decision == "BUY":
        return "pred_return_above_threshold"
    if decision == "SELL":
        return "pred_return_below_threshold"
    return "below_threshold_or_low_confidence"


async def run_decisions(
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> int:
    settings = load_settings()
    thresholds = DecisionThresholds(
        min_confidence=settings.trader_decision_agent.min_confidence,
        buy_return_threshold=settings.trader_decision_agent.buy_return_threshold,
        sell_return_threshold=settings.trader_decision_agent.sell_return_threshold,
    )

    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    inserted = 0

    try:
        async with pool.acquire() as conn:
            for symbol in symbols:
                rows = await conn.fetch(
                    """
                    SELECT ticker, ts, confidence, q10, q50, q90, direction,
                           model_version, correlation_id, idempotency_key
                    FROM predictions_tft
                    WHERE ticker = $1 AND ts >= $2 AND ts <= $3
                    ORDER BY ts
                    """,
                    symbol,
                    start,
                    end,
                )

                for row in rows:
                    decision = decide_trade_v2(
                        {
                            "direction": row["direction"],
                            "confidence": float(row["confidence"]),
                            "q10": row["q10"],
                            "q50": row["q50"],
                            "q90": row["q90"],
                        },
                        thresholds,
                    )
                    reason = _decision_reason(decision)
                    pred_return = float(row["q50"] or 0.0)
                    idempotency_key = row["idempotency_key"]
                    correlation_id = row["correlation_id"]
                    ts = row["ts"]

                    await conn.execute(
                        """
                        INSERT INTO trade_decisions (
                            ts, ticker, decision, reason, confidence,
                            pred_return, q10, q50, q90, model_version,
                            correlation_id, idempotency_key
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                        )
                        ON CONFLICT (idempotency_key) DO NOTHING
                        """,
                        ts,
                        symbol,
                        decision,
                        reason,
                        float(row["confidence"]),
                        pred_return,
                        row["q10"],
                        row["q50"],
                        row["q90"],
                        row["model_version"],
                        str(correlation_id) if correlation_id else None,
                        idempotency_key,
                    )

                    payload = {
                        "symbol": symbol,
                        "ts": ts.isoformat(),
                        "decision": decision,
                        "confidence": float(row["confidence"]),
                        "pred_return": pred_return,
                        "reason": reason,
                    }
                    await conn.execute(
                        """
                        INSERT INTO audit_events (
                            event_id, correlation_id, idempotency_key, symbol, ts,
                            schema_version, source, event_type, payload
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        ON CONFLICT (ts, idempotency_key) DO NOTHING
                        """,
                        uuid4(),
                        correlation_id,
                        idempotency_key,
                        symbol,
                        ts,
                        1,
                        "decide_trades",
                        "trade.decisions",
                        json.dumps(payload),
                    )
                    inserted += 1

    finally:
        await pool.close()

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate trade decisions from predictions")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = _parse_date(args.start)
    end = _parse_date(args.end, end=True)

    try:
        inserted = asyncio.run(run_decisions(symbols, start, end))
        print(f"Inserted {inserted} trade decisions")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
