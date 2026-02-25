from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import asyncpg

from src.core.config import Settings


def _parse_ts(ts_value: str | datetime) -> datetime:
    """Parse ISO timestamp string or pass through datetime."""
    if isinstance(ts_value, datetime):
        return ts_value
    return datetime.fromisoformat(ts_value.replace("Z", "+00:00"))


class AuditTrailLogger:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        dsn = (
            f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
            f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
            f"/{_env('POSTGRES_DB', 'awet')}"
        )
        self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def is_duplicate(self, event_type: str, idempotency_key: str) -> bool:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM audit_events WHERE event_type=$1 AND idempotency_key=$2 LIMIT 1",
                event_type,
                idempotency_key,
            )
            return row is not None

    async def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_events (
                    event_id, correlation_id, idempotency_key, symbol, ts,
                    schema_version, source, event_type, payload
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (ts, idempotency_key) DO NOTHING
                """,
                payload["event_id"],
                payload["correlation_id"],
                payload["idempotency_key"],
                payload["symbol"],
                _parse_ts(payload["ts"]),
                payload["schema_version"],
                payload["source"],
                event_type,
                json.dumps(payload, default=str),
            )

    async def write_prediction(self, payload: dict[str, Any]) -> None:
        """Write prediction event to predictions_tft table."""
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions_tft (
                    event_id, correlation_id, idempotency_key, ticker, ts,
                    horizon_minutes, direction, confidence, q10, q50, q90, model_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (ts, idempotency_key) DO NOTHING
                """,
                payload["event_id"],
                payload["correlation_id"],
                payload["idempotency_key"],
                payload["symbol"],
                _parse_ts(payload["ts"]),
                payload.get("horizon_minutes", 30),
                payload.get("direction", "neutral"),
                payload.get("confidence", 0.5),
                payload.get("q10"),
                payload.get("q50"),
                payload.get("q90"),
                payload.get("model_version", "v1"),
            )

    async def write_paper_trade(self, payload: dict[str, Any]) -> None:
        """Write paper trade execution to paper_trades table."""
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO paper_trades (
                    event_id, correlation_id, idempotency_key, ticker, ts,
                    side, qty, limit_price, fill_price, slippage_bps, status, reason
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (ts, idempotency_key) DO NOTHING
                """,
                payload["event_id"],
                payload["correlation_id"],
                payload["idempotency_key"],
                payload["symbol"],
                _parse_ts(payload["ts"]),
                payload.get("side", "hold"),
                payload.get("qty", 0),
                payload.get("limit_price"),
                payload.get("fill_price"),
                payload.get("slippage_bps"),
                payload.get("status", "blocked"),
                payload.get("reason"),
            )


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)
