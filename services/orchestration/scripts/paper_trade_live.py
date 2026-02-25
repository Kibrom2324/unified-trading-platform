#!/usr/bin/env python3
"""Paper trading live loop via risk.approved topic (Kafka)."""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import structlog

from src.core.config import load_settings
from src.core.logging import configure_logging, set_correlation_id
from src.llm.tracing import get_tracer
from src.integrations.alpaca_client import AlpacaClient
from src.integrations.trades_repository import TradesRepository
from src.models.events_risk import RiskEvent
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import RISK_APPROVED

RISK_SCHEMA_PATH = "src/schemas/risk.avsc"

logger = structlog.get_logger(__name__)


def _build_risk_event(symbol: str, side: str, qty: int, correlation_id) -> RiskEvent:
    direction = "long" if side == "buy" else "short"
    return RiskEvent(
        idempotency_key=f"paper-live-{uuid4()}",
        symbol=symbol,
        source="paper_trade_live",
        correlation_id=correlation_id,
        approved=True,
        reason="paper_live",
        risk_score=0.5,
        max_position=float(qty),
        max_notional=0.0,
        min_confidence=0.0,
        direction=direction,
    )


async def _trace_llm(settings, correlation_id: str, symbol: str, side: str, qty: int) -> None:
    tracer = get_tracer(settings)
    await tracer.trace(
        agent_name="paper_trade_live",
        correlation_id=correlation_id,
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        request_messages=[
            {
                "role": "system",
                "content": "paper trading live loop",
            },
            {
                "role": "user",
                "content": f"Place {side} {qty} {symbol}",
            },
        ],
        response_content="queued",
        latency_ms=0.0,
        status="ok",
    )


async def _publish_risk_event(settings, event: RiskEvent) -> None:
    with open(RISK_SCHEMA_PATH, "r", encoding="utf-8") as handle:
        schema_str = handle.read()
    producer = AvroProducer(settings.kafka)
    payload = event.to_avro_dict()
    producer.produce(RISK_APPROVED, schema_str, payload, key=event.symbol)
    producer.flush(10)


def _build_dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://awet:awet@localhost:5433/awet",
    )


async def _connect_db() -> asyncpg.Connection:
    return await asyncpg.connect(_build_dsn())


async def _precheck_sell(settings, symbol: str) -> bool:
    repo = TradesRepository(_build_dsn())
    await repo.connect()
    try:
        qty = await repo.get_position_qty(symbol)
        if qty is not None and qty > 0:
            return True
    finally:
        await repo.close()

    try:
        client = AlpacaClient.from_env()
        qty = await client.get_position_qty(symbol)
        await client.close()
        return qty > 0
    except Exception:
        return False


async def _wait_for_trade(
    correlation_id: str,
    timeout_s: int,
    reconcile_fn,
) -> dict | None:
    start = time.time()
    while time.time() - start < timeout_s:
        await reconcile_fn()
        conn = await _connect_db()
        try:
            row = await conn.fetchrow(
                """
                SELECT alpaca_order_id, status, filled_qty, avg_fill_price
                FROM trades
                WHERE correlation_id = $1
                ORDER BY ts DESC
                LIMIT 1
                """,
                correlation_id,
            )
            if row and row["status"] in {"filled", "blocked", "rejected", "closed"}:
                return dict(row)
        finally:
            await conn.close()
        await asyncio.sleep(2)
    return None


async def run_once(symbol: str, qty: int, side: str, timeout_s: int) -> int:
    settings = load_settings()
    configure_logging(settings.logging.level)

    correlation_uuid = uuid4()
    correlation_id = str(correlation_uuid)
    set_correlation_id(correlation_id)

    start_ts = datetime.now(tz=timezone.utc)

    await _trace_llm(settings, correlation_id, symbol, side, qty)

    if side == "sell":
        can_sell = await _precheck_sell(settings, symbol)
        if not can_sell:
            logger.warning(
                "sell_precheck_failed",
                correlation_id=correlation_id,
                symbol=symbol,
                side=side,
                qty=qty,
            )

    event = _build_risk_event(symbol, side, qty, correlation_uuid)
    await _publish_risk_event(settings, event)

    logger.info(
        "paper_trade_submitted",
        correlation_id=correlation_id,
        agent_name="paper_trade_live",
        symbol=symbol,
        side=side,
        qty=qty,
        status="submitted",
        latency_ms=0,
        alpaca_order_id=None,
    )

    from scripts.reconcile_orders import reconcile_orders

    trade = await _wait_for_trade(
        correlation_id=correlation_id,
        timeout_s=timeout_s,
        reconcile_fn=lambda: reconcile_orders(dry_run=False, verbose=False),
    )

    duration_ms = int((datetime.now(tz=timezone.utc) - start_ts).total_seconds() * 1000)
    if trade:
        logger.info(
            "paper_trade_result",
            correlation_id=correlation_id,
            agent_name="paper_trade_live",
            symbol=symbol,
            side=side,
            qty=qty,
            status=trade.get("status"),
            latency_ms=duration_ms,
            alpaca_order_id=trade.get("alpaca_order_id"),
        )
        print(
            f"correlation_id={correlation_id} order_id={trade.get('alpaca_order_id')} "
            f"status={trade.get('status')} filled_qty={trade.get('filled_qty')} "
            f"avg_fill_price={trade.get('avg_fill_price')}"
        )
        return 0

    logger.warning(
        "paper_trade_timeout",
        correlation_id=correlation_id,
        agent_name="paper_trade_live",
        symbol=symbol,
        side=side,
        qty=qty,
        status="timeout",
        latency_ms=duration_ms,
        alpaca_order_id=None,
    )
    print(f"correlation_id={correlation_id} status=timeout")
    return 1


async def main() -> int:
    parser = argparse.ArgumentParser(description="Paper trading live loop")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--qty", type=int, required=True)
    parser.add_argument("--side", choices=["buy", "sell"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--loop", action="store_true")

    args = parser.parse_args()

    if args.dry_run:
        print("dry_run requested; no trade submitted")
        return 0

    if args.loop and args.side != "buy":
        print("--loop requires --side buy")
        return 1

    if not args.loop:
        return await run_once(args.symbol.upper(), args.qty, args.side, args.timeout)

    # Loop: buy then sell
    result = await run_once(args.symbol.upper(), args.qty, "buy", args.timeout)
    if result != 0:
        return result
    return await run_once(args.symbol.upper(), args.qty, "sell", args.timeout)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
