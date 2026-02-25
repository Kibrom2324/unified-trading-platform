"""Dead Letter Queue (DLQ) for failed Kafka messages.

Messages that fail processing after max retries are written to a DLQ topic
and persisted to TimescaleDB for manual inspection and replay.

Phase 5 hardening — ensures no message is silently lost.
"""
from __future__ import annotations

import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

logger = structlog.get_logger("dlq")

DLQ_TOPIC = os.getenv("DLQ_TOPIC", "dlq.failed")
MAX_RETRIES = int(os.getenv("DLQ_MAX_RETRIES", "3"))


class DeadLetterQueue:
    """Kafka Dead Letter Queue handler.

    Usage:
        dlq = DeadLetterQueue(producer)
        try:
            process(msg)
        except Exception as e:
            dlq.send(msg, e, source_topic="market.raw", agent="feature_engineering")
    """

    def __init__(self, producer: Any = None, db_pool: Any = None):
        self._producer = producer
        self._pool = db_pool

    def send(
        self,
        original_message: bytes | str | dict,
        error: Exception,
        source_topic: str = "unknown",
        agent: str = "unknown",
        correlation_id: str = "",
        retry_count: int = 0,
    ) -> None:
        """Send failed message to DLQ topic."""
        if isinstance(original_message, bytes):
            try:
                payload_str = original_message.decode("utf-8")
            except UnicodeDecodeError:
                payload_str = str(original_message)
        elif isinstance(original_message, dict):
            payload_str = json.dumps(original_message)
        else:
            payload_str = str(original_message)

        dlq_event = {
            "dlq_id": str(uuid.uuid4()),
            "source_topic": source_topic,
            "agent": agent,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "original_payload": payload_str,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_traceback": traceback.format_exc(),
            "retry_count": retry_count,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        # Publish to DLQ topic
        if self._producer is not None:
            try:
                self._producer.produce(
                    DLQ_TOPIC,
                    key=agent.encode(),
                    value=json.dumps(dlq_event).encode(),
                )
                self._producer.poll(0)
            except Exception as pub_err:
                logger.error("dlq_publish_failed", error=str(pub_err))

        logger.error(
            "message_sent_to_dlq",
            source_topic=source_topic,
            agent=agent,
            error_type=type(error).__name__,
            error=str(error),
            correlation_id=dlq_event["correlation_id"],
        )

    async def persist_to_db(self, dlq_event: dict) -> None:
        """Persist DLQ event to TimescaleDB for inspection."""
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_events
                        (event_id, correlation_id, idempotency_key,
                         symbol, ts, schema_version, source, event_type, payload)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    uuid.UUID(dlq_event["dlq_id"]),
                    uuid.UUID(dlq_event["correlation_id"]),
                    f"dlq:{dlq_event['dlq_id']}",
                    dlq_event.get("symbol", "SYSTEM"),
                    datetime.fromisoformat(dlq_event["ts"]),
                    1,
                    dlq_event["agent"],
                    "dlq_message",
                    json.dumps(dlq_event),
                )
        except Exception as e:
            logger.error("dlq_db_persist_failed", error=str(e))


def with_retry_and_dlq(
    func,
    msg: Any,
    dlq: DeadLetterQueue,
    source_topic: str = "unknown",
    agent: str = "unknown",
    max_retries: int = MAX_RETRIES,
) -> Optional[Any]:
    """Execute func with retry logic; send to DLQ on final failure.

    Args:
        func: Callable that processes the message
        msg: The message to process
        dlq: DeadLetterQueue instance
        source_topic: Original Kafka topic
        agent: Agent/service name
        max_retries: Maximum retry attempts

    Returns:
        func result on success, None on failure (sent to DLQ)
    """
    for attempt in range(max_retries + 1):
        try:
            return func(msg)
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "processing_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    agent=agent,
                )
            else:
                dlq.send(msg, e, source_topic=source_topic,
                        agent=agent, retry_count=attempt + 1)
    return None
