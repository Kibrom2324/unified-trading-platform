from __future__ import annotations

import logging
from typing import Any

import structlog
from confluent_kafka import KafkaError
from confluent_kafka.deserializing_consumer import DeserializingConsumer
from confluent_kafka.error import ConsumeError, KeyDeserializationError, ValueDeserializationError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import StringDeserializer

from src.core.config import KafkaConfig
from src.streaming.dlq import DLQPublisher

logger = logging.getLogger(__name__)
log = structlog.get_logger("kafka_consumer")


def _on_assign(consumer, partitions):
    """Callback when partitions are assigned."""
    log.info("kafka_partitions_assigned", partitions=[f"{p.topic}[{p.partition}]" for p in partitions])


def _on_revoke(consumer, partitions):
    """Callback when partitions are revoked — commit offsets before rebalance."""
    log.warning("kafka_partitions_revoked", partitions=[f"{p.topic}[{p.partition}]" for p in partitions])
    try:
        consumer.commit(asynchronous=False)
    except Exception:
        log.exception("kafka_revoke_commit_failed")


class AvroConsumer:
    def __init__(
        self,
        config: KafkaConfig,
        group_id: str,
        schema_str: str,
        topic: str,
        dlq_publisher: DLQPublisher | None = None,
    ) -> None:
        self._schema_registry = SchemaRegistryClient({"url": config.schema_registry_url})
        self._consumer = DeserializingConsumer(
            {
                "bootstrap.servers": ",".join(config.bootstrap_servers),
                "group.id": group_id,
                "auto.offset.reset": config.auto_offset_reset,
                "enable.auto.commit": False,
                "key.deserializer": StringDeserializer("utf_8"),
                "value.deserializer": AvroDeserializer(self._schema_registry, schema_str),
                # --- Session management ---
                "session.timeout.ms": 30000,
                "max.poll.interval.ms": 300000,
                "heartbeat.interval.ms": 10000,
            }
        )
        self._consumer.subscribe([topic], on_assign=_on_assign, on_revoke=_on_revoke)
        self._topic = topic
        self._deser_error_count = 0
        self._dlq = dlq_publisher
        log.info("kafka_consumer_created", group=group_id, topic=topic)

    def poll(self, timeout: float = 1.0) -> Any | None:
        try:
            msg = self._consumer.poll(timeout)
        except (KeyDeserializationError, ValueDeserializationError) as e:
            self._deser_error_count += 1
            log.error(
                "kafka_deserialization_error",
                error=str(e),
                topic=self._topic,
                total_deser_errors=self._deser_error_count,
            )
            if self._dlq is not None:
                self._dlq.send(
                    topic=self._topic,
                    error=str(e),
                    raw=getattr(e, "value", None) if hasattr(e, "value") else None,
                )
            return None
        except ConsumeError as e:
            # Handle topic not ready errors gracefully
            if e.args[0].code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                return None
            raise
        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                return None
            raise RuntimeError(msg.error())
        return msg

    def commit(self) -> None:
        self._consumer.commit(asynchronous=False)

    def close(self) -> None:
        log.info("kafka_consumer_closing", topic=self._topic)
        try:
            self._consumer.commit(asynchronous=False)
        except Exception:
            log.warning("kafka_consumer_final_commit_failed", topic=self._topic)
        self._consumer.close()
        log.info("kafka_consumer_closed", topic=self._topic)
