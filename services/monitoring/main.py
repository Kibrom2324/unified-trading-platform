from __future__ import annotations

import asyncio
import os

import structlog
from confluent_kafka.admin import AdminClient

try:
    from confluent_kafka.admin import ConsumerGroupTopicPartitions
except ImportError:
    from confluent_kafka.admin import _ConsumerGroupTopicPartitions as ConsumerGroupTopicPartitions

from src.agents.base_agent import BaseAgent
from src.core.config import load_settings
from src.monitoring.metrics import EVENTS_PROCESSED, KAFKA_CONSUMER_LAG, KAFKA_CONSUMER_LAG_TOTAL

logger = structlog.get_logger("watchtower")

# Consumer groups to monitor
CONSUMER_GROUPS = [
    "feature-engineering",
    "prediction-agent",
    "risk-agent",
    "execution-agent",
]


class WatchtowerAgent(BaseAgent):
    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("watchtower", settings.app.http.watchtower_port)
        self._kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self._admin_client: AdminClient | None = None

    def _get_admin_client(self) -> AdminClient:
        if self._admin_client is None:
            self._admin_client = AdminClient({"bootstrap.servers": self._kafka_bootstrap})
        return self._admin_client

    async def start(self) -> None:
        self.track_task(asyncio.create_task(self._monitor_loop()))
        self.track_task(asyncio.create_task(self._lag_monitor_loop()))

    async def _monitor_loop(self) -> None:
        """Heartbeat monitor."""
        while not self.is_shutting_down:
            EVENTS_PROCESSED.labels(agent=self.name, event_type="heartbeat").inc()
            await asyncio.sleep(5)

    async def _lag_monitor_loop(self) -> None:
        """Poll Kafka consumer lag every 30 seconds."""
        while not self.is_shutting_down:
            try:
                await self._update_consumer_lag()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("lag_monitor_error", error=str(e))
            await asyncio.sleep(30)

    async def _update_consumer_lag(self) -> None:
        """Fetch consumer lag from Kafka admin API."""
        admin = self._get_admin_client()

        for group_id in CONSUMER_GROUPS:
            try:
                # Get consumer group offsets
                group_partitions = ConsumerGroupTopicPartitions(group_id)
                futures = admin.list_consumer_group_offsets([group_partitions])

                for group_result in futures.values():
                    result = group_result.result()
                    topic_partitions = result.topic_partitions

                    total_lag = 0
                    for tp in topic_partitions:
                        if tp.offset < 0:
                            continue  # No committed offset yet

                        # Get high watermark (end offset) for partition
                        watermarks = admin.list_offsets(
                            {tp: -1}  # -1 = latest offset
                        )
                        for tp_key, offset_result in watermarks.items():
                            try:
                                hw = offset_result.result().offset
                                lag = max(0, hw - tp.offset)
                                total_lag += lag

                                KAFKA_CONSUMER_LAG.labels(
                                    consumer_group=group_id,
                                    topic=tp.topic,
                                    partition=str(tp.partition),
                                ).set(lag)
                            except Exception:
                                pass

                    KAFKA_CONSUMER_LAG_TOTAL.labels(consumer_group=group_id).set(total_lag)
                    logger.debug("consumer_lag_updated", group=group_id, total_lag=total_lag)

            except Exception as e:
                logger.warning("consumer_lag_fetch_failed", group=group_id, error=str(e))


def main() -> None:
    WatchtowerAgent().run()


if __name__ == "__main__":
    main()
