"""Unit tests for reddit_ingestor.publisher — Kafka EventPublisher."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))


class TestEventPublisher:
    """Publisher unit tests (Kafka mocked)."""

    def _make_publisher(self):
        """Build an EventPublisher with mocked Kafka components."""
        with (
            patch("reddit_ingestor.publisher.SchemaRegistryClient"),
            patch("reddit_ingestor.publisher.AvroSerializer"),
            patch("reddit_ingestor.publisher.SerializingProducer") as MockProducer,
        ):
            from reddit_ingestor.publisher import EventPublisher

            mock_prod_instance = MagicMock()
            MockProducer.return_value = mock_prod_instance

            pub = EventPublisher(
                bootstrap_servers="localhost:9092",
                schema_registry_url="http://localhost:8081",
                raw_topic="social.reddit.raw",
                dlq_topic="dlq.social.reddit.raw",
                max_retries=3,
            )
            return pub, mock_prod_instance

    def test_publish_raw_calls_produce(self):
        pub, mock_prod = self._make_publisher()
        # Replace _raw_producer with our mock
        pub._raw_producer = mock_prod
        event = {"event_id": "e1", "correlation_id": "c1"}
        pub.publish_raw(event, key="k1")
        mock_prod.produce.assert_called_once()
        call_kwargs = mock_prod.produce.call_args
        assert call_kwargs.kwargs["topic"] == "social.reddit.raw"
        assert call_kwargs.kwargs["key"] == "k1"

    def test_publish_raw_increments_published_total(self):
        from reddit_ingestor.metrics import PUBLISHED_TOTAL

        pub, mock_prod = self._make_publisher()
        pub._raw_producer = mock_prod
        before = PUBLISHED_TOTAL._value.get()
        pub.publish_raw({"event_id": "e2", "correlation_id": "c2"}, key="k2")
        after = PUBLISHED_TOTAL._value.get()
        assert after == before + 1

    def test_publish_raw_routes_to_dlq_on_kafka_error(self):
        from confluent_kafka import KafkaException

        pub, mock_prod = self._make_publisher()
        pub._raw_producer = mock_prod
        mock_prod.produce.side_effect = KafkaException(
            MagicMock(code=MagicMock(return_value=-1), str=MagicMock(return_value="err"))
        )
        pub._dlq_producer = MagicMock()
        pub.publish_raw({"event_id": "e3", "correlation_id": "c3"}, key="k3")
        pub._dlq_producer.produce.assert_called_once()

    def test_poll_calls_both_producers(self):
        pub, mock_prod = self._make_publisher()
        dlq_mock = MagicMock()
        pub._raw_producer = mock_prod
        pub._dlq_producer = dlq_mock
        pub.poll(0.0)
        mock_prod.poll.assert_called_once_with(0.0)
        dlq_mock.poll.assert_called_once_with(0.0)

    def test_flush_calls_both_producers(self):
        pub, mock_prod = self._make_publisher()
        dlq_mock = MagicMock()
        pub._raw_producer = mock_prod
        pub._dlq_producer = dlq_mock
        pub.flush(5.0)
        mock_prod.flush.assert_called_once_with(5.0)
        dlq_mock.flush.assert_called_once_with(5.0)

    def test_close_flushes(self):
        pub, mock_prod = self._make_publisher()
        dlq_mock = MagicMock()
        pub._raw_producer = mock_prod
        pub._dlq_producer = dlq_mock
        pub.close()
        mock_prod.flush.assert_called_once()
        dlq_mock.flush.assert_called_once()
