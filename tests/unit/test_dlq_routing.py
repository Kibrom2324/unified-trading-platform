"""Tests for DLQ routing (src/streaming/dlq.py)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch  # noqa: I001

# ---------------------------------------------------------------------------
# DLQPublisher unit tests
# ---------------------------------------------------------------------------

class TestDLQPublisher:
    """Unit tests for the DLQPublisher class."""

    def test_import(self):
        from src.streaming.dlq import DLQPublisher
        assert DLQPublisher is not None

    def test_lazy_init(self):
        """Producer is not created on __init__."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher(bootstrap_servers="localhost:9092")
        assert pub._producer is None

    def test_send_creates_producer_lazily(self):
        """First send() triggers lazy init."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher()
        mock_producer = MagicMock()
        with patch("src.streaming.dlq.Producer", return_value=mock_producer):
            result = pub.send(topic="test.topic", error="bad data")
        assert result is True
        mock_producer.produce.assert_called_once()

    def test_send_envelope_shape(self):
        """Verify the DLQ envelope has required fields."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher()
        mock_producer = MagicMock()

        captured_value = None

        def _capture_produce(topic, value, key):
            nonlocal captured_value
            captured_value = value

        mock_producer.produce.side_effect = _capture_produce

        with patch("src.streaming.dlq.Producer", return_value=mock_producer):
            pub.send(topic="market.raw", error="ValueDeserializationError", raw=b"\x00\x01")

        assert captured_value is not None
        envelope = json.loads(captured_value)
        assert "event_id" in envelope
        assert envelope["original_topic"] == "market.raw"
        assert envelope["error_message"] == "ValueDeserializationError"
        assert "error_ts" in envelope
        assert envelope["raw_base64"] == "0001"

    def test_send_without_raw(self):
        """DLQ send works when raw bytes are None."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher()
        mock_producer = MagicMock()
        with patch("src.streaming.dlq.Producer", return_value=mock_producer):
            result = pub.send(topic="test.topic", error="oops")
        assert result is True

    def test_send_returns_false_on_produce_error(self):
        """If produce() raises, send() returns False (never re-raises)."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher()
        mock_producer = MagicMock()
        mock_producer.produce.side_effect = RuntimeError("broker down")
        with patch("src.streaming.dlq.Producer", return_value=mock_producer):
            result = pub.send(topic="test.topic", error="bad")
        assert result is False

    def test_send_returns_false_when_no_confluent_kafka(self):
        """If confluent_kafka is not available, send returns False."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher()
        with patch("src.streaming.dlq.Producer", None):
            result = pub.send(topic="test.topic", error="bad")
        assert result is False

    def test_flush_noop_when_no_producer(self):
        """flush() is safe when producer was never created."""
        from src.streaming.dlq import DLQPublisher
        pub = DLQPublisher()
        pub.flush()  # should not raise


# ---------------------------------------------------------------------------
# topics.py
# ---------------------------------------------------------------------------

def test_dlq_generic_topic_exists():
    from src.streaming.topics import DLQ_GENERIC
    assert DLQ_GENERIC == "dlq.generic"
