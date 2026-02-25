"""Unit tests for reddit_ingestor.main — orchestrator envelope building."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.main import _build_envelope
from reddit_ingestor.provider import RawPost


class TestBuildEnvelope:
    """Verify _build_envelope produces a valid Avro-compatible dict."""

    POST = RawPost(
        subreddit="wallstreetbets",
        post_id="t3_abc123",
        comment_id=None,
        author="testuser1",
        title="AAPL to the moon 🚀",
        body="Buying calls",
        url="https://www.reddit.com/r/wallstreetbets/comments/abc123/",
        score=100,
        num_comments=50,
        created_utc=datetime(2026, 2, 17, 10, 30, 0, tzinfo=UTC),
        is_submission=True,
    )

    def test_envelope_has_required_fields(self):
        env = _build_envelope(self.POST, correlation_id="cid-1", hostname="testhost")
        for field in ("event_id", "correlation_id", "event_version",
                      "produced_at_utc", "schema_version", "source", "trace", "payload"):
            assert field in env, f"Missing field: {field}"

    def test_trace_shape(self):
        env = _build_envelope(self.POST, correlation_id="cid-2", hostname="h1")
        t = env["trace"]
        assert t["service_name"] == "reddit-ingestor"
        assert t["host"] == "h1"
        assert t["attempt"] == 1
        assert t["parent_event_id"] is None

    def test_payload_fields(self):
        env = _build_envelope(self.POST, correlation_id="cid-3", hostname="h")
        p = env["payload"]
        assert p["subreddit"] == "wallstreetbets"
        assert p["post_id"] == "t3_abc123"
        assert p["comment_id"] is None
        assert p["score"] == 100
        assert p["num_comments"] == 50
        assert len(p["content_hash"]) == 64
        assert len(p["author_hash"]) == 64
        assert p["created_at_utc"] == "2026-02-17T10:30:00Z"

    def test_source_id_format(self):
        env = _build_envelope(self.POST, correlation_id="cid-4", hostname="h")
        assert env["payload"]["source_id"] == "reddit:wallstreetbets:t3_abc123"

    def test_schema_version(self):
        env = _build_envelope(self.POST, correlation_id="cid-5", hostname="h")
        assert env["schema_version"] == "social.reddit.raw.v1"
        assert env["source"] == "reddit"

    def test_correlation_id_threaded(self):
        env = _build_envelope(self.POST, correlation_id="my-cid", hostname="h")
        assert env["correlation_id"] == "my-cid"

    def test_deterministic_content_hash(self):
        """Same post → same content_hash regardless of time of envelope build."""
        e1 = _build_envelope(self.POST, correlation_id="c1", hostname="h")
        e2 = _build_envelope(self.POST, correlation_id="c2", hostname="h")
        assert e1["payload"]["content_hash"] == e2["payload"]["content_hash"]

    def test_event_ids_are_unique(self):
        e1 = _build_envelope(self.POST, correlation_id="c1", hostname="h")
        e2 = _build_envelope(self.POST, correlation_id="c2", hostname="h")
        assert e1["event_id"] != e2["event_id"]
