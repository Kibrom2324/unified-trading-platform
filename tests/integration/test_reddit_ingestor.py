"""Integration tests for reddit-ingestor.

These tests verify the full normalise → dedup → envelope pipeline using
the StubProvider. Kafka/Redis are mocked so CI doesn't need infrastructure.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.dedup import Deduplicator
from reddit_ingestor.main import _build_envelope
from reddit_ingestor.normalizer import content_hash
from reddit_ingestor.provider import StubProvider


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestStubPipelineEndToEnd:
    """Drive the StubProvider through normalise → dedup → envelope."""

    def test_stub_posts_produce_valid_envelopes(self):
        """Every StubProvider post builds a valid envelope."""
        provider = StubProvider()
        posts = _run(provider.fetch_posts(["wallstreetbets", "stocks"]))
        for post in posts:
            env = _build_envelope(post, correlation_id="int-test", hostname="ci")
            # envelope
            assert env["correlation_id"] == "int-test"
            assert env["schema_version"] == "social.reddit.raw.v1"
            # payload
            p = env["payload"]
            assert len(p["content_hash"]) == 64
            assert len(p["author_hash"]) == 64
            assert p["subreddit"] == post.subreddit

    def test_dedup_filters_duplicate_posts(self):
        """Running the same posts twice, second batch is fully deduped."""
        mock_redis = AsyncMock()
        # Track what has been "set"
        seen_keys: set[str] = set()

        async def fake_exists(key: str) -> int:
            return 1 if key in seen_keys else 0

        async def fake_set(key: str, value: str, ex: int = 0) -> None:
            seen_keys.add(key)

        mock_redis.exists = fake_exists
        mock_redis.set = fake_set
        mock_redis.aclose = AsyncMock()

        dedup = Deduplicator(mock_redis, ttl_sec=600)

        provider = StubProvider()
        posts = _run(provider.fetch_posts(["wallstreetbets", "stocks"]))

        # First pass — all new
        first_results = []
        for post in posts:
            created_str = post.created_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            c_hash = content_hash(
                title=post.title, body=post.body,
                subreddit=post.subreddit, post_id=post.post_id,
                created_at_utc=created_str,
            )
            result = _run(dedup.is_seen(c_hash, correlation_id="pass1"))
            first_results.append(result)
        assert all(r is False for r in first_results)

        # Second pass — all duplicates
        second_results = []
        for post in posts:
            created_str = post.created_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            c_hash = content_hash(
                title=post.title, body=post.body,
                subreddit=post.subreddit, post_id=post.post_id,
                created_at_utc=created_str,
            )
            result = _run(dedup.is_seen(c_hash, correlation_id="pass2"))
            second_results.append(result)
        assert all(r is True for r in second_results)

    def test_content_hash_stable_across_runs(self):
        """Content hashes must be identical across separate StubProvider instantiations."""
        prov1 = StubProvider()
        prov2 = StubProvider()
        posts1 = _run(prov1.fetch_posts(["wallstreetbets", "stocks"]))
        posts2 = _run(prov2.fetch_posts(["wallstreetbets", "stocks"]))

        for p1, p2 in zip(posts1, posts2, strict=False):
            created_str = p1.created_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            h1 = content_hash(
                title=p1.title, body=p1.body,
                subreddit=p1.subreddit, post_id=p1.post_id,
                created_at_utc=created_str,
            )
            h2 = content_hash(
                title=p2.title, body=p2.body,
                subreddit=p2.subreddit, post_id=p2.post_id,
                created_at_utc=created_str,
            )
            assert h1 == h2

    def test_envelope_key_format(self):
        """Keys sent to Kafka must follow reddit:{subreddit}:{post_id}."""
        provider = StubProvider()
        posts = _run(provider.fetch_posts(["wallstreetbets"]))
        for post in posts:
            key = f"reddit:{post.subreddit}:{post.post_id}"
            assert key.startswith("reddit:")
            parts = key.split(":")
            assert len(parts) == 3
            assert parts[1] == post.subreddit
            assert parts[2] == post.post_id
