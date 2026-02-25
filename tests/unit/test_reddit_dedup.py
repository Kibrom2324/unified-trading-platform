"""Unit tests for reddit_ingestor.dedup — Redis-backed deduplication."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.dedup import Deduplicator


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestDeduplicator:
    """Unit tests for is_seen() with mocked Redis."""

    def _make_dedup(self, **redis_overrides) -> tuple[Deduplicator, AsyncMock]:
        """Build a Deduplicator with mocked Redis."""
        mock_redis = AsyncMock()
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.set = AsyncMock()
        mock_redis.aclose = AsyncMock()
        for k, v in redis_overrides.items():
            setattr(mock_redis, k, v)
        return Deduplicator(mock_redis, ttl_sec=600), mock_redis

    # -- happy path ---------------------------------------------------------

    def test_new_hash_returns_false(self):
        dedup, mock = self._make_dedup(exists=AsyncMock(return_value=0))
        result = _run(dedup.is_seen("hash1", correlation_id="cid"))
        assert result is False
        mock.set.assert_awaited_once_with("seen:hash1", "1", ex=600)

    def test_seen_hash_returns_true(self):
        dedup, mock = self._make_dedup(exists=AsyncMock(return_value=1))
        result = _run(dedup.is_seen("hash2", correlation_id="cid"))
        assert result is True
        mock.set.assert_not_awaited()

    # -- graceful degradation -----------------------------------------------

    def test_redis_error_returns_none(self):
        """When Redis is down, is_seen returns None (proceed with event)."""
        from redis.exceptions import RedisError

        failing_redis = AsyncMock()
        failing_redis.exists = AsyncMock(side_effect=RedisError("Connection refused"))
        dedup = Deduplicator(failing_redis, ttl_sec=600)
        result = _run(dedup.is_seen("hash3", correlation_id="cid"))
        assert result is None

    def test_redis_error_on_set_returns_none(self):
        """If exists() works but set() fails, still returns None."""
        from redis.exceptions import RedisError

        mock_redis = AsyncMock()
        mock_redis.exists = AsyncMock(return_value=0)
        mock_redis.set = AsyncMock(side_effect=RedisError("Timeout"))
        dedup = Deduplicator(mock_redis, ttl_sec=600)
        # set() is called inside the try block, so RedisError → None
        result = _run(dedup.is_seen("hash4", correlation_id="cid"))
        assert result is None

    # -- TTL ----------------------------------------------------------------

    def test_ttl_passed_to_set(self):
        dedup, mock = self._make_dedup(exists=AsyncMock(return_value=0))
        dedup._ttl = 86400
        _run(dedup.is_seen("hash5", correlation_id="cid"))
        mock.set.assert_awaited_once_with("seen:hash5", "1", ex=86400)

    # -- close --------------------------------------------------------------

    def test_close_calls_aclose(self):
        dedup, mock = self._make_dedup()
        _run(dedup.close())
        mock.aclose.assert_awaited_once()
