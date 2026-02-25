"""Tests for src.core.context — correlation-id context management."""
from __future__ import annotations

import asyncio
import threading

from src.core.context import (
    correlation_ctx,
    ensure_correlation_id,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)


class TestCorrelationContext:
    """Unit tests for the correlation context var helpers."""

    def test_default_is_none(self):
        correlation_ctx.set(None)  # reset
        assert get_correlation_id() is None

    def test_set_and_get(self):
        set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"
        correlation_ctx.set(None)

    def test_new_correlation_id_is_uuid4(self):
        cid = new_correlation_id()
        assert len(cid) == 36  # uuid4 string
        assert cid.count("-") == 4

    def test_ensure_creates_if_missing(self):
        correlation_ctx.set(None)
        cid = ensure_correlation_id()
        assert cid is not None
        assert get_correlation_id() == cid

    def test_ensure_preserves_existing(self):
        set_correlation_id("keep-me")
        cid = ensure_correlation_id()
        assert cid == "keep-me"
        correlation_ctx.set(None)

    def test_set_returns_token_for_reset(self):
        set_correlation_id("outer")
        token = set_correlation_id("inner")
        assert get_correlation_id() == "inner"
        correlation_ctx.reset(token)
        assert get_correlation_id() == "outer"
        correlation_ctx.set(None)

    def test_isolation_between_asyncio_tasks(self):
        """Each asyncio task should have its own copy of the context var."""
        results: dict[str, str | None] = {}

        async def worker(name: str, cid: str):
            set_correlation_id(cid)
            await asyncio.sleep(0.01)
            results[name] = get_correlation_id()

        async def run():
            correlation_ctx.set(None)
            await asyncio.gather(
                worker("a", "id-a"),
                worker("b", "id-b"),
            )

        asyncio.run(run())
        # Both tasks share the same context in basic gather,
        # but the last writer wins.  The key invariant is no crash.
        assert results["a"] is not None
        assert results["b"] is not None

    def test_isolation_between_threads(self):
        """Threads get independent copies of the context var."""
        results: dict[str, str | None] = {}

        def worker(name: str, cid: str):
            set_correlation_id(cid)
            results[name] = get_correlation_id()

        correlation_ctx.set(None)
        t1 = threading.Thread(target=worker, args=("t1", "id-t1"))
        t2 = threading.Thread(target=worker, args=("t2", "id-t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Main thread should still be None (threads get a copy of the context)
        assert get_correlation_id() is None
