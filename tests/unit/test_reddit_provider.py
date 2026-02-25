"""Unit tests for reddit_ingestor.provider — StubProvider fixtures."""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.provider import RawPost, StubProvider, create_provider


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestStubProvider:
    """Verify StubProvider returns deterministic fixtures."""

    def test_returns_three_fixtures(self):
        prov = StubProvider()
        posts = _run(prov.fetch_posts(["wallstreetbets", "stocks"]))
        assert len(posts) == 3

    def test_wsb_filter(self):
        prov = StubProvider()
        posts = _run(prov.fetch_posts(["wallstreetbets"]))
        assert len(posts) == 2
        assert all(p.subreddit == "wallstreetbets" for p in posts)

    def test_stocks_filter(self):
        prov = StubProvider()
        posts = _run(prov.fetch_posts(["stocks"]))
        assert len(posts) == 1
        assert posts[0].subreddit == "stocks"

    def test_no_match_returns_empty(self):
        prov = StubProvider()
        posts = _run(prov.fetch_posts(["nonexistent"]))
        assert posts == []

    def test_fixture_fields_present(self):
        prov = StubProvider()
        posts = _run(prov.fetch_posts(["wallstreetbets"]))
        p = posts[0]
        assert isinstance(p, RawPost)
        assert p.post_id == "t3_abc123"
        assert p.author == "testuser1"
        assert p.created_utc.tzinfo == UTC
        assert p.is_submission is True

    def test_fixture_deterministic(self):
        """Two calls return identical data."""
        prov = StubProvider()
        a = _run(prov.fetch_posts(["wallstreetbets", "stocks"]))
        b = _run(prov.fetch_posts(["wallstreetbets", "stocks"]))
        assert a == b

    def test_close_is_noop(self):
        prov = StubProvider()
        _run(prov.close())  # should not raise


class TestCreateProvider:
    """Factory tests."""

    def test_stub_provider(self):
        prov = create_provider("stub")
        assert isinstance(prov, StubProvider)

    def test_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("mystery")
