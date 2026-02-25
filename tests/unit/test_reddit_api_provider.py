"""Unit tests for reddit_ingestor.reddit_api_provider."""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"),
)

from reddit_ingestor.provider import RawPost, create_provider
from reddit_ingestor.reddit_api_provider import RedditApiProvider


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRedditApiProviderInit:
    """Verify credential validation at construction time."""

    def test_missing_client_id_raises(self):
        with pytest.raises(ValueError, match="REDDIT_CLIENT_ID"):
            RedditApiProvider(
                client_id="",
                client_secret="secret",
                username="u",
                password="p",
            )

    def test_missing_client_secret_raises(self):
        with pytest.raises(ValueError, match="REDDIT_CLIENT_ID"):
            RedditApiProvider(
                client_id="cid",
                client_secret="",
                username="u",
                password="p",
            )

    def test_valid_credentials_accepted(self):
        prov = RedditApiProvider(
            client_id="cid",
            client_secret="csecret",
            username="u",
            password="p",
        )
        assert prov._client_id == "cid"

    def test_post_limit_clamped(self):
        prov = RedditApiProvider(
            client_id="cid",
            client_secret="csecret",
            username="u",
            password="p",
            post_limit=200,
        )
        assert prov._post_limit == 100

    def test_post_limit_floor(self):
        prov = RedditApiProvider(
            client_id="cid",
            client_secret="csecret",
            username="u",
            password="p",
            post_limit=0,
        )
        assert prov._post_limit == 1


# ---------------------------------------------------------------------------
# _to_raw_post mapping
# ---------------------------------------------------------------------------

class TestToRawPost:
    """Verify the Reddit JSON → RawPost mapping."""

    SAMPLE_DATA = {
        "name": "t3_xyz999",
        "id": "xyz999",
        "author": "real_user",
        "title": "Sample post title",
        "selftext": "Body of the post",
        "url": "https://reddit.com/r/stocks/comments/xyz999/",
        "score": 42,
        "num_comments": 7,
        "created_utc": 1771325400.0,  # 2026-02-17T14:30:00Z
    }

    def test_submission_mapping(self):
        post = RedditApiProvider._to_raw_post(self.SAMPLE_DATA, "stocks")
        assert isinstance(post, RawPost)
        assert post.subreddit == "stocks"
        assert post.post_id == "t3_xyz999"
        assert post.comment_id is None
        assert post.is_submission is True
        assert post.author == "real_user"
        assert post.title == "Sample post title"
        assert post.body == "Body of the post"
        assert post.score == 42
        assert post.num_comments == 7
        assert post.created_utc.tzinfo == UTC

    def test_comment_mapping(self):
        data = dict(self.SAMPLE_DATA, name="t1_abc", id="abc")
        post = RedditApiProvider._to_raw_post(data, "stocks")
        assert post.is_submission is False
        assert post.comment_id == "t1_abc"
        assert post.post_id == "t1_abc"

    def test_deleted_author(self):
        data = dict(self.SAMPLE_DATA)
        del data["author"]
        post = RedditApiProvider._to_raw_post(data, "test")
        assert post.author == "[deleted]"


# ---------------------------------------------------------------------------
# fetch_posts (mocked HTTP)
# ---------------------------------------------------------------------------

_LISTING_RESPONSE = {
    "data": {
        "children": [
            {
                "data": {
                    "name": "t3_aaa111",
                    "id": "aaa111",
                    "author": "poster1",
                    "title": "First post",
                    "selftext": "Hello!",
                    "url": "https://reddit.com/r/wallstreetbets/aaa111",
                    "score": 10,
                    "num_comments": 2,
                    "created_utc": 1771325400.0,
                }
            },
            {
                "data": {
                    "name": "t3_bbb222",
                    "id": "bbb222",
                    "author": "poster2",
                    "title": "Second post",
                    "selftext": "World!",
                    "url": "https://reddit.com/r/wallstreetbets/bbb222",
                    "score": 5,
                    "num_comments": 0,
                    "created_utc": 1771325400.0,
                }
            },
        ]
    }
}

_TOKEN_RESPONSE = {
    "access_token": "fake_token_abc",
    "expires_in": 3600,
    "token_type": "bearer",
    "scope": "read",
}


class TestFetchPosts:
    """fetch_posts with mocked httpx responses."""

    def _make_provider(self) -> RedditApiProvider:
        return RedditApiProvider(
            client_id="cid",
            client_secret="csecret",
            username="u",
            password="p",
            user_agent="test-agent",
        )

    def test_fetch_returns_raw_posts(self):
        """Simulates a full token + listing fetch."""
        prov = self._make_provider()

        # Mock HTTP responses
        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = _TOKEN_RESPONSE

        listing_resp = MagicMock()
        listing_resp.raise_for_status = MagicMock()
        listing_resp.json.return_value = _LISTING_RESPONSE

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=token_resp)
        mock_client.get = AsyncMock(return_value=listing_resp)
        mock_client.headers = {}

        prov._http = mock_client
        posts = _run(prov.fetch_posts(["wallstreetbets"]))
        assert len(posts) == 2
        assert posts[0].post_id == "t3_aaa111"
        assert posts[1].post_id == "t3_bbb222"

    def test_fetch_api_error_continues(self):
        """HTTP error on one subreddit should not crash the whole poll."""
        import httpx

        prov = self._make_provider()
        # Pre-set a valid token so we skip auth
        prov._access_token = "tok"
        prov._token_expires_at = float("inf")

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.headers = {}
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "403 Forbidden",
                request=MagicMock(),
                response=MagicMock(status_code=403),
            )
        )
        prov._http = mock_client
        posts = _run(prov.fetch_posts(["banned_sub"]))
        assert posts == []

    def test_multiple_subreddits(self):
        """Posts from multiple subs are merged."""
        prov = self._make_provider()
        prov._access_token = "tok"
        prov._token_expires_at = float("inf")

        listing_resp = MagicMock()
        listing_resp.raise_for_status = MagicMock()
        listing_resp.json.return_value = _LISTING_RESPONSE

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.headers = {}
        mock_client.get = AsyncMock(return_value=listing_resp)
        prov._http = mock_client

        posts = _run(prov.fetch_posts(["sub1", "sub2"]))
        # 2 posts per sub × 2 subs
        assert len(posts) == 4


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------

class TestCreateProviderRedditApi:
    """Verify the factory can produce RedditApiProvider."""

    def test_factory_creates_api_provider(self):
        prov = create_provider(
            "reddit_api",
            client_id="cid",
            client_secret="csecret",
            username="u",
            password="p",
            user_agent="test",
        )
        assert isinstance(prov, RedditApiProvider)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")

    def test_factory_stub_still_works(self):
        from reddit_ingestor.provider import StubProvider

        prov = create_provider("stub")
        assert isinstance(prov, StubProvider)
