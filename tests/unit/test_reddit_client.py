"""Unit tests for reddit_ingestor.reddit_client — sync OAuth2 Reddit client."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the service package is importable without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.reddit_client import (
    RedditApiError,
    RedditAuthError,
    RedditClient,
    RedditRateLimitError,
    _normalize_post,
    ts_to_rfc3339,
)

# ---------------------------------------------------------------------------
# ts_to_rfc3339
# ---------------------------------------------------------------------------


class TestTsToRfc3339:
    """Timestamp → RFC 3339 UTC string conversion."""

    def test_known_timestamp(self):
        # 2026-02-17T10:30:00Z
        assert ts_to_rfc3339(1771324200.0) == "2026-02-17T10:30:00Z"

    def test_epoch_zero(self):
        assert ts_to_rfc3339(0.0) == "1970-01-01T00:00:00Z"

    def test_deterministic(self):
        assert ts_to_rfc3339(1_000_000.0) == ts_to_rfc3339(1_000_000.0)

    def test_fractional_seconds_truncated(self):
        """Fractional seconds in the input float are truncated."""
        result = ts_to_rfc3339(1771324200.999)
        assert result == "2026-02-17T10:30:00Z"


# ---------------------------------------------------------------------------
# _normalize_post
# ---------------------------------------------------------------------------


class TestNormalizePost:
    """Deterministic post normalisation."""

    _ITEM: dict = {
        "name": "t3_abc123",
        "title": "AAPL\u200b to the moon",
        "selftext": "body   text\n\n\n\nhere",
        "created_utc": 1771324200.0,
        "score": 42,
        "num_comments": 7,
        "url": "https://reddit.com/r/wsb/abc123",
    }

    def test_source_id_format(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert p["source_id"] == "reddit:wsb:t3_abc123"

    def test_subreddit_passthrough(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert p["subreddit"] == "wsb"

    def test_post_id(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert p["post_id"] == "t3_abc123"

    def test_title_zero_width_stripped(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert "\u200b" not in p["title"]
        assert p["title"] == "AAPL to the moon"

    def test_body_whitespace_collapsed(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert "   " not in p["body"]
        # 3+ newlines collapsed to 2
        assert p["body"] == "body text\n\nhere"

    def test_created_at_utc_rfc3339(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert p["created_at_utc"] == "2026-02-17T10:30:00Z"

    def test_score_is_int(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert isinstance(p["score"], int) and p["score"] == 42

    def test_num_comments_is_int(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert isinstance(p["num_comments"], int) and p["num_comments"] == 7

    def test_url_preserved(self):
        p = _normalize_post(self._ITEM, "wsb")
        assert p["url"] == "https://reddit.com/r/wsb/abc123"

    def test_missing_fields_default(self):
        """Minimal item still produces a full dict without KeyError."""
        p = _normalize_post({}, "test")
        assert p["title"] == ""
        assert p["body"] == ""
        assert p["score"] == 0


# ---------------------------------------------------------------------------
# RedditClient — token retrieval
# ---------------------------------------------------------------------------


def _mock_token_response(*, status_code: int = 200, json_body: dict | None = None):
    """Build a ``requests.Response``-like mock for the token endpoint."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {
        "access_token": "test_token_abc",
        "expires_in": 3600,
        "token_type": "bearer",
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


class TestGetAccessToken:
    """OAuth2 token retrieval + caching."""

    def _make_client(self) -> RedditClient:
        return RedditClient(
            client_id="cid",
            client_secret="csec",
            username="u",
            password="p",
            user_agent="test-agent",
        )

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_success(self, mock_session_cls):
        session = mock_session_cls.return_value
        session.headers = {}
        session.post.return_value = _mock_token_response()

        client = self._make_client()
        tok = client.get_access_token()
        assert tok == "test_token_abc"

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_token_cached(self, mock_session_cls):
        """Second call within TTL must NOT hit the network again."""
        session = mock_session_cls.return_value
        session.headers = {}
        session.post.return_value = _mock_token_response()

        client = self._make_client()
        tok1 = client.get_access_token()
        tok2 = client.get_access_token()
        assert tok1 == tok2
        assert session.post.call_count == 1  # only one HTTP call

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_token_refreshed_after_expiry(self, mock_session_cls):
        """After apparent expiry, a new token request is issued."""
        session = mock_session_cls.return_value
        session.headers = {}
        session.post.return_value = _mock_token_response()

        client = self._make_client()
        client.get_access_token()

        # Simulate expiry
        client._token_expires_at = time.monotonic() - 1
        client.get_access_token()
        assert session.post.call_count == 2

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_http_error_raises_auth_error(self, mock_session_cls):
        """Non-2xx token response raises RedditAuthError."""
        import requests as req

        session = mock_session_cls.return_value
        session.headers = {}
        session.post.side_effect = req.RequestException("connection refused")

        client = self._make_client()
        with pytest.raises(RedditAuthError, match="connection refused"):
            client.get_access_token()

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_missing_access_token_field(self, mock_session_cls):
        """Response without 'access_token' key → RedditAuthError."""
        session = mock_session_cls.return_value
        session.headers = {}
        session.post.return_value = _mock_token_response(
            json_body={"error": "invalid_grant"}
        )

        client = self._make_client()
        with pytest.raises(RedditAuthError, match="Unexpected token response"):
            client.get_access_token()


# ---------------------------------------------------------------------------
# RedditClient — listing endpoints
# ---------------------------------------------------------------------------


def _mock_listing_response(
    items: list[dict], *, status_code: int = 200
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.text = ""
    resp.headers = {}
    resp.json.return_value = {
        "data": {
            "children": [{"kind": "t3", "data": it} for it in items]
        }
    }
    return resp


class TestFetchSubreddit:

    def _make_client_with_token(self) -> tuple[RedditClient, MagicMock]:
        """Return a client whose token is already cached, and its session mock."""
        with patch("reddit_ingestor.reddit_client.requests.Session") as cls:
            session = cls.return_value
            session.headers = {}
            session.post.return_value = _mock_token_response()
            client = RedditClient(
                client_id="cid",
                client_secret="csec",
                username="u",
                password="p",
                user_agent="test-agent",
            )
            # Pre-warm token
            client.get_access_token()
            return client, session

    def test_fetch_new_returns_normalised_posts(self):
        client, session = self._make_client_with_token()
        session.get.return_value = _mock_listing_response([
            {
                "name": "t3_x1",
                "title": "Hello",
                "selftext": "World",
                "created_utc": 1771324200.0,
                "score": 10,
                "num_comments": 2,
                "url": "https://reddit.com/r/test/x1",
            }
        ])

        posts = client.fetch_subreddit_new("test", limit=5)
        assert len(posts) == 1
        assert posts[0]["source_id"] == "reddit:test:t3_x1"
        assert posts[0]["created_at_utc"] == "2026-02-17T10:30:00Z"

    def test_fetch_hot_calls_hot_endpoint(self):
        client, session = self._make_client_with_token()
        session.get.return_value = _mock_listing_response([])

        client.fetch_subreddit_hot("stocks", limit=10)

        url_called = session.get.call_args[0][0]
        assert "/r/stocks/hot.json" in url_called

    def test_rate_limit_raises(self):
        client, session = self._make_client_with_token()
        resp = MagicMock()
        resp.status_code = 429
        resp.ok = False
        resp.headers = {"Retry-After": "30"}
        session.get.return_value = resp

        with pytest.raises(RedditRateLimitError, match="Rate limited"):
            client.fetch_subreddit_new("wsb")

    def test_server_error_raises(self):
        client, session = self._make_client_with_token()
        resp = MagicMock()
        resp.status_code = 500
        resp.ok = False
        resp.text = "Internal Server Error"
        session.get.return_value = resp

        with pytest.raises(RedditApiError, match="HTTP 500"):
            client.fetch_subreddit_new("wsb")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_close_closes_session(self, mock_session_cls):
        session = mock_session_cls.return_value
        session.headers = {}

        client = RedditClient(
            client_id="cid",
            client_secret="csec",
            username="u",
            password="p",
        )
        client.close()
        session.close.assert_called_once()

    @patch("reddit_ingestor.reddit_client.requests.Session")
    def test_context_manager(self, mock_session_cls):
        session = mock_session_cls.return_value
        session.headers = {}

        with RedditClient(
            client_id="cid",
            client_secret="csec",
            username="u",
            password="p",
        ) as client:
            assert isinstance(client, RedditClient)
        session.close.assert_called_once()
