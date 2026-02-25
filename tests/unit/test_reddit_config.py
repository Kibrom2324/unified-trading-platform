"""Unit tests for reddit_ingestor.config — environment-driven settings."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))


class TestSettings:
    """Verify Settings picks up env vars correctly."""

    def test_defaults(self):
        """With no env vars, defaults are sane."""
        # Re-import to pick up defaults
        from reddit_ingestor.config import Settings

        s = Settings()
        assert s.poll_interval_sec == 60
        assert s.dlq_max_retries == 3
        assert s.http_port == 8090
        assert s.provider == "stub"
        assert s.raw_topic == "social.reddit.raw"
        assert s.dlq_topic == "dlq.social.reddit.raw"

    def test_subreddits_from_env(self):
        with patch.dict(os.environ, {"SUBREDDITS": "a,b,c"}):
            from reddit_ingestor.config import Settings

            s = Settings()
            assert s.subreddits == ["a", "b", "c"]

    def test_frozen(self):
        from reddit_ingestor.config import Settings

        s = Settings()
        import pytest

        with pytest.raises(AttributeError):
            s.poll_interval_sec = 999  # type: ignore[misc]

    def test_load_settings_returns_settings(self):
        from reddit_ingestor.config import Settings, load_settings

        s = load_settings()
        assert isinstance(s, Settings)
