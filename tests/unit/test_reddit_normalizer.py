"""Unit tests for reddit_ingestor.normalizer — text normalisation & hashing."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the service package is importable without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.normalizer import author_hash, content_hash, normalize_text

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    """Deterministic text normalisation."""

    def test_strips_zero_width_space(self):
        assert normalize_text("hello\u200bworld") == "helloworld"

    def test_strips_zero_width_non_joiner(self):
        assert normalize_text("a\u200cb") == "ab"

    def test_strips_zero_width_joiner(self):
        assert normalize_text("a\u200db") == "ab"

    def test_strips_word_joiner(self):
        assert normalize_text("a\u2060b") == "ab"

    def test_strips_bom(self):
        assert normalize_text("\ufeffhello") == "hello"

    def test_strips_soft_hyphen(self):
        assert normalize_text("caf\u00ade") == "cafe"

    def test_collapses_horizontal_whitespace(self):
        assert normalize_text("a   b\t\tc") == "a b c"

    def test_collapses_excessive_newlines(self):
        result = normalize_text("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_preserves_double_newline(self):
        result = normalize_text("a\n\nb")
        assert result == "a\n\nb"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_nfc_normalization(self):
        # é composed vs decomposed
        composed = "\u00e9"                # é (single codepoint)
        decomposed = "e\u0301"             # e + combining accent
        assert normalize_text(composed) == normalize_text(decomposed)

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_whitespace(self):
        assert normalize_text("   \t  \n  ") == ""

    def test_combined_invisible_chars(self):
        text = "\ufeff  \u200b hello\u200c   world \u2060 \n\n\n\nbye "
        # After zero-width removal: "   hello   world  \n\n\n\nbye "
        # After ws collapse:        " hello world \n\n\n\nbye "
        # After newline collapse:   " hello world \n\nbye "
        # After strip:              "hello world \n\nbye"
        # Note: trailing space before \n is preserved (horizontal ws only)
        assert normalize_text(text) == "hello world \n\nbye"


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

class TestContentHash:
    """SHA-256 content hashing must be deterministic and stable."""

    def test_deterministic(self):
        """Same inputs → same hash."""
        kwargs = dict(
            title="AAPL to the moon",
            body="Buying calls",
            subreddit="wsb",
            post_id="t3_abc",
            created_at_utc="2026-02-17T10:30:00Z",
        )
        h1 = content_hash(**kwargs)
        h2 = content_hash(**kwargs)
        assert h1 == h2

    def test_hex_length(self):
        h = content_hash(
            title="x", body="y", subreddit="s", post_id="p", created_at_utc="t"
        )
        assert len(h) == 64  # sha256 hex

    def test_different_bodies_differ(self):
        common = dict(title="T", subreddit="s", post_id="p", created_at_utc="t")
        h1 = content_hash(body="alpha", **common)
        h2 = content_hash(body="beta", **common)
        assert h1 != h2

    def test_different_subreddit_differs(self):
        common = dict(title="T", body="B", post_id="p", created_at_utc="t")
        h1 = content_hash(subreddit="wsb", **common)
        h2 = content_hash(subreddit="stocks", **common)
        assert h1 != h2

    def test_normalises_before_hashing(self):
        """Zero-width chars in input should not affect the hash."""
        h_clean = content_hash(
            title="hello", body="world", subreddit="s", post_id="p", created_at_utc="t"
        )
        h_dirty = content_hash(
            title="hel\u200blo", body="wor\u200cld", subreddit="s", post_id="p", created_at_utc="t"
        )
        assert h_clean == h_dirty


# ---------------------------------------------------------------------------
# author_hash
# ---------------------------------------------------------------------------

class TestAuthorHash:
    """SHA-256 author hashing."""

    def test_deterministic(self):
        assert author_hash("user1") == author_hash("user1")

    def test_hex_length(self):
        assert len(author_hash("someone")) == 64

    def test_different_users_differ(self):
        assert author_hash("alice") != author_hash("bob")

    def test_raw_username_used(self):
        """Author hash is on the RAW username, not normalised."""
        # A username with zero-width char should hash differently to one without
        assert author_hash("user") != author_hash("us\u200ber")
