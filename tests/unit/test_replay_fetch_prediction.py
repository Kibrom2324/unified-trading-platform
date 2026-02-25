"""Unit tests for prediction fetch/validation in the Deterministic Replay Engine.

Validates:
- ``correlation_id`` is required (not optional).
- No fallback to idempotency_key-only query.
- Exactly 0 or 1 rows accepted; >1 raises ``RuntimeError``.
- ``validate_prediction_integrity()`` rejects 0 and >1 rows.
- ``replay_prediction()`` guards on correlation_id presence.
"""

from __future__ import annotations

import importlib

# ---------------------------------------------------------------------------
# Import fetch_prediction from the replay script
# ---------------------------------------------------------------------------
# ``scripts/replay_pipeline.py`` is not inside a normal package, so we
# import it by path using importlib.  We must register the module in
# ``sys.modules`` before executing so that ``@dataclass`` can resolve
# the module's ``__dict__`` (Python 3.13 requirement).
import sys
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

_spec = importlib.util.spec_from_file_location(
    "replay_pipeline",
    "scripts/replay_pipeline.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["replay_pipeline"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
fetch_prediction = _mod.fetch_prediction
validate_prediction_integrity = _mod.validate_prediction_integrity
replay_prediction = _mod.replay_prediction
AuditEvent = _mod.AuditEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**overrides: Any) -> dict[str, Any]:
    """Return a dict that behaves like an ``asyncpg.Record``."""
    base: dict[str, Any] = {
        "ticker": "AAPL",
        "ts": datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        "direction": "long",
        "confidence": 0.85,
        "q10": -0.01,
        "q50": 0.02,
        "q90": 0.05,
        "model_version": "v1.0",
    }
    base.update(overrides)
    # asyncpg.Record supports dict-style access; a plain dict suffices for
    # ``row["col"]`` usage in the function under test.
    return base


def _mock_pool(rows: list[dict[str, Any]]) -> AsyncMock:
    """Build a mock ``asyncpg.Pool`` whose connection returns *rows*."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)

    # pool.acquire() returns an async context manager yielding *conn*
    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acm
    return pool


def _mock_pool_fetchval(value: Any) -> AsyncMock:
    """Build a mock pool whose connection.fetchval() returns *value*."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=value)

    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acm
    return pool


def _make_audit_event(
    *,
    correlation_id: str = "cid-1",
    idempotency_key: str = "key-1",
    symbol: str = "AAPL",
) -> AuditEvent:
    """Build a minimal AuditEvent for testing."""
    return AuditEvent(
        event_id="ev-1",
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        symbol=symbol,
        ts=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        source="test",
        event_type="predictions.tft",
        payload={
            "direction": "long",
            "confidence": 0.85,
            "q10": -0.01,
            "q50": 0.02,
            "q90": 0.05,
            "model_version": "v1.0",
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchPredictionSignature:
    """correlation_id must be required — no default / no optional."""

    def test_correlation_id_is_required_parameter(self) -> None:
        """Calling without correlation_id must raise TypeError."""
        pool = _mock_pool([])
        with pytest.raises(TypeError, match="correlation_id"):
            # noinspection PyArgumentList
            import asyncio
            asyncio.run(
                fetch_prediction(pool, "key-1"),  # type: ignore[call-arg]
            )

    def test_signature_has_no_default_for_correlation_id(self) -> None:
        """Introspect: correlation_id should not have a default value."""
        import inspect
        sig = inspect.signature(fetch_prediction)
        param = sig.parameters["correlation_id"]
        assert param.default is inspect.Parameter.empty, (
            "correlation_id must not have a default value"
        )


class TestFetchPredictionZeroRows:
    """When no rows match, return None."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_rows(self) -> None:
        pool = _mock_pool([])
        result = await fetch_prediction(pool, "key-1", "cid-1")
        assert result is None


class TestFetchPredictionOneRow:
    """When exactly one row matches, return the normalised dict."""

    @pytest.mark.asyncio
    async def test_returns_dict_for_single_row(self) -> None:
        row = _make_row()
        pool = _mock_pool([row])
        result = await fetch_prediction(pool, "key-1", "cid-1")
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["direction"] == "long"
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_query_uses_both_params(self) -> None:
        """The executed SQL must filter by BOTH idempotency_key AND
        correlation_id — never by idempotency_key alone."""
        row = _make_row()
        pool = _mock_pool([row])
        await fetch_prediction(pool, "key-1", "cid-1")

        # Retrieve the SQL that was passed to conn.fetch()
        conn = pool.acquire.return_value.__aenter__.return_value
        sql: str = conn.fetch.call_args[0][0]
        assert "idempotency_key" in sql
        assert "correlation_id" in sql
        # Ensure both bind variables were passed
        args = conn.fetch.call_args[0][1:]
        assert args == ("key-1", "cid-1")

    @pytest.mark.asyncio
    async def test_no_limit_clause_in_query(self) -> None:
        """The SQL must NOT contain LIMIT 1 — duplicate-detection depends
        on receiving all matching rows."""
        row = _make_row()
        pool = _mock_pool([row])
        await fetch_prediction(pool, "key-1", "cid-1")

        conn = pool.acquire.return_value.__aenter__.return_value
        sql: str = conn.fetch.call_args[0][0]
        assert "LIMIT" not in sql.upper(), (
            "LIMIT clause must not be present; duplicate detection requires "
            "all rows to be fetched"
        )


class TestFetchPredictionDuplicateDetection:
    """More than one row for the same (correlation_id, idempotency_key)
    must raise RuntimeError."""

    @pytest.mark.asyncio
    async def test_raises_on_duplicate_rows(self) -> None:
        rows = [_make_row(), _make_row(direction="short")]
        pool = _mock_pool(rows)
        with pytest.raises(RuntimeError, match="Duplicate predictions_tft rows"):
            await fetch_prediction(pool, "key-1", "cid-1")

    @pytest.mark.asyncio
    async def test_error_message_includes_identifiers(self) -> None:
        rows = [_make_row(), _make_row()]
        pool = _mock_pool(rows)
        with pytest.raises(RuntimeError) as exc_info:
            await fetch_prediction(pool, "my-key", "my-cid")
        msg = str(exc_info.value)
        assert "my-cid" in msg
        assert "my-key" in msg
        assert "2 rows" in msg

    @pytest.mark.asyncio
    async def test_three_duplicates_raises(self) -> None:
        rows = [_make_row(), _make_row(), _make_row()]
        pool = _mock_pool(rows)
        with pytest.raises(RuntimeError, match="3 rows"):
            await fetch_prediction(pool, "key-1", "cid-1")


# ---------------------------------------------------------------------------
# validate_prediction_integrity tests
# ---------------------------------------------------------------------------

class TestValidatePredictionIntegrityZeroRows:
    """Zero rows → RuntimeError('Missing prediction row')."""

    @pytest.mark.asyncio
    async def test_raises_on_zero_rows(self) -> None:
        pool = _mock_pool_fetchval(0)
        with pytest.raises(RuntimeError, match="Missing prediction row"):
            await validate_prediction_integrity(pool, "cid-missing", "key-1")

    @pytest.mark.asyncio
    async def test_error_contains_correlation_id(self) -> None:
        pool = _mock_pool_fetchval(0)
        with pytest.raises(RuntimeError) as exc_info:
            await validate_prediction_integrity(pool, "cid-abc-123", "key-1")
        assert "cid-abc-123" in str(exc_info.value)


class TestValidatePredictionIntegrityOneRow:
    """Exactly one row → no exception."""

    @pytest.mark.asyncio
    async def test_passes_for_single_row(self) -> None:
        pool = _mock_pool_fetchval(1)
        # Should not raise
        await validate_prediction_integrity(pool, "cid-ok", "key-1")


class TestValidatePredictionIntegrityDuplicateRows:
    """More than one row → RuntimeError('Duplicate')."""

    @pytest.mark.asyncio
    async def test_raises_on_two_rows(self) -> None:
        pool = _mock_pool_fetchval(2)
        with pytest.raises(RuntimeError, match="Duplicate prediction rows"):
            await validate_prediction_integrity(pool, "cid-dup", "key-1")

    @pytest.mark.asyncio
    async def test_error_contains_count(self) -> None:
        pool = _mock_pool_fetchval(5)
        with pytest.raises(RuntimeError) as exc_info:
            await validate_prediction_integrity(pool, "cid-dup", "key-1")
        assert "5 rows" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_contains_correlation_id(self) -> None:
        pool = _mock_pool_fetchval(3)
        with pytest.raises(RuntimeError) as exc_info:
            await validate_prediction_integrity(pool, "cid-xyz", "key-1")
        assert "cid-xyz" in str(exc_info.value)


# ---------------------------------------------------------------------------
# replay_prediction guard tests
# ---------------------------------------------------------------------------

class TestReplayPredictionCorrelationIdGuard:
    """replay_prediction() must refuse to run without correlation_id."""

    @pytest.mark.asyncio
    async def test_raises_value_error_when_correlation_id_empty(self) -> None:
        pool = _mock_pool([])
        event = _make_audit_event(correlation_id="")
        with pytest.raises(ValueError, match="correlation_id required"):
            await replay_prediction(pool, event)
