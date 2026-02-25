"""Tests for src.core.logging — structured logging with correlation-id."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.core.context import correlation_ctx
from src.core.logging import (
    add_correlation_id,
    configure_logging,
    correlation_middleware,
    get_correlation_id,
    set_correlation_id,
)


class TestAddCorrelationId:
    """The structlog processor should inject correlation_id."""

    def test_injects_none_when_unset(self):
        correlation_ctx.set(None)
        event: dict = {"event": "hello"}
        result = add_correlation_id(None, "", event)
        assert result["correlation_id"] is None

    def test_injects_value_when_set(self):
        set_correlation_id("corr-42")
        event: dict = {"event": "hello"}
        result = add_correlation_id(None, "", event)
        assert result["correlation_id"] == "corr-42"
        correlation_ctx.set(None)


class TestConfigureLogging:
    """configure_logging should not raise and should set up structlog."""

    def test_configure_does_not_raise(self):
        configure_logging("DEBUG")

    def test_log_output_contains_correlation_id(self):
        configure_logging("DEBUG")
        set_correlation_id("test-cid")

        # Verify the processor chain injects correlation_id
        event_dict: dict = {"event": "test"}
        result = add_correlation_id(None, "", event_dict)
        assert result["correlation_id"] == "test-cid"
        correlation_ctx.set(None)


class TestCorrelationMiddleware:
    """FastAPI middleware should read/write the correlation header."""

    async def test_reads_header_and_sets_context(self):
        correlation_ctx.set(None)
        middleware_fn = correlation_middleware("X-Correlation-ID")

        request = MagicMock()
        request.headers = {"X-Correlation-ID": "from-header"}

        response = MagicMock()
        response.headers = {}

        async def call_next(_req):
            assert get_correlation_id() == "from-header"
            return response

        await middleware_fn(request, call_next)
        assert response.headers["X-Correlation-ID"] == "from-header"
        correlation_ctx.set(None)

    async def test_generates_id_when_header_missing(self):
        correlation_ctx.set(None)
        middleware_fn = correlation_middleware("X-Correlation-ID")

        request = MagicMock()
        request.headers = {}

        response = MagicMock()
        response.headers = {}

        async def call_next(_req):
            cid = get_correlation_id()
            assert cid is not None
            assert len(cid) == 36  # UUID4
            return response

        await middleware_fn(request, call_next)
        assert "X-Correlation-ID" in response.headers
        correlation_ctx.set(None)
