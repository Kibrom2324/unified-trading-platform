"""Structured logging for AWET pipeline.

Re-exports ``set_correlation_id`` / ``get_correlation_id`` from
:pymod:`src.core.context` so existing callers keep working without
changing their imports.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response

# ---- re-export from canonical context module --------------------------------
from src.core.context import (  # noqa: F401 – re-exported
    correlation_ctx,
    ensure_correlation_id,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)

__all__ = [
    "configure_logging",
    "correlation_middleware",
    "set_correlation_id",
    "get_correlation_id",
    "ensure_correlation_id",
    "new_correlation_id",
    "correlation_ctx",
]


def add_correlation_id(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor that injects *correlation_id* into every log event."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def configure_logging(
    level: str = "INFO",
    *,
    console: bool = False,
    stream: Any | None = None,
) -> None:
    """Configure structlog with correlation_id injection.

    Args:
        level: Log level string (e.g. "INFO", "DEBUG").
        console: When *True* use ``ConsoleRenderer`` instead of
            ``JSONRenderer``.  Useful for CLI scripts that write to
            a terminal.  Correlation-id and contextvars injection are
            always active regardless of renderer.
        stream: Optional file-like object for log output (e.g.
            ``sys.stderr``).  When *None* uses structlog defaults
            (stdout).  Useful when stdout is reserved for
            machine-readable output.
    """
    logging.basicConfig(level=level)
    renderer = (
        structlog.dev.ConsoleRenderer()
        if console
        else structlog.processors.JSONRenderer()
    )
    kwargs: dict[str, Any] = {
        "processors": [
            structlog.contextvars.merge_contextvars,
            add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        "wrapper_class": structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        "cache_logger_on_first_use": True,
    }
    if stream is not None:
        kwargs["logger_factory"] = structlog.PrintLoggerFactory(file=stream)
    structlog.configure(**kwargs)


def correlation_middleware(header_name: str) -> Callable[[Request, Callable], Any]:
    """FastAPI middleware that reads/writes ``correlation_id`` from/to a header."""

    async def middleware(request: Request, call_next: Callable) -> Response:
        cid = request.headers.get(header_name)
        if cid is None:
            cid = new_correlation_id()
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers[header_name] = cid
        return response

    return middleware
