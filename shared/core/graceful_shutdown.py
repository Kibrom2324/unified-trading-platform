"""Graceful shutdown handler for all services.

Ensures:
  - Kafka consumers commit offsets and close cleanly
  - Kafka producers flush pending messages
  - Database connections are closed
  - Redis state is flushed
  - In-flight processing completes before exit

Phase 5 hardening — prevents data loss on restart/deploy.
"""
from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger("graceful_shutdown")


class GracefulShutdown:
    """Manages orderly shutdown of service components.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register("kafka_consumer", consumer.close)
        shutdown.register("kafka_producer", producer.flush)
        shutdown.register("db_pool", pool.close)
        shutdown.install_signal_handlers()

        # ... run service ...

        # On SIGTERM/SIGINT, all registered handlers run in reverse order
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self._handlers: list[tuple[str, Callable]] = []
        self._shutting_down = False

    def register(self, name: str, handler: Callable) -> None:
        """Register a cleanup handler (called in reverse registration order)."""
        self._handlers.append((name, handler))
        logger.debug("shutdown_handler_registered", name=name)

    def install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT handlers."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("signal_handlers_installed")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("shutdown_signal_received", signal=sig_name)
        self.shutdown()

    def shutdown(self) -> None:
        """Execute all shutdown handlers in reverse order."""
        if self._shutting_down:
            logger.warning("shutdown_already_in_progress")
            return

        self._shutting_down = True
        logger.info("graceful_shutdown_starting", handlers=len(self._handlers))

        # Run in reverse registration order (newest first)
        for name, handler in reversed(self._handlers):
            try:
                result = handler()
                # Handle async handlers
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(result)
                        else:
                            loop.run_until_complete(result)
                    except RuntimeError:
                        asyncio.run(result)
                logger.info("shutdown_handler_completed", name=name)
            except Exception as e:
                logger.error("shutdown_handler_failed", name=name, error=str(e))

        logger.info("graceful_shutdown_complete")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down


# Global instance for convenience
_shutdown = GracefulShutdown()


def register_shutdown(name: str, handler: Callable) -> None:
    """Register a cleanup handler on the global shutdown manager."""
    _shutdown.register(name, handler)


def install_shutdown_handlers() -> None:
    """Install signal handlers on the global shutdown manager."""
    _shutdown.install_signal_handlers()


def is_shutting_down() -> bool:
    return _shutdown.is_shutting_down
