"""Centralised DB DSN construction.

Raises EnvironmentError at *call time* if POSTGRES_PASSWORD is not set,
so misconfigured deployments fail fast rather than silently connecting
with default credentials.

Usage (lazy construction — call inside the function/method that needs the DB,
NOT at module import time, to avoid breaking unit tests that don't set env):

    from shared.db.dsn import build_dsn

    async def connect(self) -> None:
        dsn = build_dsn()   # raises EnvironmentError if creds missing
        self._pool = await asyncpg.create_pool(dsn, ...)
"""
from __future__ import annotations

import os


def build_dsn() -> str:
    """Return a PostgreSQL DSN composed from environment variables.

    Raises:
        EnvironmentError: If ``POSTGRES_PASSWORD`` is not set. This prevents
            deployment with the default ``awet`` credential leaking into
            production.
    """
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise EnvironmentError(
            "POSTGRES_PASSWORD env var is required and was not set. "
            "Copy .env.example to .env and configure real credentials before "
            "starting any service that accesses the database."
        )
    user = os.getenv("POSTGRES_USER", "awet")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "awet")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
