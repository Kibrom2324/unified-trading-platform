"""Backfill checkpoint management.

Provides resumable backfill support by tracking processed files
in the backfill_checkpoints TimescaleDB table.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger("backfill_checkpoint")


class BackfillCheckpoint:
    """Manage backfill checkpoints for resumable processing."""

    def __init__(self, source: str, db_url: str | None = None) -> None:
        """Initialize checkpoint manager.

        Args:
            source: Source identifier (e.g., "polygon", "reddit")
            db_url: Database URL (defaults to DATABASE_URL env var)
        """
        self.source = source
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://awet:awet@localhost:5433/awet"
        )
        self._conn = None

    def _get_connection(self):
        """Get database connection (lazy init)."""
        if self._conn is None:
            import psycopg2
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = False
        return self._conn

    def is_file_completed(self, filename: str) -> bool:
        """Check if a file has been fully processed."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM backfill_checkpoints
                WHERE source = %s AND filename = %s AND completed_at IS NOT NULL
                """,
                (self.source, filename),
            )
            return cur.fetchone() is not None

    def get_last_offset(self, filename: str) -> int:
        """Get last processed offset for a file (for resume)."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT last_offset FROM backfill_checkpoints
                WHERE source = %s AND filename = %s
                """,
                (self.source, filename),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def update_progress(
        self,
        filename: str,
        last_offset: int,
        records_processed: int,
    ) -> None:
        """Update checkpoint progress (upsert)."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO backfill_checkpoints
                    (source, filename, last_offset, records_processed, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source, filename) DO UPDATE SET
                    last_offset = EXCLUDED.last_offset,
                    records_processed = backfill_checkpoints.records_processed + EXCLUDED.records_processed,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    self.source,
                    filename,
                    last_offset,
                    records_processed,
                    datetime.now(timezone.utc),
                ),
            )
        conn.commit()

    def mark_completed(self, filename: str, total_records: int) -> None:
        """Mark a file as fully processed."""
        conn = self._get_connection()
        now = datetime.now(timezone.utc)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO backfill_checkpoints
                    (source, filename, last_offset, records_processed, completed_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, filename) DO UPDATE SET
                    last_offset = EXCLUDED.last_offset,
                    records_processed = EXCLUDED.records_processed,
                    completed_at = EXCLUDED.completed_at,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    self.source,
                    filename,
                    total_records,
                    total_records,
                    now,
                    now,
                ),
            )
        conn.commit()
        logger.info("file_checkpoint_complete", source=self.source, filename=filename, records=total_records)

    def get_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics for this source."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total_files,
                    COUNT(completed_at) as completed_files,
                    SUM(records_processed) as total_records
                FROM backfill_checkpoints
                WHERE source = %s
                """,
                (self.source,),
            )
            row = cur.fetchone()
            return {
                "source": self.source,
                "total_files": row[0] or 0,
                "completed_files": row[1] or 0,
                "total_records": row[2] or 0,
            }

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
