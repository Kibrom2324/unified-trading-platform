"""Integration test: verify replay determinism.

Runs the demo pipeline (or reuses an existing run), captures the
``correlation_id``, then replays the full event chain and asserts all
outputs match the originals.

Requirements
    * Docker Compose services running  (``make up``)
    * ``execution/demo.py`` must be able to produce synthetic events

Run::

    pytest tests/integration/test_replay_determinism.py -v --timeout=180
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dsn() -> str:
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'awet')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5433')}"
        f"/{os.environ.get('POSTGRES_DB', 'awet')}"
    )


async def _latest_demo_cid() -> str | None:
    """Find the most recent demo correlation_id in audit_events."""
    try:
        conn = await asyncpg.connect(_dsn())
    except Exception:
        return None
    try:
        row = await conn.fetchrow(
            """
            SELECT correlation_id
            FROM audit_events
            WHERE source = 'synthetic_demo'
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        return str(row["correlation_id"]) if row else None
    finally:
        await conn.close()


async def _event_counts(cid: str) -> dict[str, int]:
    conn = await asyncpg.connect(_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT event_type, COUNT(*)::int AS cnt
            FROM audit_events
            WHERE correlation_id = $1
            GROUP BY event_type
            ORDER BY event_type
            """,
            cid,
        )
        return {r["event_type"]: r["cnt"] for r in rows}
    finally:
        await conn.close()


def _run_replay_json(
    correlation_id: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    """Invoke ``replay_pipeline.py --json-report`` and parse output."""
    proc = subprocess.run(
        [
            PYTHON, "scripts/replay_pipeline.py",
            "--correlation-id", correlation_id,
            "--json-report",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    report: dict[str, Any] | None = None
    if proc.stdout.strip():
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return proc, report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_correlation_id() -> str:
    """Return a correlation_id with at least one ``market.raw`` event.

    Reuses an existing demo run if one exists; otherwise runs a fresh
    ``execution.demo`` synchronously (requires Docker Compose stack).
    """
    # Try reusing an existing run
    existing = asyncio.run(_latest_demo_cid())
    if existing:
        counts = asyncio.run(_event_counts(existing))
        if counts.get("market.raw", 0) >= 1:
            return existing

    # Run fresh demo
    proc = subprocess.run(
        [PYTHON, "-m", "execution.demo"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        pytest.skip(f"Demo exited {proc.returncode}: {proc.stderr[-300:]}")

    cid = asyncio.run(_latest_demo_cid())
    if not cid:
        pytest.skip("No demo correlation_id in audit_events after demo run")
    return cid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.slow
class TestReplayDeterminism:
    """Verify that replaying a pipeline run produces identical results."""

    def test_demo_has_market_raw_events(self, demo_correlation_id: str) -> None:
        counts = asyncio.run(_event_counts(demo_correlation_id))
        assert counts.get("market.raw", 0) >= 1, (
            f"Expected >=1 market.raw events, got {counts}"
        )

    def test_replay_completes(self, demo_correlation_id: str) -> None:
        proc, report = _run_replay_json(demo_correlation_id)
        assert report is not None, (
            f"Replay produced no JSON\nstdout={proc.stdout[-500:]}\n"
            f"stderr={proc.stderr[-500:]}"
        )

    def test_replay_json_shape(self, demo_correlation_id: str) -> None:
        _, report = _run_replay_json(demo_correlation_id)
        assert report is not None
        for key in (
            "replay_id",
            "is_deterministic",
            "total_events",
            "replayed_events",
            "matched_events",
            "stage_summary",
        ):
            assert key in report, f"Missing key '{key}' in report"

    def test_replay_covers_all_events(self, demo_correlation_id: str) -> None:
        original = asyncio.run(_event_counts(demo_correlation_id))
        total_orig = sum(original.values())

        _, report = _run_replay_json(demo_correlation_id)
        assert report is not None
        # replay.report event written by replay itself is NOT part of the
        # original run, so total_events should equal the original total.
        assert report["total_events"] == total_orig, (
            f"original={total_orig} vs replay_total={report['total_events']}"
        )

    def test_replay_is_deterministic(self, demo_correlation_id: str) -> None:
        """Check per-stage determinism.

        The demo pipeline intentionally randomizes predictions, so
        ``predictions.tft`` direction mismatches are expected.  Market
        ingestion stages and risk stages (audit-only) must be fully
        deterministic.
        """
        _, report = _run_replay_json(demo_correlation_id)
        assert report is not None

        # No errors should occur (errors indicate missing data/logic bugs)
        assert report["errors"] == 0, (
            f"Replay had {report['errors']} errors:\n"
            + json.dumps(report.get("errors_detail", []), indent=2)
        )

        summary = report.get("stage_summary", {})

        # Market stages must be 100% deterministic
        for stage in ("market.raw", "market.engineered"):
            if stage in summary:
                s = summary[stage]
                assert s["mismatch"] == 0 and s["error"] == 0, (
                    f"{stage} not deterministic: {s}"
                )

        # Risk stages must match (audit-only fallback)
        for stage in ("risk.approved", "risk.rejected"):
            if stage in summary:
                s = summary[stage]
                assert s["error"] == 0, f"{stage} has errors: {s}"

        # predictions.tft mismatches are expected in demo mode
        # (demo generates random direction independently for each table)
        if "predictions.tft" in summary:
            s = summary["predictions.tft"]
            assert s["error"] == 0, f"predictions.tft has errors: {s}"

    def test_replay_twice_identical(self, demo_correlation_id: str) -> None:
        _, r1 = _run_replay_json(demo_correlation_id)
        _, r2 = _run_replay_json(demo_correlation_id)
        assert r1 is not None and r2 is not None
        assert r1["total_events"] == r2["total_events"]
        assert r1["matched_events"] == r2["matched_events"]
        assert r1["mismatched_events"] == r2["mismatched_events"]
        assert r1["errors"] == r2["errors"]
        assert r1["is_deterministic"] == r2["is_deterministic"]

    def test_replay_time_window_mode(self, demo_correlation_id: str) -> None:
        """Time-window replay must not crash, even with no matching events."""
        proc = subprocess.run(
            [
                PYTHON, "scripts/replay_pipeline.py",
                "--start-ts", "2026-01-15T00:00:00",
                "--end-ts", "2026-01-16T00:00:00",
                "--json-report",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        # Should exit cleanly (0 = deterministic, 1 = mismatch or 0 events)
        assert proc.returncode in (0, 1), (
            f"Unexpected exit code {proc.returncode}\n{proc.stderr[-500:]}"
        )

    def test_replay_missing_correlation_id(self) -> None:
        """Replay with a non-existent correlation_id should exit 1."""
        proc, report = _run_replay_json(
            "00000000-0000-0000-0000-000000000000"
        )
        # 0 events found → is_deterministic defaults True (vacuously)
        # but we accept either 0 or 1
        assert proc.returncode in (0, 1)
        if report:
            assert report["total_events"] == 0

    # ------------------------------------------------------------------ #
    #  Strict stdout-contract tests                                       #
    # ------------------------------------------------------------------ #

    def test_json_report_stdout_is_pure_json(
        self, demo_correlation_id: str,
    ) -> None:
        """stdout must contain exactly one valid JSON object — nothing else."""
        proc, report = _run_replay_json(demo_correlation_id)

        raw = proc.stdout.strip()
        assert raw, "stdout was empty"
        # Must start with '{' (no log lines before the JSON)
        assert raw.startswith("{"), (
            f"stdout does not start with '{{': {raw[:120]!r}"
        )
        # Must parse in one shot
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "replay_id" in parsed

    def test_json_report_no_structlog_on_stdout(
        self, demo_correlation_id: str,
    ) -> None:
        """No structlog keys (event, level, timestamp) may leak to stdout."""
        proc, _ = _run_replay_json(demo_correlation_id)

        raw = proc.stdout.strip()
        # Nothing before the opening brace
        lines_before = raw.split("{", 1)[0]
        assert lines_before.strip() == "", (
            f"Unexpected content before JSON object: {lines_before[:200]!r}"
        )

        # Nothing after the closing brace
        last_brace = raw.rfind("}")
        trailing = raw[last_brace + 1 :].strip() if last_brace != -1 else ""
        assert trailing == "", (
            f"Unexpected content after JSON object: {trailing[:200]!r}"
        )

        # No structlog-specific keys anywhere in the raw stdout
        for marker in ('"event":', '"level":', '"logger":'):
            # These keys never appear in ReplayReport.to_dict()
            assert marker not in raw, (
                f"structlog key {marker!r} found in stdout:\n{raw[:300]}"
            )

    def test_json_report_zero_events_is_valid_json(self) -> None:
        """0-event replay must still produce valid JSON on stdout."""
        proc, report = _run_replay_json(
            "00000000-0000-0000-0000-000000000000"
        )
        raw = proc.stdout.strip()
        assert raw, "stdout was empty for 0-event replay"
        assert raw.startswith("{"), (
            f"0-event stdout does not start with '{{': {raw[:120]!r}"
        )
        parsed = json.loads(raw)
        assert parsed["total_events"] == 0


# ---------------------------------------------------------------------------
# Deterministic-replay invariant enforcement tests
# ---------------------------------------------------------------------------

async def _insert_duplicate_prediction(
    dsn: str,
    correlation_id: str,
    idempotency_key: str,
    ticker: str = "TEST",
) -> None:
    """Insert two predictions_tft rows with the same correlation_id.

    Uses different ``ts`` values so the PK ``(ts, idempotency_key)``
    does not fire, simulating the duplicate scenario.
    """
    conn = await asyncpg.connect(dsn)
    try:
        ts1 = datetime(2099, 1, 1, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2099, 1, 1, 12, 0, 1, tzinfo=UTC)
        for ts_val in (ts1, ts2):
            event_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO predictions_tft (
                    event_id, ticker, ts, horizon_minutes,
                    direction, confidence,
                    q10, q50, q90, model_version,
                    correlation_id, idempotency_key, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now())
                ON CONFLICT DO NOTHING
                """,
                event_id,
                ticker,
                ts_val,
                30,
                "long",
                0.85,
                -0.01,
                0.02,
                0.05,
                "v-test",
                uuid.UUID(correlation_id),
                idempotency_key,
            )
    finally:
        await conn.close()


async def _cleanup_test_predictions(dsn: str, correlation_id: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        cid_uuid = uuid.UUID(correlation_id)
        await conn.execute(
            "DELETE FROM predictions_tft WHERE correlation_id = $1",
            cid_uuid,
        )
        # Also clean up any audit_events we may have inserted
        await conn.execute(
            "DELETE FROM audit_events WHERE correlation_id = $1",
            cid_uuid,
        )
    finally:
        await conn.close()


async def _insert_audit_event_for_prediction(
    dsn: str,
    correlation_id: str,
    idempotency_key: str,
    ticker: str = "TEST",
) -> None:
    """Insert a predictions.tft audit event to trigger replay_prediction."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO audit_events (
                event_id, correlation_id, idempotency_key, symbol,
                ts, schema_version, source, event_type, payload
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
            """,
            uuid.uuid4(),
            uuid.UUID(correlation_id),
            idempotency_key,
            ticker,
            datetime(2099, 1, 1, 12, 0, 0, tzinfo=UTC),
            1,
            "test_harness",
            "predictions.tft",
            json.dumps({
                "direction": "long",
                "confidence": 0.85,
                "q10": -0.01,
                "q50": 0.02,
                "q90": 0.05,
                "model_version": "v-test",
                "event_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
                "symbol": ticker,
            }),
        )
    finally:
        await conn.close()


@pytest.mark.e2e
@pytest.mark.slow
class TestReplayDuplicatePredictionDetection:
    """Replay must fail with RuntimeError when duplicate prediction rows exist."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self) -> Any:
        """Insert duplicates before test and clean up after."""
        self._cid = str(uuid.uuid4())
        self._ikey = f"test-dup-{self._cid[:8]}"
        dsn = _dsn()

        asyncio.run(_insert_duplicate_prediction(dsn, self._cid, self._ikey))
        asyncio.run(
            _insert_audit_event_for_prediction(dsn, self._cid, self._ikey),
        )
        yield
        asyncio.run(_cleanup_test_predictions(dsn, self._cid))

    def test_replay_fails_on_duplicate_predictions(self) -> None:
        """Replay against a correlation_id with duplicate predictions
        must exit non-zero and mention the correlation_id in stderr."""
        proc = subprocess.run(
            [
                PYTHON, "scripts/replay_pipeline.py",
                "--correlation-id", self._cid,
                "--json-report",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        # Replay must fail (exit code 1)
        assert proc.returncode != 0, (
            f"Expected non-zero exit, got {proc.returncode}\n"
            f"stdout={proc.stdout[:500]}\nstderr={proc.stderr[:500]}"
        )
        # Error message must contain the correlation_id
        combined = proc.stdout + proc.stderr
        assert self._cid in combined, (
            f"correlation_id {self._cid!r} not found in output:\n"
            f"stdout={proc.stdout[:500]}\nstderr={proc.stderr[:500]}"
        )

    def test_error_mentions_duplicate(self) -> None:
        """The error output must explicitly mention 'Duplicate'."""
        proc = subprocess.run(
            [
                PYTHON, "scripts/replay_pipeline.py",
                "--correlation-id", self._cid,
                "--json-report",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        combined = proc.stdout + proc.stderr
        assert "Duplicate" in combined or "duplicate" in combined, (
            f"'Duplicate' not found in output:\n{combined[:800]}"
        )


@pytest.mark.e2e
@pytest.mark.slow
class TestPredictionMigrationIndexExists:
    """The UNIQUE INDEX on predictions_tft must exist in the database."""

    def test_unique_index_on_correlation_id_idempotency_key(self) -> None:
        """Verify that a unique index covering (correlation_id, idempotency_key)
        exists on predictions_tft.  The exact index may include ``ts`` as required
        by TimescaleDB hypertables.
        """
        async def _check() -> list[dict[str, Any]]:
            conn = await asyncpg.connect(_dsn())
            try:
                rows = await conn.fetch(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'predictions_tft'
                      AND indexdef ILIKE '%UNIQUE%'
                      AND indexdef ILIKE '%correlation_id%'
                      AND indexdef ILIKE '%idempotency_key%'
                    """
                )
                return [dict(r) for r in rows]
            finally:
                await conn.close()

        indexes = asyncio.run(_check())
        assert len(indexes) >= 1, (
            "Missing UNIQUE index on predictions_tft covering "
            "(correlation_id, idempotency_key). "
            "Run: psql -f db/migrations/005_unique_prediction_per_run.sql"
        )


# ---------------------------------------------------------------------------
# Helpers for cryptographic hash tests
# ---------------------------------------------------------------------------

async def _insert_single_prediction(
    dsn: str,
    correlation_id: str,
    idempotency_key: str,
    ticker: str = "HASHTEST",
    direction: str = "long",
) -> None:
    """Insert exactly one prediction row for hash comparison tests."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO predictions_tft (
                event_id, ticker, ts, horizon_minutes,
                direction, confidence,
                q10, q50, q90, model_version,
                correlation_id, idempotency_key, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now())
            ON CONFLICT DO NOTHING
            """,
            uuid.uuid4(),
            ticker,
            datetime(2098, 6, 15, 10, 0, 0, tzinfo=UTC),
            30,
            direction,
            0.91,
            -0.02,
            0.03,
            0.07,
            "v-hash-test",
            uuid.UUID(correlation_id),
            idempotency_key,
        )
    finally:
        await conn.close()


async def _insert_audit_event_mismatched(
    dsn: str,
    correlation_id: str,
    idempotency_key: str,
    ticker: str = "HASHTEST",
    direction: str = "short",
) -> None:
    """Insert an audit event whose direction differs from the DB prediction."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO audit_events (
                event_id, correlation_id, idempotency_key, symbol,
                ts, schema_version, source, event_type, payload
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
            """,
            uuid.uuid4(),
            uuid.UUID(correlation_id),
            idempotency_key,
            ticker,
            datetime(2098, 6, 15, 10, 0, 0, tzinfo=UTC),
            1,
            "test_harness",
            "predictions.tft",
            json.dumps({
                "direction": direction,
                "confidence": 0.91,
                "q10": -0.02,
                "q50": 0.03,
                "q90": 0.07,
                "model_version": "v-hash-test",
                "event_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
                "symbol": ticker,
            }),
        )
    finally:
        await conn.close()


async def _insert_audit_event_matching(
    dsn: str,
    correlation_id: str,
    idempotency_key: str,
    ticker: str = "HASHTEST",
    direction: str = "long",
) -> None:
    """Insert an audit event that matches the DB prediction exactly."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO audit_events (
                event_id, correlation_id, idempotency_key, symbol,
                ts, schema_version, source, event_type, payload
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
            """,
            uuid.uuid4(),
            uuid.UUID(correlation_id),
            idempotency_key,
            ticker,
            datetime(2098, 6, 15, 10, 0, 0, tzinfo=UTC),
            1,
            "test_harness",
            "predictions.tft",
            json.dumps({
                "direction": direction,
                "confidence": 0.91,
                "q10": -0.02,
                "q50": 0.03,
                "q90": 0.07,
                "model_version": "v-hash-test",
                "event_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
                "symbol": ticker,
            }),
        )
    finally:
        await conn.close()


async def _cleanup_hash_test_data(dsn: str, correlation_id: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        cid_uuid = uuid.UUID(correlation_id)
        await conn.execute(
            "DELETE FROM predictions_tft WHERE correlation_id = $1",
            cid_uuid,
        )
        await conn.execute(
            "DELETE FROM audit_events WHERE correlation_id = $1",
            cid_uuid,
        )
    finally:
        await conn.close()


@pytest.mark.e2e
@pytest.mark.slow
class TestCryptographicHashMismatch:
    """Replay must detect field drift via SHA-256 hash comparison."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self) -> Any:
        """Insert prediction + mismatched audit event, clean up after."""
        self._cid = str(uuid.uuid4())
        self._ikey = f"hash-test-{self._cid[:8]}"
        dsn = _dsn()

        asyncio.run(
            _insert_single_prediction(dsn, self._cid, self._ikey),
        )
        yield
        asyncio.run(_cleanup_hash_test_data(dsn, self._cid))

    def test_hash_mismatch_on_direction_drift(self) -> None:
        """Intentionally different direction triggers hash mismatch."""
        dsn = _dsn()
        asyncio.run(
            _insert_audit_event_mismatched(
                dsn, self._cid, self._ikey,
                direction="short",
            ),
        )
        proc = subprocess.run(
            [
                PYTHON, "scripts/replay_pipeline.py",
                "--correlation-id", self._cid,
                "--json-report",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        # Exit code 1 is expected — mismatch means non-deterministic
        assert proc.returncode == 1, (
            f"Expected exit 1 (non-deterministic), got {proc.returncode}\n"
            f"stdout={proc.stdout[:500]}\nstderr={proc.stderr[:500]}"
        )
        report = json.loads(proc.stdout)
        assert report["is_deterministic"] is False, (
            "Replay should NOT be deterministic when direction differs"
        )
        assert report["mismatched_events"] >= 1
        # Verify stage_hashes present in mismatch entries
        for m in report["mismatches"]:
            if m["event_type"] == "predictions.tft":
                assert "stage_hashes" in m, (
                    "stage_hashes missing from mismatch entry"
                )
                sh = m["stage_hashes"]
                assert sh["match"] is False
                assert len(sh["audit_hash"]) == 64
                assert len(sh["db_hash"]) == 64
                assert sh["audit_hash"] != sh["db_hash"]

    def test_stage_hashes_present_when_matching(self) -> None:
        """When audit matches DB, stage_hashes still appear in stage_summary."""
        dsn = _dsn()
        asyncio.run(
            _insert_audit_event_matching(
                dsn, self._cid, self._ikey,
                direction="long",
            ),
        )
        proc = subprocess.run(
            [
                PYTHON, "scripts/replay_pipeline.py",
                "--correlation-id", self._cid,
                "--json-report",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, (
            f"Unexpected exit {proc.returncode}\n"
            f"stderr={proc.stderr[:500]}"
        )
        report = json.loads(proc.stdout)
        assert report["is_deterministic"] is True, (
            f"Replay should be deterministic: {report}"
        )
        # stage_hashes must be present in stage_summary
        stage = report["stage_summary"].get("predictions.tft", {})
        hashes = stage.get("stage_hashes", [])
        assert len(hashes) >= 1, (
            "stage_hashes missing from stage_summary for predictions.tft"
        )
        for h in hashes:
            assert h["match"] is True
            assert len(h["audit_hash"]) == 64
            assert len(h["db_hash"]) == 64
            # Both should be identical hex strings
            assert all(c in "0123456789abcdef" for c in h["audit_hash"])
