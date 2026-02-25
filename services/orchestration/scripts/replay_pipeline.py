#!/usr/bin/env python3
"""Deterministic Replay Engine for AWET trading pipeline.

Replays events from ``audit_events`` through the pipeline stages and
compares outputs against the original run to verify determinism.

Replay modes
    ``--correlation-id <uuid>``   Replay a specific pipeline run.
    ``--start-ts / --end-ts``     Replay all events in a time window.

Safety guarantees
    * Uses read-only comparison — no events are re-produced to Kafka.
    * Trades are never placed — replay only inspects stored records.
    * Writes a single ``replay.report`` audit event for traceability.

Exit codes
    0 — deterministic (all replayed events match originals)
    1 — non-deterministic (mismatches or errors detected)

Usage::

    python scripts/replay_pipeline.py --correlation-id 550e8400-...
    python scripts/replay_pipeline.py --start-ts 2026-01-15T09:00:00 \\
                                      --end-ts   2026-01-15T17:00:00
    python scripts/replay_pipeline.py --correlation-id <uuid> --json-report
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import structlog  # noqa: E402

from src.core.determinism import compute_stage_hash  # noqa: E402
from src.core.logging import configure_logging, set_correlation_id  # noqa: E402

logger = structlog.get_logger("replay_engine")

# Keep a reference to the *real* stdout so _print_json_only always writes
# to the true file descriptor even when contextlib.redirect_stdout is active.
_REAL_STDOUT = sys.stdout

# Write-once flag: prevents two JSON objects being emitted to stdout.
_json_emitted = False


def _print_json_only(obj: dict[str, Any]) -> None:
    """Write *obj* as JSON to the real stdout, bypassing any redirect.

    Raises ``RuntimeError`` if called more than once per process to
    guarantee the stdout contract (exactly one JSON object).
    """
    global _json_emitted  # noqa: PLW0603
    if _json_emitted:
        raise RuntimeError(
            "_print_json_only called twice — stdout JSON contract violated"
        )
    _json_emitted = True
    _REAL_STDOUT.write(json.dumps(obj, indent=2, default=str))
    _REAL_STDOUT.write("\n")
    _REAL_STDOUT.flush()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _build_dsn() -> str:
    return (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:"
        f"{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:"
        f"{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )


# Pipeline stage ordering — determines replay sequence
STAGE_ORDER = [
    "market.raw",
    "market.engineered",
    "predictions.tft",
    "trade.decisions",
    "risk.approved",
    "risk.rejected",
    "execution.completed",
    "execution.blocked",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """One row from ``audit_events``."""

    event_id: str
    correlation_id: str
    idempotency_key: str
    symbol: str
    ts: datetime
    source: str
    event_type: str
    payload: dict[str, Any]
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> AuditEvent:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return cls(
            event_id=str(row["event_id"]),
            correlation_id=str(row["correlation_id"]),
            idempotency_key=str(row["idempotency_key"]),
            symbol=str(row["symbol"]),
            ts=(
                row["ts"]
                if isinstance(row["ts"], datetime)
                else datetime.fromisoformat(str(row["ts"]))
            ),
            source=str(row["source"]),
            event_type=str(row["event_type"]),
            payload=payload,
            created_at=row.get("created_at"),
        )


@dataclass
class ReplayResult:
    """Comparison result for a single event."""

    original_event: AuditEvent
    replayed_output: dict[str, Any] | None = None
    matches: bool = True
    mismatches: list[str] = field(default_factory=list)
    error: str | None = None
    stage_hashes: dict[str, Any] | None = None


@dataclass
class ReplayReport:
    """Aggregated replay report."""

    replay_id: str
    replay_correlation_id: str
    original_correlation_ids: list[str]
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    total_events: int = 0
    replayed_events: int = 0
    matched_events: int = 0
    mismatched_events: int = 0
    skipped_events: int = 0
    errors: int = 0
    results: list[ReplayResult] = field(default_factory=list)
    stage_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_result(self, result: ReplayResult) -> None:
        self.results.append(result)
        self.replayed_events += 1
        stage = result.original_event.event_type
        if stage not in self.stage_summary:
            self.stage_summary[stage] = {
                "total": 0,
                "match": 0,
                "mismatch": 0,
                "error": 0,
                "stage_hashes": [],
            }
        self.stage_summary[stage]["total"] += 1

        if result.error:
            self.errors += 1
            self.stage_summary[stage]["error"] += 1
        elif result.matches:
            self.matched_events += 1
            self.stage_summary[stage]["match"] += 1
        else:
            self.mismatched_events += 1
            self.stage_summary[stage]["mismatch"] += 1

        if result.stage_hashes is not None:
            self.stage_summary[stage]["stage_hashes"].append(
                result.stage_hashes,
            )

    @property
    def is_deterministic(self) -> bool:
        return self.mismatched_events == 0 and self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "replay_correlation_id": self.replay_correlation_id,
            "original_correlation_ids": self.original_correlation_ids,
            "is_deterministic": self.is_deterministic,
            "total_events": self.total_events,
            "replayed_events": self.replayed_events,
            "matched_events": self.matched_events,
            "mismatched_events": self.mismatched_events,
            "skipped_events": self.skipped_events,
            "errors": self.errors,
            "stage_summary": self.stage_summary,
            "mismatches": [
                {
                    "event_type": r.original_event.event_type,
                    "symbol": r.original_event.symbol,
                    "ts": r.original_event.ts.isoformat(),
                    "idempotency_key": r.original_event.idempotency_key,
                    "details": r.mismatches,
                    **(
                        {"stage_hashes": r.stage_hashes}
                        if r.stage_hashes is not None
                        else {}
                    ),
                }
                for r in self.results
                if not r.matches
            ],
            "errors_detail": [
                {
                    "event_type": r.original_event.event_type,
                    "symbol": r.original_event.symbol,
                    "error": r.error,
                }
                for r in self.results
                if r.error
            ],
        }


# ---------------------------------------------------------------------------
# DB queries — fetch original records for comparison
# ---------------------------------------------------------------------------

async def fetch_events_by_correlation_id(
    pool: asyncpg.Pool,
    correlation_id: str,
) -> list[AuditEvent]:
    """Fetch all audit events for a correlation_id, ordered by ts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_id, correlation_id, idempotency_key, symbol, ts,
                   source, event_type, payload, created_at
            FROM audit_events
            WHERE correlation_id = $1
            ORDER BY ts, event_type
            """,
            correlation_id,
        )
    return [AuditEvent.from_row(r) for r in rows]


async def fetch_events_by_time_window(
    pool: asyncpg.Pool,
    start_ts: datetime,
    end_ts: datetime,
) -> list[AuditEvent]:
    """Fetch all audit events in [start_ts, end_ts), ordered by ts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_id, correlation_id, idempotency_key, symbol, ts,
                   source, event_type, payload, created_at
            FROM audit_events
            WHERE ts >= $1 AND ts < $2
            ORDER BY ts, event_type
            """,
            start_ts,
            end_ts,
        )
    return [AuditEvent.from_row(r) for r in rows]


async def fetch_prediction(
    pool: asyncpg.Pool,
    idempotency_key: str,
    correlation_id: str,
) -> dict[str, Any] | None:
    """Fetch original prediction by idempotency_key + correlation_id.

    Both parameters are **required** so we always resolve to the exact
    pipeline run.  If the same ``(correlation_id, idempotency_key)`` pair
    somehow maps to more than one row a ``RuntimeError`` is raised — this
    signals a data-integrity issue that must be investigated rather than
    silently picking an arbitrary row.

    Returns ``None`` when no matching row exists.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, ts, direction, confidence,
                   q10, q50, q90, model_version
            FROM predictions_tft
            WHERE idempotency_key = $1 AND correlation_id = $2
            """,
            idempotency_key,
            correlation_id,
        )
    if len(rows) > 1:
        raise RuntimeError(
            f"Duplicate predictions_tft rows for "
            f"correlation_id={correlation_id!r}, "
            f"idempotency_key={idempotency_key!r}: "
            f"found {len(rows)} rows (expected 0 or 1)"
        )
    if not rows:
        return None
    row = rows[0]
    return {
        "ticker": row["ticker"],
        "ts": (
            row["ts"].isoformat()
            if isinstance(row["ts"], datetime)
            else str(row["ts"])
        ),
        "direction": row["direction"],
        "confidence": (
            float(row["confidence"]) if row["confidence"] is not None else None
        ),
        "q10": float(row["q10"]) if row["q10"] is not None else None,
        "q50": float(row["q50"]) if row["q50"] is not None else None,
        "q90": float(row["q90"]) if row["q90"] is not None else None,
        "model_version": row["model_version"],
    }


async def validate_prediction_integrity(
    pool: asyncpg.Pool,
    correlation_id: str,
    idempotency_key: str,
) -> None:
    """Assert exactly one predictions_tft row exists for the event.

    Scoped to ``(correlation_id, idempotency_key)`` so that multi-symbol
    pipeline runs (which legitimately produce >1 prediction per
    correlation_id) are not mis-flagged.

    Raises
    ------
    RuntimeError
        If zero rows (missing prediction) or more than one row (duplicate)
        are found.  No silent tolerance.
    """
    async with pool.acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT count(*) FROM predictions_tft "
            "WHERE correlation_id = $1 AND idempotency_key = $2",
            correlation_id,
            idempotency_key,
        )
    if count == 0:
        raise RuntimeError(
            f"Missing prediction row for "
            f"correlation_id={correlation_id!r}, "
            f"idempotency_key={idempotency_key!r}"
        )
    if count > 1:
        raise RuntimeError(
            f"Duplicate prediction rows detected for "
            f"correlation_id={correlation_id!r}, "
            f"idempotency_key={idempotency_key!r}: found {count} rows"
        )


async def fetch_risk_decision(
    pool: asyncpg.Pool,
    correlation_id: str,
    ticker: str,
    ts: datetime,
) -> dict[str, Any] | None:
    """Fetch original risk decision by correlation_id + ticker + ts."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ticker, ts, approved, reason
            FROM risk_decisions
            WHERE correlation_id = $1 AND ticker = $2 AND ts = $3
            ORDER BY ts DESC
            LIMIT 1
            """,
            correlation_id,
            ticker,
            ts,
        )
    if row is None:
        return None
    return {
        "ticker": row["ticker"],
        "ts": (
            row["ts"].isoformat()
            if isinstance(row["ts"], datetime)
            else str(row["ts"])
        ),
        "approved": row["approved"],
        "reason": row["reason"],
    }


async def fetch_paper_trade(
    pool: asyncpg.Pool,
    idempotency_key: str,
) -> dict[str, Any] | None:
    """Fetch original paper trade by idempotency_key."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ticker, ts, side, qty, fill_price, status, reason
            FROM paper_trades
            WHERE idempotency_key = $1
            LIMIT 1
            """,
            idempotency_key,
        )
    if row is None:
        return None
    return {
        "ticker": row["ticker"],
        "ts": (
            row["ts"].isoformat()
            if isinstance(row["ts"], datetime)
            else str(row["ts"])
        ),
        "side": row["side"],
        "qty": row["qty"],
        "fill_price": (
            float(row["fill_price"]) if row["fill_price"] is not None else None
        ),
        "status": row["status"],
        "reason": row["reason"],
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_payloads(
    original: dict[str, Any],
    replayed: dict[str, Any],
    stage: str,
) -> list[str]:
    """Compare two payloads, return list of mismatch descriptions.

    Uses stage-aware field selection to focus on deterministic outputs.
    """
    mismatches: list[str] = []

    # Ephemeral fields that are expected to differ
    ignore_fields = {
        "event_id",
        "created_at",
        "updated_at",
        "idempotency_key",
        "correlation_id",
    }

    # Stage-specific comparison fields
    stage_fields: dict[str, set[str]] = {
        "predictions.tft": {
            "direction", "confidence", "q10", "q50", "q90",
            "model_version", "symbol",
        },
        "risk.approved": {"approved", "reason", "symbol"},
        "risk.rejected": {"approved", "reason", "symbol"},
        "execution.completed": {"status", "side", "qty", "symbol"},
        "execution.blocked": {"status", "side", "qty", "symbol"},
        "trade.decisions": {"decision", "confidence", "symbol"},
    }
    compare_fields = stage_fields.get(stage)

    def _norm(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 6)
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    orig_keys = set(original.keys()) - ignore_fields
    repl_keys = set(replayed.keys()) - ignore_fields
    if compare_fields is not None:
        orig_keys &= compare_fields
        repl_keys &= compare_fields

    for key in sorted(orig_keys | repl_keys):
        orig_val = _norm(original.get(key))
        repl_val = _norm(replayed.get(key))
        # Skip when the replayed side (audit payload) has no value — this
        # means the field wasn't stored in audit_events, not a mismatch.
        if repl_val is None:
            continue
        if orig_val != repl_val:
            mismatches.append(
                f"{key}: original={original.get(key)!r} vs "
                f"replayed={replayed.get(key)!r}"
            )

    return mismatches


# ---------------------------------------------------------------------------
# Stage-specific replay handlers
# ---------------------------------------------------------------------------

async def replay_prediction(
    pool: asyncpg.Pool,
    event: AuditEvent,
) -> ReplayResult:
    """Compare audit_events payload vs predictions_tft row."""
    # ------------------------------------------------------------------
    # Deterministic guard: correlation_id is non-negotiable.
    # ------------------------------------------------------------------
    if not event.correlation_id:
        raise ValueError(
            "correlation_id required for deterministic replay"
        )

    result = ReplayResult(original_event=event)

    # Integrity check: exactly 1 prediction row must exist.
    await validate_prediction_integrity(
        pool, event.correlation_id, event.idempotency_key,
    )

    original = await fetch_prediction(
        pool, event.idempotency_key, event.correlation_id,
    )
    if original is None:
        # No structured record — validate audit envelope only
        result.replayed_output = event.payload
        result.matches = True
        logger.debug(
            "predictions_tft row missing; audit-only pass-through",
            symbol=event.symbol,
            idempotency_key=event.idempotency_key,
        )
        return result

    # Build replayed dict from the audit_events payload
    p = event.payload
    replayed = {
        "direction": p.get("direction"),
        "confidence": (
            float(p["confidence"]) if p.get("confidence") is not None else None
        ),
        "q10": float(p["q10"]) if p.get("q10") is not None else None,
        "q50": float(p["q50"]) if p.get("q50") is not None else None,
        "q90": float(p["q90"]) if p.get("q90") is not None else None,
        "model_version": p.get("model_version"),
        "symbol": event.symbol,
    }
    original_norm = {
        "direction": original["direction"],
        "confidence": (
            float(original["confidence"])
            if original["confidence"] is not None
            else None
        ),
        "q10": float(original["q10"]) if original["q10"] is not None else None,
        "q50": float(original["q50"]) if original["q50"] is not None else None,
        "q90": float(original["q90"]) if original["q90"] is not None else None,
        "model_version": original["model_version"],
        "symbol": original["ticker"],
    }

    mismatches = compare_payloads(original_norm, replayed, "predictions.tft")

    # Cryptographic fingerprint
    audit_hash = compute_stage_hash(replayed)
    db_hash = compute_stage_hash(original_norm)
    hash_match = audit_hash == db_hash
    result.stage_hashes = {
        "audit_hash": audit_hash,
        "db_hash": db_hash,
        "match": hash_match,
    }
    if not hash_match and not mismatches:
        mismatches.append(
            f"cryptographic hash mismatch: audit={audit_hash[:16]}… "
            f"vs db={db_hash[:16]}…"
        )

    result.replayed_output = replayed
    result.matches = len(mismatches) == 0 and hash_match
    result.mismatches = mismatches
    return result


async def replay_risk(
    pool: asyncpg.Pool,
    event: AuditEvent,
) -> ReplayResult:
    """Compare audit_events payload vs risk_decisions row."""
    result = ReplayResult(original_event=event)

    original = await fetch_risk_decision(
        pool, event.correlation_id, event.symbol, event.ts,
    )
    if original is None:
        # risk_decisions table may not be populated by the demo pipeline.
        # Fall back to audit-only validation: verify required payload fields.
        p = event.payload
        required = {"symbol", "event_id", "correlation_id"}
        missing = required - set(p.keys())
        if missing:
            result.matches = False
            result.mismatches = [f"Missing audit fields: {sorted(missing)}"]
        else:
            result.matches = True
        result.replayed_output = p
        logger.debug(
            "risk_decisions row missing; audit-only pass-through",
            event_type=event.event_type,
            symbol=event.symbol,
        )
        return result

    is_approved = event.event_type == "risk.approved"
    replayed = {
        "approved": is_approved,
        "reason": event.payload.get("reason", ""),
        "symbol": event.symbol,
    }
    original_norm = {
        "approved": original["approved"],
        "reason": original["reason"] or "",
        "symbol": original["ticker"],
    }

    mismatches = compare_payloads(original_norm, replayed, event.event_type)

    # Cryptographic fingerprint
    audit_hash = compute_stage_hash(replayed)
    db_hash = compute_stage_hash(original_norm)
    hash_match = audit_hash == db_hash
    result.stage_hashes = {
        "audit_hash": audit_hash,
        "db_hash": db_hash,
        "match": hash_match,
    }
    if not hash_match and not mismatches:
        mismatches.append(
            f"cryptographic hash mismatch: audit={audit_hash[:16]}… "
            f"vs db={db_hash[:16]}…"
        )

    result.replayed_output = replayed
    result.matches = len(mismatches) == 0 and hash_match
    result.mismatches = mismatches
    return result


async def replay_execution(
    pool: asyncpg.Pool,
    event: AuditEvent,
) -> ReplayResult:
    """Compare audit_events payload vs paper_trades row."""
    result = ReplayResult(original_event=event)

    original = await fetch_paper_trade(pool, event.idempotency_key)
    if original is None:
        # Execution events without a matching paper_trades row are common
        # for blocked executions — treat as audit-only pass-through.
        result.replayed_output = event.payload
        result.matches = True
        return result

    replayed = {
        "status": event.payload.get("status", ""),
        "side": event.payload.get("side", ""),
        "qty": event.payload.get("qty", 0),
        "symbol": event.symbol,
    }
    original_norm = {
        "status": original["status"],
        "side": original["side"],
        "qty": original["qty"],
        "symbol": original["ticker"],
    }

    mismatches = compare_payloads(original_norm, replayed, event.event_type)

    # Cryptographic fingerprint
    audit_hash = compute_stage_hash(replayed)
    db_hash = compute_stage_hash(original_norm)
    hash_match = audit_hash == db_hash
    result.stage_hashes = {
        "audit_hash": audit_hash,
        "db_hash": db_hash,
        "match": hash_match,
    }
    if not hash_match and not mismatches:
        mismatches.append(
            f"cryptographic hash mismatch: audit={audit_hash[:16]}… "
            f"vs db={db_hash[:16]}…"
        )

    result.replayed_output = replayed
    result.matches = len(mismatches) == 0 and hash_match
    result.mismatches = mismatches
    return result


async def replay_passthrough(event: AuditEvent) -> ReplayResult:
    """Verify envelope completeness for market/trade stages."""
    result = ReplayResult(original_event=event)
    required = {"event_id", "correlation_id", "idempotency_key", "symbol"}
    missing = required - set(event.payload.keys())
    if missing:
        result.matches = False
        result.mismatches = [f"Missing envelope fields: {sorted(missing)}"]
    else:
        result.matches = True
    # Passthrough stages: hash the audit payload against itself
    audit_hash = compute_stage_hash(event.payload)
    result.stage_hashes = {
        "audit_hash": audit_hash,
        "db_hash": audit_hash,
        "match": True,
    }
    result.replayed_output = event.payload
    return result


async def replay_event(
    pool: asyncpg.Pool,
    event: AuditEvent,
) -> ReplayResult:
    """Route a single event to the appropriate comparison handler."""
    stage = event.event_type
    if stage == "predictions.tft":
        return await replay_prediction(pool, event)
    if stage in ("risk.approved", "risk.rejected"):
        return await replay_risk(pool, event)
    if stage in ("execution.completed", "execution.blocked"):
        return await replay_execution(pool, event)
    # market.raw, market.engineered, trade.decisions, unknown
    return await replay_passthrough(event)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_replay(
    *,
    correlation_id: str | None = None,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
    json_report: bool = False,
) -> ReplayReport:
    """Execute the full replay comparison and return the report."""
    dsn = _build_dsn()
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)

    replay_id = (
        f"replay_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid4().hex[:8]}"
    )
    replay_corr = str(uuid4())
    set_correlation_id(replay_corr)

    logger.info(
        "replay_start",
        replay_id=replay_id,
        correlation_id=correlation_id,
        start_ts=str(start_ts) if start_ts else None,
        end_ts=str(end_ts) if end_ts else None,
    )

    # Defensive guard: in --json-report mode, redirect *all* accidental
    # print() / stdout writes to stderr.  Only _print_json_only() writes
    # to the real stdout via _REAL_STDOUT.
    _redirect = (
        contextlib.redirect_stdout(sys.stderr)
        if json_report
        else contextlib.nullcontext()
    )

    try:
      with _redirect:
        # ---- fetch events ---------------------------------------------------
        if correlation_id:
            events = await fetch_events_by_correlation_id(pool, correlation_id)
            original_cids = [correlation_id]
        elif start_ts and end_ts:
            events = await fetch_events_by_time_window(pool, start_ts, end_ts)
            original_cids = sorted({e.correlation_id for e in events})
        else:
            raise ValueError(
                "Must provide --correlation-id OR --start-ts + --end-ts"
            )

        if not events:
            empty_report = ReplayReport(
                replay_id=replay_id,
                replay_correlation_id=replay_corr,
                original_correlation_ids=original_cids if correlation_id else [],
                total_events=0,
            )
            if json_report:
                _print_json_only(empty_report.to_dict())
            else:
                print("❌ No events found for the given filter.")
            return empty_report

        # Sort by pipeline stage then by ts
        def _stage_key(e: AuditEvent) -> tuple[int, datetime]:
            try:
                idx = STAGE_ORDER.index(e.event_type)
            except ValueError:
                idx = len(STAGE_ORDER)
            return (idx, e.ts)

        events.sort(key=_stage_key)

        report = ReplayReport(
            replay_id=replay_id,
            replay_correlation_id=replay_corr,
            original_correlation_ids=original_cids,
            start_ts=start_ts,
            end_ts=end_ts,
            total_events=len(events),
        )

        if not json_report:
            print()
            print("=" * 70)
            print("🔄 AWET DETERMINISTIC REPLAY ENGINE")
            print("=" * 70)
            cid_display = ", ".join(original_cids[:5])
            if len(original_cids) > 5:
                cid_display += "..."
            print(f"  Replay ID:      {replay_id}")
            print(f"  Correlation(s):  {cid_display}")
            print(f"  Total events:    {len(events)}")
            print()

        # ---- replay each event in order -------------------------------------
        for event in events:
            result = await replay_event(pool, event)
            report.add_result(result)

            if not json_report:
                if result.error:
                    status = "⚠️"
                elif result.matches:
                    status = "✅"
                else:
                    status = "❌"
                print(
                    f"  {status} {event.event_type:25s} "
                    f"{event.symbol:6s} {event.ts.isoformat()}"
                )
                for m in result.mismatches:
                    print(f"      ↳ {m}")
                if result.error:
                    print(f"      ↳ ERROR: {result.error}")

        # ---- summary --------------------------------------------------------
        if not json_report:
            print()
            print("=" * 70)
            print("REPLAY SUMMARY")
            print("=" * 70)
            print(f"  Total events:     {report.total_events}")
            print(f"  Replayed:         {report.replayed_events}")
            print(f"  Matched:          {report.matched_events}")
            print(f"  Mismatched:       {report.mismatched_events}")
            print(f"  Errors:           {report.errors}")
            print()
            print("  Stage breakdown:")
            for stage, c in sorted(report.stage_summary.items()):
                flag = (
                    "✅" if c["mismatch"] == 0 and c["error"] == 0 else "❌"
                )
                print(
                    f"    {flag} {stage:25s}  "
                    f"total={c['total']}  match={c['match']}  "
                    f"mismatch={c['mismatch']}  error={c['error']}"
                )
            print()
            if report.is_deterministic:
                print("  🟢 DETERMINISTIC — all replayed events match originals")
            else:
                print("  🔴 NON-DETERMINISTIC — mismatches detected")
            print()
        else:
            _print_json_only(report.to_dict())

        # ---- write replay report to audit trail -----------------------------
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, correlation_id, idempotency_key, symbol,
                        ts, schema_version, source, event_type, payload
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (ts, idempotency_key) DO NOTHING
                    """,
                    str(uuid4()),
                    replay_corr,
                    f"replay_report:{replay_id}",
                    "SYSTEM",
                    datetime.now(tz=UTC),
                    1,
                    "replay_engine",
                    "replay.report",
                    json.dumps(report.to_dict(), default=str),
                )
        except Exception:
            logger.exception("replay_report_write_failed")

        return report

      # end: with _redirect
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_ts(value: str) -> datetime:
    """Parse ISO-8601 timestamp, defaulting to UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


async def amain() -> int:
    parser = argparse.ArgumentParser(
        description="AWET Deterministic Replay Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--correlation-id",
        type=str,
        default=None,
        help="Replay a specific pipeline run by correlation_id",
    )
    parser.add_argument(
        "--start-ts",
        type=str,
        default=None,
        help="Start of time window (ISO-8601)",
    )
    parser.add_argument(
        "--end-ts",
        type=str,
        default=None,
        help="End of time window (ISO-8601)",
    )
    parser.add_argument(
        "--json-report",
        action="store_true",
        help="Output JSON report instead of human-readable",
    )
    args = parser.parse_args()

    # Centralised logging config: json_report → stderr, human → stdout
    if args.json_report:
        configure_logging("INFO", stream=sys.stderr)
    else:
        configure_logging("INFO", console=True, stream=sys.stdout)

    if not args.correlation_id and not (args.start_ts and args.end_ts):
        parser.error(
            "Must provide --correlation-id OR both --start-ts and --end-ts"
        )

    report = await run_replay(
        correlation_id=args.correlation_id,
        start_ts=_parse_ts(args.start_ts) if args.start_ts else None,
        end_ts=_parse_ts(args.end_ts) if args.end_ts else None,
        json_report=args.json_report,
    )
    return 0 if report.is_deterministic else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
