#!/usr/bin/env python3
"""
Run risk checks on trade decisions.

Reads from: trade_decisions
Writes to:  risk_decisions + audit_events

Usage:
    python scripts/run_risk.py --symbols AAPL --start 2024-01-01 --end 2024-01-31
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

from src.risk.engine import RiskDecision, RiskEngine, RiskInput


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.astimezone(timezone.utc)


def _decision_to_direction(decision: str) -> str:
    if decision == "BUY":
        return "long"
    if decision == "SELL":
        return "short"
    return "neutral"


async def run_risk_checks(symbols: list[str], start: datetime, end: datetime) -> int:
    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    engine = RiskEngine()
    inserted = 0

    try:
        async with pool.acquire() as conn:
            for symbol in symbols:
                rows = await conn.fetch(
                    """
                    SELECT d.ts, d.ticker, d.decision, d.confidence, d.q10, d.q50, d.q90,
                           d.correlation_id, d.idempotency_key,
                           m.close as price
                    FROM trade_decisions d
                    LEFT JOIN market_raw_day m
                      ON m.ticker = d.ticker AND m.ts = d.ts
                    WHERE d.ticker = $1 AND d.ts >= $2 AND d.ts <= $3
                    ORDER BY d.ts
                    """,
                    symbol,
                    start,
                    end,
                )

                for row in rows:
                    direction = _decision_to_direction(row["decision"])
                    risk_input = RiskInput(
                        symbol=symbol,
                        price=float(row["price"] or 1.0),
                        direction=direction,
                        confidence=float(row["confidence"] or 0.0),
                        horizon_30_q10=float(row["q10"] or -0.01),
                        horizon_30_q50=float(row["q50"] or 0.0),
                        horizon_30_q90=float(row["q90"] or 0.01),
                        volatility_5=0.02,
                        volatility_15=0.025,
                        current_position=engine.state.positions.get(symbol, 0.0),
                        portfolio_value=engine.state.portfolio_value,
                        daily_pnl=engine.state.daily_pnl,
                    )

                    risk_output = engine.evaluate(risk_input)
                    approved = risk_output.decision == RiskDecision.APPROVED
                    reason = "; ".join(risk_output.reasons)

                    await conn.execute(
                        """
                        INSERT INTO risk_decisions (
                            event_id, correlation_id, idempotency_key, ticker, ts,
                            approved, reason, risk_score, cvar_95, max_position, limits_snapshot
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                        ON CONFLICT (ts, idempotency_key) DO NOTHING
                        """,
                        uuid4(),
                        row["correlation_id"],
                        row["idempotency_key"],
                        symbol,
                        row["ts"],
                        approved,
                        reason,
                        risk_output.risk_score,
                        risk_output.cvar_95,
                        risk_output.approved_size,
                        json.dumps(risk_output.to_dict()),
                    )

                    payload = {
                        "symbol": symbol,
                        "ts": row["ts"].isoformat(),
                        "decision": row["decision"],
                        "approved": approved,
                        "reason": reason,
                        "risk_score": risk_output.risk_score,
                    }
                    await conn.execute(
                        """
                        INSERT INTO audit_events (
                            event_id, correlation_id, idempotency_key, symbol, ts,
                            schema_version, source, event_type, payload
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        ON CONFLICT (ts, idempotency_key) DO NOTHING
                        """,
                        uuid4(),
                        row["correlation_id"],
                        row["idempotency_key"],
                        symbol,
                        row["ts"],
                        1,
                        "run_risk",
                        "risk.approved" if approved else "risk.rejected",
                        json.dumps(payload),
                    )
                    inserted += 1

    finally:
        await pool.close()

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Run risk checks")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = _parse_date(args.start)
    end = _parse_date(args.end, end=True)

    try:
        inserted = asyncio.run(run_risk_checks(symbols, start, end))
        print(f"Inserted {inserted} risk decisions")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
