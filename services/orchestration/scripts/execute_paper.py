#!/usr/bin/env python3
"""
Execute approved trade decisions in paper mode (simulated fills).

Reads from: risk_decisions + trade_decisions + market_raw_day
Writes to:  trades + audit_events

Usage:
    TRADING_ENABLED=true python scripts/execute_paper.py --symbols AAPL --start 2024-01-01 --end 2024-01-31
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

from src.core.trading_safety import TradingLimits, TradingSafetyLayer
from src.integrations.trades_repository import TradeRecord, TradesRepository


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.astimezone(timezone.utc)


def _decision_to_side(decision: str) -> str | None:
    if decision == "BUY":
        return "buy"
    if decision == "SELL":
        return "sell"
    return None


async def execute_paper(symbols: list[str], start: datetime, end: datetime) -> int:
    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )

    repo = TradesRepository(dsn)
    await repo.connect()
    limits = TradingLimits.from_env()
    safety = TradingSafetyLayer(limits=limits, trades_repo=repo)

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    inserted = 0

    try:
        async with pool.acquire() as conn:
            for symbol in symbols:
                rows = await conn.fetch(
                    """
                    SELECT r.ts, r.ticker, r.approved, r.reason, r.correlation_id, r.idempotency_key,
                           d.decision, d.confidence, m.close as price
                    FROM risk_decisions r
                    LEFT JOIN trade_decisions d
                      ON d.idempotency_key = r.idempotency_key
                    LEFT JOIN market_raw_day m
                      ON m.ticker = r.ticker AND m.ts = r.ts
                    WHERE r.ticker = $1 AND r.ts >= $2 AND r.ts <= $3
                    ORDER BY r.ts
                    """,
                    symbol,
                    start,
                    end,
                )

                for row in rows:
                    decision = row["decision"] or "HOLD"
                    side = _decision_to_side(decision)
                    ts = row["ts"]
                    price = float(row["price"] or 0.0)
                    correlation_id = row["correlation_id"]
                    idempotency_key = f"{row['idempotency_key']}:exec"

                    if not row["approved"] or side is None:
                        status = "blocked"
                        reason = row["reason"] or "not_approved_or_hold"
                        qty = 0
                        notional = 0.0
                    else:
                        # Kill switch
                        kill_check = safety.check_kill_switch()
                        if not kill_check:
                            status = "blocked"
                            reason = kill_check.reason or "kill_switch"
                            qty = 0
                            notional = 0.0
                        else:
                            qty = max(1, int(limits.max_notional_per_trade / max(price, 1.0)))
                            notional = qty * price
                            checks = [
                                safety.check_notional_limit(notional, symbol),
                                safety.check_quantity_limit(qty, symbol),
                                safety.check_daily_trade_count(),
                                safety.check_daily_notional(notional),
                                await safety.check_position_limit(symbol, notional),
                                await safety.check_total_exposure(notional),
                                await safety.check_symbol_exposure(symbol, notional),
                            ]
                            failed = [c for c in checks if not c.passed]
                            if failed:
                                status = "blocked"
                                reason = "; ".join([c.reason for c in failed if c.reason])
                                qty = 0
                                notional = 0.0
                            else:
                                status = "filled"
                                reason = "ok"

                    trade = TradeRecord(
                        symbol=symbol,
                        side=side or "buy",
                        qty=qty,
                        intended_notional=notional if qty > 0 else None,
                        avg_fill_price=price if qty > 0 else None,
                        status=status,
                        alpaca_order_id=None,
                        alpaca_status=None,
                        error_message=None if status == "filled" else reason,
                        correlation_id=str(correlation_id) if correlation_id else str(uuid4()),
                        idempotency_key=idempotency_key,
                        paper_trade=True,
                        dry_run=False,
                        ts=ts,
                    )

                    await repo.insert_trade(trade)

                    await conn.execute(
                        """
                        INSERT INTO paper_trades (
                            event_id, correlation_id, idempotency_key, ticker, ts,
                            side, qty, limit_price, fill_price, slippage_bps, status, reason
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                        ON CONFLICT (ts, idempotency_key) DO NOTHING
                        """,
                        uuid4(),
                        correlation_id,
                        idempotency_key,
                        symbol,
                        ts,
                        side or "buy",
                        qty,
                        None,
                        price if qty > 0 else None,
                        0.0,
                        status,
                        reason,
                    )

                    payload = {
                        "symbol": symbol,
                        "ts": ts.isoformat(),
                        "decision": decision,
                        "status": status,
                        "qty": qty,
                        "price": price,
                        "reason": reason,
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
                        correlation_id,
                        idempotency_key,
                        symbol,
                        ts,
                        1,
                        "execute_paper",
                        "execution.completed" if status == "filled" else "execution.blocked",
                        json.dumps(payload),
                    )
                    inserted += 1

    finally:
        await pool.close()
        await repo.close()

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute paper trades")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = _parse_date(args.start)
    end = _parse_date(args.end, end=True)

    try:
        inserted = asyncio.run(execute_paper(symbols, start, end))
        print(f"Executed {inserted} paper trades")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
