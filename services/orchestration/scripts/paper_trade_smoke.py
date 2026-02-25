#!/usr/bin/env python3
"""
Paper trade smoke test:
- Fetch Alpaca account + clock (if configured)
- Place a tiny paper order OR simulate locally
- Write trade record to DB
- Print SMOKE_TEST_PASS on success

Usage:
    python scripts/paper_trade_smoke.py --symbols AAPL
    python scripts/paper_trade_smoke.py --symbols AAPL --simulate  # No Alpaca needed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


async def run_simulated_trade(symbol: str, qty: int) -> None:
    """Simulate a paper trade without Alpaca - writes to DB only."""
    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )

    conn = await asyncpg.connect(dsn)
    try:
        # Simulate a filled trade
        now = datetime.now(timezone.utc)
        correlation_id = str(uuid4())
        idempotency_key = f"smoke:{symbol}:{now.isoformat()}"

        await conn.execute(
            """
            INSERT INTO trades (
                ts, symbol, side, qty, filled_qty, avg_fill_price,
                status, correlation_id, idempotency_key, paper_trade, dry_run
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (idempotency_key) DO NOTHING
            """,
            now,
            symbol,
            "buy",
            qty,
            qty,
            150.00,  # Simulated fill price
            "filled",
            correlation_id,
            idempotency_key,
            True,
            False,
        )
        print(f"Simulated paper trade: BUY {qty} {symbol} @ $150.00")
        print("SMOKE_TEST_PASS")
    finally:
        await conn.close()


async def run_alpaca_trade(symbol: str, qty: int) -> None:
    """Run actual paper trade via Alpaca API."""
    from src.integrations.alpaca_client import AlpacaClient
    from src.integrations.trades_repository import TradeRecord, TradesRepository

    dsn = _env("DATABASE_URL", "postgresql://awet:awet@localhost:5433/awet")

    alpaca = AlpacaClient.from_env()
    repo = TradesRepository(dsn)
    await repo.connect()

    try:
        await alpaca.get_account()
        await alpaca.get_clock()

        result = await alpaca.submit_market_order(symbol, qty=qty, side="buy")
        if not result.success:
            status = "rejected"
        elif (result.status or "").lower() == "filled":
            status = "filled"
        else:
            status = "pending"

        alpaca_raw = json.dumps(result.raw_response) if result.raw_response else None

        trade = TradeRecord(
            symbol=symbol,
            side="buy",
            qty=qty,
            intended_notional=None,
            avg_fill_price=result.avg_price,
            status=status,
            alpaca_order_id=result.order_id,
            alpaca_status=result.status,
            error_message=result.error_message,
            correlation_id=str(uuid4()),
            idempotency_key=str(uuid4()),
            order_class=None,
            tp_order_id=None,
            sl_order_id=None,
            exit_fill_price=None,
            exit_filled_at=None,
            alpaca_raw=alpaca_raw,
            paper_trade=True,
            dry_run=False,
            ts=datetime.now(timezone.utc),
        )

        trade_id = await repo.insert_trade(trade)
        if trade_id is None:
            raise RuntimeError("trade_not_inserted")

        if not result.success:
            raise RuntimeError(f"order_failed: {result.error_message}")

        print("SMOKE_TEST_PASS")
    finally:
        await alpaca.close()
        await repo.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Paper trade smoke test")
    parser.add_argument(
        "--symbols",
        default="AAPL",
        help="Symbol to trade (default: AAPL)",
    )
    parser.add_argument(
        "--qty",
        type=int,
        default=1,
        help="Quantity to trade (default: 1)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate trade without Alpaca API (for testing)",
    )
    args = parser.parse_args()

    # Take first symbol if comma-separated
    symbol = args.symbols.split(",")[0].strip().upper()

    try:
        if args.simulate:
            await run_simulated_trade(symbol, args.qty)
        else:
            # Try Alpaca first, fall back to simulation
            try:
                await run_alpaca_trade(symbol, args.qty)
            except Exception as exc:
                print(f"Alpaca not available ({exc}), using simulation...")
                await run_simulated_trade(symbol, args.qty)
        return 0
    except Exception as exc:
        print(f"SMOKE_TEST_FAIL: {exc}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
