#!/usr/bin/env python3
"""
Show Last Trades - Query and display recent trades from TimescaleDB.

Usage:
    python scripts/show_last_trades.py                  # Last 20 trades
    python scripts/show_last_trades.py -n 50            # Last 50 trades
    python scripts/show_last_trades.py --json           # JSON output
    python scripts/show_last_trades.py --watch          # Continuous monitoring
    python scripts/show_last_trades.py --source=trades  # Query from trades table
    python scripts/show_last_trades.py --source=audit   # Query from audit_events (default)

This script displays:
- Symbol, Side, Status
- Quantity and intended notional
- Alpaca order ID and status
- Safety flags (paper_trade, dry_run)
- Error messages (if blocked)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import asyncpg
except ImportError:
    print("Error: asyncpg not installed. Run: pip install asyncpg")
    sys.exit(1)


def get_db_dsn() -> str:
    """Get database connection string from environment or defaults."""
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5433')}/"
        f"{os.getenv('POSTGRES_DB', 'awet')}"
    )


async def fetch_trades_from_audit(limit: int = 20) -> list[dict]:
    """Fetch recent execution events from audit_events table."""
    conn = await asyncpg.connect(get_db_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT 
                ts,
                event_type,
                payload->>'symbol' as symbol,
                payload->>'status' as status,
                payload->>'paper_trade' as paper_trade,
                payload->>'dry_run' as dry_run,
                payload->>'filled_qty' as filled_qty,
                payload->>'avg_price' as avg_price,
                payload->>'side' as side,
                payload->>'alpaca_order_id' as alpaca_order_id,
                payload->>'alpaca_status' as alpaca_status,
                payload->>'error_message' as error_message,
                payload->>'correlation_id' as correlation_id
            FROM audit_events 
            WHERE event_type LIKE 'execution.%'
            ORDER BY ts DESC 
            LIMIT $1
            """,
            limit
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def fetch_trades_from_trades_table(limit: int = 20) -> list[dict]:
    """Fetch recent trades from the dedicated trades table."""
    conn = await asyncpg.connect(get_db_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT 
                ts,
                CASE WHEN status = 'filled' THEN 'execution.completed' ELSE 'execution.blocked' END as event_type,
                symbol,
                side,
                qty as filled_qty,
                intended_notional,
                avg_fill_price as avg_price,
                status,
                alpaca_order_id,
                alpaca_status,
                error_message,
                correlation_id,
                paper_trade::text,
                dry_run::text
            FROM trades 
            ORDER BY ts DESC 
            LIMIT $1
            """,
            limit
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def fetch_trades(limit: int = 20, source: str = "audit") -> list[dict]:
    """Fetch trades from the specified source."""
    if source == "trades":
        return await fetch_trades_from_trades_table(limit)
    return await fetch_trades_from_audit(limit)


def format_table(trades: list[dict], source: str = "audit") -> str:
    """Format trades as a readable ASCII table."""
    if not trades:
        return "No execution events found."
    
    # Header
    source_label = "TRADES TABLE" if source == "trades" else "AUDIT EVENTS"
    lines = [
        "",
        "=" * 130,
        f"RECENT EXECUTION EVENTS (Source: {source_label})",
        "=" * 130,
        f"{'TIME (UTC)':<20} {'TYPE':<12} {'SYMBOL':<8} {'STATUS':<10} "
        f"{'SIDE':<6} {'QTY':<5} {'NOTIONAL':<10} {'ALPACA_STATUS':<15} {'ALPACA_ORDER_ID':<20}",
        "-" * 130,
    ]
    
    # Rows
    for trade in trades:
        ts = trade['ts']
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)[:19]
        
        event_type = trade['event_type'].replace('execution.', '')
        symbol = trade['symbol'] or '-'
        status = trade['status'] or '-'
        side = trade.get('side') or 'buy'
        qty = trade['filled_qty'] or '0'
        
        # Handle intended_notional (trades table) or calculate from avg_price (audit)
        notional = trade.get('intended_notional')
        if notional:
            notional_str = f"${float(notional):,.0f}"
        elif trade.get('avg_price') and trade.get('filled_qty'):
            try:
                notional_str = f"${float(trade['avg_price']) * float(trade['filled_qty']):,.0f}"
            except (ValueError, TypeError):
                notional_str = "-"
        else:
            notional_str = "-"
        
        alpaca_status = trade.get('alpaca_status') or '-'
        alpaca_order_id = trade.get('alpaca_order_id') or '-'
        error_msg = trade.get('error_message')
        
        # Truncate order ID for display
        if alpaca_order_id and len(alpaca_order_id) > 18:
            alpaca_order_id = alpaca_order_id[:8] + "..."
        
        # Color-code status
        if status == 'blocked':
            status_display = f"🔴 {status}"
        else:
            status_display = f"🟢 {status}"
        
        lines.append(
            f"{ts_str:<20} {event_type:<12} {symbol:<8} {status_display:<10} "
            f"{side:<6} {qty:<5} {notional_str:<10} {alpaca_status:<15} {alpaca_order_id:<20}"
        )
        
        # Show error message if present
        if error_msg:
            lines.append(f"    ⚠️  {error_msg}")
    
    lines.append("-" * 130)
    
    # Summary
    blocked = sum(1 for t in trades if t['status'] == 'blocked')
    filled = sum(1 for t in trades if t['status'] == 'filled')
    lines.append(f"Total: {len(trades)} | Blocked: {blocked} | Filled: {filled}")
    lines.append("")
    
    return "\n".join(lines)


async def watch_trades(interval: float = 5.0) -> None:
    """Continuously watch for new trades."""
    print("Watching for execution events... (Ctrl+C to stop)")
    print("-" * 60)
    
    last_ts = None
    while True:
        try:
            trades = await fetch_trades(limit=5)
            if trades:
                newest_ts = trades[0]['ts']
                if last_ts is None or newest_ts > last_ts:
                    for trade in reversed(trades):
                        if last_ts is None or trade['ts'] > last_ts:
                            ts_str = trade['ts'].strftime("%H:%M:%S") if isinstance(trade['ts'], datetime) else str(trade['ts'])[:8]
                            status = "🔴 BLOCKED" if trade['status'] == 'blocked' else "🟢 FILLED"
                            dry = " [DRY-RUN]" if trade['dry_run'] == 'true' else ""
                            print(f"[{ts_str}] {trade['symbol']:>6} {status}{dry}")
                    last_ts = newest_ts
        except Exception as e:
            print(f"Error: {e}")
        
        await asyncio.sleep(interval)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show recent execution events from AWET trading pipeline"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=20,
        help="Number of trades to show (default: 20)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch for new trades"
    )
    parser.add_argument(
        "--source",
        choices=["audit", "trades"],
        default="audit",
        help="Data source: 'audit' for audit_events table, 'trades' for trades table (default: audit)"
    )
    
    args = parser.parse_args()
    
    if args.watch:
        try:
            await watch_trades()
        except KeyboardInterrupt:
            print("\nStopped watching.")
        return
    
    trades = await fetch_trades(args.limit, source=args.source)
    
    if args.json:
        # Convert datetime to string for JSON serialization
        for trade in trades:
            if isinstance(trade['ts'], datetime):
                trade['ts'] = trade['ts'].isoformat()
        print(json.dumps(trades, indent=2))
    else:
        print(format_table(trades, source=args.source))


if __name__ == "__main__":
    asyncio.run(main())
