#!/usr/bin/env python3
"""
show_trade_chain.py - Trace a trade back to its LLM prompts and risk decisions.

This script provides end-to-end traceability by querying:
- trades table (execution result)
- llm_traces table (what agents asked the LLM)
- audit_events table (pipeline events)
- risk_decisions table (if available)

Usage:
    python scripts/show_trade_chain.py --correlation-id <uuid>
    python scripts/show_trade_chain.py --alpaca-order-id <id>
    python scripts/show_trade_chain.py --trade-id <id>

Examples:
    # Trace by correlation ID
    python scripts/show_trade_chain.py --correlation-id abc-123-def

    # Trace by Alpaca order ID
    python scripts/show_trade_chain.py --alpaca-order-id 12345678

    # JSON output for scripting
    python scripts/show_trade_chain.py --correlation-id abc-123 --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _build_dsn() -> str:
    return (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:"
        f"{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:"
        f"{_env('POSTGRES_PORT', '5433')}/"
        f"{_env('POSTGRES_DB', 'awet')}"
    )


async def get_correlation_id_from_alpaca_order(pool: Any, order_id: str) -> str | None:
    """Look up correlation_id from Alpaca order ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT correlation_id FROM trades WHERE alpaca_order_id = $1 LIMIT 1",
            order_id,
        )
        return row["correlation_id"] if row else None


async def get_correlation_id_from_trade_id(pool: Any, trade_id: int) -> str | None:
    """Look up correlation_id from trade ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT correlation_id FROM trades WHERE id = $1 LIMIT 1",
            trade_id,
        )
        return row["correlation_id"] if row else None


async def fetch_trades(pool: Any, correlation_id: str) -> list[dict[str, Any]]:
    """Fetch trade records for a correlation ID."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ts, symbol, side, qty, intended_notional, avg_fill_price,
                   status, alpaca_order_id, alpaca_status, error_message,
                   paper_trade, dry_run
            FROM trades
            WHERE correlation_id = $1
            ORDER BY ts ASC
            """,
            correlation_id,
        )
        return [dict(row) for row in rows]


async def fetch_llm_traces(pool: Any, correlation_id: str) -> list[dict[str, Any]]:
    """Fetch LLM traces for a correlation ID."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ts, agent_name, model, latency_ms, status, error_message,
                   request, response, prompt_tokens, completion_tokens
            FROM llm_traces
            WHERE correlation_id = $1
            ORDER BY ts ASC
            """,
            correlation_id,
        )
        return [dict(row) for row in rows]


async def fetch_audit_events(pool: Any, correlation_id: str) -> list[dict[str, Any]]:
    """Fetch audit events for a correlation ID."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ts, event_type, payload
            FROM audit_events
            WHERE correlation_id::text = $1
            ORDER BY ts ASC
            LIMIT 100
            """,
            correlation_id,
        )
        return [dict(row) for row in rows]


async def fetch_risk_decisions(pool: Any, correlation_id: str) -> list[dict[str, Any]]:
    """Fetch risk decisions for a correlation ID (if table exists)."""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, symbol, decision, risk_score, reason, approved_size
                FROM risk_decisions
                WHERE correlation_id::text = $1
                ORDER BY ts ASC
                """,
                correlation_id,
            )
            return [dict(row) for row in rows]
    except Exception:
        # Table might not exist
        return []


def format_timestamp(ts: datetime | None) -> str:
    """Format timestamp for display."""
    if ts is None:
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def truncate(text: str | None, max_len: int = 80) -> str:
    """Truncate text for display."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def print_timeline(chain: dict[str, Any]) -> None:
    """Print a unified timeline view."""
    print_section("TRADE CHAIN TIMELINE")
    
    correlation_id = chain.get("correlation_id", "unknown")
    print(f"\nCorrelation ID: {correlation_id}")
    
    # Build unified timeline
    events: list[tuple[datetime, str, str, str]] = []
    
    # Add trades
    for trade in chain.get("trades", []):
        ts = trade.get("ts")
        if ts:
            status_icon = "✅" if trade.get("status") == "filled" else "❌"
            desc = f"{trade.get('side', '?').upper()} {trade.get('qty', '?')} {trade.get('symbol', '?')} @ {trade.get('avg_fill_price', 'N/A')}"
            events.append((ts, "TRADE", status_icon, desc))
    
    # Add LLM traces
    for trace in chain.get("llm_traces", []):
        ts = trace.get("ts")
        if ts:
            status_icon = "🟢" if trace.get("status") == "ok" else "🔴"
            request = trace.get("request", {})
            # Parse JSON string if needed
            if isinstance(request, str):
                try:
                    request = json.loads(request)
                except json.JSONDecodeError:
                    request = {}
            messages = request.get("messages", []) if isinstance(request, dict) else []
            prompt_preview = ""
            if messages:
                last_msg = messages[-1] if messages else {}
                prompt_preview = truncate(last_msg.get("content", ""), 50)
            desc = f"{trace.get('agent_name', '?')} → {trace.get('model', '?')} ({trace.get('latency_ms', 0):.0f}ms) | {prompt_preview}"
            events.append((ts, "LLM", status_icon, desc))
    
    # Add audit events
    for event in chain.get("audit_events", []):
        ts = event.get("ts")
        if ts:
            event_type = event.get("event_type", "?")
            payload = event.get("payload", {})
            symbol = payload.get("symbol", "") if isinstance(payload, dict) else ""
            desc = f"{event_type} {symbol}"
            events.append((ts, "AUDIT", "📝", desc))
    
    # Add risk decisions
    for decision in chain.get("risk_decisions", []):
        ts = decision.get("ts")
        if ts:
            approved = decision.get("decision", "?")
            icon = "✅" if approved in ("approved", "APPROVED") else "❌"
            desc = f"{decision.get('symbol', '?')} {approved} (score={decision.get('risk_score', 'N/A')}) | {truncate(decision.get('reason', ''), 40)}"
            events.append((ts, "RISK", icon, desc))
    
    # Sort by timestamp
    events.sort(key=lambda x: x[0])
    
    if not events:
        print("\n  No events found for this correlation ID.")
        return
    
    # Print timeline
    print()
    for ts, component, status, desc in events:
        ts_str = format_timestamp(ts)
        print(f"  {ts_str}  {status} [{component:5}] {desc}")


def print_trades_detail(trades: list[dict[str, Any]]) -> None:
    """Print detailed trade information."""
    if not trades:
        return
    
    print_section("TRADE DETAILS")
    for trade in trades:
        print(f"""
  Symbol:          {trade.get('symbol')}
  Side:            {trade.get('side', '').upper()}
  Quantity:        {trade.get('qty')}
  Status:          {trade.get('status')} {'✅' if trade.get('status') == 'filled' else '❌'}
  Fill Price:      {trade.get('avg_fill_price') or 'N/A'}
  Notional:        ${trade.get('intended_notional') or 0:.2f}
  Alpaca Order ID: {trade.get('alpaca_order_id') or 'N/A'}
  Alpaca Status:   {trade.get('alpaca_status') or 'N/A'}
  Paper Trade:     {trade.get('paper_trade')}
  Dry Run:         {trade.get('dry_run')}
  Error:           {trade.get('error_message') or 'None'}
  Timestamp:       {format_timestamp(trade.get('ts'))}
""")


def print_llm_traces_detail(traces: list[dict[str, Any]]) -> None:
    """Print detailed LLM trace information."""
    if not traces:
        return
    
    print_section("LLM TRACES")
    for i, trace in enumerate(traces, 1):
        request = trace.get("request", {})
        response = trace.get("response", {})
        
        # Parse JSON strings if needed
        if isinstance(request, str):
            try:
                request = json.loads(request)
            except json.JSONDecodeError:
                request = {}
        
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {}
        
        messages = request.get("messages", []) if isinstance(request, dict) else []
        prompt = ""
        if messages:
            last_msg = messages[-1] if messages else {}
            prompt = last_msg.get("content", "")[:500]
        
        response_content = ""
        if isinstance(response, dict):
            response_content = response.get("content", "")[:500]
        
        print(f"""
  [{i}] Agent: {trace.get('agent_name')}
      Model:   {trace.get('model')}
      Latency: {trace.get('latency_ms', 0):.0f}ms
      Status:  {trace.get('status')} {'🟢' if trace.get('status') == 'ok' else '🔴'}
      Tokens:  {trace.get('prompt_tokens', 0)} → {trace.get('completion_tokens', 0)}
      
      PROMPT:
      {truncate(prompt, 300)}
      
      RESPONSE:
      {truncate(response_content, 300)}
""")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace a trade back to its LLM prompts and decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--correlation-id", "-c",
        help="Correlation ID to trace",
    )
    parser.add_argument(
        "--alpaca-order-id", "-a",
        help="Alpaca order ID to trace (will look up correlation ID)",
    )
    parser.add_argument(
        "--trade-id", "-t",
        type=int,
        help="Trade ID to trace (will look up correlation ID)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed output (prompts, responses)",
    )
    
    args = parser.parse_args()
    
    # Must specify one identifier
    if not any([args.correlation_id, args.alpaca_order_id, args.trade_id]):
        parser.error("Must specify one of: --correlation-id, --alpaca-order-id, --trade-id")
    
    try:
        import asyncpg
    except ImportError:
        print("❌ asyncpg not installed. Run: pip install asyncpg", file=sys.stderr)
        sys.exit(1)
    
    dsn = _build_dsn()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    
    try:
        # Resolve correlation ID
        correlation_id = args.correlation_id
        
        if args.alpaca_order_id:
            correlation_id = await get_correlation_id_from_alpaca_order(pool, args.alpaca_order_id)
            if not correlation_id:
                print(f"❌ No trade found with Alpaca order ID: {args.alpaca_order_id}", file=sys.stderr)
                sys.exit(1)
        
        if args.trade_id:
            correlation_id = await get_correlation_id_from_trade_id(pool, args.trade_id)
            if not correlation_id:
                print(f"❌ No trade found with ID: {args.trade_id}", file=sys.stderr)
                sys.exit(1)
        
        # Fetch all related data
        trades = await fetch_trades(pool, correlation_id)
        llm_traces = await fetch_llm_traces(pool, correlation_id)
        audit_events = await fetch_audit_events(pool, correlation_id)
        risk_decisions = await fetch_risk_decisions(pool, correlation_id)
        
        chain = {
            "correlation_id": correlation_id,
            "trades": trades,
            "llm_traces": llm_traces,
            "audit_events": audit_events,
            "risk_decisions": risk_decisions,
            "summary": {
                "trade_count": len(trades),
                "llm_trace_count": len(llm_traces),
                "audit_event_count": len(audit_events),
                "risk_decision_count": len(risk_decisions),
            },
        }
        
        if args.json:
            # JSON output (serialize datetimes)
            def json_default(obj: Any) -> str:
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            print(json.dumps(chain, indent=2, default=json_default))
        else:
            # Human-readable output
            print_timeline(chain)
            
            if args.detail or True:  # Always show details for now
                print_trades_detail(trades)
                print_llm_traces_detail(llm_traces)
            
            # Summary
            print_section("SUMMARY")
            print(f"  Correlation ID:    {correlation_id}")
            print(f"  Trades:            {len(trades)}")
            print(f"  LLM Traces:        {len(llm_traces)}")
            print(f"  Audit Events:      {len(audit_events)}")
            print(f"  Risk Decisions:    {len(risk_decisions)}")
            
            if not any([trades, llm_traces, audit_events, risk_decisions]):
                print("\n  ⚠️  No data found for this correlation ID.")
                print("     Check that the ID is correct and data exists in the database.")
    
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
