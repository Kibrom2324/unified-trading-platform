#!/usr/bin/env python3
"""
Show LLM Traces - Query and display LLM call traces from TimescaleDB.

Usage:
    python scripts/show_llm_traces.py                     # Last 20 traces
    python scripts/show_llm_traces.py -n 50               # Last 50 traces
    python scripts/show_llm_traces.py --agent RiskAgent   # Filter by agent
    python scripts/show_llm_traces.py --model llama3.2:1b # Filter by model
    python scripts/show_llm_traces.py --since "2026-01-16T00:00:00Z"
    python scripts/show_llm_traces.py --status error      # Only errors
    python scripts/show_llm_traces.py --follow            # Continuous watch

This script displays:
- Timestamp, Agent, Model
- Latency (ms), Status
- Request and Response previews
- Token usage (if available)
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


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text for display."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


async def fetch_traces(
    limit: int = 20,
    agent: str | None = None,
    model: str | None = None,
    since: datetime | None = None,
    status: str | None = None,
) -> list[dict]:
    """Fetch LLM traces from database."""
    conn = await asyncpg.connect(get_db_dsn())
    try:
        # Build query with filters
        conditions = []
        params = []
        param_idx = 1
        
        if agent:
            conditions.append(f"agent_name = ${param_idx}")
            params.append(agent)
            param_idx += 1
        
        if model:
            conditions.append(f"model = ${param_idx}")
            params.append(model)
            param_idx += 1
        
        if since:
            conditions.append(f"ts >= ${param_idx}")
            params.append(since)
            param_idx += 1
        
        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT 
                ts,
                correlation_id,
                agent_name,
                model,
                latency_ms,
                status,
                error_message,
                request,
                response,
                prompt_tokens,
                completion_tokens
            FROM llm_traces 
            WHERE {where_clause}
            ORDER BY ts DESC 
            LIMIT ${param_idx}
        """
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]
    finally:
        await conn.close()


def format_table(traces: list[dict]) -> str:
    """Format traces as a readable ASCII table."""
    if not traces:
        return "No LLM traces found."
    
    lines = [
        "",
        "=" * 140,
        "LLM TRACES",
        "=" * 140,
        f"{'TIME (UTC)':<20} {'AGENT':<18} {'MODEL':<18} {'LATENCY':<10} {'STATUS':<8} {'TOKENS':<12} {'REQUEST PREVIEW':<30}",
        "-" * 140,
    ]
    
    for trace in traces:
        ts = trace['ts']
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)[:19]
        
        agent = truncate(trace['agent_name'] or '-', 18)
        model = truncate(trace['model'] or '-', 18)
        latency = f"{trace['latency_ms']}ms"
        status = trace['status'] or '-'
        
        # Token info
        prompt_t = trace.get('prompt_tokens') or 0
        comp_t = trace.get('completion_tokens') or 0
        tokens = f"{prompt_t}/{comp_t}" if prompt_t or comp_t else "-"
        
        # Request preview (extract user message)
        request = trace.get('request', {})
        if isinstance(request, str):
            request = json.loads(request)
        messages = request.get('messages', [])
        user_msg = ""
        for msg in messages:
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')[:40]
                break
        if not user_msg and messages:
            user_msg = messages[-1].get('content', '')[:40]
        
        # Status emoji
        status_display = "🟢 ok" if status == "ok" else "🔴 err"
        
        lines.append(
            f"{ts_str:<20} {agent:<18} {model:<18} {latency:<10} {status_display:<8} {tokens:<12} {truncate(user_msg, 30):<30}"
        )
        
        # Show error if present
        if trace.get('error_message'):
            lines.append(f"    ⚠️  {truncate(trace['error_message'], 100)}")
    
    lines.append("-" * 140)
    
    # Summary
    errors = sum(1 for t in traces if t['status'] == 'error')
    ok = sum(1 for t in traces if t['status'] == 'ok')
    avg_latency = sum(t['latency_ms'] for t in traces) / len(traces) if traces else 0
    
    lines.append(f"Total: {len(traces)} | OK: {ok} | Errors: {errors} | Avg Latency: {avg_latency:.0f}ms")
    lines.append("")
    
    return "\n".join(lines)


async def watch_traces(
    interval: float = 2.0,
    agent: str | None = None,
    model: str | None = None,
) -> None:
    """Continuously watch for new traces."""
    print(f"Watching for LLM traces... (Ctrl+C to stop)")
    if agent:
        print(f"  Filter: agent={agent}")
    if model:
        print(f"  Filter: model={model}")
    print("-" * 80)
    
    last_ts = None
    while True:
        try:
            traces = await fetch_traces(limit=10, agent=agent, model=model)
            if traces:
                for trace in reversed(traces):
                    trace_ts = trace['ts']
                    if last_ts is None or trace_ts > last_ts:
                        ts_str = trace_ts.strftime("%H:%M:%S") if isinstance(trace_ts, datetime) else str(trace_ts)[:8]
                        agent_name = trace['agent_name'][:15]
                        model_name = trace['model'][:15]
                        latency = trace['latency_ms']
                        status = "🟢" if trace['status'] == 'ok' else "🔴"
                        print(f"[{ts_str}] {status} {agent_name:<15} {model_name:<15} {latency}ms")
                if traces:
                    last_ts = traces[0]['ts']
        except Exception as e:
            print(f"Error: {e}")
        
        await asyncio.sleep(interval)


async def show_summary() -> None:
    """Show daily summary statistics."""
    conn = await asyncpg.connect(get_db_dsn())
    try:
        # Today's stats
        today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        stats = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) as total_calls,
                COUNT(*) FILTER (WHERE status = 'error') as error_count,
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency
            FROM llm_traces
            WHERE ts >= $1
            """,
            today
        )
        
        top_agents = await conn.fetch(
            """
            SELECT agent_name, COUNT(*) as cnt
            FROM llm_traces
            WHERE ts >= $1
            GROUP BY agent_name
            ORDER BY cnt DESC
            LIMIT 5
            """,
            today
        )
        
        print("\n" + "=" * 60)
        print("LLM DAILY SUMMARY")
        print("=" * 60)
        print(f"Date: {today.strftime('%Y-%m-%d')}")
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Errors: {stats['error_count']}")
        print(f"Avg Latency: {stats['avg_latency']:.0f}ms" if stats['avg_latency'] else "Avg Latency: N/A")
        print(f"Max Latency: {stats['max_latency']}ms" if stats['max_latency'] else "Max Latency: N/A")
        print("\nTop Agents:")
        for row in top_agents:
            print(f"  {row['agent_name']}: {row['cnt']} calls")
        print("")
    finally:
        await conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show LLM traces from AWET trading pipeline"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=20,
        help="Number of traces to show (default: 20)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Filter by agent name"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name"
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Show traces since timestamp (ISO format: 2026-01-16T00:00:00Z)"
    )
    parser.add_argument(
        "--status",
        choices=["ok", "error"],
        help="Filter by status"
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Continuously watch for new traces"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show daily summary statistics"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    # Parse since timestamp
    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since.replace('Z', '+00:00'))
        except ValueError:
            print(f"Invalid timestamp format: {args.since}")
            sys.exit(1)
    
    if args.summary:
        await show_summary()
        return
    
    if args.follow:
        try:
            await watch_traces(agent=args.agent, model=args.model)
        except KeyboardInterrupt:
            print("\nStopped watching.")
        return
    
    traces = await fetch_traces(
        limit=args.limit,
        agent=args.agent,
        model=args.model,
        since=since,
        status=args.status,
    )
    
    if args.json:
        for trace in traces:
            if isinstance(trace['ts'], datetime):
                trace['ts'] = trace['ts'].isoformat()
        print(json.dumps(traces, indent=2))
    else:
        print(format_table(traces))


if __name__ == "__main__":
    asyncio.run(main())
