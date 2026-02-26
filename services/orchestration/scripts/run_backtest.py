from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.backtester_agent import BacktestRequest, BacktesterService, _parse_date

# Minimum recommended trading days for meaningful metrics
MIN_RECOMMENDED_DAYS = 180
WARN_IF_DAYS_LESS_THAN = 30


def _default_dates() -> tuple[str, str]:
    """Return default date range (1 year ending yesterday)."""
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _count_trading_days(start_str: str, end_str: str) -> int:
    """Estimate trading days (weekdays) in range."""
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days += 1
        current += timedelta(days=1)
    return days


def _parse_args() -> argparse.Namespace:
    default_start, default_end = _default_dates()

    parser = argparse.ArgumentParser(
        description="Run AWET backtest",
        epilog=f"""
Examples:
  python scripts/run_backtest.py --symbols AAPL --start 2023-01-01 --end 2024-12-31
  python scripts/run_backtest.py --symbols AAPL MSFT --initial-cash 50000

Recommended: Use at least {MIN_RECOMMENDED_DAYS} trading days for meaningful metrics.
        """,
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument(
        "--start",
        default=default_start,
        help=f"Start date YYYY-MM-DD (default: {default_start})",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        help=f"End date YYYY-MM-DD (default: {default_end})",
    )
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--max-pos-pct", type=float, default=0.1)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--fee-per-trade", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--force-short",
        action="store_true",
        help="Suppress warning for short date ranges",
    )
    return parser.parse_args()


def _format_metrics(metrics: dict) -> str:
    lines = [
        f"Total return: {metrics['total_return']:.2%}",
        f"CAGR: {metrics['cagr']:.2%}",
        f"Win rate: {metrics['win_rate']:.2%}",
        f"Max drawdown: {metrics['max_drawdown']:.2%}",
        f"Trades: {metrics['num_trades']}",
        f"Avg trade return: {metrics['avg_trade_return']:.2%}",
        f"Sharpe-like: {metrics['sharpe_like']:.2f}",
        f"Exposure pct: {metrics['exposure_pct']:.2%}",
        f"Avg holding (sec): {metrics['avg_holding_seconds']:.1f}",
        f"Final equity: {metrics['final_equity']:.2f}",
    ]
    return "\n".join(lines)


async def _run() -> int:
    args = _parse_args()

    # Validate date range
    trading_days = _count_trading_days(args.start, args.end)
    if trading_days < WARN_IF_DAYS_LESS_THAN and not args.force_short:
        print(f"⚠️  WARNING: Date range has only ~{trading_days} trading days.")
        print(f"   Recommended minimum: {MIN_RECOMMENDED_DAYS} days for meaningful metrics.")
        print(f"   Suggested command:")
        print(f"     python scripts/run_backtest.py --symbols {' '.join(args.symbols)} \\")
        print(f"       --start 2023-01-01 --end 2024-12-31 --initial-cash {int(args.initial_cash)}")
        print()
        print("   Use --force-short to suppress this warning.")
        print()

    request = BacktestRequest(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        max_pos_pct=args.max_pos_pct,
        slippage_bps=args.slippage_bps,
        fee_per_trade=args.fee_per_trade,
    )
    service = BacktesterService()
    try:
        preflight = await service.ensure_market_data(
            [s.upper() for s in args.symbols],
            _parse_date(request.start),
            _parse_date(request.end, end=True),
        )
        if not preflight["ok"]:
            if preflight["missing_tables"]:
                print("Missing tables:", ", ".join(preflight["missing_tables"]))
            if preflight["missing_rows"]:
                print("Missing data in:", ", ".join(preflight["missing_rows"]))
            for cmd in preflight["suggested_commands"]:
                print("Suggested:", cmd)
            return 2
        response = await service.run_backtest(request)
    finally:
        await service.close()

    metrics = response.metrics.model_dump()
    if response.message:
        print(response.message)
    print(f"Run ID: {response.run_id}")
    print(_format_metrics(metrics))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(response.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"Saved results to {output_path}")

    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
