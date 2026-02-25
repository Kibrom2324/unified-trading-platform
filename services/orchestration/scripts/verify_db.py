#!/usr/bin/env python3
"""
Verify TimescaleDB schema and print table stats.

Usage:
    python scripts/verify_db.py
    python scripts/verify_db.py --tables market_raw_day predictions_tft
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# All expected tables in the AWET schema
ALL_TABLES = [
    "market_raw_minute",
    "market_raw_day",
    "features_tft",
    "predictions_tft",
    "trade_decisions",
    "paper_trades",
    "trades",
    "audit_events",
    "risk_decisions",
    "backtest_runs",
    "models_registry",
    "positions",
    "daily_pnl_summary",
    "llm_traces",
]

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "market_raw_day": ["ticker", "ts", "open", "high", "low", "close", "volume"],
    "features_tft": [
        # Core columns
        "ticker", "ts", "price", "returns_1", "volatility_5",
        # Reddit columns
        "reddit_mentions_count", "reddit_sentiment_mean", "reddit_sentiment_weighted",
        "reddit_positive_ratio", "reddit_negative_ratio",
        # Calendar columns
        "is_weekend", "day_of_month", "week_of_year", "month_of_year", "is_month_end",
        # Reddit lag columns
        "reddit_mentions_lag1", "reddit_mentions_lag3", "reddit_mentions_lag5",
        "reddit_sentiment_mean_lag1", "reddit_sentiment_mean_lag3", "reddit_sentiment_mean_lag5",
        "reddit_sentiment_weighted_lag1", "reddit_sentiment_weighted_lag3", "reddit_sentiment_weighted_lag5",
        "reddit_positive_ratio_lag1", "reddit_positive_ratio_lag3", "reddit_positive_ratio_lag5",
        "reddit_negative_ratio_lag1", "reddit_negative_ratio_lag3", "reddit_negative_ratio_lag5",
        # Market benchmark columns
        "market_return_1", "market_volatility_5",
    ],
    "predictions_tft": ["ticker", "ts", "confidence", "q10", "q50", "q90"],
    "trade_decisions": ["ticker", "ts", "decision", "confidence", "pred_return"],
    "risk_decisions": ["ticker", "ts", "approved", "risk_score"],
    "trades": ["symbol", "ts", "side", "qty", "status"],
    "paper_trades": ["ticker", "ts", "side", "qty", "status"],
    "audit_events": ["symbol", "ts", "event_type", "payload"],
    "backtest_runs": ["run_id", "params", "metrics"],
}


async def verify_db(tables: list[str] | None = None) -> bool:
    import asyncpg

    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )

    print(f"Connecting to: {dsn.replace(_env('POSTGRES_PASSWORD', 'awet'), '***')}")

    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

    print("✅ Connected to TimescaleDB\n")

    tables_to_check = tables or ALL_TABLES

    # Check tables exist
    all_ok = True
    print("=" * 70)
    print(f"{'TABLE':<25} {'EXISTS':<10} {'ROW COUNT':<15} {'LATEST':<20}")
    print("=" * 70)

    for table in tables_to_check:
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = $1
            )
            """,
            table,
        )

        if exists:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            
            # Try to get latest timestamp
            latest = "-"
            try:
                # Try common timestamp columns
                for ts_col in ["ts", "created_at", "updated_at"]:
                    has_col = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
                        )
                        """,
                        table,
                        ts_col,
                    )
                    if has_col:
                        row = await conn.fetchrow(f"SELECT MAX({ts_col}) as latest FROM {table}")
                        if row and row["latest"]:
                            latest = str(row["latest"])[:19]
                        break
            except Exception:
                pass
            
            print(f"{table:<25} {'✅ YES':<10} {count:<15} {latest:<20}")

            # Check required columns
            required_cols = REQUIRED_COLUMNS.get(table)
            if required_cols:
                missing_cols = []
                for col in required_cols:
                    has_col = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
                        )
                        """,
                        table,
                        col,
                    )
                    if not has_col:
                        missing_cols.append(col)
                if missing_cols:
                    all_ok = False
                    print(f"  ⚠️  Missing columns: {', '.join(missing_cols)}")
        else:
            print(f"{table:<25} {'❌ NO':<10} {'-':<15} {'-':<20}")
            all_ok = False

    print("=" * 70)

    # Check hypertables
    print("\n📊 Hypertable Info:")
    hypertables = await conn.fetch(
        """
        SELECT hypertable_name, num_chunks
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'public'
        ORDER BY hypertable_name
        """
    )
    for ht in hypertables:
        print(f"  - {ht['hypertable_name']}: {ht['num_chunks']} chunks")

    await conn.close()

    if all_ok:
        print("\n✅ All tables exist")
    else:
        print("\n❌ Some tables missing - run: docker compose up -d")

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AWET database schema")
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Specific tables to check (default: all)",
    )
    args = parser.parse_args()

    try:
        ok = asyncio.run(verify_db(args.tables))
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        return 1
    except Exception as exc:
        print(f"❌ Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
