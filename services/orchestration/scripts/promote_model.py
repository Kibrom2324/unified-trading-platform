#!/usr/bin/env python3
"""
Model Promotion Gating for AWET.

Reads backtest metrics and promotes a model to "green" only if it meets thresholds.
Otherwise marks as "rejected" with reasons.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"


@dataclass
class PromotionThresholds:
    min_sharpe: float = 0.5
    max_drawdown_pct: float = 10.0
    min_trades: int = 5
    min_win_rate: float = 0.4
    min_final_equity_ratio: float = 0.9  # Final equity / initial cash


@dataclass
class PromotionResult:
    passed: bool
    model_id: str | None
    run_id: str | None
    status: str  # "green" or "rejected"
    reasons: list[str]
    metrics: dict[str, Any]


def _build_dsn() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'awet')}"
    )


async def _get_latest_backtest(run_id: str | None = None) -> dict | None:
    """Fetch latest backtest_runs row, optionally by run_id."""
    dsn = _build_dsn()
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, timeout=10)
    try:
        async with pool.acquire() as conn:
            if run_id:
                row = await conn.fetchrow(
                    """
                    SELECT run_id, params, metrics, created_at
                    FROM backtest_runs
                    WHERE run_id = $1
                    """,
                    run_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT run_id, params, metrics, created_at
                    FROM backtest_runs
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
            if not row:
                return None
            return {
                "run_id": str(row["run_id"]),
                "params": json.loads(row["params"]) if row["params"] else {},
                "metrics": json.loads(row["metrics"]) if row["metrics"] else {},
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
    finally:
        await pool.close()


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"version": 1, "updated_at": None, "models": []}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def _find_or_create_model_entry(registry: dict, run_id: str) -> tuple[dict, bool]:
    """Find model entry by run_id or create a new one."""
    for model in registry.get("models", []):
        if model.get("run_id") == run_id:
            return model, False
    # Create new entry
    new_entry = {
        "run_id": run_id,
        "model_id": f"model_{run_id}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "candidate",
        "metrics": {},
    }
    registry.setdefault("models", []).append(new_entry)
    return new_entry, True


def evaluate_thresholds(
    metrics: dict[str, Any],
    thresholds: PromotionThresholds,
) -> tuple[bool, list[str]]:
    """
    Evaluate backtest metrics against promotion thresholds.

    Returns (passed, list_of_rejection_reasons)
    """
    reasons: list[str] = []

    # Check sharpe ratio
    sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0))
    if sharpe is None:
        sharpe = 0.0
    if sharpe < thresholds.min_sharpe:
        reasons.append(f"Sharpe {sharpe:.2f} < {thresholds.min_sharpe}")

    # Check max drawdown
    drawdown = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0.0))
    if drawdown is None:
        drawdown = 0.0
    # Handle both percentage and ratio formats
    if drawdown < 1:  # Ratio format (e.g., 0.05)
        drawdown = drawdown * 100
    if drawdown > thresholds.max_drawdown_pct:
        reasons.append(f"Max drawdown {drawdown:.1f}% > {thresholds.max_drawdown_pct}%")

    # Check trade count
    trades = metrics.get("num_trades", metrics.get("trades", 0))
    if trades is None:
        trades = 0
    if trades < thresholds.min_trades:
        reasons.append(f"Trades {trades} < {thresholds.min_trades}")

    # Check win rate
    win_rate = metrics.get("win_rate", 0.0)
    if win_rate is None:
        win_rate = 0.0
    if win_rate < thresholds.min_win_rate:
        reasons.append(f"Win rate {win_rate:.1%} < {thresholds.min_win_rate:.1%}")

    # Check final equity ratio
    final_equity = metrics.get("final_equity", 0.0)
    initial_cash = metrics.get("initial_cash", 10000.0)
    if final_equity and initial_cash:
        ratio = final_equity / initial_cash
        if ratio < thresholds.min_final_equity_ratio:
            reasons.append(f"Equity ratio {ratio:.2f} < {thresholds.min_final_equity_ratio}")

    return len(reasons) == 0, reasons


def promote_model(
    run_id: str | None,
    thresholds: PromotionThresholds,
    dry_run: bool = False,
) -> PromotionResult:
    """
    Evaluate and optionally promote a model based on backtest metrics.
    """
    # Fetch backtest data
    backtest = asyncio.run(_get_latest_backtest(run_id))
    if not backtest:
        return PromotionResult(
            passed=False,
            model_id=None,
            run_id=run_id,
            status="not_found",
            reasons=["No backtest run found"],
            metrics={},
        )

    actual_run_id = backtest["run_id"]
    metrics = backtest.get("metrics", {})

    # Evaluate thresholds
    passed, reasons = evaluate_thresholds(metrics, thresholds)
    status = "green" if passed else "rejected"

    result = PromotionResult(
        passed=passed,
        model_id=f"model_{actual_run_id}",
        run_id=actual_run_id,
        status=status,
        reasons=reasons,
        metrics=metrics,
    )

    if dry_run:
        return result

    # Update registry
    registry = _load_registry()
    model_entry, is_new = _find_or_create_model_entry(registry, actual_run_id)

    model_entry["status"] = status
    model_entry["promotion_evaluated_at"] = datetime.now(tz=timezone.utc).isoformat()
    model_entry["backtest_metrics"] = metrics

    if passed:
        model_entry["promoted_at"] = datetime.now(tz=timezone.utc).isoformat()
        model_entry["rejection_reasons"] = None
    else:
        model_entry["promoted_at"] = None
        model_entry["rejection_reasons"] = reasons

    _save_registry(registry)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and promote models based on backtest metrics"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Backtest run_id to evaluate (default: latest)",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.5,
        help="Minimum Sharpe ratio (default: 0.5)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=10.0,
        help="Maximum drawdown percentage (default: 10.0)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        help="Minimum number of trades (default: 5)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.4,
        help="Minimum win rate (default: 0.4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate without updating registry",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    return parser.parse_args()


def _print_threshold_table(
    metrics: dict[str, Any],
    thresholds: PromotionThresholds,
) -> None:
    """Print a compact threshold comparison table."""
    rows = [
        ("Sharpe", thresholds.min_sharpe, metrics.get("sharpe_ratio", metrics.get("sharpe_like", 0)), ">="),
        ("Max Drawdown %", thresholds.max_drawdown_pct, metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0) * 100), "<="),
        ("Trades", thresholds.min_trades, metrics.get("num_trades", metrics.get("trades", 0)), ">="),
        ("Win Rate", thresholds.min_win_rate, metrics.get("win_rate", 0), ">="),
    ]

    print()
    print(f"{'Metric':<16} {'Threshold':<12} {'Actual':<12} {'Pass':<6}")
    print("-" * 48)
    for name, threshold, actual, op in rows:
        if actual is None:
            actual = 0
        if op == ">=":
            passed = actual >= threshold
        else:
            passed = actual <= threshold
        status = "✓" if passed else "✗"
        if isinstance(threshold, float) and threshold < 1:
            thresh_str = f"{threshold:.2f}"
        else:
            thresh_str = f"{threshold}"
        if isinstance(actual, float):
            actual_str = f"{actual:.4f}"
        else:
            actual_str = f"{actual}"
        print(f"{name:<16} {op} {thresh_str:<9} {actual_str:<12} {status}")
    print()


def _get_failure_hints(metrics: dict[str, Any], thresholds: PromotionThresholds) -> list[str]:
    """Generate actionable hints for why promotion failed."""
    hints = []
    trades = metrics.get("num_trades", metrics.get("trades", 0)) or 0
    sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe_like", 0)) or 0

    if trades < thresholds.min_trades:
        hints.append("📉 Too few trades. Possible causes:")
        hints.append("   • Date range too short (use 6+ months)")
        hints.append("   • Decision threshold too strict (try DECISION_DEBUG_LOOSEN=true)")
        hints.append("   • No prediction signals in DB (check predictions_tft table)")
        hints.append("   • Risk agent blocking trades (check risk limits)")

    if trades > 0 and sharpe < thresholds.min_sharpe:
        hints.append("📊 Low Sharpe ratio. Possible causes:")
        hints.append("   • Random/unprofitable signal (model needs retraining)")
        hints.append("   • High slippage/fees eating returns")
        hints.append("   • Market conditions unfavorable in backtest window")

    if not hints and not metrics:
        hints.append("⚠️  No metrics found. Run a backtest first:")
        hints.append("   python scripts/run_pipeline.py --source yfinance")

    return hints


def main() -> int:
    args = parse_args()

    thresholds = PromotionThresholds(
        min_sharpe=args.min_sharpe,
        max_drawdown_pct=args.max_drawdown,
        min_trades=args.min_trades,
        min_win_rate=args.min_win_rate,
    )

    result = promote_model(
        run_id=args.run_id,
        thresholds=thresholds,
        dry_run=args.dry_run,
    )

    if args.json:
        output = {
            "passed": result.passed,
            "status": result.status,
            "model_id": result.model_id,
            "run_id": result.run_id,
            "reasons": result.reasons,
            "metrics": result.metrics,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print()
        if result.passed:
            print(f"✅ Model PROMOTED to green: {result.model_id}")
        else:
            print(f"❌ Model REJECTED: {result.model_id or 'N/A'}")

        print(f"   Run ID: {result.run_id}")
        print(f"   Status: {result.status}")

        # Print threshold comparison table
        _print_threshold_table(result.metrics, thresholds)

        # Print failure hints
        if not result.passed:
            hints = _get_failure_hints(result.metrics, thresholds)
            if hints:
                print("💡 Suggestions:")
                for hint in hints:
                    print(f"   {hint}")
                print()

        # Print suggested fix command
        if not result.passed:
            trades = result.metrics.get("num_trades", 0) or 0
            if trades < thresholds.min_trades:
                print("🔧 Quick fix: Run backtest with longer window + debug mode:")
                print("   DECISION_DEBUG_LOOSEN=true python scripts/run_backtest.py \\")
                print("     --symbols AAPL MSFT --start 2023-01-01 --end 2024-12-31")
                print()

        if args.dry_run:
            print("   (dry-run: registry not updated)")
        print()

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
