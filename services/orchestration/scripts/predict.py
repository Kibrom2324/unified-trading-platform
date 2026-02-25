#!/usr/bin/env python3
"""
Generate predictions from the latest trained model.

Reads from:  features_tft or market_raw_day (TimescaleDB)
Writes to:   predictions_tft (TimescaleDB)

Modes:
  --use-synthetic   Generate random predictions for smoke tests.
  (default)         Load the green ONNX model via ONNXInferenceEngine
                    and run real inference on a lookback window of
                    engineered features.

Usage:
    python scripts/predict.py --symbols AAPL --start 2024-01-01 --end 2024-01-31
    python scripts/predict.py --symbols AAPL,MSFT --use-synthetic
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg
import numpy as np
import structlog

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logging import configure_logging
from src.ml.onnx_engine import ONNXInferenceEngine

logger = structlog.get_logger("predict")

REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_date(value: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.astimezone(timezone.utc)


def _load_registry() -> dict:
    """Load model registry."""
    if not REGISTRY_PATH.exists():
        return {"models": [], "green": None}
    return json.loads(REGISTRY_PATH.read_text())


def _get_latest_model(registry: dict) -> dict | None:
    """Get the latest model (green if available, else newest)."""
    if registry.get("green"):
        for m in registry.get("models", []):
            if m.get("run_id") == registry["green"]:
                return m
    models = registry.get("models", [])
    if models:
        return models[-1]
    return None


def _load_engine() -> ONNXInferenceEngine | None:
    """Load the ONNX inference engine with the green model.

    Returns None if no usable model exists (caller should fall back to
    synthetic or stub predictions).
    """
    engine = ONNXInferenceEngine(use_registry=True)
    if engine.load_model():
        logger.info(
            "onnx_engine_ready",
            model_path=engine.model_path,
            model_version=engine.model_version,
            lookback=engine.lookback_window,
            num_features=engine.num_features,
        )
        return engine
    logger.warning("onnx_engine_load_failed")
    return None


async def _fetch_feature_window(
    conn: asyncpg.Connection,
    symbol: str,
    ts: datetime,
    feature_cols: list[str],
    lookback: int,
) -> np.ndarray | None:
    """Fetch `lookback` rows of features ending at `ts` for `symbol`.

    Returns an ndarray of shape (lookback, num_features) or None if
    insufficient data.
    """
    col_list = ", ".join(feature_cols)
    rows = await conn.fetch(
        f"""
        SELECT {col_list}
        FROM features_tft
        WHERE ticker = $1 AND ts <= $2
        ORDER BY ts DESC
        LIMIT $3
        """,
        symbol,
        ts,
        lookback,
    )
    if len(rows) < lookback:
        return None

    # rows are newest-first; reverse to oldest-first
    rows = list(reversed(rows))
    window = np.array(
        [[float(row[c]) if row[c] is not None else 0.0 for c in feature_cols] for row in rows],
        dtype=np.float32,
    )
    return window


async def generate_predictions(
    conn: asyncpg.Connection,
    symbols: list[str],
    start: datetime,
    end: datetime,
    model_version: str,
    use_synthetic: bool = False,
    engine: ONNXInferenceEngine | None = None,
) -> int:
    """Generate predictions for given symbols and date range."""
    inserted = 0
    correlation_id = uuid4()

    # Determine feature columns and lookback from the engine
    feature_cols: list[str] = []
    lookback: int = 60
    if engine is not None:
        feature_cols = engine.feature_columns
        lookback = engine.lookback_window

    for symbol in symbols:
        # ------------------------------------------------------------
        # Fetch timestamp list to predict on
        # Prefer engineered features, fall back to market_raw_day
        # ------------------------------------------------------------
        ts_rows = await conn.fetch(
            """
            SELECT ts, price
            FROM features_tft
            WHERE ticker = $1 AND ts >= $2 AND ts <= $3
            ORDER BY ts
            """,
            symbol,
            start,
            end,
        )

        if not ts_rows:
            ts_rows = await conn.fetch(
                """
                SELECT ts, close as price
                FROM market_raw_day
                WHERE ticker = $1 AND ts >= $2 AND ts <= $3
                ORDER BY ts
                """,
                symbol,
                start,
                end,
            )

        if not ts_rows:
            logger.warning("no_market_data", symbol=symbol)
            continue

        # Pre-log for traceability
        logger.info(
            "predicting",
            symbol=symbol,
            rows=len(ts_rows),
            mode="synthetic" if use_synthetic else ("onnx" if engine else "stub"),
        )

        for row in ts_rows:
            ts = row["ts"]
            close = float(row["price"])

            if use_synthetic:
                # Generate synthetic predictions with some signal
                base_return = random.gauss(0.001, 0.02)
                q10 = base_return - 0.015
                q50 = base_return
                q90 = base_return + 0.015
                confidence = random.uniform(0.5, 0.9)
                direction = (
                    "long" if q50 > 0.003
                    else ("short" if q50 < -0.003 else "neutral")
                )
                horizon_minutes = 1440
            elif engine is not None:
                # ── Real ONNX inference ────────────────────────────
                window = await _fetch_feature_window(
                    conn, symbol, ts, feature_cols, lookback,
                )
                if window is None:
                    logger.debug(
                        "insufficient_lookback",
                        symbol=symbol,
                        ts=str(ts),
                        lookback=lookback,
                    )
                    continue

                pred = engine.predict(symbol, features=window)
                if pred is None:
                    logger.warning("onnx_predict_none", symbol=symbol, ts=str(ts))
                    continue

                # Use the first horizon's quantiles for the DB row
                first_horizon = engine.horizons[0]
                q10 = pred.get(f"horizon_{first_horizon}_q10", -0.005)
                q50 = pred.get(f"horizon_{first_horizon}_q50", 0.0)
                q90 = pred.get(f"horizon_{first_horizon}_q90", 0.005)
                confidence = pred.get("confidence", 0.5)
                direction = pred.get("direction", "neutral")
                horizon_minutes = first_horizon
            else:
                # Stub — no model available, emit neutral
                q50 = 0.0
                q10 = -0.01
                q90 = 0.01
                confidence = 0.5
                direction = "neutral"
                horizon_minutes = 1440

            event_id = uuid4()
            idempotency_key = f"{symbol}:{ts.isoformat()}:{model_version}"

            await conn.execute(
                """
                INSERT INTO predictions_tft (
                    event_id, correlation_id, idempotency_key, ticker, ts,
                    horizon_minutes, direction, confidence, q10, q50, q90, model_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (ts, idempotency_key) DO NOTHING
                """,
                event_id,
                correlation_id,
                idempotency_key,
                symbol,
                ts,
                horizon_minutes,
                direction,
                confidence,
                q10,
                q50,
                q90,
                model_version,
            )
            inserted += 1

    return inserted


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate predictions from trained model")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Generate synthetic predictions (for testing)",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version to use (default: latest from registry)",
    )
    args = parser.parse_args()

    configure_logging()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = _parse_date(args.start)
    end = _parse_date(args.end, end=True)

    # Get model version
    if args.model_version:
        model_version = args.model_version
    else:
        registry = _load_registry()
        model = _get_latest_model(registry)
        if model:
            model_version = model.get("run_id", "unknown")
        else:
            model_version = "synthetic" if args.use_synthetic else "unknown"

    # Load ONNX engine (unless synthetic mode)
    engine: ONNXInferenceEngine | None = None
    if not args.use_synthetic:
        engine = _load_engine()
        if engine is None:
            logger.warning(
                "no_model_available",
                hint="pass --use-synthetic or train a model first",
            )

    dsn = (
        f"postgresql://{_env('POSTGRES_USER', 'awet')}:{_env('POSTGRES_PASSWORD', 'awet')}"
        f"@{_env('POSTGRES_HOST', 'localhost')}:{_env('POSTGRES_PORT', '5433')}"
        f"/{_env('POSTGRES_DB', 'awet')}"
    )

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)

    try:
        async with pool.acquire() as conn:
            inserted = await generate_predictions(
                conn,
                symbols,
                start,
                end,
                model_version,
                use_synthetic=args.use_synthetic,
                engine=engine,
            )

            logger.info(
                "predictions_generated",
                symbols=symbols,
                start=str(start),
                end=str(end),
                model_version=model_version,
                rows=inserted,
                mode="synthetic" if args.use_synthetic else ("onnx" if engine else "stub"),
            )
            print(f"Inserted {inserted} prediction rows (model: {model_version})")

    finally:
        await pool.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
