"""SignalEngineService — main entry point.

Consumes:
  - Kafka[predictions.tft] — TFT model outputs (direction, confidence, q50)
  - Kafka[alpha.signals]   — Lean alpha signals (RSI, EMA cross, MACD)

Produces:
  - Kafka[signals.scored]  — unified ensemble signal with direction, confidence, score

Also persists signals to TimescaleDB[signals] table for backtesting + audit.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog
from confluent_kafka import Consumer, Producer, KafkaError

from .ensemble import SignalEnsemble, TftSignal, AlphaSignal

logger = structlog.get_logger("signal_engine")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DB_DSN = (
    f"postgresql://{os.getenv('POSTGRES_USER','awet')}:{os.getenv('POSTGRES_PASSWORD','awet')}"
    f"@{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT','5433')}"
    f"/{os.getenv('POSTGRES_DB','awet')}"
)
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.3"))


class SignalEngineService:
    """Combines TFT predictions + Lean alpha signals into ensemble signals."""

    def __init__(self) -> None:
        self._ensemble = SignalEnsemble()
        self._pool: asyncpg.Pool | None = None
        self._producer: Producer | None = None

    async def _connect_db(self) -> None:
        self._pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=5)

    async def _persist_signal(self, result: dict[str, Any]) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO signals (
                        ticker, ts, tft_direction, tft_confidence,
                        ensemble_direction, ensemble_confidence, signal_score,
                        active_alphas, regime, idempotency_key
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    ON CONFLICT (ticker, ts) DO NOTHING
                    """,
                    result["ticker"],
                    datetime.fromisoformat(result["ts"]),
                    result.get("tft_direction"),
                    result.get("tft_confidence", 0.0),
                    result["ensemble_direction"],
                    result["ensemble_confidence"],
                    result["signal_score"],
                    json.dumps(result.get("active_alphas", [])),
                    result.get("regime", "unknown"),
                    result["idempotency_key"],
                )
        except Exception as exc:
            logger.error("signal_persist_error", error=str(exc))

    def _handle_tft_message(self, payload: dict[str, Any]) -> None:
        """Process incoming TFT prediction and update ensemble."""
        symbol = payload.get("ticker") or payload.get("symbol", "")
        if not symbol:
            return
        ts_raw = payload.get("ts", datetime.now(timezone.utc).isoformat())
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)

        signal = TftSignal(
            direction=payload.get("direction", "neutral"),
            confidence=float(payload.get("confidence", 0.0)),
            q50=float(payload.get("q50", 0.0)),
            ts=ts,
        )
        self._ensemble.update_tft(symbol, signal)
        logger.debug("tft_signal_received", symbol=symbol, direction=signal.direction, confidence=signal.confidence)

    def _handle_alpha_message(self, payload: dict[str, Any]) -> None:
        """Process incoming Lean alpha signal and update ensemble."""
        symbol = payload.get("ticker") or payload.get("symbol", "")
        if not symbol:
            return
        ts_raw = payload.get("ts", datetime.now(timezone.utc).isoformat())
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)

        alpha = AlphaSignal(
            indicator=payload.get("indicator", "unknown"),
            direction=payload.get("direction", "neutral"),
            weight=float(payload.get("weight", 0.1)),
            ts=ts,
            indicator_value=payload.get("indicator_value"),
        )
        self._ensemble.update_alpha(symbol, alpha)
        logger.debug("alpha_signal_received", symbol=symbol, indicator=alpha.indicator, direction=alpha.direction)

    def _publish_ensemble_signal(self, result_dict: dict[str, Any]) -> None:
        if self._producer is None:
            return
        self._producer.produce(
            "signals.scored",
            key=result_dict["ticker"].encode(),
            value=json.dumps(result_dict).encode(),
        )
        self._producer.poll(0)

    async def _process_all_symbols(self) -> None:
        """After each message, recompute ensemble for all known symbols."""
        all_symbols = set(self._ensemble._tft.keys())
        for symbol in all_symbols:
            result = self._ensemble.compute(symbol)
            if result is None:
                continue
            if result.ensemble_confidence < MIN_SIGNAL_CONFIDENCE:
                continue
            if result.ensemble_direction == "neutral":
                continue

            result_dict = {
                "event_id": str(uuid.uuid4()),
                "ticker": symbol,
                "ts": result.ts.isoformat(),
                "tft_direction": result.tft_direction,
                "tft_confidence": result.tft_confidence,
                "ensemble_direction": result.ensemble_direction,
                "ensemble_confidence": result.ensemble_confidence,
                "signal_score": result.signal_score,
                "active_alphas": result.active_alphas,
                "regime": result.regime,
                "schema_version": "1.0",
                "source": "signal_engine",
                "idempotency_key": f"signal:{symbol}:{result.ts.isoformat()}",
            }
            self._publish_ensemble_signal(result_dict)
            await self._persist_signal(result_dict)
            logger.info(
                "ensemble_signal_published",
                symbol=symbol,
                direction=result.ensemble_direction,
                confidence=result.ensemble_confidence,
                score=result.signal_score,
                alphas=result.active_alphas,
            )

    async def run(self) -> None:
        """Main async run loop."""
        await self._connect_db()
        self._producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "signal-engine-service",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })
        consumer.subscribe(["predictions.tft", "alpha.signals"])
        logger.info("signal_engine_started", topics=["predictions.tft", "alpha.signals"])

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("kafka_error", error=str(msg.error()))
                    continue
                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    topic = msg.topic()
                    if topic == "predictions.tft":
                        self._handle_tft_message(payload)
                    elif topic == "alpha.signals":
                        self._handle_alpha_message(payload)
                    await self._process_all_symbols()
                except Exception as exc:
                    logger.error("signal_engine_process_error", error=str(exc))
        finally:
            consumer.close()
            if self._producer:
                self._producer.flush()
            if self._pool:
                await self._pool.close()
            logger.info("signal_engine_stopped")


def main() -> None:
    asyncio.run(SignalEngineService().run())


if __name__ == "__main__":
    main()
