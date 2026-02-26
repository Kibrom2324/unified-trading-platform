"""SignalEngineService — main entry point.

Consumes:
  - Kafka[predictions.tft] — TFT model outputs (Avro-encoded by model_inference)
  - Kafka[alpha.signals]   — Lean alpha signals (plain JSON from lean_alpha)
  - Kafka[market.raw]      — OHLCV bars (Avro-encoded by data_ingestion)

Produces:
  - Kafka[signals.scored]  — unified ensemble signal with direction, confidence, score

Also persists signals to TimescaleDB[signals] table for backtesting + audit.

Avro decoding note
------------------
model_inference publishes predictions.tft using confluent_kafka.serializing_producer
(SerializingProducer + AvroSerializer).  data_ingestion publishes market.raw the same
way.  Both topics carry Confluent wire-format bytes:

    byte[0]   = 0x00  (magic byte)
    byte[1:5] = schema_id (big-endian uint32)
    byte[5:]  = schemaless Avro payload

This service decodes those two topics with fastavro.schemaless_reader.
alpha.signals is published as plain UTF-8 JSON by lean_alpha and stays as-is.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import fastavro
import structlog
from confluent_kafka import Consumer, Producer, KafkaError

from .ensemble import SignalEnsemble, TftSignal, AlphaSignal

logger = structlog.get_logger("signal_engine")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
# DB_DSN is constructed lazily at connection time via shared.db.dsn.build_dsn()
# to avoid failing at import when POSTGRES_PASSWORD is not set in test environments.
from shared.db.dsn import build_dsn as _build_db_dsn
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.55"))
assert MIN_SIGNAL_CONFIDENCE <= 0.65, (
    f"Signal confidence floor ({MIN_SIGNAL_CONFIDENCE}) exceeds decision agent threshold (0.65). "
    "Raise trader_decision_agent.min_confidence in configs/app.yaml or lower MIN_SIGNAL_CONFIDENCE."
)

# ---------------------------------------------------------------------------
# Avro helpers
# ---------------------------------------------------------------------------
_SCHEMAS_DIR = Path(__file__).parent.parent.parent / "shared" / "schemas"


def _load_avro_schema(filename: str) -> dict:
    """Parse a .avsc file into a fastavro schema dict."""
    with open(_SCHEMAS_DIR / filename, "r", encoding="utf-8") as fh:
        return fastavro.parse_schema(json.loads(fh.read()))


# Confluent wire format: 0x00 magic byte + 4-byte schema_id + schemaless Avro body
_CONFLUENT_MAGIC = 0x00
_CONFLUENT_PREFIX_LEN = 5


def _decode_confluent_avro(raw: bytes, schema: dict) -> dict:
    """Decode Confluent-wire-format Avro bytes.

    Raises:
        ValueError: if the bytes do not start with the Confluent magic byte.
    """
    if len(raw) < _CONFLUENT_PREFIX_LEN or raw[0] != _CONFLUENT_MAGIC:
        raise ValueError(
            f"Expected Confluent Avro magic byte 0x00, got {raw[:1].hex()!r}"
        )
    return fastavro.schemaless_reader(io.BytesIO(raw[_CONFLUENT_PREFIX_LEN:]), schema)


class SignalEngineService:
    """Combines TFT predictions + Lean alpha signals into ensemble signals."""

    def __init__(self) -> None:
        self._ensemble = SignalEnsemble()
        self._pool: asyncpg.Pool | None = None
        self._producer: Producer | None = None
        self._dlq_producer: Producer | None = None
        # Avro schemas loaded once at init; fail fast if schema files are missing
        self._tft_schema = _load_avro_schema("prediction.avsc")
        self._market_raw_schema = _load_avro_schema("market_raw.avsc")

    async def _connect_db(self) -> None:
        self._pool = await asyncpg.create_pool(dsn=_build_db_dsn(), min_size=1, max_size=5)

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

    def _handle_market_message(self, payload: dict[str, Any]) -> None:
        """Process incoming market price tick and update regime classifier."""
        symbol = payload.get("ticker") or payload.get("symbol", "")
        price = payload.get("price") or payload.get("close") or payload.get("last")
        if symbol and price is not None:
            try:
                self._ensemble.update_price(symbol, float(price))
            except (TypeError, ValueError):
                pass

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
        self._dlq_producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "signal-engine-service",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,  # Manual commit — offsets only advance on success
        })
        consumer.subscribe(["predictions.tft", "alpha.signals", "market.raw"])
        logger.info("signal_engine_started", topics=["predictions.tft", "alpha.signals", "market.raw"])

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
                    topic = msg.topic()
                    raw = msg.value()
                    # predictions.tft and market.raw are Avro-encoded (Confluent wire
                    # format) by model_inference and data_ingestion respectively.
                    # alpha.signals is plain UTF-8 JSON published by lean_alpha.
                    if topic == "predictions.tft":
                        payload = _decode_confluent_avro(raw, self._tft_schema)
                        self._handle_tft_message(payload)
                    elif topic == "alpha.signals":
                        payload = json.loads(raw.decode("utf-8"))
                        self._handle_alpha_message(payload)
                    elif topic == "market.raw":
                        payload = _decode_confluent_avro(raw, self._market_raw_schema)
                        self._handle_market_message(payload)
                    await self._process_all_symbols()
                    # Commit only after successful processing — prevents silent message loss
                    # if the process crashes between poll and DB/produce.
                    consumer.commit(asynchronous=False)
                except Exception as exc:
                    logger.error("signal_engine_process_error", error=str(exc), topic=msg.topic())
                    # Route failed message to DLQ for inspection and replay
                    if self._dlq_producer is not None:
                        try:
                            self._dlq_producer.produce(
                                f"dlq.{msg.topic()}",
                                key=msg.key(),
                                value=msg.value(),
                                headers={
                                    "error": str(exc).encode(),
                                    "original_topic": msg.topic().encode(),
                                },
                            )
                            self._dlq_producer.poll(0)
                        except Exception:
                            pass  # DLQ failure must never crash the consumer loop
        finally:
            consumer.close()
            if self._producer:
                self._producer.flush()
            if self._dlq_producer:
                self._dlq_producer.flush()
            if self._pool:
                await self._pool.close()
            logger.info("signal_engine_stopped")


def main() -> None:
    asyncio.run(SignalEngineService().run())


if __name__ == "__main__":
    main()
