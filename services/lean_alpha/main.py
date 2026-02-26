"""LeanAlphaService — main entry point.

Consumes Kafka[market.raw] events, feeds prices to RSI, EMA-cross, and MACD
alpha models (ported from Lean), and publishes alpha signals to Kafka[alpha.signals].

Topic: alpha.signals
Schema: alpha_signal (ticker, ts, source, direction, indicator, indicator_value, correlation_id)
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from confluent_kafka import Consumer, Producer, KafkaError

from .rsi_alpha import RsiAlpha
from .ema_cross_alpha import EmaCrossAlpha
from .macd_alpha import MacdAlpha

logger = structlog.get_logger("lean_alpha")


def _kafka_config(prefix: str = "") -> dict:
    return {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    }


class LeanAlphaService:
    """Runs RSI, EMA-cross, and MACD alphas on live price stream.

    Consumes: market.raw (Avro or JSON)
    Produces: alpha.signals (JSON)
    """

    def __init__(self) -> None:
        # Per-symbol alpha model instances
        self._rsi: dict[str, RsiAlpha] = {}
        self._ema: dict[str, EmaCrossAlpha] = {}
        self._macd: dict[str, MacdAlpha] = {}

        # Signal weights (configurable via env)
        self.rsi_weight = float(os.getenv("RSI_WEIGHT", "0.2"))
        self.ema_weight = float(os.getenv("EMA_WEIGHT", "0.2"))
        self.macd_weight = float(os.getenv("MACD_WEIGHT", "0.1"))

        self._consumer: Consumer | None = None
        self._producer: Producer | None = None
        self._running = False

    def _get_or_create_models(self, symbol: str) -> tuple[RsiAlpha, EmaCrossAlpha, MacdAlpha]:
        if symbol not in self._rsi:
            self._rsi[symbol] = RsiAlpha(period=14)
            self._ema[symbol] = EmaCrossAlpha(fast_period=12, slow_period=26)
            self._macd[symbol] = MacdAlpha(fast=12, slow=26, signal=9)
        return self._rsi[symbol], self._ema[symbol], self._macd[symbol]

    def _publish_signal(
        self,
        symbol: str,
        ts: str,
        indicator: str,
        direction: str,
        indicator_value: float | None,
        weight: float,
        correlation_id: str,
    ) -> None:
        assert self._producer is not None
        payload = {
            "event_id": str(uuid.uuid4()),
            "ticker": symbol,
            "ts": ts,
            "source": f"lean_alpha:{indicator}",
            "indicator": indicator,
            "direction": direction,
            "indicator_value": indicator_value,
            "weight": weight,
            "correlation_id": correlation_id,
            "schema_version": "1.0",
        }
        self._producer.produce(
            "alpha.signals",
            key=symbol.encode(),
            value=json.dumps(payload).encode(),
        )
        self._producer.poll(0)
        logger.info("alpha_signal_published", symbol=symbol, indicator=indicator, direction=direction)

    def _process_bar(self, payload: dict[str, Any]) -> None:
        symbol = payload.get("ticker") or payload.get("symbol", "")
        close = float(payload.get("close", 0.0))
        ts = payload.get("ts", datetime.now(timezone.utc).isoformat())
        corr_id = payload.get("correlation_id", str(uuid.uuid4()))

        if not symbol or close <= 0:
            return

        rsi_model, ema_model, macd_model = self._get_or_create_models(symbol)

        rsi_signal = rsi_model.update(close)
        ema_signal = ema_model.update(close)
        macd_signal = macd_model.update(close)

        if rsi_signal:
            self._publish_signal(symbol, ts, "rsi_14", rsi_signal, rsi_model.rsi_value, self.rsi_weight, corr_id)
        if ema_signal:
            self._publish_signal(symbol, ts, "ema_cross_12_26", ema_signal, ema_model.fast_ema, self.ema_weight, corr_id)
        if macd_signal:
            self._publish_signal(symbol, ts, "macd_12_26_9", macd_signal, macd_model.histogram, self.macd_weight, corr_id)

    def start(self) -> None:
        """Run the consume loop (blocking)."""
        cfg = _kafka_config()
        self._consumer = Consumer({
            **cfg,
            "group.id": "lean-alpha-service",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,  # Manual commit — offsets only advance on success
        })
        self._producer = Producer(cfg)
        self._consumer.subscribe(["market.raw"])
        self._running = True

        logger.info("lean_alpha_service_started")
        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("kafka_error", error=str(msg.error()))
                    continue
                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    self._process_bar(payload)
                    # Commit only after successful processing
                    self._consumer.commit(asynchronous=False)
                except Exception as exc:
                    logger.error("lean_alpha_process_error", error=str(exc))
        finally:
            self._consumer.close()
            self._producer.flush()
            logger.info("lean_alpha_service_stopped")

    def stop(self) -> None:
        self._running = False


def main() -> None:
    import signal as _signal
    svc = LeanAlphaService()

    def _shutdown(sig, frame):
        logger.info("shutdown_signal_received")
        svc.stop()

    _signal.signal(_signal.SIGTERM, _shutdown)
    _signal.signal(_signal.SIGINT, _shutdown)
    svc.start()


if __name__ == "__main__":
    main()
