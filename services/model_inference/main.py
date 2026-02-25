from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent
from src.audit.trail_logger import AuditTrailLogger
from src.core.config import load_settings
from src.core.logging import set_correlation_id
from src.models.events_engineered import MarketEngineeredEvent
from src.models.events_prediction import PredictionEvent
from src.monitoring.metrics import EVENTS_PROCESSED, EVENT_LATENCY, MODEL_VERSION_INFO, PREDICTION_LAST_TIMESTAMP
from src.prediction.engine import PredictionInput, create_prediction_engine
from src.streaming.kafka_consumer import AvroConsumer
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import MARKET_ENGINEERED, PREDICTIONS_TFT

ENG_SCHEMA = "src/schemas/market_engineered.avsc"
PRED_SCHEMA = "src/schemas/prediction.avsc"


class PredictionAgent(BaseAgent):
    """TFT-based prediction agent with quantile outputs.

    Generates predictions with:
    - Multiple horizons (30s, 45s, 60s)
    - Quantile estimates (q10, q50, q90)
    - Confidence scores
    - Directional signals (long/short/neutral)
    
    Persists predictions to TimescaleDB table `predictions_tft`.
    """

    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("prediction", settings.app.http.prediction_port)
        with open(ENG_SCHEMA, "r", encoding="utf-8") as handle:
            eng_schema = handle.read()
        self.consumer = AvroConsumer(
            self.settings.kafka,
            self.settings.kafka.group_ids.prediction,
            eng_schema,
            MARKET_ENGINEERED,
        )
        self.producer = AvroProducer(self.settings.kafka)
        self.audit = AuditTrailLogger(self.settings)
        self.engine = create_prediction_engine()
        with open(PRED_SCHEMA, "r", encoding="utf-8") as handle:
            self._pred_schema = handle.read()

    async def start(self) -> None:
        self.app.add_event_handler("shutdown", self._shutdown)
        await self.audit.connect()
        await self.engine.warmup()
        self.track_task(asyncio.create_task(self._consume_loop()))

    async def _shutdown(self) -> None:
        self.consumer.close()
        self.producer.close()
        await self.audit.close()

    def _build_input(self, event: MarketEngineeredEvent) -> PredictionInput:
        """Build prediction input from engineered event."""
        return PredictionInput(
            symbol=event.symbol,
            price=event.price,
            volume=event.volume,
            returns_1=event.returns_1,
            returns_5=event.returns_5,
            returns_15=getattr(event, "returns_15", 0.0),
            volatility_5=getattr(event, "volatility_5", 0.0),
            volatility_15=getattr(event, "volatility_15", 0.0),
            sma_5=getattr(event, "sma_5", event.price),
            sma_20=getattr(event, "sma_20", event.price),
            ema_5=getattr(event, "ema_5", event.price),
            ema_20=getattr(event, "ema_20", event.price),
            rsi_14=getattr(event, "rsi_14", 50.0),
            volume_zscore=getattr(event, "volume_zscore", 0.0),
            minute_of_day=getattr(event, "minute_of_day", 0),
            day_of_week=getattr(event, "day_of_week", 0),
        )

    async def _consume_loop(self) -> None:
        while not self.is_shutting_down:
          try:
            msg = self.consumer.poll(1.0)
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            payload = msg.value()
            event = MarketEngineeredEvent.model_validate(payload)
            set_correlation_id(str(event.correlation_id))
            if await self.audit.is_duplicate(PREDICTIONS_TFT, event.idempotency_key):
                self.consumer.commit()
                continue
            start_ts = datetime.now(tz=timezone.utc)
            pred_input = self._build_input(event)
            pred_output = await self.engine.predict(pred_input)
            
            # Get model version from engine if available
            model_version = getattr(self.engine, 'model_version', 'tft_quantile_v1')
            
            pred_event = PredictionEvent(
                idempotency_key=event.idempotency_key,
                symbol=event.symbol,
                source=self.name,
                correlation_id=event.correlation_id,
                prediction=event.price * (1 + pred_output.horizon_30_q50),
                confidence=pred_output.confidence,
                horizon_minutes=1,
                model_version=model_version,
                horizon_30_q10=pred_output.horizon_30_q10,
                horizon_30_q50=pred_output.horizon_30_q50,
                horizon_30_q90=pred_output.horizon_30_q90,
                horizon_45_q10=pred_output.horizon_45_q10,
                horizon_45_q50=pred_output.horizon_45_q50,
                horizon_45_q90=pred_output.horizon_45_q90,
                horizon_60_q10=pred_output.horizon_60_q10,
                horizon_60_q50=pred_output.horizon_60_q50,
                horizon_60_q90=pred_output.horizon_60_q90,
                direction=pred_output.expected_direction(),
            )
            payload_out = pred_event.to_avro_dict()
            self.producer.produce(PREDICTIONS_TFT, self._pred_schema, payload_out, key=event.symbol)
            await self.audit.write_event(PREDICTIONS_TFT, payload_out)

            MODEL_VERSION_INFO.labels(agent=self.name, model_version=pred_event.model_version).set(1)
            PREDICTION_LAST_TIMESTAMP.labels(agent=self.name).set(pred_event.ts.timestamp())
            
            # Persist prediction to predictions_tft table
            try:
                await self.audit.write_prediction({
                    "event_id": str(pred_event.event_id),
                    "correlation_id": str(pred_event.correlation_id),
                    "idempotency_key": pred_event.idempotency_key,
                    "symbol": pred_event.symbol,
                    "ts": pred_event.ts,
                    "horizon_minutes": 30,  # Primary horizon
                    "direction": pred_output.expected_direction(),
                    "confidence": pred_output.confidence,
                    "q10": pred_output.horizon_30_q10,
                    "q50": pred_output.horizon_30_q50,
                    "q90": pred_output.horizon_30_q90,
                    "model_version": pred_event.model_version,
                })
            except Exception as e:
                self.logger.error("persist_prediction_error", error=str(e), symbol=event.symbol)
            
            self.consumer.commit()
            duration = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
            EVENTS_PROCESSED.labels(agent=self.name, event_type=PREDICTIONS_TFT).inc()
            EVENT_LATENCY.labels(agent=self.name, event_type=PREDICTIONS_TFT).observe(duration)
            self.logger.info(
                "prediction_generated",
                symbol=event.symbol,
                direction=pred_output.expected_direction(),
                confidence=pred_output.confidence,
                q50_30=pred_output.horizon_30_q50,
            )
          except asyncio.CancelledError:
              self.logger.info("consume_loop_cancelled")
              break
          except Exception:
              self.logger.exception("consume_loop_error")
              await asyncio.sleep(1.0)


def main() -> None:
    PredictionAgent().run()


if __name__ == "__main__":
    main()
