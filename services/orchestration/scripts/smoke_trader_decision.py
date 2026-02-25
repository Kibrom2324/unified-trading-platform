from __future__ import annotations

import sys
import time
from uuid import uuid4

from src.core.config import load_settings
from src.models.events_prediction import PredictionEvent
from src.models.events_trade_decision import TradeDecisionEvent
from src.streaming.kafka_consumer import AvroConsumer
from src.streaming.kafka_producer import AvroProducer

PRED_SCHEMA = "src/schemas/prediction.avsc"
DECISION_SCHEMA = "src/schemas/trade_decision.avsc"


def main() -> int:
    settings = load_settings()
    with open(PRED_SCHEMA, "r", encoding="utf-8") as handle:
        pred_schema = handle.read()
    with open(DECISION_SCHEMA, "r", encoding="utf-8") as handle:
        decision_schema = handle.read()

    producer = AvroProducer(settings.kafka)
    consumer = AvroConsumer(
        settings.kafka,
        "smoke-trader-decision",
        decision_schema,
        settings.kafka.topics.trade_decisions,
    )

    correlation_id = uuid4()
    idempotency_key = f"smoke:{correlation_id}"
    event = PredictionEvent(
        idempotency_key=idempotency_key,
        symbol="AAPL",
        source="smoke_trader_decision",
        correlation_id=correlation_id,
        prediction=195.0,
        confidence=0.9,
        horizon_minutes=1,
        model_version="smoke",
        direction="long",
        horizon_30_q10=-0.01,
        horizon_30_q50=0.01,
        horizon_30_q90=0.03,
    )

    producer.produce(
        settings.kafka.topics.predictions_tft,
        pred_schema,
        event.to_avro_dict(),
        key=event.symbol,
    )
    producer.flush(10.0)

    deadline = time.time() + 10
    while time.time() < deadline:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        payload = msg.value()
        decision_event = TradeDecisionEvent.model_validate(payload)
        print(decision_event.model_dump())
        consumer.commit()
        return 0 if decision_event.decision == "BUY" else 1

    print("No decision message received")
    return 1


if __name__ == "__main__":
    sys.exit(main())
