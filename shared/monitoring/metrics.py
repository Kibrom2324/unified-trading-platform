from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, CollectorRegistry, generate_latest

try:
    from fastapi import Response
except ImportError:  # pragma: no cover - optional in non-HTTP runtimes
    Response = None  # type: ignore[assignment]

REGISTRY = CollectorRegistry()

EVENTS_PROCESSED = Counter(
    "events_processed_total",
    "Total processed events",
    ["agent", "event_type"],
    registry=REGISTRY,
)

EVENTS_FAILED = Counter(
    "events_failed_total",
    "Total failed events",
    ["agent", "event_type"],
    registry=REGISTRY,
)

EVENT_LATENCY = Histogram(
    "event_latency_seconds",
    "Processing latency in seconds",
    ["agent", "event_type"],
    registry=REGISTRY,
)

# Risk decision metrics
RISK_DECISIONS = Counter(
    "risk_decisions_total",
    "Total risk decisions by outcome",
    ["decision"],
    registry=REGISTRY,
)

# Prediction model metadata
MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Current model version in use (labelled)",
    ["agent", "model_version"],
    registry=REGISTRY,
)

PREDICTION_LAST_TIMESTAMP = Gauge(
    "prediction_last_timestamp_seconds",
    "Unix timestamp of last prediction event",
    ["agent"],
    registry=REGISTRY,
)

# Kafka consumer lag metrics
KAFKA_CONSUMER_LAG = Gauge(
    "kafka_consumer_lag",
    "Kafka consumer group lag (messages behind)",
    ["consumer_group", "topic", "partition"],
    registry=REGISTRY,
)

KAFKA_CONSUMER_LAG_TOTAL = Gauge(
    "kafka_consumer_lag_total",
    "Total Kafka consumer group lag across all partitions",
    ["consumer_group"],
    registry=REGISTRY,
)

# Live ingestion metrics
LIVE_BARS_INGESTED = Counter(
    "live_bars_ingested_total",
    "Total bars ingested via live ingestion",
    ["symbol", "source"],
    registry=REGISTRY,
)

LIVE_INGESTION_LATENCY = Histogram(
    "live_ingestion_latency_seconds",
    "Live ingestion duration in seconds",
    ["source"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

LIVE_INGESTION_FAILURES = Counter(
    "live_ingestion_failures_total",
    "Total live ingestion failures by symbol",
    ["symbol", "source"],
    registry=REGISTRY,
)

# Reconciliation / portfolio metrics
RECONCILE_ERRORS = Counter(
    "reconcile_errors_total",
    "Total reconciliation errors",
    ["type"],
    registry=REGISTRY,
)

PENDING_ORDERS = Gauge(
    "pending_orders",
    "Number of pending orders",
    registry=REGISTRY,
)

OPEN_POSITIONS = Gauge(
    "open_positions",
    "Number of open positions",
    registry=REGISTRY,
)

PORTFOLIO_VALUE = Gauge(
    "portfolio_value_usd",
    "Total portfolio market value (USD)",
    registry=REGISTRY,
)


def metrics_response() -> Response:
    payload = generate_latest(REGISTRY)
    if Response is None:
        return payload  # type: ignore[return-value]
    return Response(payload, media_type=CONTENT_TYPE_LATEST)
