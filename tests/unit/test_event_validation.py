"""Event validation tests.

Verifies all event types have required BaseEvent fields and validates
schema contracts across the entire pipeline.
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from src.models.base import BaseEvent
from src.models.events_market import MarketRawEvent
from src.models.events_engineered import MarketEngineeredEvent
from src.models.events_prediction import PredictionEvent
from src.models.events_risk import RiskEvent
from src.models.events_execution import ExecutionEvent


class TestBaseEventValidation:
    """Test that all event types require BaseEvent fields."""

    REQUIRED_FIELDS = {
        "event_id",
        "correlation_id",
        "idempotency_key",
        "symbol",
        "ts",
        "schema_version",
        "source",
    }

    def test_event_missing_required_field(self) -> None:
        """Event should fail validation without required fields."""
        with pytest.raises(Exception):
            MarketRawEvent(
                idempotency_key="x",
                source="test",
                price=1.0,
            )

    def test_base_event_fields_present(self) -> None:
        """BaseEvent must define all required fields."""
        base_fields = set(BaseEvent.model_fields.keys())
        for field in self.REQUIRED_FIELDS:
            assert field in base_fields, f"BaseEvent missing required field: {field}"

    def test_market_raw_event_has_all_fields(self) -> None:
        """MarketRawEvent must have all required fields."""
        event = MarketRawEvent(
            idempotency_key="test-key-001",
            symbol="AAPL",
            source="polygon",
            price=150.0,
            volume=1000,
            open=149.50,
            high=150.50,
            low=149.00,
            close=150.0,
        )
        self._assert_required_fields(event)

    def test_engineered_event_has_all_fields(self) -> None:
        """MarketEngineeredEvent must have all required fields."""
        event = MarketEngineeredEvent(
            idempotency_key="test-key-002",
            symbol="MSFT",
            source="feature-engineering",
            price=300.0,
            volume=500,
            returns_1=0.001,
            returns_5=0.005,
            returns_15=0.01,
            volatility_5=0.02,
            volatility_15=0.03,
            sma_5=299.0,
            sma_20=298.0,
            ema_5=299.5,
            ema_20=298.5,
            rsi_14=55.0,
            volume_zscore=0.5,
            minute_of_day=600,
            day_of_week=1,
        )
        self._assert_required_fields(event)

    def test_prediction_event_has_all_fields(self) -> None:
        """PredictionEvent must have all required fields."""
        event = PredictionEvent(
            idempotency_key="test-key-003",
            symbol="NVDA",
            source="prediction-agent",
            prediction=0.005,
            confidence=0.75,
            horizon_minutes=30,
            model_version="v1.0.0",
            horizon_30_q10=-0.01,
            horizon_30_q50=0.005,
            horizon_30_q90=0.02,
            horizon_45_q10=-0.015,
            horizon_45_q50=0.01,
            horizon_45_q90=0.025,
            horizon_60_q10=-0.02,
            horizon_60_q50=0.015,
            horizon_60_q90=0.03,
            direction="long",
        )
        self._assert_required_fields(event)

    def test_risk_event_has_all_fields(self) -> None:
        """RiskEvent must have all required fields."""
        event = RiskEvent(
            idempotency_key="test-key-004",
            symbol="TSLA",
            source="risk-agent",
            approved=True,
            reason="All checks passed",
            risk_score=0.3,
            max_position=10.0,
            max_notional=1500.0,
            min_confidence=0.3,
            cvar_95=0.02,
            max_loss=500.0,
            direction="long",
        )
        self._assert_required_fields(event)

    def test_execution_event_has_all_fields(self) -> None:
        """ExecutionEvent must have all required fields."""
        event = ExecutionEvent(
            idempotency_key="test-key-005",
            symbol="AMD",
            source="execution-agent",
            status="filled",
            filled_qty=10,
            avg_price=120.0,
            paper_trade=True,
        )
        self._assert_required_fields(event)

    def _assert_required_fields(self, event: BaseEvent) -> None:
        """Helper to assert all required fields are present and valid."""
        # Check field presence
        event_dict = event.model_dump()
        for field in self.REQUIRED_FIELDS:
            assert field in event_dict, f"Event missing required field: {field}"

        # Check field types
        assert isinstance(event.event_id, UUID)
        assert isinstance(event.correlation_id, UUID)
        assert isinstance(event.idempotency_key, str)
        assert len(event.idempotency_key) > 0
        assert isinstance(event.symbol, str)
        assert len(event.symbol) > 0
        assert isinstance(event.ts, datetime)
        assert event.ts.tzinfo is not None  # Must be timezone-aware
        assert isinstance(event.schema_version, int)
        assert event.schema_version >= 1
        assert isinstance(event.source, str)
        assert len(event.source) > 0


class TestEventSerialization:
    """Test event serialization for Kafka/Avro compatibility."""

    def test_to_avro_dict_contains_all_fields(self) -> None:
        """to_avro_dict() must include all required fields."""
        event = MarketRawEvent(
            idempotency_key="serial-test-001",
            symbol="AAPL",
            source="test",
            price=150.0,
            volume=1000,
            open=149.50,
            high=150.50,
            low=149.00,
            close=150.0,
        )
        avro_dict = event.to_avro_dict()

        required = {"event_id", "correlation_id", "idempotency_key", "symbol", "ts", "schema_version", "source"}
        for field in required:
            assert field in avro_dict, f"to_avro_dict missing: {field}"

    def test_avro_dict_has_json_serializable_types(self) -> None:
        """All values in to_avro_dict() must be JSON-serializable."""
        import json

        event = MarketRawEvent(
            idempotency_key="json-test-001",
            symbol="MSFT",
            source="test",
            price=300.0,
            volume=500,
            open=299.50,
            high=300.50,
            low=299.00,
            close=300.0,
        )
        avro_dict = event.to_avro_dict()

        # Should not raise
        json_str = json.dumps(avro_dict)
        assert len(json_str) > 0

        # UUID should be string in avro dict
        assert isinstance(avro_dict["event_id"], str)
        assert isinstance(avro_dict["correlation_id"], str)

