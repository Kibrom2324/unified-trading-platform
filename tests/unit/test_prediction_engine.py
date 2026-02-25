"""Unit tests for prediction engine."""

from __future__ import annotations

import pytest
import numpy as np
from src.prediction.engine import (
    PredictionInput,
    PredictionOutput,
    StubPredictionEngine,
    ONNXPredictionEngine,
    create_prediction_engine,
)


@pytest.fixture
def sample_input() -> PredictionInput:
    return PredictionInput(
        symbol="AAPL",
        price=150.0,
        volume=1000000.0,
        returns_1=0.002,
        returns_5=0.01,
        returns_15=0.015,
        volatility_5=0.02,
        volatility_15=0.025,
        sma_5=149.5,
        sma_20=148.0,
        ema_5=149.8,
        ema_20=148.5,
        rsi_14=55.0,
        volume_zscore=0.5,
        minute_of_day=570,
        day_of_week=1,
    )


class TestPredictionInput:
    def test_to_array(self, sample_input: PredictionInput) -> None:
        arr = sample_input.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (15,)
        assert arr[0] == 150.0
        assert arr.dtype == np.float32


class TestPredictionOutput:
    def test_expected_direction_long(self) -> None:
        output = PredictionOutput(
            symbol="AAPL",
            horizon_30_q10=0.001,
            horizon_30_q50=0.005,
            horizon_30_q90=0.01,
            horizon_45_q10=0.002,
            horizon_45_q50=0.006,
            horizon_45_q90=0.012,
            horizon_60_q10=0.003,
            horizon_60_q50=0.007,
            horizon_60_q90=0.014,
            confidence=0.7,
        )
        assert output.expected_direction() == "long"

    def test_expected_direction_short(self) -> None:
        output = PredictionOutput(
            symbol="AAPL",
            horizon_30_q10=-0.01,
            horizon_30_q50=-0.005,
            horizon_30_q90=-0.001,
            horizon_45_q10=-0.012,
            horizon_45_q50=-0.006,
            horizon_45_q90=-0.002,
            horizon_60_q10=-0.014,
            horizon_60_q50=-0.007,
            horizon_60_q90=-0.003,
            confidence=0.7,
        )
        assert output.expected_direction() == "short"

    def test_expected_direction_neutral(self) -> None:
        output = PredictionOutput(
            symbol="AAPL",
            horizon_30_q10=-0.001,
            horizon_30_q50=0.0001,
            horizon_30_q90=0.001,
            horizon_45_q10=-0.001,
            horizon_45_q50=0.0002,
            horizon_45_q90=0.001,
            horizon_60_q10=-0.001,
            horizon_60_q50=0.0003,
            horizon_60_q90=0.001,
            confidence=0.5,
        )
        assert output.expected_direction() == "neutral"

    def test_to_dict(self) -> None:
        output = PredictionOutput(
            symbol="AAPL",
            horizon_30_q10=0.001,
            horizon_30_q50=0.005,
            horizon_30_q90=0.01,
            horizon_45_q10=0.002,
            horizon_45_q50=0.006,
            horizon_45_q90=0.012,
            horizon_60_q10=0.003,
            horizon_60_q50=0.007,
            horizon_60_q90=0.014,
            confidence=0.7,
        )
        d = output.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["direction"] == "long"
        assert d["confidence"] == 0.7


class TestStubPredictionEngine:
    @pytest.mark.asyncio
    async def test_warmup(self) -> None:
        engine = StubPredictionEngine(seed=42)
        await engine.warmup()
        assert engine._warmed_up

    @pytest.mark.asyncio
    async def test_predict(self, sample_input: PredictionInput) -> None:
        engine = StubPredictionEngine(seed=42)
        await engine.warmup()
        output = await engine.predict(sample_input)
        
        assert output.symbol == "AAPL"
        assert 0.0 < output.confidence <= 1.0
        assert output.horizon_30_q10 < output.horizon_30_q50 < output.horizon_30_q90 or \
               output.horizon_30_q10 == output.horizon_30_q50 == output.horizon_30_q90 == 0.0

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self, sample_input: PredictionInput) -> None:
        engine1 = StubPredictionEngine(seed=42)
        engine2 = StubPredictionEngine(seed=42)
        await engine1.warmup()
        await engine2.warmup()
        
        output1 = await engine1.predict(sample_input)
        output2 = await engine2.predict(sample_input)
        
        assert output1.horizon_30_q50 == output2.horizon_30_q50


class TestCreatePredictionEngine:
    def test_create_stub(self) -> None:
        engine = create_prediction_engine("stub")
        assert isinstance(engine, StubPredictionEngine)

    def test_create_auto_fallback(self) -> None:
        # Auto mode will use ONNX if a green model exists in registry,
        # otherwise falls back to stub. We just verify it returns a valid engine.
        engine = create_prediction_engine("auto")
        assert engine is not None
        # Should be either StubPredictionEngine or ONNXPredictionEngine
        assert isinstance(engine, (StubPredictionEngine, ONNXPredictionEngine))
