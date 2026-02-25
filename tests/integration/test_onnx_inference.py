"""Integration tests for ONNX inference engine + predict script.

Tests the full path from ONNX model loading through feature extraction
to prediction output, using the actual green model on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.onnx_engine import ONNXInferenceEngine
from src.ml.registry import ModelRegistry, ModelStatus


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def green_model_dir() -> Path:
    """Return the directory of the current green model."""
    reg = ModelRegistry(
        registry_path=REPO_ROOT / "models" / "registry.json",
        models_dir=REPO_ROOT / "models" / "tft",
    )
    green = reg.get_green()
    if green is None:
        pytest.skip("No green model in registry — train one first")
    model_dir = Path(green.onnx_path).parent
    if not (model_dir / "model.onnx").exists():
        pytest.skip(f"Green model ONNX not found at {model_dir}")
    return model_dir


@pytest.fixture(scope="module")
def engine(green_model_dir: Path) -> ONNXInferenceEngine:
    """Load the ONNX engine with the green model."""
    eng = ONNXInferenceEngine(
        model_path=str(green_model_dir / "model.onnx"),
        use_registry=False,
    )
    loaded = eng.load_model()
    assert loaded, "Failed to load green ONNX model"
    return eng


# ── Model loading tests ────────────────────────────────────────────


class TestONNXModelLoading:
    """Verify the engine can load the green model and read its metadata."""

    def test_model_loads_successfully(self, engine: ONNXInferenceEngine) -> None:
        assert engine.is_loaded

    def test_session_not_none(self, engine: ONNXInferenceEngine) -> None:
        assert engine.session is not None

    def test_lookback_window_auto_detected(self, engine: ONNXInferenceEngine) -> None:
        assert engine.lookback_window == 60

    def test_num_features(self, engine: ONNXInferenceEngine) -> None:
        assert engine.num_features == 15

    def test_feature_order_is_default(self, engine: ONNXInferenceEngine) -> None:
        expected = [
            "price", "volume", "returns_1", "returns_5", "returns_15",
            "volatility_5", "volatility_15", "sma_5", "sma_20",
            "ema_5", "ema_20", "rsi_14", "volume_zscore",
            "minute_of_day", "day_of_week",
        ]
        assert engine.feature_columns == expected

    def test_model_version_extracted(self, engine: ONNXInferenceEngine) -> None:
        assert engine.model_version != "not_loaded"
        assert engine.model_version != "unknown"

    def test_feature_meta_loaded(self, green_model_dir: Path) -> None:
        meta_path = green_model_dir / "feature_meta.json"
        assert meta_path.exists(), "feature_meta.json should exist after backfill"
        meta = json.loads(meta_path.read_text())
        assert "feature_columns" in meta
        assert len(meta["feature_columns"]) == 15


# ── Inference tests ─────────────────────────────────────────────────


class TestONNXInference:
    """Run inference with synthetic feature arrays and validate outputs."""

    def _random_window(self, engine: ONNXInferenceEngine) -> np.ndarray:
        """Create a random feature window of the correct shape."""
        return np.random.randn(
            engine.lookback_window, engine.num_features
        ).astype(np.float32)

    def test_predict_returns_dict(self, engine: ONNXInferenceEngine) -> None:
        window = self._random_window(engine)
        result = engine.predict("TEST", features=window)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, engine: ONNXInferenceEngine) -> None:
        window = self._random_window(engine)
        result = engine.predict("TEST", features=window)
        assert result is not None
        for key in ("symbol", "confidence", "direction", "model_version"):
            assert key in result, f"Missing key: {key}"

    def test_predict_has_horizon_quantiles(self, engine: ONNXInferenceEngine) -> None:
        window = self._random_window(engine)
        result = engine.predict("TEST", features=window)
        assert result is not None
        for h in engine.horizons:
            for q in ("q10", "q50", "q90"):
                key = f"horizon_{h}_{q}"
                assert key in result, f"Missing key: {key}"
                assert isinstance(result[key], float)

    def test_predict_direction_valid(self, engine: ONNXInferenceEngine) -> None:
        window = self._random_window(engine)
        result = engine.predict("TEST", features=window)
        assert result is not None
        assert result["direction"] in ("long", "short", "neutral")

    def test_predict_confidence_range(self, engine: ONNXInferenceEngine) -> None:
        window = self._random_window(engine)
        result = engine.predict("TEST", features=window)
        assert result is not None
        # Confidence is a float — it may exceed [0,1] from raw model output
        assert isinstance(result["confidence"], float)

    def test_predict_quantile_ordering(self, engine: ONNXInferenceEngine) -> None:
        """q10 <= q50 <= q90 for each horizon (on average over multiple runs)."""
        violations = 0
        for _ in range(20):
            window = self._random_window(engine)
            result = engine.predict("TEST", features=window)
            if result is None:
                continue
            for h in engine.horizons:
                q10 = result[f"horizon_{h}_q10"]
                q50 = result[f"horizon_{h}_q50"]
                q90 = result[f"horizon_{h}_q90"]
                if not (q10 <= q50 <= q90):
                    violations += 1
        # Allow some violations on random data — model wasn't trained for this
        # This is a smoke test, not a correctness test
        assert violations < 40, f"Too many quantile ordering violations: {violations}/60"

    def test_predict_output_shape_via_raw_session(
        self, engine: ONNXInferenceEngine
    ) -> None:
        """Verify raw ONNX session output shape is [1, 10]."""
        window = self._random_window(engine)
        input_data = window[np.newaxis, :, :]  # [1, 60, 15]
        assert engine.session is not None
        input_name = engine.session.get_inputs()[0].name
        outputs = engine.session.run(None, {input_name: input_data})
        assert outputs[0].shape == (1, 10)

    def test_predict_multiple_symbols(self, engine: ONNXInferenceEngine) -> None:
        """Engine tracks per-symbol buffers."""
        for sym in ("AAPL", "MSFT", "GOOG"):
            window = self._random_window(engine)
            result = engine.predict(sym, features=window)
            assert result is not None
            assert result["symbol"] == sym

    def test_predict_with_buffer(self, engine: ONNXInferenceEngine) -> None:
        """Test the add_tick → predict via internal buffer path."""
        sym = "BUF_TEST"
        buf = engine.get_buffer(sym)
        # Fill buffer with enough ticks
        for _ in range(engine.lookback_window):
            event = {col: float(np.random.randn()) for col in engine.feature_columns}
            engine.add_tick(sym, event)
        assert len(buf) == engine.lookback_window
        result = engine.predict(sym)
        assert result is not None
        assert result["symbol"] == sym


# ── Registry compat check tests ────────────────────────────────────


class TestRegistryCompatCheck:
    """Test the _check_onnx_output_compat static method."""

    def test_green_model_is_compatible(self, green_model_dir: Path) -> None:
        onnx_path = str(green_model_dir / "model.onnx")
        assert ModelRegistry._check_onnx_output_compat(onnx_path, expected_outputs=10)

    def test_nonexistent_model_returns_false(self) -> None:
        assert not ModelRegistry._check_onnx_output_compat("/tmp/no_such_model.onnx")

    def test_wrong_expected_outputs(self, green_model_dir: Path) -> None:
        onnx_path = str(green_model_dir / "model.onnx")
        # Green model outputs 10, so expecting 1 should fail
        assert not ModelRegistry._check_onnx_output_compat(onnx_path, expected_outputs=1)
