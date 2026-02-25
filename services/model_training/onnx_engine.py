"""ONNX-based TFT inference engine.

Loads trained TFT model from ONNX format and performs
efficient inference using onnxruntime. Integrates with
model registry to load the current "green" model.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("onnx_engine")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime_not_installed", hint="pip install onnxruntime")


@dataclass
class InferenceBuffer:
    """Per-symbol sliding buffer for inference."""
    symbol: str
    max_length: int = 500
    features: deque[list[float]] = field(default_factory=lambda: deque(maxlen=500))

    def add_features(self, feature_vector: list[float]) -> None:
        """Add feature vector to buffer."""
        self.features.append(feature_vector)

    def get_window(self, window_size: int) -> np.ndarray | None:
        """Get last window_size features as numpy array."""
        if len(self.features) < window_size:
            return None
        window = list(self.features)[-window_size:]
        return np.array(window, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features)


class ONNXInferenceEngine:
    """ONNX-based inference engine for TFT model with registry integration.

    Supports dynamic feature lists loaded from ``feature_meta.json``
    saved alongside the ONNX model during training.  When the metadata
    file is present the engine uses its ``feature_columns`` list;
    otherwise it falls back to the default 15-feature baseline.
    """

    # Default 15-feature baseline — used when feature_meta.json is absent
    DEFAULT_FEATURE_ORDER = [
        "price", "volume", "returns_1", "returns_5", "returns_15",
        "volatility_5", "volatility_15", "sma_5", "sma_20",
        "ema_5", "ema_20", "rsi_14", "volume_zscore",
        "minute_of_day", "day_of_week",
    ]

    # Kept for backward compat — callers that reference FEATURE_ORDER directly
    FEATURE_ORDER = DEFAULT_FEATURE_ORDER
    
    # Output layout: 9 quantiles (3 horizons × 3 quantiles) + 1 confidence
    HORIZONS = [30, 45, 60]
    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(
        self,
        model_path: str | None = None,
        lookback_window: int = 120,
        horizons: list[int] | None = None,
        use_registry: bool = True,
    ) -> None:
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime required: pip install onnxruntime")
        
        self._explicit_path = model_path
        self._use_registry = use_registry
        self.model_path: str | None = None
        self.lookback_window = lookback_window
        self.horizons = horizons or self.HORIZONS
        self.session: ort.InferenceSession | None = None
        self._buffers: dict[str, InferenceBuffer] = {}
        self._loaded = False
        self._model_version: str | None = None
        # Dynamic feature metadata — populated from feature_meta.json
        self._feature_order: list[str] = list(self.DEFAULT_FEATURE_ORDER)
        self._feature_meta: dict[str, Any] | None = None
        # CF-4: Per-column normalization stats loaded from norm_stats.json
        self._norm_means: np.ndarray | None = None
        self._norm_stds: np.ndarray | None = None
    
    def _resolve_model_path(self) -> str | None:
        """Resolve model path from registry or explicit path."""
        if self._explicit_path:
            return self._explicit_path
        
        if self._use_registry:
            try:
                from src.ml.registry import get_green_model_path
                path = get_green_model_path()
                if path:
                    return path
            except Exception as e:
                logger.warning("registry_lookup_failed", error=str(e))
        
        # Fallback to env var
        return os.getenv("ONNX_MODEL_PATH", "models/tft/model.onnx")

    def load_model(self) -> bool:
        """Load ONNX model from registry or explicit path.

        After loading the ONNX session this method also attempts to
        load ``feature_meta.json`` from the same directory.  If found
        the engine switches to the feature list stored by training,
        which may contain more than the default 15 features.
        """
        self.model_path = self._resolve_model_path()
        
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning("onnx_model_not_found", path=self.model_path)
            return False
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available_providers = ort.get_available_providers()
            providers = [p for p in providers if p in available_providers]
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )
            self._loaded = True
            
            # Extract model version from path
            import re
            match = re.search(r'models/tft/([^/]+)/', self.model_path)
            self._model_version = match.group(1) if match else "unknown"
            
            # Auto-detect lookback_window from model input shape
            input_info = self.session.get_inputs()[0]
            if len(input_info.shape) >= 2 and isinstance(input_info.shape[1], int):
                self.lookback_window = input_info.shape[1]
                logger.info("auto_detected_lookback", lookback=self.lookback_window)
            
            # Load dynamic feature metadata if available
            self._load_feature_meta()
            
            # CF-4: Load normalization stats sidecar if available
            self._load_norm_stats()
            
            logger.info(
                "onnx_model_loaded",
                path=self.model_path,
                providers=providers,
                version=self._model_version,
                lookback=self.lookback_window,
                num_features=len(self._feature_order),
                feature_source="feature_meta.json" if self._feature_meta else "default",
            )
            return True
        except Exception as e:
            logger.error("onnx_load_error", error=str(e))
            return False

    def _load_feature_meta(self) -> None:
        """Load feature_meta.json from the model directory.

        If the file exists the engine uses its ``feature_columns`` list
        for feature extraction.  This allows models trained with
        extended feature sets (Reddit sentiment, calendar, market
        benchmark, etc.) to work without code changes.
        """
        if not self.model_path:
            return
        model_dir = Path(self.model_path).parent
        meta_path = model_dir / "feature_meta.json"
        if not meta_path.exists():
            logger.debug("feature_meta_not_found", dir=str(model_dir))
            return
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            feature_cols = meta.get("feature_columns")
            if not feature_cols or not isinstance(feature_cols, list):
                logger.warning("feature_meta_invalid", path=str(meta_path))
                return
            self._feature_meta = meta
            self._feature_order = feature_cols
            # Update instance-level FEATURE_ORDER for callers that read it
            self.FEATURE_ORDER = feature_cols  # type: ignore[assignment]
            logger.info(
                "feature_meta_loaded",
                path=str(meta_path),
                num_features=len(feature_cols),
                sample=feature_cols[:5],
            )
        except Exception as e:
            logger.warning("feature_meta_load_error", error=str(e))

    def _load_norm_stats(self) -> None:
        """Load norm_stats.json sidecar from the model directory.

        CF-4 fix: ensures inference uses the same per-column mean/std
        statistics that were computed during training.
        """
        if not self.model_path:
            return
        model_dir = Path(self.model_path).parent
        norm_path = model_dir / "norm_stats.json"
        if not norm_path.exists():
            logger.debug("norm_stats_not_found", dir=str(model_dir))
            return
        try:
            with open(norm_path) as f:
                data = json.load(f)
            self._norm_means = np.array(data["means"], dtype=np.float32)
            self._norm_stds = np.array(data["stds"], dtype=np.float32)
            logger.info(
                "norm_stats_loaded",
                path=str(norm_path),
                num_features=len(self._norm_means),
            )
        except Exception as e:
            logger.warning("norm_stats_load_error", error=str(e))
    
    def reload_if_needed(self) -> bool:
        """Reload model if registry has a new green model."""
        if not self._use_registry:
            return False
        
        new_path = self._resolve_model_path()
        if new_path and new_path != self.model_path:
            logger.info("reloading_model", old=self.model_path, new=new_path)
            return self.load_model()
        return False
    
    @property
    def model_version(self) -> str:
        """Get current model version."""
        return self._model_version or "not_loaded"

    def get_buffer(self, symbol: str) -> InferenceBuffer:
        """Get or create buffer for symbol."""
        if symbol not in self._buffers:
            self._buffers[symbol] = InferenceBuffer(symbol=symbol)
        return self._buffers[symbol]

    @property
    def feature_columns(self) -> list[str]:
        """Return the active feature column list."""
        return list(self._feature_order)

    @property
    def num_features(self) -> int:
        """Number of features expected by the loaded model."""
        return len(self._feature_order)

    def extract_features(self, event_data: dict[str, Any]) -> list[float]:
        """Extract feature vector from event data.

        Uses the dynamic ``_feature_order`` list which is either loaded
        from ``feature_meta.json`` or falls back to the default 15
        features.  Missing keys default to ``0.0``.
        """
        features = []
        for col in self._feature_order:
            val = event_data.get(col, 0.0)
            features.append(float(val) if val is not None else 0.0)
        return features

    def add_tick(self, symbol: str, event_data: dict[str, Any]) -> None:
        """Add a tick to the symbol buffer."""
        buffer = self.get_buffer(symbol)
        features = self.extract_features(event_data)
        buffer.add_features(features)

    def predict(self, symbol: str, features: np.ndarray | None = None) -> dict[str, Any] | None:
        """Generate prediction for symbol.

        Args:
            symbol: Symbol to predict
            features: Optional pre-built feature array (lookback_window, num_features).
                     If None, uses the internal buffer.

        Returns:
            Prediction dict or None if insufficient data or model not loaded.
        """
        if not self._loaded:
            if not self.load_model():
                return self._generate_stub_prediction(symbol)
        
        # Use provided features or get from buffer
        if features is not None:
            window = features
        else:
            buffer = self.get_buffer(symbol)
            window = buffer.get_window(self.lookback_window)
            if window is None:
                return None
        
        # Add quantile predictions to result for compatibility
        result = self._run_inference(symbol, window)
        if result and "quantile_predictions" not in result:
            # Build quantile array for backward compatibility
            qp = []
            for h in self.horizons:
                qp.append([
                    result.get(f"horizon_{h}_q10", -0.005),
                    result.get(f"horizon_{h}_q50", 0.0),
                    result.get(f"horizon_{h}_q90", 0.005),
                ])
            result["quantile_predictions"] = np.array(qp)
        return result
    
    def _run_inference(self, symbol: str, window: np.ndarray) -> dict[str, Any]:
        """Run inference on feature window."""
        if self.session is None:
            return self._generate_stub_prediction(symbol)
        
        # Ensure window is exactly lookback_window size
        if window.shape[0] > self.lookback_window:
            window = window[-self.lookback_window:]
        elif window.shape[0] < self.lookback_window:
            logger.warning("window_too_small", got=window.shape[0], expected=self.lookback_window)
            return self._generate_stub_prediction(symbol)
        
        input_data = window[np.newaxis, :, :]
        # CF-4: Apply training normalization stats if available
        if self._norm_means is not None and self._norm_stds is not None:
            num_feats = min(input_data.shape[2], len(self._norm_means))
            input_data = input_data.copy()
            input_data[:, :, :num_feats] = (
                (input_data[:, :, :num_feats] - self._norm_means[:num_feats])
                / self._norm_stds[:num_feats]
            )
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})
            predictions = outputs[0][0]
            num_quantiles = 3
            num_horizons = len(self.horizons)
            quantiles = predictions[:num_horizons * num_quantiles].reshape(num_horizons, num_quantiles)
            confidence = predictions[-1] if len(predictions) > num_horizons * num_quantiles else 0.5
            result = {
                "symbol": symbol,
                "confidence": float(confidence),
                "model_version": self._model_version,
            }
            for i, horizon in enumerate(self.horizons):
                result[f"horizon_{horizon}_q10"] = float(quantiles[i, 0])
                result[f"horizon_{horizon}_q50"] = float(quantiles[i, 1])
                result[f"horizon_{horizon}_q90"] = float(quantiles[i, 2])
            avg_q50 = np.mean([result[f"horizon_{h}_q50"] for h in self.horizons])
            if avg_q50 > 0.001:
                result["direction"] = "long"
            elif avg_q50 < -0.001:
                result["direction"] = "short"
            else:
                result["direction"] = "neutral"
            return result
        except Exception as e:
            logger.error("inference_error", error=str(e), symbol=symbol)
            return self._generate_stub_prediction(symbol)

    def _generate_stub_prediction(self, symbol: str) -> dict[str, Any]:
        """Generate stub prediction when model unavailable."""
        result = {
            "symbol": symbol,
            "confidence": 0.5,
            "direction": "neutral",
            "model_version": "stub",
        }
        qp = []
        for horizon in self.horizons:
            result[f"horizon_{horizon}_q10"] = -0.005
            result[f"horizon_{horizon}_q50"] = 0.0
            result[f"horizon_{horizon}_q90"] = 0.005
            qp.append([-0.005, 0.0, 0.005])
        result["quantile_predictions"] = np.array(qp)
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded
