"""TFT Prediction Engine with quantile outputs.

This module provides a prediction interface that can:
1. Use a stub implementation for testing (default)
2. Load a real PyTorch TFT model if available
3. Interface with external prediction services via HTTP

The engine outputs quantile predictions (q10, q50, q90) for
multiple horizons (30s, 45s, 60s).
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("prediction_engine")


@dataclass
class PredictionInput:
    """Input features for prediction."""
    symbol: str
    price: float
    volume: float
    returns_1: float
    returns_5: float
    returns_15: float
    volatility_5: float
    volatility_15: float
    sma_5: float
    sma_20: float
    ema_5: float
    ema_20: float
    rsi_14: float
    volume_zscore: float
    minute_of_day: int
    day_of_week: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input (15-feature baseline)."""
        return np.array([
            self.price,
            self.volume,
            self.returns_1,
            self.returns_5,
            self.returns_15,
            self.volatility_5,
            self.volatility_15,
            self.sma_5,
            self.sma_20,
            self.ema_5,
            self.ema_20,
            self.rsi_14,
            self.volume_zscore,
            float(self.minute_of_day),
            float(self.day_of_week),
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for flexible feature extraction.

        ONNXInferenceEngine.extract_features() accepts a dict and
        dynamically selects which features to use based on model
        metadata.  Callers with extended features can merge additional
        keys into the returned dict before passing it.
        """
        return {
            "price": self.price,
            "volume": self.volume,
            "returns_1": self.returns_1,
            "returns_5": self.returns_5,
            "returns_15": self.returns_15,
            "volatility_5": self.volatility_5,
            "volatility_15": self.volatility_15,
            "sma_5": self.sma_5,
            "sma_20": self.sma_20,
            "ema_5": self.ema_5,
            "ema_20": self.ema_20,
            "rsi_14": self.rsi_14,
            "volume_zscore": self.volume_zscore,
            "minute_of_day": self.minute_of_day,
            "day_of_week": self.day_of_week,
        }


@dataclass
class PredictionOutput:
    """Quantile prediction outputs."""
    symbol: str
    horizon_30_q10: float
    horizon_30_q50: float
    horizon_30_q90: float
    horizon_45_q10: float
    horizon_45_q50: float
    horizon_45_q90: float
    horizon_60_q10: float
    horizon_60_q50: float
    horizon_60_q90: float
    confidence: float

    def expected_direction(self) -> str:
        """Derive expected direction from q50 predictions."""
        avg_q50 = (self.horizon_30_q50 + self.horizon_45_q50 + self.horizon_60_q50) / 3
        if avg_q50 > 0.001:
            return "long"
        elif avg_q50 < -0.001:
            return "short"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizon_30_q10": self.horizon_30_q10,
            "horizon_30_q50": self.horizon_30_q50,
            "horizon_30_q90": self.horizon_30_q90,
            "horizon_45_q10": self.horizon_45_q10,
            "horizon_45_q50": self.horizon_45_q50,
            "horizon_45_q90": self.horizon_45_q90,
            "horizon_60_q10": self.horizon_60_q10,
            "horizon_60_q50": self.horizon_60_q50,
            "horizon_60_q90": self.horizon_60_q90,
            "confidence": self.confidence,
            "direction": self.expected_direction(),
        }


class BasePredictionEngine(ABC):
    """Base class for prediction engines."""

    @abstractmethod
    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Generate prediction from input features."""
        pass

    @abstractmethod
    async def warmup(self) -> None:
        """Warmup the model (load weights, compile, etc)."""
        pass


class StubPredictionEngine(BasePredictionEngine):
    """Stub prediction engine that generates realistic random predictions.

    Uses a combination of input features to generate correlated predictions
    for testing and development.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._warmed_up = False

    async def warmup(self) -> None:
        self._warmed_up = True
        logger.info("stub_engine_warmup_complete")

    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        momentum = input_data.returns_5 * 0.3 + input_data.returns_15 * 0.2
        vol_adj = 1.0 + input_data.volatility_5 * 2.0
        rsi_signal = (input_data.rsi_14 - 50) / 100.0
        base_pred = momentum * 0.5 + rsi_signal * 0.002
        noise_scale = 0.002 * vol_adj
        noise_30 = self.rng.normal(0, noise_scale)
        noise_45 = self.rng.normal(0, noise_scale * 1.2)
        noise_60 = self.rng.normal(0, noise_scale * 1.5)
        spread_30 = 0.005 * vol_adj
        spread_45 = 0.007 * vol_adj
        spread_60 = 0.01 * vol_adj
        confidence = 1.0 / (1.0 + input_data.volatility_5 * 10)
        confidence = min(0.95, max(0.1, confidence))

        return PredictionOutput(
            symbol=input_data.symbol,
            horizon_30_q10=base_pred - spread_30 + noise_30,
            horizon_30_q50=base_pred + noise_30,
            horizon_30_q90=base_pred + spread_30 + noise_30,
            horizon_45_q10=base_pred * 1.5 - spread_45 + noise_45,
            horizon_45_q50=base_pred * 1.5 + noise_45,
            horizon_45_q90=base_pred * 1.5 + spread_45 + noise_45,
            horizon_60_q10=base_pred * 2.0 - spread_60 + noise_60,
            horizon_60_q50=base_pred * 2.0 + noise_60,
            horizon_60_q90=base_pred * 2.0 + spread_60 + noise_60,
            confidence=confidence,
        )


class TFTModelEngine(BasePredictionEngine):
    """Real TFT model engine using PyTorch.

    .. deprecated::
        This engine uses ``torch.jit.load()`` which is incompatible
        with current training checkpoints (saved via
        ``model.state_dict()``).  Use :class:`ONNXPredictionEngine`
        for production inference.  This class is retained only for
        backward compatibility and will fall back to
        :class:`StubPredictionEngine` when the model cannot be loaded.

    Loads a trained TFT model from disk and generates predictions.
    Falls back to stub if model not available.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv("TFT_MODEL_PATH", "models/tft_model.pt")
        self.model = None
        self._fallback = StubPredictionEngine()

    async def warmup(self) -> None:
        """Load TFT model if available."""
        if os.path.exists(self.model_path):
            try:
                import torch
                self.model = torch.jit.load(self.model_path)
                self.model.eval()
                logger.info("tft_model_loaded", path=self.model_path)
            except Exception as exc:
                logger.warning("tft_model_load_failed", error=str(exc))
                await self._fallback.warmup()
        else:
            logger.info("tft_model_not_found", path=self.model_path, using="stub")
            await self._fallback.warmup()

    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Generate prediction using TFT model or fallback."""
        if self.model is None:
            return await self._fallback.predict(input_data)

        try:
            import torch
            features = torch.tensor(input_data.to_array()).unsqueeze(0)
            with torch.no_grad():
                output = self.model(features)
            quantiles = output.squeeze().numpy()
            return PredictionOutput(
                symbol=input_data.symbol,
                horizon_30_q10=float(quantiles[0]),
                horizon_30_q50=float(quantiles[1]),
                horizon_30_q90=float(quantiles[2]),
                horizon_45_q10=float(quantiles[3]),
                horizon_45_q50=float(quantiles[4]),
                horizon_45_q90=float(quantiles[5]),
                horizon_60_q10=float(quantiles[6]),
                horizon_60_q50=float(quantiles[7]),
                horizon_60_q90=float(quantiles[8]),
                confidence=float(quantiles[9]) if len(quantiles) > 9 else 0.5,
            )
        except Exception as exc:
            logger.warning("tft_inference_failed", error=str(exc))
            return await self._fallback.predict(input_data)


class ONNXPredictionEngine(BasePredictionEngine):
    """ONNX-based prediction engine for production inference.

    Uses onnxruntime for efficient inference with CUDA support.
    Integrates with model registry to automatically load the "green" model.
    Supports hot-reload when a new green model is promoted.
    """

    def __init__(
        self,
        model_path: str | None = None,
        lookback_window: int = 120,
        use_registry: bool = True,
        hot_reload_interval: int = 60,
    ) -> None:
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.use_registry = use_registry
        self.hot_reload_interval = hot_reload_interval
        self._engine = None
        self._fallback = StubPredictionEngine()
        self._model_version: str | None = None
        self._reload_task: asyncio.Task | None = None

    async def warmup(self) -> None:
        """Load ONNX model from registry or explicit path."""
        try:
            from src.ml.onnx_engine import ONNXInferenceEngine
            self._engine = ONNXInferenceEngine(
                model_path=self.model_path,
                lookback_window=self.lookback_window,
                use_registry=self.use_registry,
            )
            if self._engine.load_model():
                self._model_version = self._engine.model_version
                logger.info(
                    "onnx_engine_loaded",
                    path=self._engine.model_path,
                    version=self._model_version,
                )
                # Start hot-reload background task
                if self.use_registry and self.hot_reload_interval > 0:
                    self._reload_task = asyncio.create_task(self._hot_reload_loop())
            else:
                logger.info("onnx_model_not_found", using="stub")
                await self._fallback.warmup()
        except Exception as exc:
            logger.warning("onnx_engine_init_failed", error=str(exc))
            await self._fallback.warmup()

    async def _hot_reload_loop(self) -> None:
        """Background task to check for new green model and reload."""
        while True:
            await asyncio.sleep(self.hot_reload_interval)
            try:
                if self._engine and self._engine.reload_if_needed():
                    old_version = self._model_version
                    self._model_version = self._engine.model_version
                    logger.info(
                        "model_hot_reloaded",
                        old_version=old_version,
                        new_version=self._model_version,
                    )
            except Exception as e:
                logger.warning("hot_reload_failed", error=str(e))

    async def shutdown(self) -> None:
        """Clean shutdown of background tasks."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass

    @property
    def model_version(self) -> str:
        """Get current model version."""
        return self._model_version or "stub"

    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Generate prediction using ONNX model with rolling window."""
        if self._engine is None or not self._engine.is_loaded:
            return await self._fallback.predict(input_data)
        
        # Build event data dict — ONNXInferenceEngine.extract_features()
        # dynamically selects which keys to use based on feature_meta.json
        event_data = input_data.to_dict()
        
        # Add tick to rolling buffer
        self._engine.add_tick(input_data.symbol, event_data)
        
        # Get prediction (returns None if insufficient data in buffer)
        result = self._engine.predict(input_data.symbol)
        if result is None:
            # Not enough data in buffer yet, return stub prediction
            return await self._fallback.predict(input_data)
        
        return PredictionOutput(
            symbol=result["symbol"],
            horizon_30_q10=result.get("horizon_30_q10", 0.0),
            horizon_30_q50=result.get("horizon_30_q50", 0.0),
            horizon_30_q90=result.get("horizon_30_q90", 0.0),
            horizon_45_q10=result.get("horizon_45_q10", 0.0),
            horizon_45_q50=result.get("horizon_45_q50", 0.0),
            horizon_45_q90=result.get("horizon_45_q90", 0.0),
            horizon_60_q10=result.get("horizon_60_q10", 0.0),
            horizon_60_q50=result.get("horizon_60_q50", 0.0),
            horizon_60_q90=result.get("horizon_60_q90", 0.0),
            confidence=result.get("confidence", 0.5),
        )


def create_prediction_engine(engine_type: str | None = None) -> BasePredictionEngine:
    """Factory function to create prediction engine.

    Args:
        engine_type: "stub", "tft", "onnx", or None (auto-detect from registry)

    Returns:
        Configured prediction engine
    """
    engine_type = engine_type or os.getenv("PREDICTION_ENGINE", "auto")

    if engine_type == "stub":
        return StubPredictionEngine()
    elif engine_type == "tft":
        return TFTModelEngine()
    elif engine_type == "onnx":
        return ONNXPredictionEngine(use_registry=True)
    else:
        # Auto-detect: Try registry first, then explicit paths
        try:
            from src.ml.registry import get_green_model_path
            green_path = get_green_model_path()
            if green_path:
                logger.info("using_registry_green_model", path=green_path)
                return ONNXPredictionEngine(use_registry=True)
        except Exception:
            pass
        
        # Fallback to explicit paths
        onnx_path = os.getenv("ONNX_MODEL_PATH", "models/tft_model.onnx")
        if os.path.exists(onnx_path):
            return ONNXPredictionEngine(model_path=onnx_path, use_registry=False)
        
        pt_path = os.getenv("TFT_MODEL_PATH", "models/tft_model.pt")
        if os.path.exists(pt_path):
            return TFTModelEngine(pt_path)
        
        # Default to stub
        logger.info("no_model_found_using_stub")
        return StubPredictionEngine()
