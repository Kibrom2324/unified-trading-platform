"""Tests for TFT model and ONNX export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if onnxruntime is available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTFTModel:
    """Tests for TFT model architecture."""
    
    def test_model_forward_pass(self):
        """Test model forward pass produces correct output shapes."""
        from src.ml.tft.model import TFTConfig, TemporalFusionTransformerCore
        
        config = TFTConfig(
            input_dim=15,
            hidden_dim=32,
            num_heads=2,
            num_encoder_layers=1,
            lookback_window=60,
            horizons=[30, 45, 60],
            quantiles=[0.1, 0.5, 0.9],
        )
        
        model = TemporalFusionTransformerCore(config)
        model.eval()
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, config.lookback_window, config.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shapes
        assert "quantiles" in output
        assert "confidence" in output
        
        # quantiles: (batch, num_horizons, num_quantiles)
        assert output["quantiles"].shape == (batch_size, 3, 3)
        
        # confidence: (batch,)
        assert output["confidence"].shape == (batch_size,)
        
        # Confidence should be between 0 and 1
        assert (output["confidence"] >= 0).all()
        assert (output["confidence"] <= 1).all()
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        from src.ml.tft.model import TFTConfig, TemporalFusionTransformerCore, QuantileLoss
        
        config = TFTConfig(hidden_dim=16, num_heads=2)
        model = TemporalFusionTransformerCore(config)
        
        x = torch.randn(2, config.lookback_window, config.input_dim)
        y = torch.randn(2, 3)  # 3 horizons
        
        output = model(x)
        loss = QuantileLoss()(output["quantiles"], y)
        loss.backward()
        
        # Check that at least some gradients exist (some may be None due to unused paths)
        has_grads = any(
            p.grad is not None 
            for p in model.parameters() 
            if p.requires_grad
        )
        assert has_grads, "No gradients computed for any parameter"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestQuantileLoss:
    """Tests for quantile loss function."""
    
    def test_loss_computation(self):
        """Test basic loss computation."""
        from src.ml.tft.model import QuantileLoss
        
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        # Perfect predictions should have low loss
        predictions = torch.zeros(4, 3, 3)  # (batch, horizons, quantiles)
        targets = torch.zeros(4, 3)  # (batch, horizons)
        
        loss = criterion(predictions, targets)
        assert loss.item() < 0.1
    
    def test_loss_positive(self):
        """Test that loss is always positive."""
        from src.ml.tft.model import QuantileLoss
        
        criterion = QuantileLoss()
        predictions = torch.randn(4, 3, 3)
        targets = torch.randn(4, 3)
        
        loss = criterion(predictions, targets)
        assert loss.item() >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestONNXExport:
    """Tests for ONNX export functionality."""
    
    def test_export_to_onnx(self):
        """Test ONNX export creates valid file."""
        from src.ml.tft.model import TFTConfig, TemporalFusionTransformerCore, export_to_onnx
        
        config = TFTConfig(hidden_dim=16, num_heads=2)
        model = TemporalFusionTransformerCore(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            
            result_path = export_to_onnx(
                model,
                str(onnx_path),
                lookback_window=config.lookback_window,
                input_dim=config.input_dim,
            )
            
            assert Path(result_path).exists()
            assert Path(result_path).stat().st_size > 0
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")
    def test_onnx_inference(self):
        """Test ONNX model inference produces correct shapes."""
        from src.ml.tft.model import TFTConfig, TemporalFusionTransformerCore, export_to_onnx
        
        config = TFTConfig(hidden_dim=16, num_heads=2)
        model = TemporalFusionTransformerCore(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                model,
                str(onnx_path),
                lookback_window=config.lookback_window,
                input_dim=config.input_dim,
            )
            
            # Load and run inference
            session = ort.InferenceSession(str(onnx_path))
            
            # Create input
            x = np.random.randn(1, config.lookback_window, config.input_dim).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {"features": x})
            predictions = outputs[0]
            
            # Expected output: 9 quantiles (3 horizons × 3 quantiles) + 1 confidence = 10
            assert predictions.shape == (1, 10)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")
class TestONNXInferenceEngine:
    """Tests for ONNX inference engine."""
    
    @pytest.fixture
    def engine_with_model(self):
        """Create engine with a trained model."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required for this test")
        
        from src.ml.tft.model import TFTConfig, TemporalFusionTransformerCore, export_to_onnx
        from src.ml.onnx_engine import ONNXInferenceEngine
        
        config = TFTConfig(hidden_dim=16, num_heads=2, lookback_window=60)
        model = TemporalFusionTransformerCore(config)
        
        tmpdir = tempfile.mkdtemp()
        onnx_path = Path(tmpdir) / "model.onnx"
        export_to_onnx(
            model,
            str(onnx_path),
            lookback_window=config.lookback_window,
            input_dim=config.input_dim,
        )
        
        engine = ONNXInferenceEngine(
            model_path=str(onnx_path),
            lookback_window=config.lookback_window,
            use_registry=False,
        )
        engine.load_model()
        
        return engine
    
    def test_buffer_management(self, engine_with_model):
        """Test per-symbol buffer management."""
        engine = engine_with_model
        
        # Add ticks for two symbols
        for i in range(10):
            engine.add_tick("AAPL", {"price": 150.0 + i, "volume": 100000})
            engine.add_tick("MSFT", {"price": 300.0 + i, "volume": 200000})
        
        # Buffers should exist for both
        assert "AAPL" in engine._buffers
        assert "MSFT" in engine._buffers
        
        # Buffer lengths should match
        assert len(engine.get_buffer("AAPL")) == 10
        assert len(engine.get_buffer("MSFT")) == 10
    
    def test_predict_insufficient_data(self, engine_with_model):
        """Test prediction with insufficient data returns None."""
        engine = engine_with_model
        
        # Add only a few ticks (less than lookback window)
        for i in range(5):
            engine.add_tick("AAPL", {"price": 150.0, "volume": 100000})
        
        result = engine.predict("AAPL")
        assert result is None  # Should return None due to insufficient data
    
    def test_predict_with_features(self, engine_with_model):
        """Test prediction with pre-built features."""
        engine = engine_with_model
        
        # Create feature array directly
        features = np.random.randn(engine.lookback_window, 15).astype(np.float32)
        
        result = engine.predict("AAPL", features=features)
        
        assert result is not None
        assert "symbol" in result
        assert "confidence" in result
        assert "horizon_30_q50" in result
        assert "direction" in result
    
    def test_stub_prediction(self):
        """Test stub prediction when model not loaded."""
        from src.ml.onnx_engine import ONNXInferenceEngine
        
        # Engine without a valid model
        engine = ONNXInferenceEngine(
            model_path="/nonexistent/model.onnx",
            use_registry=False,
        )
        
        # Should fall back to stub
        stub = engine._generate_stub_prediction("AAPL")
        
        assert stub["symbol"] == "AAPL"
        assert stub["confidence"] == 0.5
        assert stub["direction"] == "neutral"
        assert "horizon_30_q50" in stub
