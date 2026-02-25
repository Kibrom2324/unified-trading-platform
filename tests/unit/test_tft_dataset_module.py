"""Tests for TFT dataset module."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.tft.dataset import (
    FEATURE_COLUMNS,
    DatasetConfig,
    NormStats,
    build_sequences,
    compute_norm_stats,
)


@pytest.fixture
def sample_symbol_data():
    """Generate sample symbol data for testing."""
    # Generate 500 samples for one symbol
    np.random.seed(42)
    n_samples = 500
    
    symbol_data = {"AAPL": []}
    base_price = 150.0
    
    for i in range(n_samples):
        price = base_price + np.random.randn() * 2
        symbol_data["AAPL"].append({
            "ts": f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00",
            "price": price,
            "volume": 100000 + np.random.randint(0, 50000),
            "returns_1": np.random.randn() * 0.01,
            "returns_5": np.random.randn() * 0.02,
            "returns_15": np.random.randn() * 0.03,
            "volatility_5": abs(np.random.randn()) * 0.01,
            "volatility_15": abs(np.random.randn()) * 0.02,
            "sma_5": price,
            "sma_20": price,
            "ema_5": price,
            "ema_20": price,
            "rsi_14": 50 + np.random.randn() * 15,
            "volume_zscore": np.random.randn(),
            "minute_of_day": i % (24 * 60),
            "day_of_week": i % 5,
        })
    
    return symbol_data


@pytest.fixture
def config():
    """Default dataset config."""
    return DatasetConfig(
        lookback_window=60,  # Smaller for testing
        horizons=[30, 45, 60],
        train_split=0.8,
        normalize=True,
    )


class TestFeatureColumns:
    """Tests for feature column definitions."""
    
    def test_feature_count(self):
        """Verify expected number of features."""
        assert len(FEATURE_COLUMNS) == 15
    
    def test_required_features_present(self):
        """Verify critical features are included."""
        required = ["price", "volume", "returns_1", "rsi_14"]
        for feat in required:
            assert feat in FEATURE_COLUMNS


class TestNormStats:
    """Tests for normalization statistics."""
    
    def test_normalize_denormalize(self):
        """Test normalization round-trip."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = NormStats(
            means=np.array([2.5, 3.5, 4.5]),
            stds=np.array([1.5, 1.5, 1.5]),
        )
        
        normalized = stats.normalize(data)
        denormalized = stats.denormalize(normalized)
        
        np.testing.assert_array_almost_equal(data, denormalized)
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        stats = NormStats(
            means=np.array([1.0, 2.0]),
            stds=np.array([0.5, 0.5]),
        )
        
        d = stats.to_dict()
        restored = NormStats.from_dict(d)
        
        np.testing.assert_array_equal(stats.means, restored.means)
        np.testing.assert_array_equal(stats.stds, restored.stds)


class TestComputeNormStats:
    """Tests for compute_norm_stats function."""
    
    def test_computes_stats(self, sample_symbol_data):
        """Test that stats are computed correctly."""
        stats = compute_norm_stats(sample_symbol_data)
        
        assert stats.means.shape == (15,)
        assert stats.stds.shape == (15,)
        assert all(stats.stds > 0), "All stds should be positive"


class TestBuildSequences:
    """Tests for sequence building."""
    
    def test_output_shapes(self, sample_symbol_data, config):
        """Test output array shapes."""
        X, y, sample_ids, symbols = build_sequences(sample_symbol_data, config)
        
        # Should have samples
        assert len(X) > 0, "Should generate samples"
        
        # X shape: (samples, lookback_window, features)
        assert X.shape[1] == config.lookback_window
        assert X.shape[2] == len(FEATURE_COLUMNS)
        
        # y shape: (samples, num_horizons)
        assert y.shape[1] == len(config.horizons)
        
        # sample_ids and symbols should match X length
        assert len(sample_ids) == len(X)
        assert len(symbols) == len(X)
    
    def test_insufficient_data(self, config):
        """Test handling of insufficient data."""
        small_data = {"AAPL": [
            {"price": 100, "volume": 1000, "ts": "2024-01-01T00:00:00+00:00"}
            for _ in range(10)
        ]}
        
        X, y, _, _ = build_sequences(small_data, config)
        
        # Should not generate samples with insufficient data
        assert len(X) == 0
    
    def test_multiple_symbols(self, config):
        """Test handling of multiple symbols."""
        np.random.seed(42)
        
        # Create data for two symbols
        multi_symbol_data = {}
        for symbol in ["AAPL", "MSFT"]:
            multi_symbol_data[symbol] = []
            for i in range(300):
                multi_symbol_data[symbol].append({
                    "ts": f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00",
                    "price": 150.0 + np.random.randn(),
                    "volume": 100000,
                    "returns_1": 0.001,
                    "returns_5": 0.002,
                    "returns_15": 0.003,
                    "volatility_5": 0.01,
                    "volatility_15": 0.02,
                    "sma_5": 150.0,
                    "sma_20": 150.0,
                    "ema_5": 150.0,
                    "ema_20": 150.0,
                    "rsi_14": 50.0,
                    "volume_zscore": 0.0,
                    "minute_of_day": i,
                    "day_of_week": 0,
                })
        
        X, y, _, symbols = build_sequences(multi_symbol_data, config)
        
        # Should have samples from both symbols
        unique_symbols = set(symbols)
        assert len(unique_symbols) == 2
        assert "AAPL" in unique_symbols
        assert "MSFT" in unique_symbols
    
    def test_normalization_applied(self, sample_symbol_data, config):
        """Test that normalization is applied when enabled."""
        config.normalize = True
        X_norm, _, _, _ = build_sequences(sample_symbol_data, config)
        
        config.normalize = False
        X_raw, _, _, _ = build_sequences(sample_symbol_data, config)
        
        # Normalized data should have different values
        # (assuming raw data isn't already normalized)
        assert not np.allclose(X_norm, X_raw)
    
    def test_target_values_are_returns(self, sample_symbol_data, config):
        """Test that targets are return values."""
        X, y, _, _ = build_sequences(sample_symbol_data, config)
        
        # Returns should be small values (roughly -0.1 to 0.1 for normal markets)
        assert np.abs(y).max() < 1.0, "Targets should be returns, not prices"


class TestDatasetConfig:
    """Tests for DatasetConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()
        
        assert config.lookback_window == 120
        assert config.horizons == [30, 45, 60]
        assert config.train_split == 0.8
        assert config.normalize is True
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = DatasetConfig(
            lookback_window=60,
            horizons=[15, 30],
            train_split=0.9,
            batch_size=128,
        )
        
        assert config.lookback_window == 60
        assert config.horizons == [15, 30]
        assert config.train_split == 0.9
        assert config.batch_size == 128
