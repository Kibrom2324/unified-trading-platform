"""Unit tests for TFT dataset builder and training."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.ml.dataset import TFTDatasetBuilder, TFTSample


@pytest.fixture
def sample_symbol_data() -> dict[str, list[dict]]:
    """Generate sample symbol data for testing."""
    data = {"AAPL": []}
    base_price = 150.0
    base_time = datetime(2025, 7, 1, 9, 30, 0, tzinfo=timezone.utc)
    for i in range(200):
        ts = base_time + timedelta(minutes=i)
        price = base_price + np.sin(i / 10) * 2 + np.random.randn() * 0.5
        data["AAPL"].append({
            "ts": ts,
            "price": price,
            "volume": 10000 + np.random.randint(0, 5000),
            "returns_1": 0.001 * np.random.randn(),
            "returns_5": 0.005 * np.random.randn(),
            "returns_15": 0.01 * np.random.randn(),
            "volatility_5": 0.02,
            "volatility_15": 0.025,
            "sma_5": price,
            "sma_20": price - 1,
            "ema_5": price,
            "ema_20": price - 0.5,
            "rsi_14": 50 + np.random.randint(-10, 10),
            "volume_zscore": np.random.randn(),
            "minute_of_day": 570 + i,
            "day_of_week": 1,
        })
    return data


class TestTFTDatasetBuilder:
    def test_build_sequences(self, sample_symbol_data: dict) -> None:
        """Should generate valid training sequences."""
        builder = TFTDatasetBuilder(lookback_window=50, horizons=[30, 45, 60])
        samples = list(builder.build_sequences(sample_symbol_data))
        assert len(samples) > 0
        assert all(isinstance(s, TFTSample) for s in samples)

    def test_sample_shapes(self, sample_symbol_data: dict) -> None:
        """Samples should have correct shapes."""
        builder = TFTDatasetBuilder(lookback_window=50, horizons=[30, 45, 60])
        samples = list(builder.build_sequences(sample_symbol_data))
        sample = samples[0]
        assert sample.features.shape == (50, 15)
        assert sample.targets.shape == (3,)

    def test_to_numpy(self, sample_symbol_data: dict) -> None:
        """Should convert to numpy arrays."""
        builder = TFTDatasetBuilder(lookback_window=50, horizons=[30, 45, 60])
        samples = list(builder.build_sequences(sample_symbol_data))
        X, y, ids = builder.to_numpy(samples)
        assert X.ndim == 3
        assert y.ndim == 2
        assert len(ids) == len(samples)

    def test_insufficient_data(self) -> None:
        """Should skip symbols with insufficient data."""
        builder = TFTDatasetBuilder(lookback_window=100, horizons=[30, 45, 60])
        short_data = {"AAPL": [{"ts": datetime.now(), "price": 100}] * 50}
        samples = list(builder.build_sequences(short_data))
        assert len(samples) == 0

    def test_sample_id_unique(self, sample_symbol_data: dict) -> None:
        """Sample IDs should be unique."""
        builder = TFTDatasetBuilder(lookback_window=50, horizons=[30, 45, 60])
        samples = list(builder.build_sequences(sample_symbol_data))
        ids = [s.sample_id for s in samples]
        assert len(ids) == len(set(ids))


class TestTFTSample:
    def test_input_length(self) -> None:
        """Should report correct input length."""
        sample = TFTSample(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            features=np.zeros((120, 15)),
            targets=np.zeros(3),
            sample_id="test123",
        )
        assert sample.input_length == 120
