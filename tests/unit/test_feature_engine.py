"""Unit tests for feature engineering engine."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from src.features.engine import FeatureComputer, FeatureState


class TestFeatureComputer:
    """Tests for FeatureComputer calculations."""

    def test_returns_simple(self) -> None:
        """Test returns calculation."""
        prices = [100.0, 102.0, 105.0]
        assert FeatureComputer.returns(prices, 1) == pytest.approx(0.0294, rel=0.01)
        assert FeatureComputer.returns(prices, 2) == pytest.approx(0.05, rel=0.01)

    def test_returns_insufficient_data(self) -> None:
        """Returns 0 when insufficient data."""
        assert FeatureComputer.returns([100.0], 1) == 0.0
        assert FeatureComputer.returns([100.0, 102.0], 5) == 0.0

    def test_rolling_volatility(self) -> None:
        """Test volatility calculation."""
        returns = [0.01, -0.02, 0.015, -0.005, 0.02]
        vol = FeatureComputer.rolling_volatility(returns, 5)
        assert vol > 0.0
        assert vol < 0.1

    def test_sma(self) -> None:
        """Test simple moving average."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        sma = FeatureComputer.sma(prices, 3)
        assert sma == pytest.approx(106.0, rel=0.01)

    def test_sma_insufficient_data(self) -> None:
        """SMA returns last price when insufficient data."""
        prices = [100.0, 102.0]
        sma = FeatureComputer.sma(prices, 5)
        assert sma == 102.0

    def test_ema(self) -> None:
        """Test exponential moving average."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        ema = FeatureComputer.ema(prices, 3)
        assert ema > 106.0
        assert ema < 108.0

    def test_rsi_neutral(self) -> None:
        """Test RSI with mixed gains/losses."""
        prices = [100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.5, 
                  102.5, 102.0, 103.0, 102.5, 103.5, 103.0, 104.0, 103.5]
        rsi = FeatureComputer.rsi(prices, 14)
        assert 30.0 < rsi < 70.0

    def test_rsi_insufficient_data(self) -> None:
        """RSI returns 50 when insufficient data."""
        prices = [100.0, 102.0, 104.0]
        assert FeatureComputer.rsi(prices, 14) == 50.0

    def test_volume_zscore(self) -> None:
        """Test volume z-score calculation."""
        volumes = [1000.0] * 19 + [2000.0]
        zscore = FeatureComputer.volume_zscore(volumes, 20)
        assert zscore > 2.0

    def test_minute_of_day(self) -> None:
        """Test minute of day calculation."""
        ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert FeatureComputer.minute_of_day(ts) == 870

    def test_day_of_week(self) -> None:
        """Test day of week calculation (Monday=0)."""
        monday = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert FeatureComputer.day_of_week(monday) == 0


class TestFeatureState:
    """Tests for FeatureState rolling window."""

    def test_add_tick(self) -> None:
        """Test adding ticks to state."""
        state = FeatureState(symbol="AAPL")
        state.add_tick(100.0, 1000.0, "2024-01-15T14:30:00Z")
        state.add_tick(102.0, 1200.0, "2024-01-15T14:31:00Z")
        
        assert len(state.prices) == 2
        assert len(state.volumes) == 2
        assert len(state.returns) == 2
        assert state.returns[0] == 0.0
        assert state.returns[1] == pytest.approx(0.02, rel=0.01)

    def test_serialization(self) -> None:
        """Test state serialization and deserialization."""
        state = FeatureState(symbol="AAPL")
        state.add_tick(100.0, 1000.0, "2024-01-15T14:30:00Z")
        state.add_tick(102.0, 1200.0, "2024-01-15T14:31:00Z")
        
        data = state.to_dict()
        restored = FeatureState.from_dict(data)
        
        assert restored.symbol == state.symbol
        assert list(restored.prices) == list(state.prices)
        assert list(restored.volumes) == list(state.volumes)
        assert list(restored.returns) == list(state.returns)

    def test_max_window(self) -> None:
        """Test that state respects max window size."""
        state = FeatureState(symbol="AAPL", max_window=5)
        state.prices = state.prices.__class__(maxlen=5)
        
        for i in range(10):
            state.add_tick(100.0 + i, 1000.0, f"2024-01-15T14:{30+i}:00Z")
        
        assert len(state.prices) == 5
        assert state.prices[0] == 105.0
