from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog


@dataclass
class FeatureState:
    """Rolling window state for a single symbol."""
    symbol: str
    max_window: int = 500
    prices: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    volumes: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    returns: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    timestamps: deque[str] = field(default_factory=lambda: deque(maxlen=500))

    def add_tick(self, price: float, volume: float, ts: str) -> None:
        if self.prices:
            ret = (price - self.prices[-1]) / max(self.prices[-1], 1e-9)
        else:
            ret = 0.0
        self.prices.append(price)
        self.volumes.append(volume)
        self.returns.append(ret)
        self.timestamps.append(ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "prices": list(self.prices),
            "volumes": list(self.volumes),
            "returns": list(self.returns),
            "timestamps": list(self.timestamps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], max_window: int = 500) -> "FeatureState":
        state = cls(symbol=data["symbol"], max_window=max_window)
        for p, v, r, t in zip(
            data.get("prices", []),
            data.get("volumes", []),
            data.get("returns", []),
            data.get("timestamps", []),
        ):
            state.prices.append(p)
            state.volumes.append(v)
            state.returns.append(r)
            state.timestamps.append(t)
        return state


class FeatureComputer:
    """Deterministic feature computation engine."""

    @staticmethod
    def returns(prices: list[float], period: int) -> float:
        """Compute returns over period."""
        if len(prices) < period + 1:
            return 0.0
        return (prices[-1] - prices[-period - 1]) / max(prices[-period - 1], 1e-9)

    @staticmethod
    def rolling_volatility(returns: list[float], window: int) -> float:
        """Rolling standard deviation of returns."""
        if len(returns) < window:
            return 0.0
        arr = np.array(returns[-window:])
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    @staticmethod
    def sma(prices: list[float], period: int) -> float:
        """Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return float(np.mean(prices[-period:]))

    @staticmethod
    def ema(prices: list[float], period: int) -> float:
        """Exponential Moving Average."""
        if not prices:
            return 0.0
        if len(prices) < period:
            return prices[-1]
        alpha = 2.0 / (period + 1)
        ema_val = prices[0]
        for p in prices[1:]:
            ema_val = alpha * p + (1 - alpha) * ema_val
        return float(ema_val)

    @staticmethod
    def rsi(prices: list[float], period: int = 14) -> float:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def volume_zscore(volumes: list[float], window: int = 20) -> float:
        """Z-score of current volume vs rolling window."""
        if len(volumes) < window:
            return 0.0
        arr = np.array(volumes[-window:])
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std == 0:
            return 0.0
        return (volumes[-1] - mean) / std

    @staticmethod
    def minute_of_day(ts: datetime) -> int:
        """Minutes since midnight."""
        return ts.hour * 60 + ts.minute

    @staticmethod
    def day_of_week(ts: datetime) -> int:
        """Day of week (0=Monday, 6=Sunday)."""
        return ts.weekday()


class FeatureStore:
    """Redis-backed feature state store for restart safety."""

    def __init__(
        self,
        redis_host: str | None = None,
        redis_port: int | None = None,
        key_prefix: str = "awet:features:",
    ) -> None:
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.key_prefix = key_prefix
        self._client: redis.Redis | None = None
        self._local_cache: dict[str, FeatureState] = {}
        self.logger = structlog.get_logger("FeatureStore")

    async def connect(self) -> None:
        if self._client is None:
            self._client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
            )
            try:
                await self._client.ping()
                self.logger.info("redis_connected", host=self.redis_host, port=self.redis_port)
            except Exception as exc:
                self.logger.warning("redis_unavailable", error=str(exc))
                self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_state(self, symbol: str) -> FeatureState:
        """Get feature state for symbol, restoring from Redis if available."""
        if symbol in self._local_cache:
            return self._local_cache[symbol]
        if self._client:
            try:
                key = f"{self.key_prefix}{symbol}"
                data = await self._client.get(key)
                if data:
                    state = FeatureState.from_dict(json.loads(data))
                    self._local_cache[symbol] = state
                    self.logger.info("state_restored", symbol=symbol, ticks=len(state.prices))
                    return state
            except Exception as exc:
                self.logger.warning("redis_get_error", symbol=symbol, error=str(exc))
        state = FeatureState(symbol=symbol)
        self._local_cache[symbol] = state
        return state

    async def save_state(self, state: FeatureState) -> None:
        """Persist feature state to Redis."""
        self._local_cache[state.symbol] = state
        if self._client:
            try:
                key = f"{self.key_prefix}{state.symbol}"
                await self._client.set(key, json.dumps(state.to_dict()), ex=86400)
            except Exception as exc:
                self.logger.warning("redis_save_error", symbol=state.symbol, error=str(exc))
