from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

from src.core.circuit_breaker import CircuitBreaker
from src.core.retry import retry_async


@dataclass
class OHLCVBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trades: int | None = None


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limit_per_minute: int = 5,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timeout_seconds = timeout_seconds
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        self._request_times: list[float] = []
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _rate_limit(self) -> None:
        """Token bucket rate limiting."""
        now = asyncio.get_event_loop().time()
        window_start = now - 60.0
        self._request_times = [t for t in self._request_times if t > window_start]
        if len(self._request_times) >= self.rate_limit_per_minute:
            sleep_time = 60.0 - (now - self._request_times[0]) + 0.1
            self.logger.info("rate_limit_wait", sleep_seconds=sleep_time)
            await asyncio.sleep(sleep_time)
        self._request_times.append(asyncio.get_event_loop().time())

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with rate limiting, retries, and circuit breaker."""
        if not self._circuit_breaker.allow():
            raise RuntimeError("Circuit breaker open - provider unavailable")

        await self._rate_limit()

        async def do_request() -> dict[str, Any]:
            session = await self._get_session()
            async with session.request(method, url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    raise RuntimeError("Rate limited by provider")
                if resp.status >= 500:
                    raise RuntimeError(f"Server error: {resp.status}")
                resp.raise_for_status()
                return await resp.json()

        try:
            result = await retry_async(do_request, retries=3, base_delay=1.0, max_delay=10.0)
            self._circuit_breaker.record_success()
            return result
        except Exception:
            self._circuit_breaker.record_failure()
            raise

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for a symbol."""
        ...

    @abstractmethod
    async def get_latest_bar(self, symbol: str) -> OHLCVBar | None:
        """Fetch the latest bar for a symbol."""
        ...


class PolygonProvider(MarketDataProvider):
    """Polygon.io market data provider."""

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        multiplier, span = self._parse_timeframe(timeframe)
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{span}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        data = await self._request("GET", url, headers=headers, params=params)
        results = data.get("results", [])
        bars = []
        for r in results:
            ts = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc)
            if start <= ts <= end:
                bars.append(
                    OHLCVBar(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(r["o"]),
                        high=float(r["h"]),
                        low=float(r["l"]),
                        close=float(r["c"]),
                        volume=float(r["v"]),
                        vwap=float(r.get("vw", 0)) if r.get("vw") else None,
                        trades=int(r.get("n", 0)) if r.get("n") else None,
                    )
                )
        self.logger.info("fetched_bars", symbol=symbol, count=len(bars))
        return bars

    async def get_latest_bar(self, symbol: str) -> OHLCVBar | None:
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"adjusted": "true"}
        data = await self._request("GET", url, headers=headers, params=params)
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return OHLCVBar(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc),
            open=float(r["o"]),
            high=float(r["h"]),
            low=float(r["l"]),
            close=float(r["c"]),
            volume=float(r["v"]),
            vwap=float(r.get("vw", 0)) if r.get("vw") else None,
            trades=int(r.get("n", 0)) if r.get("n") else None,
        )

    def _parse_timeframe(self, timeframe: str) -> tuple[int, str]:
        """Parse timeframe string like '1Min' -> (1, 'minute')."""
        mapping = {
            "1Min": (1, "minute"),
            "5Min": (5, "minute"),
            "15Min": (15, "minute"),
            "1H": (1, "hour"),
            "1D": (1, "day"),
        }
        if timeframe in mapping:
            return mapping[timeframe]
        raise ValueError(f"Unsupported timeframe: {timeframe}")


class AlpacaProvider(MarketDataProvider):
    """Alpaca market data provider."""

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        url = f"{self.base_url}/v2/stocks/{symbol}/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key.split(":")[0] if ":" in self.api_key else self.api_key,
            "APCA-API-SECRET-KEY": self.api_key.split(":")[1] if ":" in self.api_key else "",
        }
        params = {
            "timeframe": self._convert_timeframe(timeframe),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "all",
            "limit": 10000,
        }
        data = await self._request("GET", url, headers=headers, params=params)
        results = data.get("bars", [])
        bars = []
        for r in results:
            ts = datetime.fromisoformat(r["t"].replace("Z", "+00:00"))
            bars.append(
                OHLCVBar(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=float(r["v"]),
                    vwap=float(r.get("vw", 0)) if r.get("vw") else None,
                    trades=int(r.get("n", 0)) if r.get("n") else None,
                )
            )
        self.logger.info("fetched_bars", symbol=symbol, count=len(bars))
        return bars

    async def get_latest_bar(self, symbol: str) -> OHLCVBar | None:
        url = f"{self.base_url}/v2/stocks/{symbol}/bars/latest"
        headers = {
            "APCA-API-KEY-ID": self.api_key.split(":")[0] if ":" in self.api_key else self.api_key,
            "APCA-API-SECRET-KEY": self.api_key.split(":")[1] if ":" in self.api_key else "",
        }
        data = await self._request("GET", url, headers=headers)
        bar = data.get("bar")
        if not bar:
            return None
        return OHLCVBar(
            symbol=symbol,
            timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
            open=float(bar["o"]),
            high=float(bar["h"]),
            low=float(bar["l"]),
            close=float(bar["c"]),
            volume=float(bar["v"]),
            vwap=float(bar.get("vw", 0)) if bar.get("vw") else None,
            trades=int(bar.get("n", 0)) if bar.get("n") else None,
        )

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe to Alpaca format."""
        mapping = {
            "1Min": "1Min",
            "5Min": "5Min",
            "15Min": "15Min",
            "1H": "1Hour",
            "1D": "1Day",
        }
        if timeframe in mapping:
            return mapping[timeframe]
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def create_provider(
    provider_name: str,
    api_key: str,
    base_url: str,
    rate_limit_per_minute: int = 5,
    timeout_seconds: float = 30.0,
) -> MarketDataProvider:
    """Factory function to create market data provider."""
    providers = {
        "polygon": PolygonProvider,
        "alpaca": AlpacaProvider,
        "yfinance": YFinanceProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: {list(providers.keys())}")
    return providers[provider_name](
        api_key=api_key,
        base_url=base_url,
        rate_limit_per_minute=rate_limit_per_minute,
        timeout_seconds=timeout_seconds,
    )


class YFinanceProvider(MarketDataProvider):
    """
    Yahoo Finance market data provider (FREE, no API key required).
    
    Supports daily and intraday data for US stocks.
    Rate limits are more relaxed than paid providers.
    Best for: daily data, paper trading, development/testing.
    
    Limitations:
    - Intraday data limited to last 60 days
    - 1-minute data limited to last 7 days
    - May have slight delays (15-20 min for some data)
    """

    def __init__(
        self,
        api_key: str = "",  # Not required
        base_url: str = "",  # Not required
        rate_limit_per_minute: int = 30,  # More relaxed
        timeout_seconds: float = 30.0,
    ) -> None:
        # yfinance doesn't need api_key or base_url
        super().__init__(
            api_key=api_key or "none",
            base_url=base_url or "https://query1.finance.yahoo.com",
            rate_limit_per_minute=rate_limit_per_minute,
            timeout_seconds=timeout_seconds,
        )
        self._yf = None

    def _get_yf(self):
        """Lazy load yfinance."""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars using yfinance."""
        await self._rate_limit()
        
        yf = self._get_yf()
        interval = self._convert_timeframe(timeframe)
        
        # yfinance is synchronous, run in executor
        loop = asyncio.get_event_loop()
        
        def fetch_data():
            ticker = yf.Ticker(symbol)
            # yfinance needs timezone-naive or will warn
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            
            df = ticker.history(
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=True,
            )
            return df
        
        try:
            df = await loop.run_in_executor(None, fetch_data)
        except Exception as e:
            self.logger.error("yfinance_fetch_error", symbol=symbol, error=str(e))
            return []
        
        if df.empty:
            self.logger.warning("yfinance_no_data", symbol=symbol)
            return []
        
        bars = []
        for idx, row in df.iterrows():
            # idx is the timestamp
            ts = idx.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            
            bars.append(
                OHLCVBar(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    vwap=None,  # yfinance doesn't provide VWAP
                    trades=None,  # yfinance doesn't provide trade count
                )
            )
        
        self.logger.info("yfinance_fetched_bars", symbol=symbol, count=len(bars))
        return bars

    async def get_latest_bar(self, symbol: str) -> OHLCVBar | None:
        """Fetch the latest bar using yfinance."""
        await self._rate_limit()
        
        yf = self._get_yf()
        loop = asyncio.get_event_loop()
        
        def fetch_latest():
            ticker = yf.Ticker(symbol)
            # Get last 2 days to ensure we get at least 1 bar
            df = ticker.history(period="2d", interval="1d")
            return df
        
        try:
            df = await loop.run_in_executor(None, fetch_latest)
        except Exception as e:
            self.logger.error("yfinance_latest_error", symbol=symbol, error=str(e))
            return None
        
        if df.empty:
            return None
        
        # Get the last row
        row = df.iloc[-1]
        ts = df.index[-1].to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        
        return OHLCVBar(
            symbol=symbol,
            timestamp=ts,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            vwap=None,
            trades=None,
        )

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe to yfinance interval format."""
        mapping = {
            "1Min": "1m",
            "2Min": "2m",
            "5Min": "5m",
            "15Min": "15m",
            "30Min": "30m",
            "1H": "1h",
            "1D": "1d",
            "1W": "1wk",
            "1M": "1mo",
        }
        if timeframe in mapping:
            return mapping[timeframe]
        raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")
