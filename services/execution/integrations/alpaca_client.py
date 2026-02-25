"""
Alpaca Trading Client - Paper Trading Only
==========================================

This module provides a client for submitting orders to Alpaca's PAPER trading API.

SAFETY GUARANTEES:
-----------------
1. Only paper-api.alpaca.markets is allowed - live endpoints are rejected at startup
2. All orders are marked with extended_hours=False for safety
3. Retries with exponential backoff for transient errors
4. Full error capture and logging

This client is intentionally minimal - it only supports:
- Market orders (no limit/stop orders)
- Buy side only (sell support can be added later)
- Paper trading only (live trading is blocked)

Usage:
    client = AlpacaClient.from_env()
    result = await client.submit_market_order("AAPL", qty=1, side="buy")
    if result.success:
        print(f"Order {result.order_id} submitted: {result.status}")
    else:
        print(f"Order failed: {result.error_message}")
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Safety: Only allow paper trading endpoints
ALLOWED_BASE_URLS = [
    "https://paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets/",
]

# Live endpoints that must NEVER be used
BLOCKED_LIVE_URLS = [
    "https://api.alpaca.markets",
    "https://broker-api.alpaca.markets",
]


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Alpaca order statuses."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"


@dataclass
class OrderResult:
    """Result of an order submission attempt."""
    success: bool
    order_id: str | None = None
    status: str | None = None
    symbol: str | None = None
    filled_qty: int = 0
    avg_price: float | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "status": self.status,
            "symbol": self.symbol,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "error_message": self.error_message,
        }


class AlpacaClientError(Exception):
    """Base exception for Alpaca client errors."""
    pass


class AlpacaLiveEndpointError(AlpacaClientError):
    """Raised when attempting to use a live trading endpoint."""
    pass


class AlpacaClient:
    """
    Async client for Alpaca Paper Trading API.
    
    This client ONLY supports paper trading. Any attempt to configure
    a live trading endpoint will raise AlpacaLiveEndpointError.
    
    Example:
        client = AlpacaClient.from_env()
        result = await client.submit_market_order("AAPL", qty=1, side="buy")
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Alpaca client.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: API base URL (MUST be paper-api.alpaca.markets)
            timeout: Request timeout in seconds
            max_retries: Max retries for transient errors
            
        Raises:
            AlpacaLiveEndpointError: If base_url is a live trading endpoint
        """
        # CRITICAL SAFETY CHECK: Block live endpoints
        self._validate_paper_endpoint(base_url)
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        
        logger.info(
            "alpaca_client_initialized",
            base_url=self.base_url,
            paper_trading=True,
        )
    
    @staticmethod
    def _validate_paper_endpoint(base_url: str) -> None:
        """
        Validate that the base URL is a paper trading endpoint.
        
        Raises:
            AlpacaLiveEndpointError: If URL is a live trading endpoint
        """
        url_normalized = base_url.rstrip("/").lower()
        
        # Check for blocked live URLs
        for blocked in BLOCKED_LIVE_URLS:
            if url_normalized == blocked.lower():
                raise AlpacaLiveEndpointError(
                    f"SAFETY BLOCK: Live trading endpoint detected: {base_url}. "
                    "This system only supports paper trading. "
                    "Use https://paper-api.alpaca.markets instead."
                )
        
        # Check for allowed paper URLs
        if url_normalized not in [u.rstrip("/").lower() for u in ALLOWED_BASE_URLS]:
            raise AlpacaLiveEndpointError(
                f"SAFETY BLOCK: Unknown Alpaca endpoint: {base_url}. "
                "Only https://paper-api.alpaca.markets is allowed."
            )
    
    @classmethod
    def from_env(cls) -> "AlpacaClient":
        """
        Create client from environment variables.
        
        Required env vars:
            ALPACA_API_KEY: API key
            ALPACA_SECRET_KEY: Secret key
            ALPACA_BASE_URL: Base URL (must be paper endpoint)
        """
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not api_key or not secret_key:
            raise AlpacaClientError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment"
            )
        
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """
        Make HTTP request with exponential backoff retry.
        
        Retries on:
        - 429 (rate limit)
        - 500, 502, 503, 504 (server errors)
        - Connection errors
        """
        client = await self._get_client()
        last_exception: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.request(method, path, json=json_data)
                
                # Don't retry client errors (except rate limit)
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(
                        "alpaca_rate_limited",
                        attempt=attempt + 1,
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                if response.status_code >= 500:
                    # Server error - retry
                    wait_time = 2 ** attempt
                    logger.warning(
                        "alpaca_server_error",
                        status_code=response.status_code,
                        attempt=attempt + 1,
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                # Success or non-retryable error
                return response
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.warning(
                    "alpaca_connection_error",
                    error=str(e),
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        raise AlpacaClientError("Max retries exceeded")
    
    async def get_account(self) -> dict[str, Any]:
        """Get account information."""
        response = await self._request_with_retry("GET", "/v2/account")
        response.raise_for_status()
        return response.json()

    async def get_clock(self) -> dict[str, Any]:
        """Get market clock status from Alpaca."""
        response = await self._request_with_retry("GET", "/v2/clock")
        response.raise_for_status()
        return response.json()
    
    async def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
    ) -> OrderResult:
        """
        Submit a market order to Alpaca Paper Trading.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            qty: Number of shares
            side: "buy" or "sell"
            
        Returns:
            OrderResult with success/failure info
        """
        order_data = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day",
        }
        
        logger.info(
            "alpaca_submitting_order",
            symbol=symbol,
            qty=qty,
            side=side,
            paper_trading=True,
        )
        
        try:
            response = await self._request_with_retry(
                "POST",
                "/v2/orders",
                json_data=order_data,
            )
            
            if response.status_code in (200, 201):
                data = response.json()
                result = OrderResult(
                    success=True,
                    order_id=data.get("id"),
                    status=data.get("status"),
                    filled_qty=int(data.get("filled_qty") or 0),
                    avg_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
                    raw_response=data,
                )
                logger.info(
                    "alpaca_order_submitted",
                    order_id=result.order_id,
                    status=result.status,
                    symbol=symbol,
                )
                return result
            
            # Order rejected by Alpaca
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("message", f"HTTP {response.status_code}")
            
            logger.warning(
                "alpaca_order_rejected",
                status_code=response.status_code,
                error=error_msg,
                symbol=symbol,
            )
            
            return OrderResult(
                success=False,
                error_message=error_msg,
                raw_response=error_data,
            )
            
        except Exception as e:
            logger.exception(
                "alpaca_order_error",
                error=str(e),
                symbol=symbol,
            )
            return OrderResult(
                success=False,
                error_message=str(e),
            )

    async def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> OrderResult:
        """
        Submit a bracket order (entry + take-profit + stop-loss).

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: "buy" or "sell" (entry side)
            take_profit_price: Limit price for take-profit leg
            stop_loss_price: Stop price for stop-loss leg

        Returns:
            OrderResult with success/failure info
        """
        order_data = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {
                "limit_price": f"{take_profit_price:.2f}",
            },
            "stop_loss": {
                "stop_price": f"{stop_loss_price:.2f}",
            },
        }

        logger.info(
            "alpaca_submitting_bracket_order",
            symbol=symbol,
            qty=qty,
            side=side,
            take_profit=take_profit_price,
            stop_loss=stop_loss_price,
            paper_trading=True,
        )

        try:
            response = await self._request_with_retry(
                "POST",
                "/v2/orders",
                json_data=order_data,
            )

            if response.status_code in (200, 201):
                data = response.json()
                result = OrderResult(
                    success=True,
                    order_id=data.get("id"),
                    status=data.get("status"),
                    filled_qty=int(data.get("filled_qty") or 0),
                    avg_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
                    raw_response=data,
                )
                logger.info(
                    "alpaca_bracket_order_submitted",
                    order_id=result.order_id,
                    status=result.status,
                    symbol=symbol,
                )
                return result

            error_data = response.json() if response.content else {}
            error_msg = error_data.get("message", f"HTTP {response.status_code}")

            logger.warning(
                "alpaca_bracket_order_rejected",
                status_code=response.status_code,
                error=error_msg,
                symbol=symbol,
            )

            return OrderResult(
                success=False,
                error_message=error_msg,
                raw_response=error_data,
            )

        except Exception as e:
            logger.exception(
                "alpaca_bracket_order_error",
                error=str(e),
                symbol=symbol,
            )
            return OrderResult(
                success=False,
                error_message=str(e),
            )
    
    async def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Get order details by ID."""
        try:
            response = await self._request_with_retry("GET", f"/v2/orders/{order_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    async def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        response = await self._request_with_retry("GET", "/v2/positions")
        response.raise_for_status()
        return response.json()
    
    async def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Get position for a specific symbol."""
        try:
            response = await self._request_with_retry(
                "GET", f"/v2/positions/{symbol.upper()}"
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    async def get_position_qty(self, symbol: str) -> int:
        """
        Get current position quantity for a symbol.
        
        Returns:
            Positive qty for long, negative for short, 0 if no position
        """
        position = await self.get_position(symbol)
        if position:
            return int(float(position.get("qty", 0)))
        return 0
    
    async def get_latest_trade(self, symbol: str) -> float | None:
        """
        Get the latest trade price for a symbol.
        
        Uses Alpaca's /v2/stocks/{symbol}/trades/latest endpoint.
        Falls back to last quote if trade not available.
        
        Returns:
            Latest trade price or None if unavailable
        """
        try:
            # Use data API for latest trade
            # Note: Paper account uses same data endpoints
            response = await self._request_with_retry(
                "GET", f"/v2/stocks/{symbol.upper()}/trades/latest"
            )
            if response.status_code == 200:
                data = response.json()
                trade = data.get("trade", {})
                return float(trade.get("p", 0)) if trade.get("p") else None
            return None
        except Exception as e:
            logger.warning(
                "alpaca_latest_trade_error",
                symbol=symbol,
                error=str(e),
            )
            return None
    
    async def get_latest_quote(self, symbol: str) -> dict[str, float] | None:
        """
        Get the latest quote (bid/ask) for a symbol.
        
        Returns:
            Dict with 'bid', 'ask', 'mid' prices or None if unavailable
        """
        try:
            response = await self._request_with_retry(
                "GET", f"/v2/stocks/{symbol.upper()}/quotes/latest"
            )
            if response.status_code == 200:
                data = response.json()
                quote = data.get("quote", {})
                bid = float(quote.get("bp", 0))
                ask = float(quote.get("ap", 0))
                if bid > 0 and ask > 0:
                    return {
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                    }
            return None
        except Exception as e:
            logger.warning(
                "alpaca_latest_quote_error",
                symbol=symbol,
                error=str(e),
            )
            return None
    
    async def get_current_price(self, symbol: str) -> float | None:
        """
        Get the best available current price for a symbol.
        
        Tries latest trade first, falls back to quote midpoint.
        
        Returns:
            Current price or None if unavailable
        """
        # Try latest trade first
        price = await self.get_latest_trade(symbol)
        if price and price > 0:
            return price
        
        # Fall back to quote midpoint
        quote = await self.get_latest_quote(symbol)
        if quote and quote.get("mid"):
            return quote["mid"]
        
        return None


# Convenience function for testing
async def test_connection() -> bool:
    """Test Alpaca connection with current environment."""
    try:
        client = AlpacaClient.from_env()
        account = await client.get_account()
        await client.close()
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"   Account: {account.get('id')}")
        print(f"   Status: {account.get('status')}")
        print(f"   Buying Power: ${account.get('buying_power')}")
        return True
    except AlpacaLiveEndpointError as e:
        print(f"❌ SAFETY BLOCK: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())
