"""
Trades Repository - Persist executed trades to PostgreSQL/TimescaleDB.

This module provides a repository for storing and querying trade records.
All trades (filled, blocked, rejected) are persisted for audit and analytics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TradeRecord:
    """A single trade record for persistence."""
    symbol: str
    side: str
    qty: int
    intended_notional: float | None
    avg_fill_price: float | None
    status: str
    alpaca_order_id: str | None
    alpaca_status: str | None
    error_message: str | None
    correlation_id: str
    idempotency_key: str
    order_class: str | None = None
    tp_order_id: str | None = None
    sl_order_id: str | None = None
    exit_fill_price: float | None = None
    exit_filled_at: datetime | None = None
    alpaca_raw: dict[str, Any] | None = None
    paper_trade: bool = True
    dry_run: bool = False
    ts: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "intended_notional": self.intended_notional,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status,
            "alpaca_order_id": self.alpaca_order_id,
            "alpaca_status": self.alpaca_status,
            "order_class": self.order_class,
            "tp_order_id": self.tp_order_id,
            "sl_order_id": self.sl_order_id,
            "exit_fill_price": self.exit_fill_price,
            "exit_filled_at": self.exit_filled_at,
            "alpaca_raw": self.alpaca_raw,
            "error_message": self.error_message,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "paper_trade": self.paper_trade,
            "dry_run": self.dry_run,
        }


class TradesRepository:
    """
    Repository for trade persistence and queries.
    
    Uses asyncpg for async PostgreSQL access to TimescaleDB.
    """
    
    def __init__(self, dsn: str):
        """
        Initialize repository with database connection string.
        
        Args:
            dsn: PostgreSQL connection string
        """
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None
    
    async def connect(self) -> None:
        """Establish database connection pool."""
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        logger.info("trades_repository_connected", dsn=self.dsn[:30] + "...")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def insert_trade(self, trade: TradeRecord) -> int | None:
        """
        Insert a trade record.
        
        First checks if record exists (by idempotency_key), then inserts if new.
        
        Returns:
            Trade ID or None if duplicate
        """
        if not self._pool:
            logger.warning("trades_repository_not_connected")
            return None
        
        async with self._pool.acquire() as conn:
            try:
                # Check for existing record (idempotency)
                existing = await conn.fetchval(
                    "SELECT id FROM trades WHERE idempotency_key = $1 LIMIT 1",
                    trade.idempotency_key,
                )
                if existing:
                    logger.debug("trade_duplicate_skipped", idempotency_key=trade.idempotency_key)
                    return None
                
                row = await conn.fetchrow(
                    """
                    INSERT INTO trades (
                        ts, symbol, side, qty, intended_notional, avg_fill_price,
                        status, alpaca_order_id, alpaca_status, order_class,
                        tp_order_id, sl_order_id, exit_fill_price, exit_filled_at,
                        alpaca_raw, error_message, correlation_id, idempotency_key,
                        paper_trade, dry_run
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20
                    )
                    RETURNING id
                    """,
                    trade.ts or datetime.now(tz=timezone.utc),
                    trade.symbol,
                    trade.side,
                    trade.qty,
                    trade.intended_notional,
                    trade.avg_fill_price,
                    trade.status,
                    trade.alpaca_order_id,
                    trade.alpaca_status,
                    trade.order_class,
                    trade.tp_order_id,
                    trade.sl_order_id,
                    trade.exit_fill_price,
                    trade.exit_filled_at,
                    trade.alpaca_raw,
                    trade.error_message,
                    trade.correlation_id,
                    trade.idempotency_key,
                    trade.paper_trade,
                    trade.dry_run,
                )
                
                if row:
                    trade_id = row["id"]
                    logger.info(
                        "trade_inserted",
                        trade_id=trade_id,
                        symbol=trade.symbol,
                        side=trade.side,
                        status=trade.status,
                    )
                    return trade_id
                else:
                    logger.debug(
                        "trade_duplicate_skipped",
                        idempotency_key=trade.idempotency_key,
                    )
                    return None
                    
            except Exception as e:
                logger.exception(
                    "trade_insert_error",
                    error=str(e),
                    symbol=trade.symbol,
                )
                return None
    
    async def get_recent_trades(
        self,
        limit: int = 20,
        symbol: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get recent trades with optional filters.
        
        Args:
            limit: Max number of trades to return
            symbol: Filter by symbol
            status: Filter by status (filled, blocked, rejected)
            
        Returns:
            List of trade dictionaries
        """
        if not self._pool:
            return []
        
        async with self._pool.acquire() as conn:
            query = """
                SELECT 
                    id, ts, symbol, side, qty, intended_notional, avg_fill_price,
                    status, alpaca_order_id, alpaca_status, error_message,
                    correlation_id, idempotency_key, paper_trade, dry_run
                FROM trades
                WHERE 1=1
            """
            params = []
            param_idx = 1
            
            if symbol:
                query += f" AND symbol = ${param_idx}"
                params.append(symbol.upper())
                param_idx += 1
            
            if status:
                query += f" AND status = ${param_idx}"
                params.append(status)
                param_idx += 1
            
            query += f" ORDER BY ts DESC LIMIT ${param_idx}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def get_trade_by_order_id(self, alpaca_order_id: str) -> dict[str, Any] | None:
        """Get a trade by Alpaca order ID."""
        if not self._pool:
            return None
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM trades WHERE alpaca_order_id = $1",
                alpaca_order_id,
            )
            return dict(row) if row else None
    
    async def get_trades_summary(self) -> dict[str, Any]:
        """Get summary statistics of trades."""
        if not self._pool:
            return {}
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(*) FILTER (WHERE status = 'filled') as filled,
                    COUNT(*) FILTER (WHERE status = 'blocked') as blocked,
                    COUNT(*) FILTER (WHERE status = 'rejected') as rejected,
                    COUNT(*) FILTER (WHERE side = 'buy') as buys,
                    COUNT(*) FILTER (WHERE side = 'sell') as sells,
                    SUM(qty) FILTER (WHERE status = 'filled') as total_qty_filled,
                    SUM(intended_notional) FILTER (WHERE status = 'filled') as total_notional
                FROM trades
                """
            )
            return dict(row) if row else {}

    # =========================================================================
    # THROTTLE QUERY METHODS
    # =========================================================================

    async def count_orders_in_window(self, minutes: int = 1) -> int:
        """
        Count orders placed in the last N minutes (for rate limiting).
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            Number of orders placed in window
        """
        if not self._pool:
            return 0
        
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM trades
                WHERE ts > NOW() - INTERVAL '1 minute' * $1
                  AND status != 'blocked'
                """,
                minutes,
            )
            return count or 0

    async def count_orders_for_symbol_today(self, symbol: str) -> int:
        """
        Count orders for a specific symbol today (for per-symbol daily limit).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Number of orders for symbol today
        """
        if not self._pool:
            return 0
        
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM trades
                WHERE symbol = $1
                  AND ts::date = CURRENT_DATE
                  AND status != 'blocked'
                """,
                symbol.upper(),
            )
            return count or 0

    async def count_open_orders(self) -> int:
        """
        Count currently open/pending orders (for max open orders check).
        
        Returns:
            Number of open orders
        """
        if not self._pool:
            return 0
        
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM trades
                WHERE alpaca_order_id IS NOT NULL
                  AND alpaca_status IN ('new', 'accepted', 'pending_new', 'partially_filled')
                """
            )
            return count or 0

    async def get_last_trade_time(self, symbol: str) -> datetime | None:
        """
        Get timestamp of most recent trade for a symbol (for cooldown check).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Datetime of last trade, or None if no trades
        """
        if not self._pool:
            return None
        
        async with self._pool.acquire() as conn:
            ts = await conn.fetchval(
                """
                SELECT MAX(ts)
                FROM trades
                WHERE symbol = $1
                  AND status != 'blocked'
                """,
                symbol.upper(),
            )
            return ts

    async def check_idempotency_exists(self, idempotency_key: str) -> bool:
        """
        Check if an idempotency key already exists (for duplicate prevention).
        
        Args:
            idempotency_key: The key to check
            
        Returns:
            True if exists, False otherwise
        """
        if not self._pool:
            return False
        
        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM trades WHERE idempotency_key = $1)",
                idempotency_key,
            )
            return exists or False

    async def get_pending_orders(self) -> list[dict[str, Any]]:
        """
        Get trades with pending Alpaca orders (not yet filled/rejected).
        
        Returns:
            List of trades that need status reconciliation
        """
        if not self._pool:
            return []
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, ts, symbol, alpaca_order_id, alpaca_status, correlation_id,
                             order_class, tp_order_id, sl_order_id, status
                FROM trades
                WHERE alpaca_order_id IS NOT NULL
                    AND paper_trade = TRUE
                    AND (
                        alpaca_status IN ('new', 'accepted', 'pending_new', 'partially_filled')
                        OR (order_class = 'bracket' AND status <> 'closed')
                    )
                ORDER BY ts DESC
                LIMIT 100
                """
            )
            return [dict(row) for row in rows]

    async def update_bracket_exit(
        self,
        exit_order_id: str,
        status: str,
        exit_fill_price: float | None = None,
        exit_filled_at: datetime | None = None,
    ) -> bool:
        """
        Update trade record when a bracket leg fills.

        Args:
            exit_order_id: Alpaca order ID for TP/SL leg
            status: New status (filled/canceled/rejected)
            exit_fill_price: Fill price for exit leg
            exit_filled_at: Fill timestamp for exit leg

        Returns:
            True if updated, False otherwise
        """
        if not self._pool:
            return False

        async with self._pool.acquire() as conn:
            updates = ["updated_at = now()"]
            params: list[Any] = [exit_order_id]
            idx = 2

            if exit_fill_price is not None:
                updates.append(f"exit_fill_price = ${idx}")
                params.append(exit_fill_price)
                idx += 1

            if exit_filled_at is not None:
                updates.append(f"exit_filled_at = ${idx}")
                params.append(exit_filled_at)
                idx += 1

            if status == "filled":
                updates.append(f"status = ${idx}")
                params.append("closed")
                idx += 1

            query = f"""
                UPDATE trades
                SET {', '.join(updates)}
                WHERE tp_order_id = $1 OR sl_order_id = $1
            """

            result = await conn.execute(query, *params)
            return result.split()[-1] != "0"

    async def update_order_status(
        self,
        alpaca_order_id: str,
        status: str,
        filled_qty: int | None = None,
        avg_fill_price: float | None = None,
        filled_at: datetime | None = None,
    ) -> bool:
        """
        Update trade record with final order status from Alpaca.
        
        Args:
            alpaca_order_id: Alpaca order ID
            status: New status (filled, canceled, rejected, etc.)
            filled_qty: Final filled quantity
            avg_fill_price: Average fill price
            filled_at: Timestamp when order was filled
            
        Returns:
            True if updated, False otherwise
        """
        if not self._pool:
            return False
        
        async with self._pool.acquire() as conn:
            # Build dynamic update - always update updated_at
            updates = ["alpaca_status = $2", "updated_at = now()"]
            params: list[Any] = [alpaca_order_id, status]
            idx = 3
            
            if filled_qty is not None:
                updates.append(f"filled_qty = ${idx}")
                params.append(filled_qty)
                idx += 1
            
            if avg_fill_price is not None:
                updates.append(f"avg_fill_price = ${idx}")
                params.append(avg_fill_price)
                idx += 1
            
            if filled_at is not None:
                updates.append(f"filled_at = ${idx}")
                params.append(filled_at)
                idx += 1
            
            # Update status field if order is terminal
            if status in ("filled", "canceled", "rejected", "expired"):
                if status == "filled":
                    updates.append(f"status = ${idx}")
                    params.append("filled")
                else:
                    updates.append(f"status = ${idx}")
                    params.append("rejected")
                idx += 1
            
            query = f"""
                UPDATE trades
                SET {', '.join(updates)}
                WHERE alpaca_order_id = $1
            """
            
            result = await conn.execute(query, *params)
            updated = result.split()[-1] != "0"
            
            if updated:
                logger.info(
                    "trade_status_updated",
                    alpaca_order_id=alpaca_order_id,
                    new_status=status,
                    filled_qty=filled_qty,
                    avg_fill_price=avg_fill_price,
                )
            
            return updated

    # =========================================================================
    # POSITION QUERY METHODS (Portfolio Truth)
    # =========================================================================

    async def get_position_qty(self, symbol: str) -> int | None:
        """
        Get position quantity from positions table.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position qty, or None if not found (caller should fallback to Alpaca)
        """
        if not self._pool:
            return None
        
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    "SELECT qty FROM positions WHERE symbol = $1",
                    symbol.upper(),
                )
                if row:
                    return int(float(row["qty"]))
                return None
            except asyncpg.exceptions.UndefinedTableError:
                logger.warning("positions_table_missing", method="get_position_qty")
                return None

    async def get_total_exposure(self) -> float:
        """
        Get total market value of all positions.
        
        Returns:
            Total market value in USD
        """
        if not self._pool:
            return 0.0
        
        async with self._pool.acquire() as conn:
            try:
                total = await conn.fetchval(
                    "SELECT COALESCE(SUM(market_value), 0) FROM positions WHERE qty > 0"
                )
                return float(total or 0)
            except asyncpg.exceptions.UndefinedTableError:
                logger.warning("positions_table_missing", method="get_total_exposure")
                return 0.0

    async def get_symbol_exposure(self, symbol: str) -> float:
        """
        Get market value for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Market value in USD for that symbol
        """
        if not self._pool:
            return 0.0
        
        async with self._pool.acquire() as conn:
            try:
                mv = await conn.fetchval(
                    "SELECT COALESCE(market_value, 0) FROM positions WHERE symbol = $1 AND qty > 0",
                    symbol.upper(),
                )
                return float(mv or 0)
            except asyncpg.exceptions.UndefinedTableError:
                logger.warning("positions_table_missing", method="get_symbol_exposure")
                return 0.0


def get_trades_repository(settings) -> TradesRepository:
    """
    Create a TradesRepository from settings.
    
    Args:
        settings: Settings object with database config
        
    Returns:
        TradesRepository instance
    """
    import os
    
    dsn = os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'awet')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5433')}/"
        f"{os.getenv('POSTGRES_DB', 'awet')}"
    )
    return TradesRepository(dsn)
