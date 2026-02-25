"""
ExitAgent - Automatic Position Closing
======================================

This agent monitors open positions and automatically closes them based on:
1. Take profit: PnL% >= take_profit_pct
2. Stop loss: PnL% <= -stop_loss_pct
3. Time exit: holding_time > max_holding_minutes

SAFETY ARCHITECTURE:
-------------------
1. Paper Trading Only: All trades marked paper_trade=True
2. Dry-Run Gate: Respects execution_dry_run config
3. Approval File Gate: Requires approval file to execute
4. Position Validation: Checks position exists before selling

FLOW:
-----
    Every exit_check_interval_seconds:
        ↓
    Read positions from DB (or fallback to Alpaca)
        ↓
    For each position:
        - Fetch current price
        - Calculate PnL%
        - Check exit conditions
        ↓
    If exit condition met → Submit SELL order
        ↓
    Persist to trades table with reason
        ↓
    Emit ExecutionEvent to audit trail
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import asyncpg
import structlog

# Add project root to path for script usage
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import load_settings
from src.core.logging import configure_logging, set_correlation_id
from src.integrations.alpaca_client import AlpacaClient, AlpacaClientError
from src.integrations.trades_repository import TradeRecord, TradesRepository, get_trades_repository

logger = structlog.get_logger(__name__)


class ExitReason:
    """Exit reason constants."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_EXIT = "time_exit"


class ExitAgent:
    """
    Monitors positions and triggers automatic exits.
    
    Exit conditions:
    - Take profit: PnL% >= take_profit_pct
    - Stop loss: PnL% <= -stop_loss_pct
    - Time exit: holding_time > max_holding_minutes
    """
    
    def __init__(self):
        self.settings = load_settings()
        configure_logging(self.settings.logging.level)
        self.logger = structlog.get_logger("exit_agent")
        
        self._alpaca_client: AlpacaClient | None = None
        self._trades_repo: TradesRepository | None = None
        self._db_pool: asyncpg.Pool | None = None
        
        # Config
        self.enabled = getattr(self.settings.app, "enable_exit_logic", True)
        self.take_profit_pct = getattr(self.settings.app, "take_profit_pct", 1.0)
        self.stop_loss_pct = getattr(self.settings.app, "stop_loss_pct", 0.5)
        self.max_holding_minutes = getattr(self.settings.app, "max_holding_minutes", 60)
        self.check_interval = getattr(self.settings.app, "exit_check_interval_seconds", 60)
        
        self.logger.info(
            "exit_agent_initialized",
            enabled=self.enabled,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
            max_holding_minutes=self.max_holding_minutes,
            check_interval=self.check_interval,
        )
    
    async def connect(self) -> None:
        """Initialize connections."""
        # Alpaca client
        try:
            self._alpaca_client = AlpacaClient.from_env()
            self.logger.info("alpaca_client_connected")
        except AlpacaClientError as e:
            self.logger.warning("alpaca_client_not_available", error=str(e))
        
        # Database
        dsn = os.getenv(
            "DATABASE_URL",
            f"postgresql://{os.getenv('POSTGRES_USER', 'awet')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'awet')}"
            f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5433')}/"
            f"{os.getenv('POSTGRES_DB', 'awet')}"
        )
        self._db_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        self._trades_repo = get_trades_repository(self.settings)
        await self._trades_repo.connect()
        self.logger.info("database_connected")
    
    async def close(self) -> None:
        """Cleanup connections."""
        if self._alpaca_client:
            await self._alpaca_client.close()
        if self._trades_repo:
            await self._trades_repo.close()
        if self._db_pool:
            await self._db_pool.close()
    
    def _check_safety_gates(self) -> tuple[bool, str | None]:
        """
        Check if exits are allowed (same gates as ExecutionAgent).
        
        Returns:
            Tuple of (allowed, error_message)
        """
        # Check dry-run mode
        if self.settings.app.execution_dry_run:
            return (False, "Blocked by execution_dry_run=True")
        
        # Check approval file
        approval_file = self.settings.app.execution_approval_file
        if not os.path.exists(approval_file):
            return (False, f"Blocked: approval file missing ({approval_file})")
        
        return (True, None)
    
    async def _get_positions(self) -> list[dict[str, Any]]:
        """
        Get open positions from DB first, fallback to Alpaca.
        
        Returns:
            List of position dicts with symbol, qty, avg_entry_price, updated_at
        """
        positions = []
        
        # Try DB first
        if self._db_pool:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT symbol, qty, avg_entry_price, market_value, updated_at
                    FROM positions
                    WHERE qty > 0
                    """
                )
                for row in rows:
                    positions.append({
                        "symbol": row["symbol"],
                        "qty": float(row["qty"]),
                        "avg_entry_price": float(row["avg_entry_price"] or 0),
                        "market_value": float(row["market_value"] or 0),
                        "updated_at": row["updated_at"],
                        "source": "db",
                    })
        
        # Fallback to Alpaca if no DB positions
        if not positions and self._alpaca_client:
            try:
                alpaca_positions = await self._alpaca_client.get_positions()
                for pos in alpaca_positions:
                    positions.append({
                        "symbol": pos.get("symbol", ""),
                        "qty": float(pos.get("qty", 0)),
                        "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                        "market_value": float(pos.get("market_value", 0)),
                        "updated_at": datetime.now(tz=timezone.utc),
                        "source": "alpaca",
                    })
            except Exception as e:
                self.logger.warning("alpaca_positions_fetch_failed", error=str(e))
        
        return positions
    
    async def _get_current_price(self, symbol: str) -> float | None:
        """Get current price from Alpaca."""
        if not self._alpaca_client:
            return None
        try:
            return await self._alpaca_client.get_current_price(symbol)
        except Exception as e:
            self.logger.warning("price_fetch_failed", symbol=symbol, error=str(e))
            return None
    
    async def _get_entry_time(self, symbol: str) -> datetime | None:
        """
        Get the earliest trade time for this symbol (approximate entry time).
        """
        if not self._db_pool:
            return None
        
        async with self._db_pool.acquire() as conn:
            # Get earliest BUY trade for this symbol that's still open
            row = await conn.fetchrow(
                """
                SELECT MIN(ts) as entry_time
                FROM trades
                WHERE symbol = $1 AND side = 'buy' AND status = 'filled'
                """,
                symbol.upper(),
            )
            return row["entry_time"] if row and row["entry_time"] else None
    
    def _calculate_pnl_pct(
        self, avg_entry_price: float, current_price: float
    ) -> float:
        """Calculate PnL percentage."""
        if avg_entry_price <= 0:
            return 0.0
        return ((current_price - avg_entry_price) / avg_entry_price) * 100
    
    def _check_exit_condition(
        self,
        pnl_pct: float,
        holding_minutes: float,
    ) -> tuple[bool, str | None]:
        """
        Check if any exit condition is met.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return (True, ExitReason.TAKE_PROFIT)
        
        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return (True, ExitReason.STOP_LOSS)
        
        # Time exit
        if holding_minutes >= self.max_holding_minutes:
            return (True, ExitReason.TIME_EXIT)
        
        return (False, None)
    
    async def _submit_exit(
        self,
        symbol: str,
        qty: int,
        reason: str,
        pnl_pct: float,
        current_price: float,
    ) -> bool:
        """
        Submit a SELL order to close position.
        
        Returns:
            True if successful, False otherwise
        """
        correlation_id = str(uuid4())
        idempotency_key = f"exit-{symbol}-{reason}-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"
        set_correlation_id(correlation_id)
        
        self.logger.info(
            "exit_triggered",
            symbol=symbol,
            qty=qty,
            reason=reason,
            pnl_pct=round(pnl_pct, 2),
            current_price=current_price,
            correlation_id=correlation_id,
        )
        
        # Check safety gates
        allowed, gate_error = self._check_safety_gates()
        
        status = "blocked"
        filled_qty = 0
        avg_price: float | None = None
        alpaca_order_id: str | None = None
        alpaca_status: str | None = None
        error_message: str | None = gate_error
        
        if allowed and self._alpaca_client:
            try:
                result = await self._alpaca_client.submit_market_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                )
                if result.success:
                    status = "filled"
                    filled_qty = result.filled_qty or qty
                    avg_price = result.avg_price or current_price
                    alpaca_order_id = result.order_id
                    alpaca_status = result.status
                    self.logger.info(
                        "exit_order_submitted",
                        symbol=symbol,
                        order_id=alpaca_order_id,
                        filled_qty=filled_qty,
                        reason=reason,
                    )
                else:
                    error_message = result.error_message
                    alpaca_status = "rejected"
            except Exception as e:
                error_message = f"Alpaca API error: {str(e)}"
                self.logger.exception("exit_order_failed", symbol=symbol, error=str(e))
        
        # Persist to trades table
        trade = TradeRecord(
            symbol=symbol,
            side="sell",
            qty=filled_qty,
            intended_notional=current_price * qty if current_price else None,
            avg_fill_price=avg_price,
            status=status,
            alpaca_order_id=alpaca_order_id,
            alpaca_status=alpaca_status,
            error_message=f"Exit: {reason}" + (f" - {error_message}" if error_message else ""),
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            paper_trade=True,
            dry_run=self.settings.app.execution_dry_run,
        )
        
        if self._trades_repo:
            await self._trades_repo.insert_trade(trade)
        
        return status == "filled"
    
    async def check_positions(self) -> dict[str, Any]:
        """
        Check all positions for exit conditions.
        
        Returns:
            Summary dict with counts
        """
        if not self.enabled:
            self.logger.debug("exit_logic_disabled")
            return {"enabled": False, "checked": 0, "exits": 0}
        
        positions = await self._get_positions()
        
        checked = 0
        exits_triggered = 0
        exits_blocked = 0
        
        for pos in positions:
            symbol = pos["symbol"]
            qty = int(pos["qty"])
            avg_entry = pos["avg_entry_price"]
            
            if qty <= 0:
                continue
            
            checked += 1
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                self.logger.warning("exit_check_skipped_no_price", symbol=symbol)
                continue
            
            # Calculate PnL%
            pnl_pct = self._calculate_pnl_pct(avg_entry, current_price)
            
            # Calculate holding time
            entry_time = await self._get_entry_time(symbol)
            if entry_time:
                holding_minutes = (datetime.now(tz=timezone.utc) - entry_time).total_seconds() / 60
            else:
                # Use position updated_at as fallback
                holding_minutes = (datetime.now(tz=timezone.utc) - pos["updated_at"]).total_seconds() / 60
            
            # Check exit conditions
            should_exit, reason = self._check_exit_condition(pnl_pct, holding_minutes)
            
            if should_exit and reason:
                self.logger.info(
                    "exit_condition_met",
                    symbol=symbol,
                    reason=reason,
                    pnl_pct=round(pnl_pct, 2),
                    holding_minutes=round(holding_minutes, 1),
                )
                
                success = await self._submit_exit(
                    symbol=symbol,
                    qty=qty,
                    reason=reason,
                    pnl_pct=pnl_pct,
                    current_price=current_price,
                )
                
                if success:
                    exits_triggered += 1
                else:
                    exits_blocked += 1
            else:
                self.logger.debug(
                    "exit_check_no_action",
                    symbol=symbol,
                    pnl_pct=round(pnl_pct, 2),
                    holding_minutes=round(holding_minutes, 1),
                )
        
        return {
            "enabled": True,
            "checked": checked,
            "exits_triggered": exits_triggered,
            "exits_blocked": exits_blocked,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    
    async def run_once(self) -> dict[str, Any]:
        """Run a single exit check cycle."""
        await self.connect()
        try:
            return await self.check_positions()
        finally:
            await self.close()
    
    async def run_loop(self) -> None:
        """Run exit checks in a loop."""
        await self.connect()
        
        self.logger.info(
            "exit_agent_loop_started",
            interval=self.check_interval,
            take_profit=self.take_profit_pct,
            stop_loss=self.stop_loss_pct,
            max_holding=self.max_holding_minutes,
        )
        
        try:
            while True:
                try:
                    result = await self.check_positions()
                    self.logger.info("exit_check_completed", **result)
                except Exception as e:
                    self.logger.exception("exit_check_error", error=str(e))
                
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            self.logger.info("exit_agent_loop_stopped")
        finally:
            await self.close()


async def main():
    parser = argparse.ArgumentParser(description="Exit Agent - Automatic Position Closing")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--watch", action="store_true", help="Run in loop mode")
    args = parser.parse_args()
    
    agent = ExitAgent()
    
    if args.once or not args.watch:
        result = await agent.run_once()
        print(f"\n📊 Exit Check Result:")
        print(f"   Enabled: {result.get('enabled', False)}")
        print(f"   Positions Checked: {result.get('checked', 0)}")
        print(f"   Exits Triggered: {result.get('exits_triggered', 0)}")
        print(f"   Exits Blocked: {result.get('exits_blocked', 0)}")
    else:
        try:
            await agent.run_loop()
        except KeyboardInterrupt:
            print("\n👋 Exit agent stopped")


if __name__ == "__main__":
    asyncio.run(main())
