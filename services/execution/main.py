"""
ExecutionAgent - Paper Trade Execution with Safety Gates
=========================================================

This agent is the final step in the AWET trading pipeline. It consumes
risk-approved signals and produces execution events (either completed or blocked).

SAFETY ARCHITECTURE:
-------------------
1. **Paper Trading Only**: This agent ALWAYS sets `paper_trade=True` on all
   ExecutionEvent objects. The Alpaca client blocks live endpoints at startup.

2. **Approval File Gate**: The agent checks for an approval file on disk
   (configured via `execution_approval_file`). If the file does NOT exist,
   all trades are blocked and routed to `execution.blocked` topic.

3. **Dry-Run Mode**: When `execution_dry_run=True` (default), ALL trades are
   blocked regardless of the approval file. This is the safest default.
   
   To enable paper trading:
   - Set `execution_dry_run: false` in config/app.yaml
   - Run `make approve` to create the approval file

4. **Alpaca Paper Endpoint Validation**: The AlpacaClient refuses to start
   if ALPACA_BASE_URL is not https://paper-api.alpaca.markets.

TRADING LOGIC:
--------------
1. **Side Decision**: Based on RiskEvent.direction:
   - "long" → BUY
   - "short" → SELL
   - anything else → BLOCKED

2. **Position Sizing**: qty = min(max_notional / price, max_qty)
   - Fetches current price from Alpaca
   - Enforces min_qty and max_qty bounds

3. **Sell Validation**: Before SELL, checks current position
   - If no position exists, trade is BLOCKED

FLOW:
-----
    RiskEvent (from risk.approved topic)
        ↓
    Check dry_run flag → if True → BLOCKED (no API call)
        ↓
    Check approval file → if missing → BLOCKED (no API call)
        ↓
    Determine side from direction → if neutral → BLOCKED
        ↓
    For SELL: check position → if no position → BLOCKED
        ↓
    Fetch current price → calculate qty
        ↓
    Submit order to Alpaca Paper API
        ↓
    Create ExecutionEvent + persist to trades table
        ↓
    Publish to execution.completed or execution.blocked topic
        ↓
    Write to audit trail
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent
from src.audit.trail_logger import AuditTrailLogger
from src.core.config import load_settings
from src.core.logging import set_correlation_id
from src.integrations.alpaca_client import (
    AlpacaClient,
    AlpacaClientError,
    AlpacaLiveEndpointError,
)
from src.integrations.trades_repository import (
    TradeRecord,
    TradesRepository,
    get_trades_repository,
)
from src.models.events_execution import ExecutionEvent
from src.models.events_risk import RiskEvent
from src.monitoring.metrics import EVENTS_PROCESSED, EVENT_LATENCY
from src.streaming.kafka_consumer import AvroConsumer
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import EXECUTION_BLOCKED, EXECUTION_COMPLETED, RISK_APPROVED

RISK_SCHEMA = "src/schemas/risk.avsc"
EXEC_SCHEMA = "src/schemas/execution.avsc"


class ExecutionAgent(BaseAgent):
    """
    Executes paper trades based on risk-approved signals.
    
    This agent implements multiple safety gates to ensure no real-money
    trades can ever be submitted:
    
    1. paper_trade=True is hardcoded on all ExecutionEvents
    2. execution_dry_run config flag blocks all trades when True (default)
    3. Approval file must exist on disk to allow "filled" status
    4. AlpacaClient blocks any non-paper endpoint at initialization
    
    Trading Logic:
    - Side: BUY for "long" direction, SELL for "short"
    - Qty: Calculated from max_trade_notional_usd / current_price
    - Sell validation: Checks position exists before selling
    """
    
    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("execution", settings.app.http.execution_port)
        
        with open(RISK_SCHEMA, "r", encoding="utf-8") as handle:
            risk_schema = handle.read()
        
        self.consumer = AvroConsumer(
            self.settings.kafka,
            self.settings.kafka.group_ids.execution,
            risk_schema,
            RISK_APPROVED,
        )
        self.producer = AvroProducer(self.settings.kafka)
        self.audit = AuditTrailLogger(self.settings)
        
        with open(EXEC_SCHEMA, "r", encoding="utf-8") as handle:
            self._exec_schema = handle.read()
        
        self._alpaca_client: AlpacaClient | None = None
        self._init_alpaca_client()
        
        self._trades_repo: TradesRepository = get_trades_repository(self.settings)
        
        self.logger.info(
            "execution_agent_initialized",
            dry_run=self.settings.app.execution_dry_run,
            default_qty=self.settings.app.execution_default_qty,
            max_notional=self.settings.app.max_trade_notional_usd,
            max_qty=self.settings.app.max_qty_per_trade,
            approval_file=self.settings.app.execution_approval_file,
            paper_trade=True,
            alpaca_enabled=self._alpaca_client is not None,
        )
    
    def _init_alpaca_client(self) -> None:
        try:
            self._alpaca_client = AlpacaClient.from_env()
            self.logger.info(
                "alpaca_client_ready",
                base_url=self._alpaca_client.base_url,
                paper_trading=True,
            )
        except AlpacaLiveEndpointError as e:
            self.logger.error("alpaca_live_endpoint_blocked", error=str(e))
            raise
        except AlpacaClientError as e:
            self.logger.warning(
                "alpaca_client_not_configured",
                error=str(e),
                message="Orders will be simulated locally",
            )
            self._alpaca_client = None

    async def start(self) -> None:
        self.app.add_event_handler("shutdown", self._shutdown)
        await self.audit.connect()
        await self._trades_repo.connect()
        self.track_task(asyncio.create_task(self._consume_loop()))

    async def _shutdown(self) -> None:
        self.consumer.close()
        self.producer.close()
        await self.audit.close()
        await self._trades_repo.close()
        if self._alpaca_client:
            await self._alpaca_client.close()

    async def _consume_loop(self) -> None:
        while not self.is_shutting_down:
          try:
            try:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    await asyncio.sleep(0.1)
                    continue
                await self._process_message(msg)
            except Exception as e:
                self.logger.exception(
                    "execution_processing_error",
                    message="Exception in consume loop - continuing",
                    error=str(e),
                )
                await asyncio.sleep(1.0)
          except asyncio.CancelledError:
              self.logger.info("consume_loop_cancelled")
              break

    def _determine_side(self, direction: str) -> str | None:
        direction_lower = direction.lower() if direction else ""
        if direction_lower in ("long", "bullish", "buy"):
            return "buy"
        elif direction_lower in ("short", "bearish", "sell"):
            return "sell"
        return None

    async def _calculate_qty(self, symbol: str, current_price: float | None) -> int:
        max_notional = self.settings.app.max_trade_notional_usd
        max_qty = self.settings.app.max_qty_per_trade
        min_qty = self.settings.app.min_qty_per_trade
        default_qty = self.settings.app.execution_default_qty
        
        if current_price is None or current_price <= 0:
            self.logger.warning(
                "price_unavailable_using_default",
                symbol=symbol,
                default_qty=default_qty,
            )
            return default_qty
        
        qty = int(max_notional / current_price)
        qty = min(qty, max_qty)
        qty = max(qty, min_qty)
        return qty

    async def _check_can_sell(self, symbol: str) -> tuple[bool, int, str | None]:
        """
        Check if we can sell a symbol by verifying position exists.
        
        Priority:
        1. Check positions table (DB truth)
        2. If not in DB, fallback to Alpaca API once
        
        Returns:
            Tuple of (can_sell, position_qty, error_message)
        """
        if not self.settings.app.check_position_before_sell:
            return (True, 0, None)
        
        # 1. Try DB first (positions table)
        db_qty = await self._trades_repo.get_position_qty(symbol)
        if db_qty is not None:
            if db_qty <= 0:
                return (False, db_qty, f"Cannot SELL {symbol}: no position in DB (qty={db_qty})")
            self.logger.debug("sell_check_db_hit", symbol=symbol, qty=db_qty)
            return (True, db_qty, None)
        
        # 2. Fallback to Alpaca API if DB doesn't have the symbol
        if not self._alpaca_client:
            # No DB and no Alpaca client - allow but warn
            self.logger.warning("sell_check_no_source", symbol=symbol)
            return (True, 0, None)
        
        try:
            self.logger.debug("sell_check_alpaca_fallback", symbol=symbol)
            position_qty = await self._alpaca_client.get_position_qty(symbol)
            if position_qty <= 0:
                return (False, position_qty, f"Cannot SELL {symbol}: no position (qty={position_qty})")
            return (True, position_qty, None)
        except Exception as e:
            return (False, 0, f"Position check failed: {str(e)}")

    async def _check_can_buy(self, symbol: str) -> tuple[bool, str | None]:
        """
        Check if we can open/add to a position based on allow_add_to_position.

        Returns:
            Tuple of (can_buy, error_message)
        """
        allow_add = getattr(self.settings.app, "allow_add_to_position", False)
        if allow_add:
            return (True, None)

        db_qty = await self._trades_repo.get_position_qty(symbol)
        if db_qty is not None:
            if db_qty > 0:
                return (False, f"Cannot BUY {symbol}: already holding qty={db_qty}")
            return (True, None)

        if not self._alpaca_client:
            return (True, None)

        try:
            alpaca_qty = await self._alpaca_client.get_position_qty(symbol)
            if alpaca_qty > 0:
                return (False, f"Cannot BUY {symbol}: already holding qty={alpaca_qty}")
            return (True, None)
        except Exception as e:
            return (False, f"Position check failed: {str(e)}")

    def _extract_bracket_legs(self, raw: dict | None) -> tuple[str | None, str | None]:
        """Extract TP/SL order IDs from Alpaca bracket response."""
        if not raw:
            return (None, None)
        legs = raw.get("legs") or []
        tp_id = None
        sl_id = None
        for leg in legs:
            leg_type = (leg.get("type") or "").lower()
            if leg_type == "limit" and tp_id is None:
                tp_id = leg.get("id")
            elif leg_type == "stop" and sl_id is None:
                sl_id = leg.get("id")
        return (tp_id, sl_id)

    async def _check_exposure_caps(
        self, symbol: str, side: str, intended_notional: float
    ) -> tuple[bool, str | None]:
        """
        Check if order would exceed exposure caps.
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            intended_notional: Estimated USD value of order
            
        Returns:
            Tuple of (passed, reason) where passed=True means within limits
        """
        # Only check BUY orders (sells reduce exposure)
        if side != "buy":
            return (True, None)
        
        max_total = getattr(self.settings.app, "max_total_exposure_usd", 50000)
        max_per_symbol = getattr(self.settings.app, "max_exposure_per_symbol_usd", 10000)
        
        # Check total portfolio exposure
        current_total = await self._trades_repo.get_total_exposure()
        if current_total + intended_notional > max_total:
            return (
                False,
                f"exposure_cap: total ${current_total + intended_notional:.0f} "
                f"would exceed ${max_total:.0f} cap",
            )
        
        # Check per-symbol exposure
        current_symbol = await self._trades_repo.get_symbol_exposure(symbol)
        if current_symbol + intended_notional > max_per_symbol:
            return (
                False,
                f"symbol_exposure_cap: {symbol} ${current_symbol + intended_notional:.0f} "
                f"would exceed ${max_per_symbol:.0f} cap",
            )
        
        return (True, None)

    async def _check_throttles(self, symbol: str, idempotency_key: str) -> tuple[bool, str | None]:
        """
        Check execution throttles before placing an order.
        
        Returns:
            Tuple of (passed, reason) where passed=True means throttle checks passed,
            or passed=False with reason explaining why blocked.
        """
        # 1. Check duplicate idempotency key
        if await self._trades_repo.check_idempotency_exists(idempotency_key):
            return (False, "duplicate_idempotency_key")
        
        # 2. Check rate limit (orders per minute)
        max_per_minute = getattr(self.settings.app, "max_orders_per_minute", 5)
        orders_in_minute = await self._trades_repo.count_orders_in_window(minutes=1)
        if orders_in_minute >= max_per_minute:
            return (False, f"rate_limit: {orders_in_minute}/{max_per_minute} orders in last minute")
        
        # 3. Check per-symbol daily limit
        max_per_symbol_day = getattr(self.settings.app, "max_orders_per_symbol_per_day", 3)
        symbol_orders_today = await self._trades_repo.count_orders_for_symbol_today(symbol)
        if symbol_orders_today >= max_per_symbol_day:
            return (False, f"max_symbol_orders: {symbol_orders_today}/{max_per_symbol_day} for {symbol} today")
        
        # 4. Check max open orders
        max_open = getattr(self.settings.app, "max_open_orders_total", 20)
        open_orders = await self._trades_repo.count_open_orders()
        if open_orders >= max_open:
            return (False, f"max_open_orders: {open_orders}/{max_open} open orders")
        
        # 5. Check cooldown per symbol
        cooldown_seconds = getattr(self.settings.app, "cooldown_seconds_per_symbol", 900)
        last_trade_time = await self._trades_repo.get_last_trade_time(symbol)
        if last_trade_time:
            elapsed = (datetime.now(tz=timezone.utc) - last_trade_time).total_seconds()
            if elapsed < cooldown_seconds:
                remaining = int(cooldown_seconds - elapsed)
                return (False, f"cooldown: {symbol} traded {int(elapsed)}s ago, wait {remaining}s more")
        
        return (True, None)

    async def _process_message(self, msg) -> None:
        payload = msg.value()
        event = RiskEvent.model_validate(payload)
        set_correlation_id(str(event.correlation_id))
        
        if await self.audit.is_duplicate(EXECUTION_COMPLETED, event.idempotency_key):
            self.consumer.commit()
            return
        if await self.audit.is_duplicate(EXECUTION_BLOCKED, event.idempotency_key):
            self.consumer.commit()
            return
        
        start_ts = datetime.now(tz=timezone.utc)
        dry_run = self.settings.app.execution_dry_run
        approval_file = self.settings.app.execution_approval_file
        approval_file_exists = os.path.exists(approval_file)
        should_trade = (not dry_run) and approval_file_exists
        
        status = "blocked"
        filled_qty = 0
        intended_notional: float | None = None
        current_price: float | None = None
        avg_price: float | None = None
        alpaca_order_id: str | None = None
        alpaca_status: str | None = None
        order_class: str | None = None
        tp_order_id: str | None = None
        sl_order_id: str | None = None
        alpaca_raw: dict | None = None
        error_message: str | None = None
        
        side = self._determine_side(event.direction)
        
        # First check throttles if we might trade (this avoids unnecessary Alpaca calls)
        throttle_blocked = False
        if should_trade and side is not None:
            throttle_passed, throttle_reason = await self._check_throttles(
                event.symbol, event.idempotency_key
            )
            if not throttle_passed:
                throttle_blocked = True
                error_message = f"Throttle blocked: {throttle_reason}"
                self.logger.warning(
                    "execution_throttled",
                    symbol=event.symbol,
                    reason=throttle_reason,
                    correlation_id=str(event.correlation_id),
                )
        
        if not should_trade:
            if dry_run:
                error_message = "Blocked by execution_dry_run=True"
            elif not approval_file_exists:
                error_message = "Blocked: approval file missing (run 'make approve')"
            side = side or "buy"
        elif side is None:
            error_message = f"Blocked: neutral direction (direction={event.direction})"
            side = "hold"
        elif throttle_blocked:
            # Already blocked by throttle, skip Alpaca call
            pass
        else:
            if self._alpaca_client:
                current_price = await self._alpaca_client.get_current_price(event.symbol)
            
            qty = await self._calculate_qty(event.symbol, current_price)
            intended_notional = (current_price or 0) * qty
            
            # Check exposure caps (only for BUY orders)
            if side == "buy" and intended_notional:
                exposure_passed, exposure_reason = await self._check_exposure_caps(
                    event.symbol, side, intended_notional
                )
                if not exposure_passed:
                    error_message = f"Exposure blocked: {exposure_reason}"
                    status = "blocked"
                    self.logger.warning(
                        "execution_exposure_blocked",
                        symbol=event.symbol,
                        reason=exposure_reason,
                        intended_notional=intended_notional,
                        correlation_id=str(event.correlation_id),
                    )
            
            if side == "sell" and error_message is None:
                can_sell, position_qty, sell_error = await self._check_can_sell(event.symbol)
                if not can_sell:
                    error_message = sell_error
                    status = "blocked"
                else:
                    qty = min(qty, position_qty) if position_qty > 0 else qty

            if side == "buy" and error_message is None:
                can_buy, buy_error = await self._check_can_buy(event.symbol)
                if not can_buy:
                    error_message = buy_error
                    status = "blocked"
            
            if error_message is None:
                if self._alpaca_client:
                    try:
                        use_bracket = bool(getattr(self.settings.app, "use_bracket_orders", False))
                        take_profit_pct = float(getattr(self.settings.app, "take_profit_pct", 0))
                        stop_loss_pct = float(getattr(self.settings.app, "stop_loss_pct", 0))

                        if side == "buy" and use_bracket:
                            if current_price and current_price > 0:
                                tp_price = current_price * (1 + take_profit_pct)
                                sl_price = current_price * (1 - stop_loss_pct)
                                result = await self._alpaca_client.submit_bracket_order(
                                    symbol=event.symbol,
                                    qty=qty,
                                    side=side,
                                    take_profit_price=tp_price,
                                    stop_loss_price=sl_price,
                                )
                                order_class = "bracket"
                            else:
                                self.logger.warning(
                                    "bracket_fallback_no_price",
                                    symbol=event.symbol,
                                )
                                result = await self._alpaca_client.submit_market_order(
                                    symbol=event.symbol,
                                    qty=qty,
                                    side=side,
                                )
                        else:
                            result = await self._alpaca_client.submit_market_order(
                                symbol=event.symbol,
                                qty=qty,
                                side=side,
                            )
                        if result.success:
                            status = "filled"
                            filled_qty = result.filled_qty or qty
                            avg_price = result.avg_price or current_price
                            alpaca_order_id = result.order_id
                            alpaca_status = result.status
                            alpaca_raw = result.raw_response
                            if order_class == "bracket":
                                tp_order_id, sl_order_id = self._extract_bracket_legs(alpaca_raw)
                        else:
                            status = "blocked"
                            error_message = result.error_message
                            alpaca_status = "rejected"
                    except Exception as e:
                        status = "blocked"
                        error_message = f"Alpaca API error: {str(e)}"
                        self.logger.exception("alpaca_order_exception", symbol=event.symbol, error=str(e))
                else:
                    status = "filled"
                    filled_qty = qty
                    avg_price = current_price or event.risk_score
                    alpaca_status = "simulated"
                    self.logger.info("execution_simulated", symbol=event.symbol, side=side, qty=qty)
        
        trade = TradeRecord(
            symbol=event.symbol,
            side=side or "unknown",
            qty=filled_qty,
            intended_notional=intended_notional,
            avg_fill_price=avg_price,
            status=status,
            alpaca_order_id=alpaca_order_id,
            alpaca_status=alpaca_status,
            order_class=order_class,
            tp_order_id=tp_order_id,
            sl_order_id=sl_order_id,
            alpaca_raw=alpaca_raw,
            error_message=error_message,
            correlation_id=str(event.correlation_id),
            idempotency_key=event.idempotency_key,
            paper_trade=True,
            dry_run=dry_run,
            ts=start_ts,
        )
        await self._trades_repo.insert_trade(trade)
        
        execution = ExecutionEvent(
            idempotency_key=event.idempotency_key,
            symbol=event.symbol,
            source=self.name,
            correlation_id=event.correlation_id,
            status=status,
            filled_qty=filled_qty,
            avg_price=avg_price or event.risk_score,
            paper_trade=True,
            dry_run=dry_run,
            side=side or "unknown",
            alpaca_order_id=alpaca_order_id,
            alpaca_status=alpaca_status,
            error_message=error_message,
        )
        
        payload_out = execution.to_avro_dict()
        output_topic = EXECUTION_COMPLETED if status == "filled" else EXECUTION_BLOCKED
        
        self.producer.produce(output_topic, self._exec_schema, payload_out, key=event.symbol)
        await self.audit.write_event(output_topic, payload_out)
        self.consumer.commit()
        
        duration = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
        EVENTS_PROCESSED.labels(agent=self.name, event_type=output_topic).inc()
        EVENT_LATENCY.labels(agent=self.name, event_type=output_topic).observe(duration)
        
        self.logger.info(
            "execution_processed",
            symbol=event.symbol,
            status=status,
            topic=output_topic,
            direction=event.direction,
            side=side,
            qty=filled_qty,
            intended_notional=intended_notional,
            current_price=current_price,
            avg_fill_price=avg_price,
            approved=should_trade,
            dry_run=dry_run,
            paper_trade=True,
            approval_file_exists=approval_file_exists,
            alpaca_order_id=alpaca_order_id,
            alpaca_status=alpaca_status,
            error_message=error_message,
            correlation_id=str(event.correlation_id),
        )


def main() -> None:
    ExecutionAgent().run()


if __name__ == "__main__":
    main()
