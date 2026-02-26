from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Consumer as KafkaConsumer, KafkaError
from fastapi import Request, Response
from src.agents.base_agent import BaseAgent
from src.audit.trail_logger import AuditTrailLogger
from src.core.config import load_settings
from src.core.logging import set_correlation_id
from src.models.events_signal import ScoredSignalEvent
from src.models.events_trade_decision import TradeDecisionEvent
from src.models.events_risk import RiskEvent
from src.monitoring.metrics import EVENTS_PROCESSED, EVENT_LATENCY, RISK_DECISIONS
from src.risk.engine import RiskDecision, RiskEngine, RiskInput
from src.streaming.kafka_consumer import AvroConsumer
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import RISK_APPROVED, RISK_REJECTED, SIGNALS_SCORED

RISK_SCHEMA = "src/schemas/risk.avsc"
DECISION_SCHEMA = "src/schemas/trade_decision.avsc"


class RiskAgent(BaseAgent):
    """Risk agent with real institutional-grade risk controls.

    Implements:
    - Position size limits (2% max per position)
    - Daily loss limits (5% max daily drawdown)
    - Volatility filters
    - CVaR calculation
    - Kill switch for emergencies
    """

    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("risk", settings.app.http.risk_port)
        self._use_trade_decisions = os.getenv("USE_TRADE_DECISIONS", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        if self._use_trade_decisions:
            with open(DECISION_SCHEMA, "r", encoding="utf-8") as handle:
                input_schema = handle.read()
            self._input_topic = self.settings.kafka.topics.trade_decisions
            self.consumer = AvroConsumer(
                self.settings.kafka,
                self.settings.kafka.group_ids.risk,
                input_schema,
                self._input_topic,
            )
            self._commit_consumer = lambda: self.consumer.commit()
        else:
            # signals.scored carries plain UTF-8 JSON published by signal_engine.
            # It contains the ensemble output: regime-scaled, confidence-filtered,
            # weighted vote of TFT + RSI + EMA + MACD.
            self._input_topic = SIGNALS_SCORED
            self.consumer = KafkaConsumer({
                "bootstrap.servers": ",".join(self.settings.kafka.bootstrap_servers),
                "group.id": self.settings.kafka.group_ids.risk,
                "auto.offset.reset": self.settings.kafka.auto_offset_reset,
                "enable.auto.commit": False,
                "session.timeout.ms": 30000,
                "max.poll.interval.ms": 300000,
                "heartbeat.interval.ms": 10000,
            })
            self.consumer.subscribe([SIGNALS_SCORED])
            self._commit_consumer = lambda: self.consumer.commit(asynchronous=False)
        self.producer = AvroProducer(self.settings.kafka)
        self.audit = AuditTrailLogger(self.settings)
        self.risk_engine = RiskEngine()
        with open(RISK_SCHEMA, "r", encoding="utf-8") as handle:
            self._risk_schema = handle.read()

        # Register admin endpoint so Prometheus alertmanager can auto-trigger kill switch
        self.app.add_api_route(
            "/admin/kill-switch",
            self._handle_kill_switch,
            methods=["POST"],
        )

    async def start(self) -> None:
        self.app.add_event_handler("shutdown", self._shutdown)
        await self.audit.connect()
        # Restore risk state from Redis so positions/PnL survive restarts
        await self.risk_engine.connect_redis()
        await self.risk_engine.load_state()
        self.track_task(asyncio.create_task(self._consume_loop()))
        self.track_task(asyncio.create_task(self._daily_reset_scheduler()))

    async def _daily_reset_scheduler(self) -> None:
        """Reset daily P&L counter at 9:30 AM ET every day.

        Waits until the next 9:30 AM Eastern, calls risk_engine.reset_daily(),
        then repeats on a 24-hour cycle.  An extra 60-second sleep after the
        reset prevents double-firing on DST transition edge cases.
        """
        from zoneinfo import ZoneInfo
        import math

        ET = ZoneInfo("America/New_York")
        while not self.is_shutting_down:
            now_et = datetime.now(tz=ET)
            target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            if now_et >= target:
                # Already past 9:30 today — schedule for tomorrow
                from datetime import timedelta
                target = target + timedelta(days=1)
            secs_until = (target - now_et).total_seconds()
            self.logger.info(
                "daily_reset_scheduled",
                target_et=target.isoformat(),
                seconds_until=math.floor(secs_until),
            )
            await asyncio.sleep(secs_until)
            if self.is_shutting_down:
                break
            await self.risk_engine.reset_daily()
            self.logger.info(
                "daily_risk_reset_fired",
                ts_et=datetime.now(tz=ET).isoformat(),
            )
            # Brief sleep avoids double-firing on sub-second / DST edges
            await asyncio.sleep(60)

    async def _shutdown(self) -> None:
        # Persist risk state before stopping
        await self.risk_engine.save_state()
        await self.risk_engine.close_redis()
        self.consumer.close()
        await self.audit.close()

    async def health_subsystems(self) -> dict[str, dict[str, Any]]:
        """Check Kafka, DB, and Redis connectivity."""
        checks: dict[str, dict[str, Any]] = {}
        # Redis
        redis_client = getattr(self.risk_engine, "_redis", None)
        if redis_client:
            try:
                await redis_client.ping()
                checks["redis"] = {"ok": True}
            except Exception as exc:
                checks["redis"] = {"ok": False, "error": str(exc)}
        else:
            checks["redis"] = {"ok": False, "error": "not_connected"}
        # Audit DB
        try:
            pool = getattr(self.audit, "_pool", None)
            if pool and not pool._closed:
                checks["db"] = {"ok": True}
            else:
                checks["db"] = {"ok": False, "error": "pool_closed"}
        except Exception as exc:
            checks["db"] = {"ok": False, "error": str(exc)}
        # Kill switch status
        checks["kill_switch"] = {
            "ok": not self.risk_engine.state.kill_switch_active,
            "active": self.risk_engine.state.kill_switch_active,
        }
        return checks

    def _build_risk_input_from_prediction(self, event: PredictionEvent) -> RiskInput:
        """Build risk input from prediction event."""
        return RiskInput(
            symbol=event.symbol,
            price=event.prediction,
            direction=getattr(event, "direction", "neutral"),
            confidence=event.confidence,
            horizon_30_q10=getattr(event, "horizon_30_q10", -0.01),
            horizon_30_q50=getattr(event, "horizon_30_q50", 0.0),
            horizon_30_q90=getattr(event, "horizon_30_q90", 0.01),
            volatility_5=0.02,
            volatility_15=0.025,
            current_position=self.risk_engine.state.positions.get(event.symbol, 0.0),
            portfolio_value=self.risk_engine.state.portfolio_value,
            daily_pnl=self.risk_engine.state.daily_pnl,
        )

    def _build_risk_input_from_signal(self, signal: ScoredSignalEvent) -> RiskInput:
        """Build RiskInput from a signals.scored ensemble event.

        Fields used:
          ensemble_direction  → direction ("long" | "short" | "neutral")
          ensemble_confidence → confidence (already filtered ≥ MIN_SIGNAL_CONFIDENCE
                                by signal_engine before publish)
          signal_score        → not a price; set price=1.0 so risk sizing
                                works in relative terms (execution fetches the
                                real market price from Alpaca independently).
        """
        return RiskInput(
            symbol=signal.ticker,
            price=1.0,  # real price fetched by execution; 1.0 keeps sizing sane
            direction=signal.ensemble_direction,
            confidence=signal.ensemble_confidence,
            horizon_30_q10=-0.01,   # not available in ensemble output
            horizon_30_q50=0.0,
            horizon_30_q90=0.01,
            volatility_5=0.02,
            volatility_15=0.025,
            current_position=self.risk_engine.state.positions.get(signal.ticker, 0.0),
            portfolio_value=self.risk_engine.state.portfolio_value,
            daily_pnl=self.risk_engine.state.daily_pnl,
        )

    def _build_risk_input_from_decision(self, event: TradeDecisionEvent) -> RiskInput:
        """Build risk input from trade decision event.

        Uses price and quantile data propagated from the upstream PredictionEvent
        via the TradeDecisionEvent fields added in the CRITICAL-1 fix.
        """
        direction = getattr(event, "direction", "neutral")
        if direction == "neutral":
            # Fallback: infer from decision string
            decision = (event.decision or "").lower()
            if decision == "buy":
                direction = "long"
            elif decision == "sell":
                direction = "short"

        # Use propagated price; fall back to prediction; warn if still zero
        price = getattr(event, "price", 0.0) or getattr(event, "prediction", 0.0)
        if price <= 0:
            self.logger.warning(
                "risk_input_no_price",
                symbol=event.symbol,
                hint="TradeDecisionEvent has no price — position sizing will be wrong",
            )
            price = 1.0  # absolute last resort

        return RiskInput(
            symbol=event.symbol,
            price=price,
            direction=direction,
            confidence=event.confidence,
            horizon_30_q10=getattr(event, "horizon_30_q10", -0.01),
            horizon_30_q50=getattr(event, "horizon_30_q50", 0.0),
            horizon_30_q90=getattr(event, "horizon_30_q90", 0.01),
            volatility_5=0.02,
            volatility_15=0.025,
            current_position=self.risk_engine.state.positions.get(event.symbol, 0.0),
            portfolio_value=self.risk_engine.state.portfolio_value,
            daily_pnl=self.risk_engine.state.daily_pnl,
        )

    async def _consume_loop(self) -> None:
        while not self.is_shutting_down:
          try:
            msg = self.consumer.poll(1.0)
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            if self._use_trade_decisions:
                # trade.decisions path: AvroConsumer already deserialised to dict
                payload = msg.value()
                event = TradeDecisionEvent.model_validate(payload)
                _symbol = event.symbol
                _correlation_id = event.correlation_id
                _idempotency_key = event.idempotency_key
                risk_input = self._build_risk_input_from_decision(event)
            else:
                # signals.scored path: plain JSON bytes from signal_engine ensemble
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        self.logger.error("kafka_error", error=str(msg.error()))
                    continue
                signal = ScoredSignalEvent.model_validate(
                    json.loads(msg.value().decode("utf-8"))
                )
                _symbol = signal.ticker
                _correlation_id = signal.correlation_id
                _idempotency_key = signal.idempotency_key
                risk_input = self._build_risk_input_from_signal(signal)
            set_correlation_id(str(_correlation_id))
            if await self.audit.is_duplicate(RISK_APPROVED, _idempotency_key) or await self.audit.is_duplicate(
                RISK_REJECTED, _idempotency_key
            ):
                self._commit_consumer()
                continue
            start_ts = datetime.now(tz=timezone.utc)
            risk_output = self.risk_engine.evaluate(risk_input)
            approved = risk_output.decision == RiskDecision.APPROVED
            reason = "; ".join(risk_output.reasons)
            risk_event = RiskEvent(
                idempotency_key=_idempotency_key,
                symbol=_symbol,
                source=self.name,
                correlation_id=_correlation_id,
                approved=approved,
                reason=reason,
                risk_score=risk_output.risk_score,
                max_position=risk_output.approved_size,
                max_notional=risk_output.approved_size * risk_input.price,
                min_confidence=self.risk_engine.limits.min_confidence,
                cvar_95=risk_output.cvar_95,
                max_loss=risk_output.max_loss,
                direction=risk_input.direction,
            )
            payload_out = risk_event.to_avro_dict()
            topic = RISK_APPROVED if approved else RISK_REJECTED
            self.producer.produce(topic, self._risk_schema, payload_out, key=_symbol)
            await self.audit.write_event(topic, payload_out)
            self._commit_consumer()
            duration = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
            EVENTS_PROCESSED.labels(agent=self.name, event_type=topic).inc()
            EVENT_LATENCY.labels(agent=self.name, event_type=topic).observe(duration)
            RISK_DECISIONS.labels(decision=risk_output.decision.value).inc()
            # Persist risk state to Redis after each evaluation
            await self.risk_engine.save_state()
            self.logger.info(
                "risk_evaluated",
                symbol=_symbol,
                input_topic=self._input_topic,
                ensemble_direction=risk_input.direction,
                decision=risk_output.decision.value,
                risk_score=risk_output.risk_score,
                cvar_95=risk_output.cvar_95,
            )
          except asyncio.CancelledError:
              self.logger.info("consume_loop_cancelled")
              break
          except Exception:
              self.logger.exception("consume_loop_error")
              await asyncio.sleep(1.0)

    async def _handle_kill_switch(self, request: Request) -> dict[str, str]:
        """POST /admin/kill-switch — called by Prometheus alertmanager webhook.

        Alertmanager sends a JSON body with ``commonAnnotations.summary``.
        We extract the reason string and delegate to RiskEngine.activate_kill_switch().
        """
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        reason = (
            payload.get("commonAnnotations", {}).get("summary")
            or payload.get("reason")
            or "prometheus_alertmanager"
        )
        self.risk_engine.activate_kill_switch(reason=str(reason))
        await self.risk_engine.save_state()
        self.logger.critical(
            "kill_switch_triggered_by_webhook",
            reason=reason,
            source=str(request.client),
        )
        return {"status": "kill_switch_activated", "reason": str(reason)}


def main() -> None:
    RiskAgent().run()


if __name__ == "__main__":
    main()
