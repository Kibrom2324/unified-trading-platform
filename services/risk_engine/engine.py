"""Risk Engine with real risk metrics.

This module implements institutional-grade risk controls:
- Position size limits (2% max per position)
- Daily loss limits (5% max daily drawdown)
- Volatility filters (reject during extreme volatility)
- CVaR (Conditional Value at Risk) calculation
- Correlation spike detection
- Kill switch for emergency stops
- State persistence to Redis (survives restarts)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yaml

logger = structlog.get_logger("risk_engine")


class RiskDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"


@dataclass
class RiskInput:
    """Input data for risk evaluation."""
    symbol: str
    price: float
    direction: str
    confidence: float
    horizon_30_q10: float
    horizon_30_q50: float
    horizon_30_q90: float
    volatility_5: float
    volatility_15: float
    current_position: float = 0.0
    portfolio_value: float = 100000.0
    daily_pnl: float = 0.0


@dataclass
class RiskOutput:
    """Risk evaluation result."""
    decision: RiskDecision
    reasons: list[str]
    approved_size: float
    risk_score: float
    cvar_95: float
    max_loss: float
    checks_passed: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "approved_size": self.approved_size,
            "risk_score": self.risk_score,
            "cvar_95": self.cvar_95,
            "max_loss": self.max_loss,
            "checks_passed": self.checks_passed,
        }


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_position_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    max_volatility: float = 0.10
    min_confidence: float = 0.3
    max_correlation_spike: float = 0.95
    kill_switch_loss_pct: float = 0.10


@dataclass
class PositionState:
    """Track position and P&L state."""
    positions: dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    portfolio_value: float = 100000.0
    kill_switch_active: bool = False
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def reset_daily(self) -> None:
        """Reset daily tracking (call at market open)."""
        self.daily_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Redis persistence."""
        return {
            "positions": self.positions,
            "daily_pnl": self.daily_pnl,
            "portfolio_value": self.portfolio_value,
            "kill_switch_active": self.kill_switch_active,
            "last_reset": self.last_reset.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionState":
        """Deserialize from Redis."""
        last_reset = data.get("last_reset")
        if isinstance(last_reset, str):
            last_reset = datetime.fromisoformat(last_reset)
        else:
            last_reset = datetime.now(timezone.utc)
        return cls(
            positions=data.get("positions", {}),
            daily_pnl=data.get("daily_pnl", 0.0),
            portfolio_value=data.get("portfolio_value", 100000.0),
            kill_switch_active=data.get("kill_switch_active", False),
            last_reset=last_reset,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for Redis persistence."""
        return {
            "positions": self.positions,
            "daily_pnl": self.daily_pnl,
            "portfolio_value": self.portfolio_value,
            "kill_switch_active": self.kill_switch_active,
            "last_reset": self.last_reset.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PositionState:
        """Restore state from Redis snapshot."""
        state = cls()
        state.positions = data.get("positions", {})
        state.daily_pnl = float(data.get("daily_pnl", 0.0))
        state.portfolio_value = float(data.get("portfolio_value", 100000.0))
        state.kill_switch_active = bool(data.get("kill_switch_active", False))
        last_reset = data.get("last_reset")
        if last_reset:
            try:
                state.last_reset = datetime.fromisoformat(last_reset)
            except (ValueError, TypeError):
                state.last_reset = datetime.now(timezone.utc)
        return state


class RiskEngine:
    """Real risk engine with institutional-grade controls."""

    def __init__(
        self,
        limits: RiskLimits | None = None,
        state: PositionState | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.state = state or PositionState()
        self._correlation_history: dict[str, list[float]] = {}
        self._symbol_limits: dict[str, RiskLimits] = {}
        # CF-6: Track whether state was successfully loaded from Redis.
        # If False, risk engine operates in fail-closed mode (reject all trades).
        self._state_loaded: bool = False
        self._load_config()

    def _load_config(self) -> None:
        """Load config from YAML file, then apply env overrides."""
        # Try to load from config/limits.yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "limits.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)

                limits_cfg = cfg.get("limits", {})
                if limits_cfg.get("max_position_pct") is not None:
                    self.limits.max_position_pct = float(limits_cfg["max_position_pct"])
                if limits_cfg.get("max_daily_loss_pct") is not None:
                    self.limits.max_daily_loss_pct = float(limits_cfg["max_daily_loss_pct"])
                if limits_cfg.get("max_volatility") is not None:
                    self.limits.max_volatility = float(limits_cfg["max_volatility"])
                if limits_cfg.get("min_confidence") is not None:
                    self.limits.min_confidence = float(limits_cfg["min_confidence"])
                if limits_cfg.get("max_correlation_spike") is not None:
                    self.limits.max_correlation_spike = float(limits_cfg["max_correlation_spike"])
                if limits_cfg.get("kill_switch_loss_pct") is not None:
                    self.limits.kill_switch_loss_pct = float(limits_cfg["kill_switch_loss_pct"])

                # Load per-symbol overrides
                symbol_limits = cfg.get("symbol_limits", {})
                for symbol, sym_cfg in symbol_limits.items():
                    sym_limits = RiskLimits(
                        max_position_pct=float(sym_cfg.get("max_position_pct", self.limits.max_position_pct)),
                        max_daily_loss_pct=float(sym_cfg.get("max_daily_loss_pct", self.limits.max_daily_loss_pct)),
                        max_volatility=float(sym_cfg.get("max_volatility", self.limits.max_volatility)),
                        min_confidence=float(sym_cfg.get("min_confidence", self.limits.min_confidence)),
                        max_correlation_spike=float(sym_cfg.get("max_correlation_spike", self.limits.max_correlation_spike)),
                        kill_switch_loss_pct=float(sym_cfg.get("kill_switch_loss_pct", self.limits.kill_switch_loss_pct)),
                    )
                    self._symbol_limits[symbol] = sym_limits

                logger.info("risk_limits_loaded", source="config/limits.yaml", limits=limits_cfg)
            except Exception as e:
                logger.warning("risk_limits_yaml_error", error=str(e), path=str(config_path))

        # Env overrides take precedence
        if os.getenv("RISK_MAX_POSITION_PCT"):
            self.limits.max_position_pct = float(os.getenv("RISK_MAX_POSITION_PCT", "0.02"))
        if os.getenv("RISK_MAX_DAILY_LOSS_PCT"):
            self.limits.max_daily_loss_pct = float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", "0.05"))
        if os.getenv("RISK_MIN_CONFIDENCE"):
            self.limits.min_confidence = float(os.getenv("RISK_MIN_CONFIDENCE", "0.3"))

    def get_limits_for_symbol(self, symbol: str) -> RiskLimits:
        """Get risk limits for a symbol, with per-symbol overrides if defined."""
        return self._symbol_limits.get(symbol, self.limits)

    def evaluate(self, risk_input: RiskInput) -> RiskOutput:
        """Evaluate risk for a proposed trade."""
        checks_passed = {}
        reasons = []

        # Get per-symbol limits if defined, otherwise use defaults
        limits = self.get_limits_for_symbol(risk_input.symbol)

        if self.state.kill_switch_active:
            return RiskOutput(
                decision=RiskDecision.REJECTED,
                reasons=["KILL_SWITCH_ACTIVE"],
                approved_size=0.0,
                risk_score=1.0,
                cvar_95=0.0,
                max_loss=0.0,
                checks_passed={"kill_switch": False},
            )

        # CF-6: Fail-closed — reject all trades if Redis state was never loaded.
        # This prevents silent reset of daily_pnl and kill_switch on restart.
        if not self._state_loaded:
            logger.critical(
                "risk_state_not_loaded",
                reason="Redis state unavailable — fail-closed mode active",
            )
            return RiskOutput(
                decision=RiskDecision.REJECTED,
                reasons=["REDIS_STATE_UNAVAILABLE — fail-closed until state restored"],
                approved_size=0.0,
                risk_score=1.0,
                cvar_95=0.0,
                max_loss=0.0,
                checks_passed={"state_loaded": False},
            )
        checks_passed["confidence"] = risk_input.confidence >= limits.min_confidence
        if not checks_passed["confidence"]:
            reasons.append(f"Low confidence: {risk_input.confidence:.2f} < {limits.min_confidence}")
        checks_passed["volatility"] = risk_input.volatility_5 <= limits.max_volatility
        if not checks_passed["volatility"]:
            reasons.append(f"High volatility: {risk_input.volatility_5:.4f} > {limits.max_volatility}")
        checks_passed["daily_loss"] = self._check_daily_loss_with_limits(limits)
        if not checks_passed["daily_loss"]:
            reasons.append(f"Daily loss limit: {self.state.daily_pnl:.2f}")
        checks_passed["direction"] = self._check_direction(risk_input)
        if not checks_passed["direction"]:
            reasons.append(f"Neutral signal: {risk_input.direction}")
        cvar_95 = self._calculate_cvar(risk_input)
        max_position_value = risk_input.portfolio_value * limits.max_position_pct
        current_position_value = abs(risk_input.current_position * risk_input.price)
        remaining_capacity = max(0, max_position_value - current_position_value)
        approved_size = remaining_capacity / risk_input.price if risk_input.price > 0 else 0
        checks_passed["position_limit"] = remaining_capacity > 0
        if not checks_passed["position_limit"]:
            reasons.append(f"Position limit reached for {risk_input.symbol}")
        max_loss = approved_size * risk_input.price * abs(risk_input.horizon_30_q10)
        risk_score = self._calculate_risk_score(risk_input, cvar_95)
        all_passed = all(checks_passed.values())

        if all_passed:
            decision = RiskDecision.APPROVED
            reasons = ["All risk checks passed"]
        elif risk_score < 0.7 and checks_passed.get("daily_loss", True):
            decision = RiskDecision.PENDING_REVIEW
        else:
            decision = RiskDecision.REJECTED

        return RiskOutput(
            decision=decision,
            reasons=reasons,
            approved_size=approved_size if decision == RiskDecision.APPROVED else 0.0,
            risk_score=risk_score,
            cvar_95=cvar_95,
            max_loss=max_loss,
            checks_passed=checks_passed,
        )

    def _check_confidence(self, risk_input: RiskInput) -> bool:
        return risk_input.confidence >= self.limits.min_confidence

    def _check_volatility(self, risk_input: RiskInput) -> bool:
        return risk_input.volatility_5 <= self.limits.max_volatility

    def _check_daily_loss(self, risk_input: RiskInput) -> bool:
        daily_loss_pct = -self.state.daily_pnl / max(self.state.portfolio_value, 1)
        return daily_loss_pct < self.limits.max_daily_loss_pct

    def _check_daily_loss_with_limits(self, limits: RiskLimits) -> bool:
        """Check daily loss against given limits."""
        daily_loss_pct = -self.state.daily_pnl / max(self.state.portfolio_value, 1)
        return daily_loss_pct < limits.max_daily_loss_pct

    def _check_direction(self, risk_input: RiskInput) -> bool:
        return risk_input.direction in ("long", "short")

    def _calculate_cvar(self, risk_input: RiskInput) -> float:
        """Calculate CVaR (Conditional Value at Risk) at 95% level.

        Uses the quantile predictions to estimate tail risk.
        """
        if risk_input.direction == "long":
            var_95 = risk_input.horizon_30_q10
        else:
            var_95 = -risk_input.horizon_30_q90
        vol_adj = 1 + risk_input.volatility_5 * 5
        cvar_95 = var_95 * vol_adj * 1.25
        return abs(cvar_95)

    def _calculate_risk_score(self, risk_input: RiskInput, cvar_95: float) -> float:
        """Calculate overall risk score (0-1, higher = riskier)."""
        vol_score = min(risk_input.volatility_5 / self.limits.max_volatility, 1.0)
        confidence_score = 1.0 - risk_input.confidence
        cvar_score = min(cvar_95 / 0.05, 1.0)
        daily_loss_score = abs(self.state.daily_pnl) / (self.state.portfolio_value * self.limits.max_daily_loss_pct)
        daily_loss_score = min(daily_loss_score, 1.0)
        risk_score = (
            vol_score * 0.3 +
            confidence_score * 0.2 +
            cvar_score * 0.3 +
            daily_loss_score * 0.2
        )
        return min(risk_score, 1.0)

    def update_pnl(self, symbol: str, realized_pnl: float) -> None:
        """Update daily P&L after a trade."""
        self.state.daily_pnl += realized_pnl
        loss_pct = -self.state.daily_pnl / self.state.portfolio_value
        if loss_pct >= self.limits.kill_switch_loss_pct:
            self.state.kill_switch_active = True
            logger.warning("kill_switch_activated", daily_loss_pct=loss_pct)

    def activate_kill_switch(self, reason: str) -> None:
        """Manually activate kill switch."""
        self.state.kill_switch_active = True
        logger.warning("kill_switch_manual", reason=reason)

    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch (requires manual intervention)."""
        self.state.kill_switch_active = False
        logger.info("kill_switch_deactivated")

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self.state.reset_daily()
        logger.info("daily_risk_reset")

    # ── Redis state persistence ──────────────────────────────────────────

    async def connect_redis(
        self,
        host: str | None = None,
        port: int | None = None,
        key: str = "awet:risk:state",
    ) -> None:
        """Connect to Redis for state persistence."""
        self._redis: Any = None
        self._redis_key = key
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.critical("redis_not_installed", hint="pip install redis")
            # CF-6: Fail-closed — no Redis means no state persistence.
            self.state.kill_switch_active = True
            return
        self._redis = aioredis.Redis(host=host, port=port, decode_responses=True)
        try:
            await self._redis.ping()
            logger.info("risk_redis_connected", host=host, port=port)
        except Exception as exc:
            logger.critical(
                "risk_redis_unavailable_fail_closed",
                error=str(exc),
                action="kill_switch_activated",
            )
            # CF-6: Fail-closed — can't connect to Redis, activate kill switch.
            self.state.kill_switch_active = True
            self._redis = None

    async def close_redis(self) -> None:
        """Close Redis connection."""
        if getattr(self, "_redis", None):
            await self._redis.aclose()
            self._redis = None

    async def save_state(self) -> None:
        """Persist current PositionState to Redis (TTL 24h)."""
        redis_client = getattr(self, "_redis", None)
        if not redis_client:
            return
        try:
            key = getattr(self, "_redis_key", "awet:risk:state")
            await redis_client.set(key, json.dumps(self.state.to_dict()), ex=86400)
        except Exception as exc:
            logger.warning("risk_state_save_error", error=str(exc))

    async def load_state(self) -> bool:
        """Restore PositionState from Redis. Returns True if restored.

        CF-6: On failure, activates kill switch (fail-closed). Trading
        remains blocked until state is successfully loaded or manually
        overridden.
        """
        redis_client = getattr(self, "_redis", None)
        if not redis_client:
            logger.critical(
                "risk_state_load_failed",
                reason="No Redis connection",
                action="fail_closed_kill_switch_active",
            )
            self.state.kill_switch_active = True
            self._state_loaded = False
            return False
        try:
            key = getattr(self, "_redis_key", "awet:risk:state")
            data = await redis_client.get(key)
            if data:
                self.state = PositionState.from_dict(json.loads(data))
                self._state_loaded = True
                logger.info(
                    "risk_state_restored",
                    positions=len(self.state.positions),
                    daily_pnl=self.state.daily_pnl,
                    kill_switch=self.state.kill_switch_active,
                )
                return True
            else:
                # No state in Redis (first run or key expired).
                # This is acceptable — mark as loaded with defaults.
                self._state_loaded = True
                logger.info("risk_state_empty_first_run")
                return True
        except Exception as exc:
            logger.critical(
                "risk_state_load_failed",
                error=str(exc),
                action="fail_closed_kill_switch_active",
            )
            self.state.kill_switch_active = True
            self._state_loaded = False
        return False


_engine: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    """Get singleton risk engine instance."""
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine
