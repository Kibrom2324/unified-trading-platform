"""Unit tests for risk engine."""

from __future__ import annotations

import pytest
from src.risk.engine import (
    RiskDecision,
    RiskEngine,
    RiskInput,
    RiskLimits,
    PositionState,
)


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine(
        limits=RiskLimits(
            max_position_pct=0.02,
            max_daily_loss_pct=0.05,
            max_volatility=0.10,
            min_confidence=0.3,
        ),
        state=PositionState(portfolio_value=100000.0),
    )


@pytest.fixture
def sample_input() -> RiskInput:
    return RiskInput(
        symbol="AAPL",
        price=150.0,
        direction="long",
        confidence=0.6,
        horizon_30_q10=-0.01,
        horizon_30_q50=0.005,
        horizon_30_q90=0.02,
        volatility_5=0.02,
        volatility_15=0.025,
        current_position=0.0,
        portfolio_value=100000.0,
        daily_pnl=0.0,
    )


class TestRiskEngine:
    def test_approve_valid_trade(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        result = risk_engine.evaluate(sample_input)
        assert result.decision == RiskDecision.APPROVED
        assert result.approved_size > 0
        assert result.risk_score < 1.0

    def test_reject_low_confidence(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        sample_input.confidence = 0.1
        result = risk_engine.evaluate(sample_input)
        assert result.decision in (RiskDecision.REJECTED, RiskDecision.PENDING_REVIEW)
        assert "confidence" in result.reasons[0].lower()
        assert result.checks_passed["confidence"] is False

    def test_reject_high_volatility(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        sample_input.volatility_5 = 0.15
        result = risk_engine.evaluate(sample_input)
        assert result.decision in (RiskDecision.REJECTED, RiskDecision.PENDING_REVIEW)
        assert result.checks_passed["volatility"] is False

    def test_reject_neutral_direction(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        sample_input.direction = "neutral"
        result = risk_engine.evaluate(sample_input)
        assert result.decision in (RiskDecision.REJECTED, RiskDecision.PENDING_REVIEW)
        assert result.checks_passed["direction"] is False

    def test_kill_switch_blocks_all(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        risk_engine.state.kill_switch_active = True
        result = risk_engine.evaluate(sample_input)
        assert result.decision == RiskDecision.REJECTED
        assert "KILL_SWITCH_ACTIVE" in result.reasons

    def test_position_limit(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        sample_input.current_position = 20
        result = risk_engine.evaluate(sample_input)
        assert result.approved_size < 2000.0 / 150.0

    def test_daily_loss_limit(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        risk_engine.state.daily_pnl = -6000.0
        result = risk_engine.evaluate(sample_input)
        assert result.decision == RiskDecision.REJECTED
        assert result.checks_passed["daily_loss"] is False

    def test_cvar_calculation(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        result = risk_engine.evaluate(sample_input)
        assert result.cvar_95 > 0

    def test_risk_score_bounds(self, risk_engine: RiskEngine, sample_input: RiskInput) -> None:
        result = risk_engine.evaluate(sample_input)
        assert 0.0 <= result.risk_score <= 1.0


class TestPositionState:
    def test_reset_daily(self) -> None:
        state = PositionState(daily_pnl=-1000.0)
        state.reset_daily()
        assert state.daily_pnl == 0.0


class TestKillSwitch:
    def test_auto_activate_on_loss(self, risk_engine: RiskEngine) -> None:
        risk_engine.update_pnl("AAPL", -12000.0)
        assert risk_engine.state.kill_switch_active is True

    def test_manual_deactivate(self, risk_engine: RiskEngine) -> None:
        risk_engine.state.kill_switch_active = True
        risk_engine.deactivate_kill_switch()
        assert risk_engine.state.kill_switch_active is False
