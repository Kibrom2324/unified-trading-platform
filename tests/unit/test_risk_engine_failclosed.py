"""Tests for CF-6: Risk engine fail-closed on Redis failure.

T-6: Start risk engine, process decisions, simulate Redis outage,
     restart engine, verify trading stays blocked.

Also tests:
  - evaluate() rejects when _state_loaded=False
  - load_state() with no Redis activates kill switch
  - Explicit mark_state_loaded bypasses fail-closed
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import asyncio
from datetime import datetime, timezone

from services.risk_engine.engine import (
    RiskDecision,
    RiskEngine,
    RiskInput,
    RiskLimits,
    PositionState,
)


def _make_input(**overrides) -> RiskInput:
    """Create a standard valid RiskInput for testing."""
    defaults = dict(
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
    defaults.update(overrides)
    return RiskInput(**defaults)


# ===========================================================================
# Unit tests — fail-closed behavior
# ===========================================================================

class TestFailClosedBehavior:
    """Risk engine must reject all trades when state is not loaded."""

    def test_new_engine_rejects_by_default(self):
        """A fresh RiskEngine (no load_state called) rejects all trades."""
        engine = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0),
        )
        # _state_loaded defaults to False
        assert engine._state_loaded is False

        result = engine.evaluate(_make_input())
        assert result.decision == RiskDecision.REJECTED, (
            f"Expected REJECTED, got {result.decision}"
        )
        assert "REDIS_STATE_UNAVAILABLE" in result.reasons[0]
        assert result.checks_passed.get("state_loaded") is False

    def test_engine_approves_after_state_loaded(self):
        """After _state_loaded=True, engine processes normally."""
        engine = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0),
        )
        engine._state_loaded = True

        result = engine.evaluate(_make_input())
        assert result.decision == RiskDecision.APPROVED, (
            f"Expected APPROVED, got {result.decision}. Reasons: {result.reasons}"
        )

    def test_kill_switch_takes_priority_over_state_loaded(self):
        """Kill switch should fire before the state_loaded check."""
        engine = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0, kill_switch_active=True),
        )
        engine._state_loaded = True

        result = engine.evaluate(_make_input())
        assert result.decision == RiskDecision.REJECTED
        assert "KILL_SWITCH_ACTIVE" in result.reasons


# ===========================================================================
# T-6: Integration test — Redis outage simulation
# ===========================================================================

class TestRedisOutageScenario:
    """T-6: Simulate Redis outage and verify fail-closed behavior."""

    def test_load_state_without_redis_activates_kill_switch(self):
        """If no Redis connection, load_state must activate kill switch."""
        engine = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0),
        )
        # Don't call connect_redis — simulate "Redis not available"
        # _redis is not set (no attribute), so load_state should fail-closed
        result = asyncio.get_event_loop().run_until_complete(engine.load_state())

        assert result is False, "load_state should return False without Redis"
        assert engine.state.kill_switch_active is True, (
            "Kill switch must be active after failed load_state"
        )
        assert engine._state_loaded is False

    def test_full_outage_scenario(self):
        """
        T-6 full scenario:
        1. Create engine, mark state as loaded (simulates successful startup)
        2. Process some decisions — should work
        3. Simulate restart with no Redis — create new engine, call load_state
        4. Verify trading is blocked
        """
        # Step 1: Normal operation
        engine_v1 = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0),
        )
        engine_v1._state_loaded = True

        # Step 2: Process some trades — should succeed
        for _ in range(3):
            result = engine_v1.evaluate(_make_input())
            assert result.decision == RiskDecision.APPROVED, (
                f"Pre-outage trade should be APPROVED, got {result.decision}"
            )

        # Record some P&L
        engine_v1.update_pnl("AAPL", -500.0)
        assert engine_v1.state.daily_pnl == -500.0

        # Step 3: Simulate restart — new engine, Redis is down
        engine_v2 = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0),
        )
        # Try to load state without Redis connection
        loaded = asyncio.get_event_loop().run_until_complete(engine_v2.load_state())

        assert loaded is False, "load_state should fail without Redis"

        # Step 4: Verify trading is blocked
        result = engine_v2.evaluate(_make_input())
        assert result.decision == RiskDecision.REJECTED, (
            f"Post-outage trade should be REJECTED, got {result.decision}"
        )

        # Verify state was NOT silently reset
        assert engine_v2.state.kill_switch_active is True, (
            "Kill switch must be active after failed state restore"
        )
        assert engine_v2._state_loaded is False

    def test_state_loaded_flag_not_set_without_explicit_load(self):
        """Engine constructed with state= arg still needs load_state to trade."""
        engine = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(
                portfolio_value=100000.0,
                daily_pnl=-2000.0,
                kill_switch_active=False,
            ),
        )
        # Even with explicit state, _state_loaded is False
        result = engine.evaluate(_make_input())
        assert result.decision == RiskDecision.REJECTED
        assert "REDIS_STATE_UNAVAILABLE" in result.reasons[0]

    def test_manual_override_allows_trading(self):
        """Manual override: setting _state_loaded=True bypasses fail-closed."""
        engine = RiskEngine(
            limits=RiskLimits(),
            state=PositionState(portfolio_value=100000.0),
        )
        assert engine._state_loaded is False

        # Manual override (e.g., operator confirms state is OK)
        engine._state_loaded = True

        result = engine.evaluate(_make_input())
        assert result.decision == RiskDecision.APPROVED


# ===========================================================================
# Runner
# ===========================================================================
if __name__ == "__main__":
    import traceback as tb
    test_classes = [TestFailClosedBehavior, TestRedisOutageScenario]
    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        for method_name in sorted(m for m in dir(cls) if m.startswith("test_")):
            try:
                getattr(instance, method_name)()
                print(f"  ✅ {cls.__name__}.{method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls.__name__}.{method_name}: {e}")
                tb.print_exc()
                failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
