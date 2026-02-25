"""Unit tests for Phases 2-5: walk-forward, regime detector, filters,
calibration, portfolio risk, metrics, DLQ, and graceful shutdown.
"""
import sys
import os
import importlib.util

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from datetime import datetime, timezone, timedelta

# Direct module import to bypass services/model_training/__init__.py
# which has a broken import chain (src.ml.dataset not on path)
def _import_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_wf = _import_module(
    "walk_forward",
    os.path.join(_PROJECT_ROOT, "services", "model_training", "walk_forward.py"),
)
generate_folds = _wf.generate_folds
compute_oos_metrics = _wf.compute_oos_metrics
WalkForwardValidator = _wf.WalkForwardValidator
FoldResult = _wf.FoldResult

# Phase 3 — Regime, filters, calibration, portfolio risk
from services.signal_engine.regime import RegimeClassifier, Regime
from services.signal_engine.filters import SignalFilter
from services.signal_engine.confidence_calibration import PlattScaler
from services.risk_engine.portfolio_risk import (
    MaximumDrawdownPortfolio, TrailingStopRisk, MaximumSectorExposure,
    PositionState, UnifiedPortfolioRisk,
)

# Phase 4 — Metrics
from services.backtester.metrics import compute_metrics

# Phase 5 — DLQ, graceful shutdown
from shared.kafka.dlq import DeadLetterQueue, with_retry_and_dlq
from shared.core.graceful_shutdown import GracefulShutdown


# ===========================================================================
# Phase 2: Walk-forward tests
# ===========================================================================

class TestWalkForward:
    def test_fold_generation(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        folds = generate_folds(start, end, train_months=12, oos_months=1)
        assert len(folds) >= 6, f"Expected >= 6 folds, got {len(folds)}"
        # Each fold must have is_end == oos_start
        for f in folds:
            assert f.in_sample_end == f.oos_start
            assert f.oos_end > f.oos_start

    def test_oos_metrics_positive_returns(self):
        np.random.seed(42)
        preds = np.random.randn(100) * 0.01
        actuals = preds + np.random.randn(100) * 0.001  # correlated
        metrics = compute_oos_metrics(preds, actuals)
        assert metrics["oos_sharpe"] > 0, "Correlated predictions should give positive Sharpe"
        assert 0 <= metrics["oos_win_rate"] <= 1

    def test_oos_metrics_empty(self):
        metrics = compute_oos_metrics(np.array([]), np.array([]))
        assert metrics["oos_samples"] == 0
        assert metrics["oos_sharpe"] == 0.0

    def test_validator_rejects_low_sharpe(self):
        validator = WalkForwardValidator(min_oos_sharpe=0.5, min_folds=2)
        results = [
            FoldResult(fold_number=0, in_sample_start=datetime.now(timezone.utc),
                      in_sample_end=datetime.now(timezone.utc),
                      oos_start=datetime.now(timezone.utc),
                      oos_end=datetime.now(timezone.utc),
                      model_id="m1", onnx_path="test.onnx",
                      oos_sharpe=-0.5, oos_win_rate=0.3),
            FoldResult(fold_number=1, in_sample_start=datetime.now(timezone.utc),
                      in_sample_end=datetime.now(timezone.utc),
                      oos_start=datetime.now(timezone.utc),
                      oos_end=datetime.now(timezone.utc),
                      model_id="m2", onnx_path="test.onnx",
                      oos_sharpe=0.1, oos_win_rate=0.35),
        ]
        best = validator.select_best(results)
        assert best is None, "Should reject model with low OOS Sharpe"

    def test_validator_accepts_good_model(self):
        validator = WalkForwardValidator(min_oos_sharpe=0.5, min_folds=2, min_win_rate=0.4)
        results = [
            FoldResult(fold_number=0, in_sample_start=datetime.now(timezone.utc),
                      in_sample_end=datetime.now(timezone.utc),
                      oos_start=datetime.now(timezone.utc),
                      oos_end=datetime.now(timezone.utc),
                      model_id="m1", onnx_path="m1.onnx",
                      oos_sharpe=1.2, oos_win_rate=0.55),
            FoldResult(fold_number=1, in_sample_start=datetime.now(timezone.utc),
                      in_sample_end=datetime.now(timezone.utc),
                      oos_start=datetime.now(timezone.utc),
                      oos_end=datetime.now(timezone.utc),
                      model_id="m2", onnx_path="m2.onnx",
                      oos_sharpe=0.8, oos_win_rate=0.52),
        ]
        best = validator.select_best(results)
        assert best is not None, "Should accept model with good OOS metrics"
        assert best.promoted is True
        # CF-2 fix: selects latest fold (fold_number=1), not best Sharpe
        assert best.fold_number == 1, f"Expected latest fold 1, got {best.fold_number}"
        assert best.oos_sharpe == 0.8


# ===========================================================================
# Phase 3: Regime classifier tests
# ===========================================================================

class TestRegimeClassifier:
    def test_initial_regime_is_sideways(self):
        rc = RegimeClassifier()
        regime = rc.update("AAPL", 100.0)
        assert regime == Regime.SIDEWAYS  # not enough data

    def test_bull_regime(self):
        rc = RegimeClassifier(fast_period=5, slow_period=10)
        # Feed rising prices
        for i in range(20):
            regime = rc.update("AAPL", 100.0 + i * 5)
        assert regime == Regime.BULL

    def test_bear_regime(self):
        rc = RegimeClassifier(fast_period=5, slow_period=10)
        # Start high, then crash
        for _ in range(15):
            rc.update("MSFT", 200.0)
        for _ in range(20):
            regime = rc.update("MSFT", 50.0)
        assert regime == Regime.BEAR

    def test_confidence_scale(self):
        rc = RegimeClassifier(fast_period=5, slow_period=10)
        for i in range(20):
            rc.update("TEST", 100.0 + i * 5)
        scale = rc.confidence_scale("TEST")
        assert scale == 1.0  # Bull regime


# ===========================================================================
# Phase 3: Signal filters tests
# ===========================================================================

class TestSignalFilters:
    def test_market_hours_pass(self):
        f = SignalFilter()
        ts = datetime(2025, 1, 6, 19, 0, tzinfo=timezone.utc)  # 14:00 ET
        ok, _ = f.check_market_hours(ts)
        assert ok

    def test_market_hours_reject_weekend(self):
        f = SignalFilter()
        ts = datetime(2025, 1, 4, 19, 0, tzinfo=timezone.utc)  # Saturday
        ok, reason = f.check_market_hours(ts)
        assert not ok
        assert "weekend" in reason

    def test_low_liquidity_reject(self):
        f = SignalFilter(min_avg_volume=1_000_000)
        ok, reason = f.check_liquidity("AAPL", 500_000)
        assert not ok
        assert "low_liquidity" in reason

    def test_wide_spread_reject(self):
        f = SignalFilter(max_spread_pct=0.001)
        ok, reason = f.check_spread("AAPL", 100.0, 100.5)  # 0.5% spread
        assert not ok
        assert "wide_spread" in reason

    def test_earnings_blackout(self):
        f = SignalFilter(earnings_blackout_days=2)
        f.set_earnings_dates("AAPL", [datetime(2025, 1, 30, tzinfo=timezone.utc)])
        ok, reason = f.check_earnings_blackout("AAPL", datetime(2025, 1, 29, tzinfo=timezone.utc))
        assert not ok
        assert "earnings_blackout" in reason


# ===========================================================================
# Phase 3: Platt scaling tests
# ===========================================================================

class TestPlattScaler:
    def test_fit_and_transform(self):
        scaler = PlattScaler()
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.6, 0.4, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])
        scaler.fit(scores, labels)
        calibrated = scaler.transform(scores)
        assert len(calibrated) == len(scores)
        assert all(0 <= c <= 1 for c in calibrated), "Calibrated values must be in [0,1]"

    def test_unfitted_passthrough(self):
        scaler = PlattScaler()
        scores = np.array([0.5, 0.6])
        result = scaler.transform(scores)
        np.testing.assert_array_equal(result, scores)

    def test_single_transform(self):
        scaler = PlattScaler()
        scaler.A = -2.0
        scaler.B = 1.0
        scaler._fitted = True
        val = scaler.transform_single(0.5)
        assert 0 <= val <= 1


# ===========================================================================
# Phase 3: Portfolio risk tests
# ===========================================================================

class TestPortfolioRisk:
    def test_drawdown_kill(self):
        dd = MaximumDrawdownPortfolio(max_drawdown_pct=0.10)
        dd.update(100_000)  # peak
        should_kill, reason = dd.update(89_000)  # 11% drawdown
        assert should_kill
        assert "portfolio_drawdown_kill" in reason

    def test_drawdown_ok(self):
        dd = MaximumDrawdownPortfolio(max_drawdown_pct=0.10)
        dd.update(100_000)
        should_kill, _ = dd.update(95_000)  # 5% drawdown
        assert not should_kill

    def test_trailing_stop(self):
        ts = TrailingStopRisk(trailing_pct=0.05)
        pos = PositionState(symbol="AAPL", qty=100, avg_entry_price=150.0,
                           current_price=160.0, highest_price=160.0)
        # Price drops 6% from high
        pos.current_price = 150.0
        should_exit, reason = ts.check(pos)
        assert should_exit
        assert "trailing_stop" in reason

    def test_trailing_stop_ok(self):
        ts = TrailingStopRisk(trailing_pct=0.05)
        pos = PositionState(symbol="AAPL", qty=100, avg_entry_price=150.0,
                           current_price=158.0, highest_price=160.0)
        should_exit, _ = ts.check(pos)
        assert not should_exit  # 1.25% drop < 5%

    def test_sector_exposure_breach(self):
        sec = MaximumSectorExposure(max_sector_pct=0.30)
        positions = [
            PositionState(symbol="AAPL", qty=100, avg_entry_price=150, current_price=150, highest_price=150),
            PositionState(symbol="MSFT", qty=200, avg_entry_price=300, current_price=300, highest_price=300),
        ]
        # AAPL=$15k + MSFT=$60k = $75k tech out of $100k = 75% > 30%
        violations = sec.check_exposure(positions, 100_000)
        assert len(violations) > 0
        assert "technology" in violations[0][0]

    def test_sector_new_position_blocked(self):
        sec = MaximumSectorExposure(max_sector_pct=0.30)
        positions = [
            PositionState(symbol="AAPL", qty=100, avg_entry_price=150, current_price=150, highest_price=150),
        ]
        ok, reason = sec.can_add_position("MSFT", 20_000, positions, 100_000)
        assert not ok  # 15k + 20k = 35k/100k = 35% > 30%

    def test_unified_risk(self):
        risk = UnifiedPortfolioRisk(
            max_portfolio_drawdown=0.10,
            trailing_stop_pct=0.05,
            max_sector_pct=0.30,
        )
        positions = [
            PositionState(symbol="AAPL", qty=10, avg_entry_price=150, current_price=145, highest_price=160),
        ]
        violations = risk.check_portfolio(100_000, positions)
        # Should have trailing stop (160→145 = 9.4% > 5%)
        assert any("trailing_stop" in v for v in violations)


# ===========================================================================
# Phase 4: Metrics tests
# ===========================================================================

class TestMetrics:
    def test_positive_returns(self):
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001  # slight positive drift
        report = compute_metrics(returns)
        assert report.total_return > 0
        assert report.sharpe_ratio > 0
        assert 0 <= report.win_rate <= 1
        assert report.max_drawdown >= 0

    def test_empty_returns(self):
        report = compute_metrics(np.array([]))
        assert report.total_return == 0.0
        assert report.sharpe_ratio == 0.0

    def test_with_benchmark(self):
        np.random.seed(42)
        strategy = np.random.randn(252) * 0.01 + 0.001
        benchmark = np.random.randn(252) * 0.008
        report = compute_metrics(strategy, benchmark)
        assert report.beta != 0.0  # Should have some correlation


# ===========================================================================
# Phase 5: DLQ tests
# ===========================================================================

class TestDLQ:
    def test_dlq_send_no_crash(self):
        dlq = DeadLetterQueue(producer=None)
        dlq.send({"key": "value"}, ValueError("test error"),
                source_topic="market.raw", agent="test")

    def test_retry_and_dlq(self):
        call_count = 0

        def failing_func(msg):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always fails")

        dlq = DeadLetterQueue(producer=None)
        result = with_retry_and_dlq(
            failing_func, "test_msg", dlq,
            source_topic="test", agent="test", max_retries=2
        )
        assert result is None
        assert call_count == 3  # initial + 2 retries

    def test_retry_succeeds(self):
        call_count = 0

        def sometimes_fails(msg):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail once")
            return "success"

        dlq = DeadLetterQueue(producer=None)
        result = with_retry_and_dlq(
            sometimes_fails, "test_msg", dlq,
            source_topic="test", agent="test", max_retries=3
        )
        assert result == "success"
        assert call_count == 2


# ===========================================================================
# Phase 5: Graceful shutdown tests
# ===========================================================================

class TestGracefulShutdown:
    def test_handlers_called_in_reverse(self):
        order = []
        gs = GracefulShutdown()
        gs.register("first", lambda: order.append("first"))
        gs.register("second", lambda: order.append("second"))
        gs.register("third", lambda: order.append("third"))
        gs.shutdown()
        assert order == ["third", "second", "first"]

    def test_double_shutdown_ignored(self):
        gs = GracefulShutdown()
        count = [0]
        gs.register("counter", lambda: count.__setitem__(0, count[0] + 1))
        gs.shutdown()
        gs.shutdown()  # should be no-op
        assert count[0] == 1

    def test_failed_handler_doesnt_block(self):
        order = []
        gs = GracefulShutdown()
        gs.register("good1", lambda: order.append("good1"))
        gs.register("bad", lambda: 1/0)  # will raise
        gs.register("good2", lambda: order.append("good2"))
        gs.shutdown()
        assert "good2" in order
        assert "good1" in order


# ===========================================================================
# Runner
# ===========================================================================
if __name__ == "__main__":
    import traceback as tb
    test_classes = [
        TestWalkForward, TestRegimeClassifier, TestSignalFilters,
        TestPlattScaler, TestPortfolioRisk, TestMetrics,
        TestDLQ, TestGracefulShutdown,
    ]
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
