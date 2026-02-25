"""Tests for CF-2 (latest fold selection), CF-3 (embargo gap), CF-4 (annualization).

T-1: Embargo gap — no OOS sample overlaps IS window.
T-3: Annualization consistency — minute and daily Sharpe agree.
T-4: Latest fold selection — select_best returns latest, not max-Sharpe.
"""
import sys
import os
import importlib.util
import json
import tempfile

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

_ds = _import_module(
    "dataset",
    os.path.join(_PROJECT_ROOT, "services", "model_training", "dataset.py"),
)
TFTDatasetBuilder = _ds.TFTDatasetBuilder

from services.backtester.metrics import compute_metrics


# ===========================================================================
# T-4: Latest fold selection (CF-2)
# ===========================================================================

class TestLatestFoldSelection:
    """CF-2: select_best must return the latest fold, not the best OOS Sharpe."""

    def test_selects_latest_fold_not_best_sharpe(self):
        """Fold 0 has higher Sharpe but fold 2 (latest) should be selected."""
        validator = WalkForwardValidator(min_oos_sharpe=0.3, min_folds=2, min_win_rate=0.4)
        now = datetime.now(timezone.utc)
        results = [
            FoldResult(fold_number=0, in_sample_start=now, in_sample_end=now,
                       oos_start=now, oos_end=now, model_id="m0",
                       onnx_path="m0.onnx", oos_sharpe=2.5, oos_win_rate=0.60),
            FoldResult(fold_number=1, in_sample_start=now, in_sample_end=now,
                       oos_start=now, oos_end=now, model_id="m1",
                       onnx_path="m1.onnx", oos_sharpe=1.0, oos_win_rate=0.55),
            FoldResult(fold_number=2, in_sample_start=now, in_sample_end=now,
                       oos_start=now, oos_end=now, model_id="m2",
                       onnx_path="m2.onnx", oos_sharpe=0.8, oos_win_rate=0.50),
        ]
        best = validator.select_best(results)
        assert best is not None
        assert best.fold_number == 2, (
            f"Expected latest fold (2), got fold {best.fold_number}"
        )
        assert best.model_id == "m2"
        assert best.promoted is True

    def test_quality_gate_still_rejects_bad_models(self):
        """Even with latest-fold logic, median Sharpe < threshold => None."""
        validator = WalkForwardValidator(min_oos_sharpe=1.0, min_folds=2, min_win_rate=0.4)
        now = datetime.now(timezone.utc)
        results = [
            FoldResult(fold_number=0, in_sample_start=now, in_sample_end=now,
                       oos_start=now, oos_end=now, model_id="m0",
                       onnx_path="m0.onnx", oos_sharpe=0.3, oos_win_rate=0.50),
            FoldResult(fold_number=1, in_sample_start=now, in_sample_end=now,
                       oos_start=now, oos_end=now, model_id="m1",
                       onnx_path="m1.onnx", oos_sharpe=0.5, oos_win_rate=0.50),
        ]
        best = validator.select_best(results)
        assert best is None, "Median Sharpe 0.4 < threshold 1.0, should reject"


# ===========================================================================
# T-1: Embargo gap (CF-3)
# ===========================================================================

class TestEmbargoGap:
    """CF-3: No OOS sample's time window should overlap with IS."""

    def test_embargo_creates_gap_between_is_and_oos(self):
        """With embargo_bars > 0, oos_start must be strictly after is_end."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, tzinfo=timezone.utc)
        folds = generate_folds(
            start, end,
            train_months=12, oos_months=1,
            embargo_bars=180, bar_duration_minutes=1,
        )
        assert len(folds) >= 2, f"Need at least 2 folds, got {len(folds)}"

        for f in folds:
            gap = f.oos_start - f.in_sample_end
            expected_gap = timedelta(minutes=180)
            assert gap == expected_gap, (
                f"Fold {f.fold_number}: gap={gap}, expected={expected_gap}"
            )
            assert f.oos_start > f.in_sample_end, (
                f"Fold {f.fold_number}: OOS start must be after IS end"
            )

    def test_no_overlap_between_is_and_oos(self):
        """Verify no OOS window overlaps with its own IS window."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, tzinfo=timezone.utc)
        folds = generate_folds(
            start, end,
            train_months=12, oos_months=1,
            embargo_bars=180, bar_duration_minutes=1,
        )
        for f in folds:
            assert f.oos_start > f.in_sample_end, (
                f"Fold {f.fold_number}: OOS [{f.oos_start}] must be "
                f"after IS [{f.in_sample_end}]"
            )
            # With embargo, the gap should be exactly embargo_bars * bar_duration
            gap_minutes = (f.oos_start - f.in_sample_end).total_seconds() / 60
            assert gap_minutes >= 180, (
                f"Fold {f.fold_number}: gap={gap_minutes}min < 180min embargo"
            )

    def test_zero_embargo_preserves_old_behavior(self):
        """embargo_bars=0 means OOS starts right after IS (backward compat)."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, tzinfo=timezone.utc)
        folds = generate_folds(
            start, end,
            train_months=12, oos_months=1,
            embargo_bars=0,
        )
        for f in folds:
            assert f.oos_start == f.in_sample_end, (
                f"Fold {f.fold_number}: with embargo=0, OOS should start at IS end"
            )

    def test_daily_bar_embargo(self):
        """Embargo with daily bars: 180 bars × 1440 min = 180 days gap."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        folds = generate_folds(
            start, end,
            train_months=12, oos_months=1,
            embargo_bars=180, bar_duration_minutes=1440,
        )
        assert len(folds) >= 1, "Should generate at least 1 fold"
        for f in folds:
            gap_days = (f.oos_start - f.in_sample_end).days
            assert gap_days == 180, (
                f"Fold {f.fold_number}: daily embargo gap={gap_days}d, expected=180d"
            )


# ===========================================================================
# T-3: Annualization consistency (CF-4)
# ===========================================================================

class TestAnnualizationConsistency:
    """CF-4: Sharpe computed at minute and daily frequency should agree."""

    def test_daily_and_minute_sharpe_consistent(self):
        """Same returns array produces Sharpe that scales by sqrt(bars_per_day).

        This verifies the annualization formula: Sharpe_minute / Sharpe_daily ≈ sqrt(390).
        We use the SAME return array and only change bars_per_day, which is the
        correct way to test the annualization factor in isolation.
        """
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 252)

        sharpe_daily = compute_oos_metrics(
            returns, returns, risk_free_rate=0.0, bars_per_day=1,
        )["oos_sharpe"]

        sharpe_minute = compute_oos_metrics(
            returns, returns, risk_free_rate=0.0, bars_per_day=390,
        )["oos_sharpe"]

        # The ratio should be sqrt(390) ≈ 19.75
        if sharpe_daily != 0:
            ratio = sharpe_minute / sharpe_daily
            expected_ratio = np.sqrt(390)
            assert abs(ratio - expected_ratio) < 0.1, (
                f"Ratio={ratio:.4f}, expected≈{expected_ratio:.4f}"
            )

    def test_annualization_factor_correct(self):
        """Verify Sharpe scales as sqrt(bars_per_year)."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)

        sharpe_daily = compute_oos_metrics(returns, returns, bars_per_day=1)["oos_sharpe"]
        sharpe_minute = compute_oos_metrics(returns, returns, bars_per_day=390)["oos_sharpe"]

        # Minute annualization uses sqrt(252*390) vs daily sqrt(252)
        # Ratio should be approximately sqrt(390) ≈ 19.75
        if sharpe_daily != 0:
            ratio = sharpe_minute / sharpe_daily
            expected_ratio = np.sqrt(390)
            assert abs(ratio - expected_ratio) < 1.0, (
                f"Ratio={ratio:.2f}, expected≈{expected_ratio:.2f}"
            )

    def test_backtester_metrics_bars_per_day(self):
        """Verify backtester compute_metrics respects bars_per_day."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)

        report_daily = compute_metrics(returns, bars_per_day=1)
        report_minute = compute_metrics(returns, bars_per_day=390)

        # With minute bars, 252 bars = 252/390 days ≈ 0.65 days
        # With daily bars, 252 bars = 252 days = 1 year
        # CAGR should be much higher for minute (shorter period, same total return)
        assert report_daily.sharpe_ratio != 0
        assert report_minute.sharpe_ratio != 0
        # Minute Sharpe should be larger (higher annualization factor)
        assert abs(report_minute.sharpe_ratio) > abs(report_daily.sharpe_ratio), (
            f"Minute Sharpe={report_minute.sharpe_ratio} should exceed "
            f"Daily Sharpe={report_daily.sharpe_ratio}"
        )

    def test_default_bars_per_day_is_one(self):
        """Default bars_per_day=1 preserves backward compatibility."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        # Default call (no bars_per_day) should produce same result as explicit 1
        metrics_default = compute_oos_metrics(returns, returns)
        metrics_explicit = compute_oos_metrics(returns, returns, bars_per_day=1)
        assert metrics_default["oos_sharpe"] == metrics_explicit["oos_sharpe"]


# ===========================================================================
# T-2: Normalization stats round-trip (CF-4)
# ===========================================================================

class TestNormStatsRoundTrip:
    """CF-4: Normalization stats saved during training must match at inference."""

    def test_save_load_round_trip(self):
        """Save norm stats from training; load and verify match <1e-6."""
        np.random.seed(42)
        # Simulate a training feature matrix: 1000 samples × 15 features
        all_features = np.random.randn(1000, 15).astype(np.float32)
        all_features[:, 0] *= 100  # price-scale
        all_features[:, 1] *= 1e6  # volume-scale

        builder = TFTDatasetBuilder(normalize=True)
        builder.fit_normalization(all_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm_path = os.path.join(tmpdir, "norm_stats.json")
            builder.save_norm_stats(norm_path)

            # Load stats back
            loaded_stats = TFTDatasetBuilder.load_norm_stats(norm_path)

            # Verify round-trip accuracy
            for i in range(15):
                orig_mean, orig_std = builder._feature_stats[i]
                loaded_mean, loaded_std = loaded_stats[i]
                assert abs(orig_mean - loaded_mean) < 1e-6, (
                    f"Feature {i}: mean mismatch {orig_mean} vs {loaded_mean}"
                )
                assert abs(orig_std - loaded_std) < 1e-6, (
                    f"Feature {i}: std mismatch {orig_std} vs {loaded_std}"
                )

    def test_fitted_normalization_consistent(self):
        """Normalizing the same data twice with fitted stats gives identical results."""
        np.random.seed(42)
        all_features = np.random.randn(500, 15).astype(np.float32)

        builder = TFTDatasetBuilder(normalize=True)
        builder.fit_normalization(all_features)

        chunk1 = all_features[0:120]
        chunk2 = all_features[0:120].copy()

        norm1 = builder._normalize_features(chunk1)
        norm2 = builder._normalize_features(chunk2)

        max_err = float(np.max(np.abs(norm1 - norm2)))
        assert max_err < 1e-6, f"Normalization not deterministic: max error={max_err}"

    def test_unfitted_uses_per_sample_stats(self):
        """Without fit, _normalize_features falls back to per-sample stats."""
        np.random.seed(42)
        features = np.random.randn(120, 15).astype(np.float32)

        builder = TFTDatasetBuilder(normalize=True)
        # Do NOT call fit_normalization
        result = builder._normalize_features(features)

        # Per-sample z-score: each column should have mean≈0, std≈1
        for i in range(15):
            col_mean = float(np.mean(result[:, i]))
            col_std = float(np.std(result[:, i]))
            assert abs(col_mean) < 0.01, f"Feature {i}: mean={col_mean}, expected ~0"
            assert abs(col_std - 1.0) < 0.01, f"Feature {i}: std={col_std}, expected ~1"

    def test_json_sidecar_contains_feature_columns(self):
        """norm_stats.json includes feature_columns for traceability."""
        np.random.seed(42)
        features = np.random.randn(200, 15).astype(np.float32)

        builder = TFTDatasetBuilder(normalize=True)
        builder.fit_normalization(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm_path = os.path.join(tmpdir, "norm_stats.json")
            builder.save_norm_stats(norm_path)

            with open(norm_path) as f:
                data = json.load(f)
            assert "feature_columns" in data
            assert data["feature_columns"] == builder.FEATURE_COLUMNS
            assert len(data["means"]) == 15
            assert len(data["stds"]) == 15


# ===========================================================================
# Runner (for running outside pytest)
# ===========================================================================
if __name__ == "__main__":
    import traceback as tb
    test_classes = [
        TestLatestFoldSelection, TestEmbargoGap, TestAnnualizationConsistency,
        TestNormStatsRoundTrip,
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
