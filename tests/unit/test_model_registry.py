"""Tests for model registry system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.ml.registry import (
    ModelEntry,
    ModelMetrics,
    ModelRegistry,
    ModelStatus,
)


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.json"
        models_dir = Path(tmpdir) / "models"
        registry = ModelRegistry(
            registry_path=registry_path,
            models_dir=models_dir,
        )
        yield registry


@pytest.fixture
def sample_metrics():
    """Sample training metrics."""
    return ModelMetrics(
        train_loss=0.05,
        val_loss=0.07,
        epochs=50,
        samples=10000,
        train_duration_seconds=120.5,
    )


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_to_dict(self, sample_metrics):
        """Test metrics serialization."""
        d = sample_metrics.to_dict()
        assert d["train_loss"] == 0.05
        assert d["val_loss"] == 0.07
        assert d["epochs"] == 50
        assert d["samples"] == 10000
        assert d["train_duration_seconds"] == 120.5

    def test_from_dict(self):
        """Test metrics deserialization."""
        d = {
            "train_loss": 0.03,
            "val_loss": 0.04,
            "epochs": 100,
            "samples": 5000,
            "train_duration_seconds": 60.0,
        }
        metrics = ModelMetrics.from_dict(d)
        assert metrics.train_loss == 0.03
        assert metrics.val_loss == 0.04
        assert metrics.epochs == 100


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_empty_registry(self, temp_registry):
        """Test empty registry state."""
        assert len(temp_registry.list_models()) == 0
        assert temp_registry.get_green() is None

    def test_register_model(self, temp_registry, sample_metrics, tmp_path):
        """Test model registration."""
        # Create dummy model files
        onnx_path = tmp_path / "model.onnx"
        checkpoint_path = tmp_path / "checkpoint.pt"
        onnx_path.write_bytes(b"dummy onnx")
        checkpoint_path.write_bytes(b"dummy checkpoint")

        entry = temp_registry.register(
            run_id="test_run_001",
            onnx_path=str(onnx_path),
            checkpoint_path=str(checkpoint_path),
            metrics=sample_metrics,
            config={"hidden_dim": 64},
        )

        assert entry.status == ModelStatus.CANDIDATE
        assert "test_run_001" in entry.model_id
        assert entry.metrics.val_loss == 0.07
        assert entry.config["hidden_dim"] == 64

    def test_promote_to_green(self, temp_registry, sample_metrics, tmp_path):
        """Test model promotion to green status."""
        # Create and register a model
        onnx_path = tmp_path / "model.onnx"
        checkpoint_path = tmp_path / "checkpoint.pt"
        onnx_path.write_bytes(b"dummy")
        checkpoint_path.write_bytes(b"dummy")

        entry = temp_registry.register(
            run_id="test_run_002",
            onnx_path=str(onnx_path),
            checkpoint_path=str(checkpoint_path),
            metrics=sample_metrics,
        )

        # Promote to green
        promoted = temp_registry.promote_to_green(entry.model_id)
        assert promoted.status == ModelStatus.GREEN
        assert promoted.promoted_at is not None

        # Verify it's the green model
        green = temp_registry.get_green()
        assert green is not None
        assert green.model_id == entry.model_id

    def test_promote_deprecates_old_green(self, temp_registry, sample_metrics, tmp_path):
        """Test that promoting a new model deprecates the old green."""
        # Create first model
        onnx1 = tmp_path / "model1.onnx"
        cp1 = tmp_path / "checkpoint1.pt"
        onnx1.write_bytes(b"dummy1")
        cp1.write_bytes(b"dummy1")

        entry1 = temp_registry.register(
            run_id="run_001",
            onnx_path=str(onnx1),
            checkpoint_path=str(cp1),
            metrics=sample_metrics,
        )
        temp_registry.promote_to_green(entry1.model_id)

        # Create second model
        onnx2 = tmp_path / "model2.onnx"
        cp2 = tmp_path / "checkpoint2.pt"
        onnx2.write_bytes(b"dummy2")
        cp2.write_bytes(b"dummy2")

        entry2 = temp_registry.register(
            run_id="run_002",
            onnx_path=str(onnx2),
            checkpoint_path=str(cp2),
            metrics=sample_metrics,
        )
        temp_registry.promote_to_green(entry2.model_id)

        # Verify old model is deprecated
        old_entry = temp_registry.get(entry1.model_id)
        assert old_entry.status == ModelStatus.DEPRECATED
        assert old_entry.deprecated_at is not None

        # Verify new model is green
        green = temp_registry.get_green()
        assert green.model_id == entry2.model_id

    def test_list_models_by_status(self, temp_registry, sample_metrics, tmp_path):
        """Test filtering models by status."""
        # Create multiple models
        for i in range(3):
            onnx = tmp_path / f"model{i}.onnx"
            cp = tmp_path / f"checkpoint{i}.pt"
            onnx.write_bytes(b"dummy")
            cp.write_bytes(b"dummy")

            temp_registry.register(
                run_id=f"run_{i:03d}",
                onnx_path=str(onnx),
                checkpoint_path=str(cp),
                metrics=sample_metrics,
            )

        # All should be candidates
        candidates = temp_registry.list_models(status=ModelStatus.CANDIDATE)
        assert len(candidates) == 3

        greens = temp_registry.list_models(status=ModelStatus.GREEN)
        assert len(greens) == 0

    def test_auto_promote_best(self, temp_registry, tmp_path):
        """Test auto-promotion of best candidate."""
        # Create models with different val_loss
        for i, val_loss in enumerate([0.10, 0.05, 0.08]):
            onnx = tmp_path / f"model{i}.onnx"
            cp = tmp_path / f"checkpoint{i}.pt"
            onnx.write_bytes(b"dummy")
            cp.write_bytes(b"dummy")

            metrics = ModelMetrics(
                train_loss=0.03,
                val_loss=val_loss,
                epochs=50,
                samples=5000,
            )

            temp_registry.register(
                run_id=f"run_{i:03d}",
                onnx_path=str(onnx),
                checkpoint_path=str(cp),
                metrics=metrics,
            )

        # Auto-promote should pick model with val_loss=0.05
        promoted = temp_registry.auto_promote_best(min_samples=1000)
        assert promoted is not None
        assert promoted.metrics.val_loss == 0.05

    def test_persistence(self, tmp_path, sample_metrics):
        """Test registry persistence across instances."""
        registry_path = tmp_path / "registry.json"
        models_dir = tmp_path / "models"

        # Create and populate registry
        registry1 = ModelRegistry(registry_path, models_dir)

        onnx = tmp_path / "model.onnx"
        cp = tmp_path / "checkpoint.pt"
        onnx.write_bytes(b"dummy")
        cp.write_bytes(b"dummy")

        entry = registry1.register(
            run_id="persist_test",
            onnx_path=str(onnx),
            checkpoint_path=str(cp),
            metrics=sample_metrics,
        )
        registry1.promote_to_green(entry.model_id)

        # Create new registry instance from same file
        registry2 = ModelRegistry(registry_path, models_dir)

        # Verify data persisted
        green = registry2.get_green()
        assert green is not None
        assert green.model_id == entry.model_id
        assert green.metrics.val_loss == 0.07

    def test_delete_model(self, temp_registry, sample_metrics, tmp_path):
        """Test model deletion."""
        onnx = tmp_path / "model.onnx"
        cp = tmp_path / "checkpoint.pt"
        onnx.write_bytes(b"dummy")
        cp.write_bytes(b"dummy")

        entry = temp_registry.register(
            run_id="delete_test",
            onnx_path=str(onnx),
            checkpoint_path=str(cp),
            metrics=sample_metrics,
        )

        assert temp_registry.delete(entry.model_id)
        assert temp_registry.get(entry.model_id) is None
        assert len(temp_registry.list_models()) == 0

    def test_get_green_onnx_path(self, temp_registry, sample_metrics, tmp_path):
        """Test getting green model ONNX path."""
        # No green model yet
        assert temp_registry.get_green_onnx_path() is None

        # Register and promote
        onnx = tmp_path / "model.onnx"
        cp = tmp_path / "checkpoint.pt"
        onnx.write_bytes(b"dummy")
        cp.write_bytes(b"dummy")

        entry = temp_registry.register(
            run_id="onnx_path_test",
            onnx_path=str(onnx),
            checkpoint_path=str(cp),
            metrics=sample_metrics,
        )
        temp_registry.promote_to_green(entry.model_id)

        # Now should have path
        path = temp_registry.get_green_onnx_path()
        assert path is not None
        assert "model.onnx" in path
