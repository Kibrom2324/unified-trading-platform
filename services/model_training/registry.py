"""Model registry for TFT model versioning and deployment.

Manages model lifecycle:
- candidate: Newly trained, not yet validated
- green: Active production model (only one at a time)
- deprecated: Previously active, kept for rollback

Uses both file-based registry (JSON) and optional TimescaleDB table.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("model_registry")


class ModelStatus(str, Enum):
    """Model deployment status."""
    CANDIDATE = "candidate"
    GREEN = "green"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """Training metrics for a model."""
    train_loss: float
    val_loss: float
    epochs: int
    samples: int
    train_duration_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetrics:
        return cls(**data)


@dataclass
class ModelEntry:
    """Registry entry for a model version."""
    model_id: str
    run_id: str
    status: ModelStatus
    onnx_path: str
    checkpoint_path: str
    created_at: str
    metrics: ModelMetrics
    config: dict[str, Any] = field(default_factory=dict)
    promoted_at: str | None = None
    deprecated_at: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "onnx_path": self.onnx_path,
            "checkpoint_path": self.checkpoint_path,
            "created_at": self.created_at,
            "metrics": self.metrics.to_dict(),
            "config": self.config,
            "promoted_at": self.promoted_at,
            "deprecated_at": self.deprecated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        return cls(
            model_id=data["model_id"],
            run_id=data["run_id"],
            status=ModelStatus(data["status"]),
            onnx_path=data["onnx_path"],
            checkpoint_path=data["checkpoint_path"],
            created_at=data["created_at"],
            metrics=ModelMetrics.from_dict(data["metrics"]),
            config=data.get("config", {}),
            promoted_at=data.get("promoted_at"),
            deprecated_at=data.get("deprecated_at"),
        )


class ModelRegistry:
    """File-based model registry with JSON storage."""
    
    def __init__(
        self,
        registry_path: str | Path = "models/registry.json",
        models_dir: str | Path = "models/tft",
    ) -> None:
        self.registry_path = Path(registry_path)
        self.models_dir = Path(models_dir)
        self._entries: dict[str, ModelEntry] = {}
        self._load()
    
    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry_data in data.get("models", []):
                    entry = ModelEntry.from_dict(entry_data)
                    self._entries[entry.model_id] = entry
                logger.info("registry_loaded", models=len(self._entries))
            except Exception as e:
                logger.error("registry_load_error", error=str(e))
                self._entries = {}
        else:
            logger.info("registry_not_found_creating", path=str(self.registry_path))
    
    def _save(self) -> None:
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "models": [e.to_dict() for e in self._entries.values()],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("registry_saved", models=len(self._entries))
    
    def register(
        self,
        run_id: str,
        onnx_path: str | Path,
        checkpoint_path: str | Path,
        metrics: ModelMetrics,
        config: dict[str, Any] | None = None,
    ) -> ModelEntry:
        """Register a new model version as candidate.
        
        Args:
            run_id: Training run identifier
            onnx_path: Path to ONNX model file
            checkpoint_path: Path to PyTorch checkpoint
            metrics: Training metrics
            config: Model configuration
            
        Returns:
            Created ModelEntry
        """
        now = datetime.now(tz=timezone.utc)
        model_id = f"tft_{run_id}_{now.strftime('%Y%m%d_%H%M%S')}"
        
        # Copy model files to registry location (if not already there)
        model_dir = self.models_dir / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        dest_onnx = model_dir / "model.onnx"
        dest_checkpoint = model_dir / "checkpoint.pt"
        
        # Only copy if source is different from destination
        src_onnx = Path(onnx_path).resolve()
        src_checkpoint = Path(checkpoint_path).resolve()
        
        if src_onnx.exists() and src_onnx != dest_onnx.resolve():
            shutil.copy2(onnx_path, dest_onnx)
        elif src_onnx.exists():
            dest_onnx = src_onnx  # Already in place
            
        if src_checkpoint.exists() and src_checkpoint != dest_checkpoint.resolve():
            shutil.copy2(checkpoint_path, dest_checkpoint)
        elif src_checkpoint.exists():
            dest_checkpoint = src_checkpoint  # Already in place
        
        entry = ModelEntry(
            model_id=model_id,
            run_id=run_id,
            status=ModelStatus.CANDIDATE,
            onnx_path=str(dest_onnx),
            checkpoint_path=str(dest_checkpoint),
            created_at=now.isoformat(),
            metrics=metrics,
            config=config or {},
        )
        
        self._entries[model_id] = entry
        self._save()
        
        logger.info(
            "model_registered",
            model_id=model_id,
            status=entry.status.value,
            val_loss=metrics.val_loss,
        )
        return entry
    
    def promote_to_green(self, model_id: str) -> ModelEntry:
        """Promote model to green (production) status.
        
        Deprecates any existing green model.
        
        Args:
            model_id: ID of model to promote
            
        Returns:
            Updated ModelEntry
        """
        if model_id not in self._entries:
            raise ValueError(f"Model {model_id} not found in registry")
        
        now = datetime.now(tz=timezone.utc).isoformat()
        
        # Deprecate current green model
        current_green = self.get_green()
        if current_green:
            current_green.status = ModelStatus.DEPRECATED
            current_green.deprecated_at = now
            logger.info("model_deprecated", model_id=current_green.model_id)
        
        # Promote new model
        entry = self._entries[model_id]
        entry.status = ModelStatus.GREEN
        entry.promoted_at = now
        
        self._save()
        
        logger.info("model_promoted", model_id=model_id)
        return entry
    
    def get_green(self) -> ModelEntry | None:
        """Get the current green (production) model."""
        for entry in self._entries.values():
            if entry.status == ModelStatus.GREEN:
                return entry
        return None
    
    def get_green_onnx_path(self) -> str | None:
        """Get ONNX path for green model."""
        green = self.get_green()
        if green and Path(green.onnx_path).exists():
            return green.onnx_path
        return None
    
    def get(self, model_id: str) -> ModelEntry | None:
        """Get model entry by ID."""
        return self._entries.get(model_id)
    
    def list_models(
        self,
        status: ModelStatus | None = None,
    ) -> list[ModelEntry]:
        """List models, optionally filtered by status."""
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)
    
    def delete(self, model_id: str, delete_files: bool = False) -> bool:
        """Delete model from registry.
        
        Args:
            model_id: ID of model to delete
            delete_files: Also delete model files from disk
            
        Returns:
            True if deleted, False if not found
        """
        if model_id not in self._entries:
            return False
        
        entry = self._entries[model_id]
        
        if delete_files:
            model_dir = Path(entry.onnx_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info("model_files_deleted", path=str(model_dir))
        
        del self._entries[model_id]
        self._save()
        
        logger.info("model_deleted", model_id=model_id)
        return True
    
    @staticmethod
    def _check_onnx_output_compat(onnx_path: str, expected_outputs: int = 10) -> bool:
        """Verify the ONNX model output dimension matches the pipeline.

        The inference engine expects ``[batch, expected_outputs]`` where
        the default 10 = 3 horizons × 3 quantiles + 1 confidence.
        """
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_path)
            out_shape = sess.get_outputs()[0].shape
            n_out = out_shape[1] if len(out_shape) > 1 and isinstance(out_shape[1], int) else -1
            return n_out == expected_outputs
        except Exception as exc:
            logger.warning("onnx_compat_check_failed", path=onnx_path, error=str(exc))
            return False

    def auto_promote_best(self, min_samples: int = 1000) -> ModelEntry | None:
        """Auto-promote best candidate model if it beats current green.

        Candidates are only eligible when:
        1. They have at least ``min_samples`` training samples.
        2. Their ONNX output shape is compatible with the inference
           pipeline (10 outputs by default).
        3. Their validation loss is strictly lower than the current
           green model's.

        Args:
            min_samples: Minimum training samples required
            
        Returns:
            Promoted model or None
        """
        candidates = self.list_models(status=ModelStatus.CANDIDATE)
        candidates = [c for c in candidates if c.metrics.samples >= min_samples]

        # Filter out models with incompatible ONNX output dimensions
        compat_candidates = []
        for c in candidates:
            if Path(c.onnx_path).exists() and self._check_onnx_output_compat(c.onnx_path):
                compat_candidates.append(c)
            else:
                logger.info(
                    "candidate_incompatible_output",
                    model_id=c.model_id,
                    onnx_path=c.onnx_path,
                )
        candidates = compat_candidates
        
        if not candidates:
            logger.info("no_eligible_candidates", min_samples=min_samples)
            return None
        
        # Sort by validation loss (lower is better)
        candidates.sort(key=lambda c: c.metrics.val_loss)
        best_candidate = candidates[0]
        
        current_green = self.get_green()
        if current_green:
            if best_candidate.metrics.val_loss >= current_green.metrics.val_loss:
                logger.info(
                    "candidate_not_better",
                    candidate_loss=best_candidate.metrics.val_loss,
                    green_loss=current_green.metrics.val_loss,
                )
                return None
        
        return self.promote_to_green(best_candidate.model_id)


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_green_model_path() -> str | None:
    """Convenience function to get green ONNX path."""
    return get_registry().get_green_onnx_path()
