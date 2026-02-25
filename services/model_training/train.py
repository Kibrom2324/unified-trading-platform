#!/usr/bin/env python3
"""TFT training CLI with TimescaleDB integration and model registry.

.. deprecated::
    This training script uses ``TemporalFusionTransformerCore`` from
    ``src/ml/tft/model.py`` (Lightning-based).  The canonical training
    pipeline is ``scripts/train_tft_baseline.py`` which trains the
    production ``SimpleTFTModel``, exports to ONNX with
    ``feature_meta.json``, and registers in the model registry.
    Use that instead.

Usage:
    python -m src.ml.train --from-db --epochs 50 --promote
    python -m src.ml.train --export-only --model-id <id>
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("tft_train")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("torch_not_installed", hint="pip install torch")

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        LIGHTNING_AVAILABLE = True
    except ImportError:
        LIGHTNING_AVAILABLE = False
        logger.warning("lightning_not_installed", hint="pip install pytorch-lightning")


def train_tft(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    lookback_window: int = 120,
    horizons: list[int] | None = None,
    symbols: list[str] | None = None,
    db_url: str | None = None,
    output_dir: str = "models/tft",
    use_lightning: bool = True,
    early_stopping_patience: int = 10,
) -> dict[str, Any]:
    """Train TFT model from TimescaleDB features.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        hidden_dim: Hidden dimension for model
        lookback_window: Input sequence length
        horizons: Prediction horizons in timesteps
        symbols: List of symbols to train on (None for all)
        db_url: Database connection URL
        output_dir: Output directory for model files
        use_lightning: Use PyTorch Lightning trainer
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Dict with training metrics and paths
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required: pip install torch")
    
    from src.ml.tft.dataset import TFTDataModule, DatasetConfig
    from src.ml.tft.model import TFTConfig, TFTLightning, export_to_onnx, TemporalFusionTransformerCore, QuantileLoss
    from src.ml.registry import get_registry, ModelMetrics
    
    horizons = horizons or [30, 45, 60]
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    
    logger.info(
        "training_start",
        run_id=run_id,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        symbols=symbols,
    )
    
    # Prepare data
    data_config = DatasetConfig(
        lookback_window=lookback_window,
        horizons=horizons,
        batch_size=batch_size,
        train_split=0.8,
        normalize=True,
    )
    
    data_module = TFTDataModule(
        config=data_config,
        db_url=db_url,
        symbols=symbols,
    )
    data_module.prepare_data()
    
    # Model config
    model_config = TFTConfig(
        input_dim=15,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_encoder_layers=2,
        dropout=0.1,
        lookback_window=lookback_window,
        horizons=horizons,
        quantiles=[0.1, 0.5, 0.9],
        learning_rate=learning_rate,
        weight_decay=1e-5,
    )
    
    # Output paths
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    onnx_path = run_dir / "model.onnx"
    
    start_time = time.time()
    
    if use_lightning and LIGHTNING_AVAILABLE:
        # Use PyTorch Lightning
        model = TFTLightning(model_config)
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=str(run_dir),
                filename="best-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )
        
        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )
        
        # Get best metrics
        best_val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        train_loss = float(trainer.callback_metrics.get("train_loss", float("inf")))
        actual_epochs = trainer.current_epoch + 1
        
        # Save final checkpoint
        torch.save(model.model.state_dict(), checkpoint_path)
        
        # Export to ONNX
        export_to_onnx(
            model,
            str(onnx_path),
            lookback_window=lookback_window,
            input_dim=15,
        )
        
    else:
        # Manual training loop (fallback)
        model = TemporalFusionTransformerCore(model_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out["quantiles"], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    out = model(batch_x)
                    loss = criterion(out["quantiles"], batch_y)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            logger.info(
                "epoch_complete",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("early_stopping", epoch=epoch + 1)
                    break
        
        actual_epochs = epoch + 1
        
        # Load best and export
        model.load_state_dict(torch.load(checkpoint_path))
        export_to_onnx(
            model,
            str(onnx_path),
            lookback_window=lookback_window,
            input_dim=15,
        )
    
    train_duration = time.time() - start_time
    n_samples = len(data_module.train_dataset) + len(data_module.val_dataset)
    
    # Register model
    registry = get_registry()
    metrics = ModelMetrics(
        train_loss=float(train_loss),
        val_loss=float(best_val_loss),
        epochs=actual_epochs,
        samples=n_samples,
        train_duration_seconds=train_duration,
    )
    
    entry = registry.register(
        run_id=run_id,
        onnx_path=str(onnx_path),
        checkpoint_path=str(checkpoint_path),
        metrics=metrics,
        config=asdict(model_config),
    )
    
    logger.info(
        "training_complete",
        run_id=run_id,
        model_id=entry.model_id,
        val_loss=best_val_loss,
        epochs=actual_epochs,
        duration_seconds=train_duration,
    )
    
    return {
        "run_id": run_id,
        "model_id": entry.model_id,
        "train_loss": float(train_loss),
        "val_loss": float(best_val_loss),
        "epochs": actual_epochs,
        "samples": n_samples,
        "duration_seconds": train_duration,
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(checkpoint_path),
    }


def export_model(model_id: str, output_path: str | None = None) -> str:
    """Export existing model to ONNX.
    
    Args:
        model_id: Model ID from registry
        output_path: Custom output path (optional)
        
    Returns:
        Path to ONNX file
    """
    from src.ml.tft.model import TFTConfig, TemporalFusionTransformerCore, export_to_onnx
    from src.ml.registry import get_registry
    
    registry = get_registry()
    entry = registry.get(model_id)
    
    if not entry:
        raise ValueError(f"Model {model_id} not found")
    
    checkpoint_path = entry.checkpoint_path
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model config
    config = TFTConfig(**entry.config) if entry.config else TFTConfig()
    model = TemporalFusionTransformerCore(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    output_path = output_path or entry.onnx_path
    export_to_onnx(
        model,
        output_path,
        lookback_window=config.lookback_window,
        input_dim=config.input_dim,
    )
    
    return output_path


def promote_model(model_id: str | None = None, auto: bool = False) -> str | None:
    """Promote model to green status.
    
    Args:
        model_id: Specific model to promote (or None with auto=True)
        auto: Auto-promote best candidate
        
    Returns:
        Promoted model ID or None
    """
    from src.ml.registry import get_registry
    
    registry = get_registry()
    
    if auto:
        entry = registry.auto_promote_best()
        if entry:
            logger.info("model_auto_promoted", model_id=entry.model_id)
            return entry.model_id
        return None
    
    if not model_id:
        raise ValueError("model_id required when auto=False")
    
    entry = registry.promote_to_green(model_id)
    logger.info("model_promoted", model_id=entry.model_id)
    return entry.model_id


def list_models(status: str | None = None) -> None:
    """List models in registry."""
    from src.ml.registry import get_registry, ModelStatus
    
    registry = get_registry()
    status_filter = ModelStatus(status) if status else None
    models = registry.list_models(status=status_filter)
    
    print(f"\n{'Model ID':<50} {'Status':<12} {'Val Loss':<12} {'Created':<20}")
    print("-" * 100)
    
    for m in models:
        print(f"{m.model_id:<50} {m.status.value:<12} {m.metrics.val_loss:<12.6f} {m.created_at[:19]}")
    
    print(f"\nTotal: {len(models)} models")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="TFT Training Pipeline")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train TFT model")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--lookback", type=int, default=120)
    train_parser.add_argument("--symbols", nargs="+", help="Symbols to train on")
    train_parser.add_argument("--promote", action="store_true", help="Auto-promote if best")
    train_parser.add_argument("--no-lightning", action="store_true", help="Don't use Lightning")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model-id", required=True)
    export_parser.add_argument("--output", help="Output path")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to green")
    promote_parser.add_argument("--model-id", help="Model ID to promote")
    promote_parser.add_argument("--auto", action="store_true", help="Auto-promote best")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List models in registry")
    list_parser.add_argument("--status", choices=["candidate", "green", "deprecated"])
    
    args = parser.parse_args()
    
    if args.command == "train":
        result = train_tft(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            lookback_window=args.lookback,
            symbols=args.symbols,
            use_lightning=not args.no_lightning,
        )
        print(f"\nTraining complete!")
        print(f"  Model ID: {result['model_id']}")
        print(f"  Val Loss: {result['val_loss']:.6f}")
        print(f"  ONNX: {result['onnx_path']}")
        
        if args.promote:
            promoted = promote_model(auto=True)
            if promoted:
                print(f"  Promoted: {promoted}")
    
    elif args.command == "export":
        path = export_model(args.model_id, args.output)
        print(f"Exported to: {path}")
    
    elif args.command == "promote":
        if args.auto:
            promoted = promote_model(auto=True)
        else:
            promoted = promote_model(model_id=args.model_id)
        if promoted:
            print(f"Promoted: {promoted}")
        else:
            print("No model promoted")
    
    elif args.command == "list":
        list_models(args.status)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
