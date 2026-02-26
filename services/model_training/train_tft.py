"""TFT training pipeline with MLflow tracking and ONNX export.

.. deprecated::
    This training script uses the legacy ``TemporalFusionTransformer``
    from ``src/ml/tft_model.py``.  The canonical training pipeline is
    ``scripts/train_tft_baseline.py`` which trains ``SimpleTFTModel``,
    exports to ONNX, saves ``feature_meta.json``, and registers in
    the model registry.  Use that instead.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src.core.config import load_settings
from src.ml.dataset import TFTDatasetBuilder, TFTSample

logger = structlog.get_logger("tft_trainer")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class TFTTrainer:
    """TFT model trainer with MLflow integration."""

    def __init__(
        self,
        settings: Any = None,
        device: str | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required: pip install torch")
        self.settings = settings or load_settings()
        self.training_config = self.settings.training
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info("trainer_init", device=str(self.device))
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    async def build_dataset_from_db(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training dataset from features_tft table in TimescaleDB.

        Args:
            start_date: Filter start date (ISO format)
            end_date: Filter end date (ISO format)
            symbols: Filter to specific symbols

        Returns:
            (X, y) numpy arrays for training
        """
        import asyncpg
        import os

        from shared.db.dsn import build_dsn
        dsn = os.getenv("DATABASE_URL") or build_dsn()

        conn = await asyncpg.connect(dsn)
        logger.info("connected_to_db", dsn=dsn.split("@")[1])

        # Build query
        query = """
            SELECT ticker, ts, price, volume,
                   returns_1, returns_5, returns_15,
                   volatility_5, volatility_15,
                   sma_5, sma_20, ema_5, ema_20,
                   rsi_14, volume_zscore,
                   minute_of_day, day_of_week
            FROM features_tft
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if start_date:
            query += f" AND ts >= ${param_idx}"
            params.append(datetime.fromisoformat(start_date))
            param_idx += 1

        if end_date:
            query += f" AND ts <= ${param_idx}"
            params.append(datetime.fromisoformat(end_date))
            param_idx += 1

        if symbols:
            query += f" AND ticker = ANY(${param_idx})"
            params.append(symbols)
            param_idx += 1

        query += " ORDER BY ticker, ts"

        rows = await conn.fetch(query, *params)
        await conn.close()

        logger.info("loaded_features_from_db", rows=len(rows))

        if not rows:
            raise ValueError("No features found in database")

        # Group by symbol
        symbol_data: dict[str, list[dict]] = {}
        for row in rows:
            ticker = row["ticker"]
            if ticker not in symbol_data:
                symbol_data[ticker] = []
            symbol_data[ticker].append({
                "ts": row["ts"],
                "price": float(row["price"]),
                "volume": float(row["volume"]),
                "returns_1": float(row["returns_1"] or 0),
                "returns_5": float(row["returns_5"] or 0),
                "returns_15": float(row["returns_15"] or 0),
                "volatility_5": float(row["volatility_5"] or 0),
                "volatility_15": float(row["volatility_15"] or 0),
                "sma_5": float(row["sma_5"] or row["price"]),
                "sma_20": float(row["sma_20"] or row["price"]),
                "ema_5": float(row["ema_5"] or row["price"]),
                "ema_20": float(row["ema_20"] or row["price"]),
                "rsi_14": float(row["rsi_14"] or 50),
                "volume_zscore": float(row["volume_zscore"] or 0),
                "minute_of_day": int(row["minute_of_day"] or 0),
                "day_of_week": int(row["day_of_week"] or 0),
            })

        builder = TFTDatasetBuilder(
            lookback_window=self.training_config.lookback_window,
            horizons=self.training_config.horizons,
        )
        samples = list(builder.build_sequences(symbol_data))
        logger.info("dataset_built_from_db", samples=len(samples))

        if not samples:
            raise ValueError("No training samples generated from DB features")

        X, y, _ = builder.to_numpy(samples)
        return X, y

    def build_dataset_from_backfill(
        self,
        cache_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training dataset from backfilled data.

        This loads data from TimescaleDB audit trail or cached npz file.
        """
        if cache_path and Path(cache_path).exists():
            logger.info("loading_cached_dataset", path=cache_path)
            X, y, _ = TFTDatasetBuilder.load_dataset(cache_path)
            return X, y
        logger.info("building_dataset_from_polygon")
        from src.backfill.polygon_loader import PolygonCSVLoader
        data_config = self.settings.data
        data_dir = f"{data_config.data_root}/{data_config.polygon_minute_dir}"
        loader = PolygonCSVLoader(
            data_dir=data_dir,
            symbols=self.settings.app.symbols,
            start_date=datetime.fromisoformat(data_config.start_date) if data_config.start_date else None,
            end_date=datetime.fromisoformat(data_config.end_date) if data_config.end_date else None,
        )
        symbol_data: dict[str, list[dict]] = {}
        for bar in loader.load_all():
            if bar.ticker not in symbol_data:
                symbol_data[bar.ticker] = []
            vwap = bar.vwap if bar.vwap else (bar.open + bar.high + bar.low + bar.close) / 4
            symbol_data[bar.ticker].append({
                "ts": bar.timestamp,
                "price": bar.close,
                "volume": bar.volume,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "returns_1": 0.0,
                "returns_5": 0.0,
                "returns_15": 0.0,
                "volatility_5": 0.0,
                "volatility_15": 0.0,
                "sma_5": bar.close,
                "sma_20": bar.close,
                "ema_5": bar.close,
                "ema_20": bar.close,
                "rsi_14": 50.0,
                "volume_zscore": 0.0,
                "minute_of_day": bar.timestamp.hour * 60 + bar.timestamp.minute,
                "day_of_week": bar.timestamp.weekday(),
            })
        for symbol in symbol_data:
            symbol_data[symbol].sort(key=lambda x: x["ts"])
            self._compute_features(symbol_data[symbol])
        builder = TFTDatasetBuilder(
            lookback_window=self.training_config.lookback_window,
            horizons=self.training_config.horizons,
        )
        samples = list(builder.build_sequences(symbol_data))
        logger.info("dataset_built", samples=len(samples))
        if not samples:
            raise ValueError("No training samples generated")
        X, y, _ = builder.to_numpy(samples)
        if cache_path:
            builder.save_dataset(samples, cache_path)
        return X, y

    def _compute_features(self, records: list[dict]) -> None:
        """Compute rolling features for records (in-place)."""
        prices = []
        volumes = []
        for i, rec in enumerate(records):
            prices.append(rec["price"])
            volumes.append(rec["volume"])
            if i > 0:
                rec["returns_1"] = (prices[-1] - prices[-2]) / max(prices[-2], 1e-8)
            if i >= 5:
                rec["returns_5"] = (prices[-1] - prices[-6]) / max(prices[-6], 1e-8)
                rec["volatility_5"] = np.std(prices[-5:]) / np.mean(prices[-5:]) if np.mean(prices[-5:]) > 0 else 0
                rec["sma_5"] = np.mean(prices[-5:])
                alpha5 = 2 / 6
                ema5 = prices[-5]
                for p in prices[-4:]:
                    ema5 = alpha5 * p + (1 - alpha5) * ema5
                rec["ema_5"] = ema5
            if i >= 15:
                rec["returns_15"] = (prices[-1] - prices[-16]) / max(prices[-16], 1e-8)
                rec["volatility_15"] = np.std(prices[-15:]) / np.mean(prices[-15:]) if np.mean(prices[-15:]) > 0 else 0
            if i >= 20:
                rec["sma_20"] = np.mean(prices[-20:])
                alpha20 = 2 / 21
                ema20 = prices[-20]
                for p in prices[-19:]:
                    ema20 = alpha20 * p + (1 - alpha20) * ema20
                rec["ema_20"] = ema20
                vol_mean = np.mean(volumes[-20:])
                vol_std = np.std(volumes[-20:])
                rec["volume_zscore"] = (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0
            if i >= 14:
                deltas = np.diff(prices[-(15):])
                gains = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0
                losses = np.mean([-d for d in deltas if d < 0]) if any(d < 0 for d in deltas) else 0
                if losses > 0:
                    rs = gains / losses
                    rec["rsi_14"] = 100 - (100 / (1 + rs))
                else:
                    rec["rsi_14"] = 100 if gains > 0 else 50

    def create_model(self) -> nn.Module:
        """Create TFT model instance."""
        from src.ml.tft_model import TemporalFusionTransformer
        model = TemporalFusionTransformer(
            input_dim=15,
            hidden_dim=self.training_config.hidden_size,
            num_heads=self.training_config.num_attention_heads,
            num_layers=2,
            num_horizons=len(self.training_config.horizons),
            num_quantiles=3,
            dropout=self.training_config.dropout,
        )
        return model.to(self.device)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
    ) -> dict[str, Any]:
        """Train the TFT model.

        Args:
            X: Input features (batch, seq_len, features)
            y: Targets (batch, num_horizons)
            validation_split: Fraction for validation

        Returns:
            Training metrics dict
        """
        from src.ml.tft_model import QuantileLoss

        logger.info("training_start", X_shape=X.shape, y_shape=y.shape)
        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
        )
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.training_config.mlflow_experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                "lookback_window": self.training_config.lookback_window,
                "horizons": str(self.training_config.horizons),
                "hidden_size": self.training_config.hidden_size,
                "learning_rate": self.training_config.learning_rate,
                "batch_size": self.training_config.batch_size,
                "epochs": self.training_config.epochs,
            })
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.training_config.epochs):
            self.model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output["quantiles"], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = criterion(val_output["quantiles"], y_val_t).item()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, step=epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
            if epoch % 10 == 0:
                logger.info(
                    "epoch_complete",
                    epoch=epoch,
                    train_loss=f"{train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                )

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.end_run()
        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "epochs": self.training_config.epochs,
        }

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        output_dir = Path(self.training_config.model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info("checkpoint_saved", path=str(path))

    def export_onnx(self, output_path: str | None = None) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Output path (uses config default if None)

        Returns:
            Path to exported ONNX file
        """
        if self.model is None:
            raise ValueError("No model to export. Train first.")
        output_path = output_path or self.training_config.onnx_model_path
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        dummy_input = torch.randn(
            1,
            self.training_config.lookback_window,
            15,
            device=self.device,
        )

        class ONNXWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.model(x)
                batch_size = x.size(0)
                quantiles = out["quantiles"].reshape(batch_size, -1)
                confidence = out["confidence"].unsqueeze(-1)
                return torch.cat([quantiles, confidence], dim=-1)

        wrapper = ONNXWrapper(self.model)
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=["features"],
            output_names=["predictions"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "predictions": {0: "batch_size"},
            },
            opset_version=14,
        )
        logger.info("onnx_exported", path=output_path)
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
                mlflow.set_experiment(self.training_config.mlflow_experiment_name)
                with mlflow.start_run():
                    mlflow.log_artifact(output_path, "onnx_model")
            except Exception as e:
                logger.warning("mlflow_artifact_upload_failed", error=str(e))
        return output_path


def main() -> None:
    """CLI entrypoint for TFT training."""
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument(
        "--config",
        default="config/app.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--cache",
        default=".tmp/tft_dataset.npz",
        help="Dataset cache path",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing model to ONNX",
    )
    args = parser.parse_args()
    settings = load_settings(os.path.dirname(args.config))
    trainer = TFTTrainer(settings=settings, device=args.device)
    if args.export_only:
        checkpoint_path = Path(settings.training.model_output_dir) / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        trainer.model = trainer.create_model()
        trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device))
        onnx_path = trainer.export_onnx()
        print(f"ONNX model exported to: {onnx_path}")
        return
    X, y = trainer.build_dataset_from_backfill(cache_path=args.cache)
    metrics = trainer.train(X, y)
    print(f"Training complete: {metrics}")
    onnx_path = trainer.export_onnx()
    print(f"ONNX model exported to: {onnx_path}")


if __name__ == "__main__":
    main()
