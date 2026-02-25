"""TFT Dataset builder for training.

Converts engineered market features into supervised sequences
for Temporal Fusion Transformer training.
"""

from __future__ import annotations

import json

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import structlog

logger = structlog.get_logger("tft_dataset")


@dataclass
class TFTSample:
    """Single training sample for TFT."""
    symbol: str
    timestamp: datetime
    features: np.ndarray
    targets: np.ndarray
    sample_id: str

    @property
    def input_length(self) -> int:
        return self.features.shape[0]


class TFTDatasetBuilder:
    """Build TFT training dataset from engineered features."""

    FEATURE_COLUMNS = [
        "price", "volume", "returns_1", "returns_5", "returns_15",
        "volatility_5", "volatility_15", "sma_5", "sma_20",
        "ema_5", "ema_20", "rsi_14", "volume_zscore",
        "minute_of_day", "day_of_week",
    ]

    def __init__(
        self,
        lookback_window: int = 120,
        horizons: list[int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.lookback_window = lookback_window
        self.horizons = horizons or [30, 45, 60]
        self.normalize = normalize
        # Per-column normalization stats: {col_index: (mean, std)}
        # When fitted, these are used for all subsequent normalize calls
        # to ensure training and inference use identical statistics.
        self._feature_stats: dict[int, tuple[float, float]] = {}
        self._fitted: bool = False

    def _compute_stats(self, data: np.ndarray) -> tuple[float, float]:
        """Compute mean and std for normalization."""
        mean = float(np.nanmean(data))
        std = float(np.nanstd(data))
        return mean, std if std > 1e-8 else 1.0

    def fit_normalization(self, all_features: np.ndarray) -> None:
        """Compute and store per-column normalization stats from training data.

        Args:
            all_features: 2D array of shape (total_rows, num_features)
                          representing the full in-sample training set.
        """
        self._feature_stats = {}
        for i in range(all_features.shape[1]):
            mean, std = self._compute_stats(all_features[:, i])
            self._feature_stats[i] = (mean, std)
        self._fitted = True
        logger.info(
            "normalization_fitted",
            num_features=all_features.shape[1],
            num_rows=all_features.shape[0],
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize features.

        Uses pre-fitted per-column stats when available (CF-4 fix).
        Falls back to per-sample stats for backward compatibility.
        """
        if not self.normalize:
            return features
        normalized = np.zeros_like(features)
        for i in range(features.shape[1]):
            col = features[:, i]
            if self._fitted and i in self._feature_stats:
                mean, std = self._feature_stats[i]
            else:
                mean, std = self._compute_stats(col)
            normalized[:, i] = (col - mean) / std
        return normalized

    def save_norm_stats(self, path: str | Path) -> None:
        """Save normalization stats as JSON sidecar.

        Args:
            path: Path to write norm_stats.json
        """
        if not self._fitted:
            logger.warning("save_norm_stats_called_without_fit")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "means": [self._feature_stats[i][0] for i in sorted(self._feature_stats)],
            "stds": [self._feature_stats[i][1] for i in sorted(self._feature_stats)],
            "feature_columns": list(self.FEATURE_COLUMNS),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("norm_stats_saved", path=str(path), features=len(self._feature_stats))

    @classmethod
    def load_norm_stats(cls, path: str | Path) -> dict[int, tuple[float, float]]:
        """Load normalization stats from JSON sidecar.

        Args:
            path: Path to norm_stats.json

        Returns:
            Dict mapping column index to (mean, std) tuple
        """
        with open(path) as f:
            data = json.load(f)
        stats: dict[int, tuple[float, float]] = {}
        means = data["means"]
        stds = data["stds"]
        for i in range(len(means)):
            stats[i] = (means[i], stds[i])
        return stats

    def build_sequences(
        self,
        symbol_data: dict[str, list[dict]],
    ) -> Iterator[TFTSample]:
        """Build training sequences from symbol data.

        Args:
            symbol_data: Dict mapping symbol to list of feature dicts,
                         sorted by timestamp ascending.

        Yields:
            TFTSample objects for training.
        """
        max_horizon = max(self.horizons)
        min_length = self.lookback_window + max_horizon

        for symbol, records in symbol_data.items():
            if len(records) < min_length:
                logger.warning(
                    "insufficient_data",
                    symbol=symbol,
                    records=len(records),
                    required=min_length,
                )
                continue
            features_list = []
            prices = []
            timestamps = []

            for rec in records:
                row = []
                for col in self.FEATURE_COLUMNS:
                    val = rec.get(col, 0.0)
                    row.append(float(val) if val is not None else 0.0)
                features_list.append(row)
                prices.append(rec.get("price", rec.get("close", 0.0)))
                timestamps.append(rec.get("ts", rec.get("timestamp")))

            features_array = np.array(features_list, dtype=np.float32)
            prices_array = np.array(prices, dtype=np.float32)
            for i in range(self.lookback_window, len(records) - max_horizon):
                input_features = features_array[i - self.lookback_window:i]
                input_features = self._normalize_features(input_features)
                current_price = prices_array[i - 1]
                targets = []
                for h in self.horizons:
                    future_price = prices_array[i + h - 1]
                    ret = (future_price - current_price) / max(current_price, 1e-8)
                    targets.append(ret)
                targets_array = np.array(targets, dtype=np.float32)
                ts = timestamps[i - 1]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                sample_id = hashlib.sha256(
                    f"{symbol}:{ts.isoformat()}".encode()
                ).hexdigest()[:16]

                yield TFTSample(
                    symbol=symbol,
                    timestamp=ts,
                    features=input_features,
                    targets=targets_array,
                    sample_id=sample_id,
                )

    def to_numpy(
        self,
        samples: list[TFTSample],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Convert samples to numpy arrays for training.

        Returns:
            (X, y, sample_ids) tuple
        """
        X = np.stack([s.features for s in samples])
        y = np.stack([s.targets for s in samples])
        ids = [s.sample_id for s in samples]
        return X, y, ids

    def save_dataset(
        self,
        samples: list[TFTSample],
        output_path: str | Path,
    ) -> None:
        """Save dataset to npz file.

        Also saves norm_stats.json sidecar if normalization has been fitted.
        """
        X, y, ids = self.to_numpy(samples)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            sample_ids=np.array(ids),
            lookback_window=self.lookback_window,
            horizons=np.array(self.horizons),
        )
        # CF-4: auto-save norm stats sidecar alongside dataset
        if self._fitted:
            norm_path = output_path.parent / "norm_stats.json"
            self.save_norm_stats(norm_path)
        logger.info(
            "dataset_saved",
            path=str(output_path),
            samples=len(samples),
            X_shape=X.shape,
            y_shape=y.shape,
        )

    @staticmethod
    def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from npz file."""
        data = np.load(path, allow_pickle=True)
        return data["X"], data["y"], data["sample_ids"]
