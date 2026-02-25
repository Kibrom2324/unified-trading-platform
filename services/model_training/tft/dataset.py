"""TFT Dataset module - queries TimescaleDB and builds time-series sequences.

Loads features from features_tft table, groups by symbol, creates
sliding window sequences for TFT training with proper train/val splits.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np
import structlog

logger = structlog.get_logger("tft_dataset")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for TFT dataset."""
    lookback_window: int = 120
    horizons: list[int] = field(default_factory=lambda: [30, 45, 60])
    train_split: float = 0.8
    normalize: bool = True
    min_samples_per_symbol: int = 500
    batch_size: int = 64
    num_workers: int = 4


FEATURE_COLUMNS = [
    "price", "volume", "returns_1", "returns_5", "returns_15",
    "volatility_5", "volatility_15", "sma_5", "sma_20",
    "ema_5", "ema_20", "rsi_14", "volume_zscore",
    "minute_of_day", "day_of_week",
]


@dataclass
class NormStats:
    """Normalization statistics for features."""
    means: np.ndarray
    stds: np.ndarray
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return (data - self.means) / self.stds
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return data * self.stds + self.means
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormStats:
        return cls(
            means=np.array(data["means"], dtype=np.float32),
            stds=np.array(data["stds"], dtype=np.float32),
        )


def query_features_from_db(
    db_url: str | None = None,
    symbols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    split: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Query features_tft table from TimescaleDB.
    
    Args:
        db_url: Database connection URL
        symbols: List of symbols to query (None for all)
        start_date: Start timestamp filter
        end_date: End timestamp filter
        split: Filter by split column ('train', 'val', 'test')
        
    Returns:
        Dict mapping symbol to list of feature records, sorted by timestamp
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 required: pip install psycopg2-binary")
    
    db_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/awet")
    
    # Build query
    query = f"""
        SELECT 
            ticker,
            ts,
            {', '.join(FEATURE_COLUMNS)}
        FROM features_tft
        WHERE 1=1
    """
    params: list[Any] = []
    
    if symbols:
        query += " AND ticker = ANY(%s)"
        params.append(symbols)
    if start_date:
        query += " AND ts >= %s"
        params.append(start_date)
    if end_date:
        query += " AND ts <= %s"
        params.append(end_date)
    if split:
        query += " AND split = %s"
        params.append(split)
    
    query += " ORDER BY ticker, ts ASC"
    
    logger.info("querying_features", symbols=symbols, start_date=start_date, end_date=end_date)
    
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()
    
    # Group by symbol
    symbol_data: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        symbol = row["ticker"]
        if symbol not in symbol_data:
            symbol_data[symbol] = []
        symbol_data[symbol].append(dict(row))
    
    logger.info(
        "features_loaded",
        total_rows=len(rows),
        symbols=list(symbol_data.keys()),
        rows_per_symbol={s: len(r) for s, r in symbol_data.items()},
    )
    
    return symbol_data


def compute_norm_stats(symbol_data: dict[str, list[dict]]) -> NormStats:
    """Compute normalization statistics across all data."""
    all_features = []
    for records in symbol_data.values():
        for rec in records:
            row = [float(rec.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]
            all_features.append(row)
    
    arr = np.array(all_features, dtype=np.float32)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)  # Prevent div by zero
    
    return NormStats(means=means, stds=stds)


def build_sequences(
    symbol_data: dict[str, list[dict]],
    config: DatasetConfig,
    norm_stats: NormStats | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build training sequences from symbol data.
    
    Args:
        symbol_data: Dict mapping symbol to list of feature dicts
        config: Dataset configuration
        norm_stats: Optional normalization stats (computed if None)
        
    Returns:
        (X, y, sample_ids, symbols) - features, targets, IDs, and symbol for each sample
    """
    max_horizon = max(config.horizons)
    min_length = config.lookback_window + max_horizon
    
    X_list = []
    y_list = []
    sample_ids = []
    sample_symbols = []
    
    if norm_stats is None and config.normalize:
        norm_stats = compute_norm_stats(symbol_data)
    
    for symbol, records in symbol_data.items():
        if len(records) < min_length:
            logger.warning(
                "insufficient_data",
                symbol=symbol,
                records=len(records),
                required=min_length,
            )
            continue
        
        # Extract features and prices
        features_list = []
        prices = []
        timestamps = []
        
        for rec in records:
            row = [float(rec.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]
            features_list.append(row)
            prices.append(float(rec.get("price", rec.get("close", 0.0)) or 0.0))
            timestamps.append(rec.get("ts"))
        
        features_array = np.array(features_list, dtype=np.float32)
        prices_array = np.array(prices, dtype=np.float32)
        
        # Normalize
        if config.normalize and norm_stats:
            features_array = norm_stats.normalize(features_array)
        
        # Build sequences
        for i in range(config.lookback_window, len(records) - max_horizon):
            input_features = features_array[i - config.lookback_window:i]
            current_price = prices_array[i - 1]
            
            # Compute target returns for each horizon
            targets = []
            for h in config.horizons:
                future_price = prices_array[i + h - 1]
                ret = (future_price - current_price) / max(current_price, 1e-8)
                targets.append(ret)
            
            ts = timestamps[i - 1]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            
            sample_id = hashlib.sha256(
                f"{symbol}:{ts.isoformat()}".encode()
            ).hexdigest()[:16]
            
            X_list.append(input_features)
            y_list.append(targets)
            sample_ids.append(sample_id)
            sample_symbols.append(symbol)
    
    X = np.stack(X_list) if X_list else np.empty((0, config.lookback_window, len(FEATURE_COLUMNS)))
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0, len(config.horizons)))
    
    logger.info(
        "sequences_built",
        total_samples=len(X),
        X_shape=X.shape,
        y_shape=y.shape,
        unique_symbols=len(set(sample_symbols)),
    )
    
    return X, y, sample_ids, sample_symbols


class TFTDataset(Dataset):
    """PyTorch Dataset for TFT training."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: list[str] | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required: pip install torch")
        
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.sample_ids = sample_ids or [str(i) for i in range(len(X))]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TFTDataModule:
    """Data module for TFT training with TimescaleDB backend."""
    
    def __init__(
        self,
        config: DatasetConfig | None = None,
        db_url: str | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self.db_url = db_url
        self.symbols = symbols
        
        self.norm_stats: NormStats | None = None
        self.train_dataset: TFTDataset | None = None
        self.val_dataset: TFTDataset | None = None
        self._prepared = False
    
    def prepare_data(self) -> None:
        """Load and prepare data from TimescaleDB."""
        if self._prepared:
            return
        
        # Query features from DB
        symbol_data = query_features_from_db(
            db_url=self.db_url,
            symbols=self.symbols,
        )
        
        if not symbol_data:
            raise ValueError("No data found in features_tft table")
        
        # Compute normalization stats on all data
        self.norm_stats = compute_norm_stats(symbol_data)
        
        # Build sequences
        X, y, sample_ids, sample_symbols = build_sequences(
            symbol_data,
            self.config,
            self.norm_stats,
        )
        
        if len(X) < 100:
            raise ValueError(f"Insufficient samples: {len(X)}, need at least 100")
        
        # Train/val split (stratified by symbol would be better, but simple random for now)
        n_train = int(len(X) * self.config.train_split)
        indices = np.random.permutation(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        self.train_dataset = TFTDataset(
            X[train_idx],
            y[train_idx],
            [sample_ids[i] for i in train_idx],
        )
        self.val_dataset = TFTDataset(
            X[val_idx],
            y[val_idx],
            [sample_ids[i] for i in val_idx],
        )
        
        self._prepared = True
        
        logger.info(
            "data_prepared",
            train_samples=len(self.train_dataset),
            val_samples=len(self.val_dataset),
            feature_dim=X.shape[-1],
            lookback=self.config.lookback_window,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if not self._prepared:
            self.prepare_data()
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if not self._prepared:
            self.prepare_data()
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
    
    def get_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/val splits as numpy arrays.
        
        Returns:
            (X_train, y_train, X_val, y_val)
        """
        if not self._prepared:
            self.prepare_data()
        
        return (
            self.train_dataset.X.numpy(),
            self.train_dataset.y.numpy(),
            self.val_dataset.X.numpy(),
            self.val_dataset.y.numpy(),
        )


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    output_path: str,
    norm_stats: NormStats | None = None,
    config: DatasetConfig | None = None,
) -> None:
    """Save dataset to NPZ file for caching."""
    import json
    
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        sample_ids=np.array(sample_ids),
        norm_stats_means=norm_stats.means if norm_stats else None,
        norm_stats_stds=norm_stats.stds if norm_stats else None,
        config=json.dumps(config.__dict__) if config else None,
    )
    logger.info("dataset_saved", path=output_path, samples=len(X))


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str], NormStats | None]:
    """Load dataset from NPZ file."""
    import json
    
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    sample_ids = data["sample_ids"].tolist()
    
    norm_stats = None
    if data["norm_stats_means"] is not None:
        norm_stats = NormStats(
            means=data["norm_stats_means"],
            stds=data["norm_stats_stds"],
        )
    
    logger.info("dataset_loaded", path=path, samples=len(X))
    return X, y, sample_ids, norm_stats
