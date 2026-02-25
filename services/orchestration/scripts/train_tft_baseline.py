#!/usr/bin/env python3
"""
Train TFT baseline model from features_tft table.

Reads from: features_tft (TimescaleDB)
Outputs:    models/tft/<run_id>/model.onnx
            models/registry.json (updated)

SINGLE SOURCE OF TRUTH: config/universe.csv
  - Training uses symbols from universe.csv if it exists
  - Override with --symbols flag

Usage:
    python scripts/train_tft_baseline.py                   # Train on universe.csv symbols
    python scripts/train_tft_baseline.py --max-epochs 50
    python scripts/train_tft_baseline.py --hidden-size 32 --attention-heads 2
    python scripts/train_tft_baseline.py --dry-run         # Validate data + show ticker count
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = structlog.get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://awet:awet@localhost:5433/awet"
)

REQUIRE_REDDIT = os.getenv("REQUIRE_REDDIT", "0") == "1"

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "tft"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
UNIVERSE_FILE = PROJECT_ROOT / "config" / "universe.csv"

# Default hyperparameters
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_ATTENTION_HEADS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEQUENCE_LENGTH = 60  # 60 minutes of history
DEFAULT_HORIZON = 30  # minutes ahead (matches build_features default)

# Guardrail: minimum number of tickers for production training
MIN_TICKERS_WARNING = 50


FEATURE_CATALOG = {
    "price/volume": [
        ("price", "features_tft.price", "build_features.compute_features: closes[i]"),
        ("open", "features_tft.open", "build_features.compute_features: opens[i]"),
        ("high", "features_tft.high", "build_features.compute_features: highs[i]"),
        ("low", "features_tft.low", "build_features.compute_features: lows[i]"),
        ("close", "features_tft.close", "build_features.compute_features: closes[i]"),
        ("volume", "features_tft.volume", "build_features.compute_features: volumes[i]"),
        ("returns_1", "features_tft.returns_1", "build_features.compute_features: log_returns_1"),
        ("returns_5", "features_tft.returns_5", "build_features.compute_features: log_returns_5"),
        ("returns_15", "features_tft.returns_15", "build_features.compute_features: log_returns_15"),
        ("volatility_5", "features_tft.volatility_5", "build_features.compute_features: rolling std"),
        ("volatility_15", "features_tft.volatility_15", "build_features.compute_features: rolling std"),
    ],
    "technical indicators": [
        ("sma_5", "features_tft.sma_5", "build_features.compute_features: sma_5"),
        ("sma_20", "features_tft.sma_20", "build_features.compute_features: sma_20"),
        ("ema_5", "features_tft.ema_5", "build_features.compute_features: ema_5"),
        ("ema_20", "features_tft.ema_20", "build_features.compute_features: ema_20"),
        ("rsi_14", "features_tft.rsi_14", "build_features.compute_features: rsi_14"),
        ("volume_zscore", "features_tft.volume_zscore", "build_features.compute_features: vol_zscore"),
    ],
    "calendar/time": [
        ("minute_of_day", "features_tft.minute_of_day", "build_features.compute_features: ts.hour*60+ts.minute"),
        ("hour_of_day", "features_tft.hour_of_day", "build_features.compute_features: ts.hour"),
        ("day_of_week", "features_tft.day_of_week", "build_features.compute_features: ts.weekday()"),
    ],
    "reddit": [
        ("reddit_mentions_count", "features_tft.reddit_mentions_count", "reddit_features: mentions_count"),
        ("reddit_sentiment_mean", "features_tft.reddit_sentiment_mean", "reddit_features: sentiment_mean"),
        ("reddit_sentiment_weighted", "features_tft.reddit_sentiment_weighted", "reddit_features: sentiment_weighted"),
        ("reddit_positive_ratio", "features_tft.reddit_positive_ratio", "reddit_features: positive_ratio"),
        ("reddit_negative_ratio", "features_tft.reddit_negative_ratio", "reddit_features: negative_ratio"),
    ],
}

REQUIRED_REDDIT_COLUMNS = [
    "reddit_mentions_count",
    "reddit_sentiment_mean",
    "reddit_sentiment_weighted",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
]


def print_feature_catalog(include_reddit: bool) -> None:
    print("Feature catalog (final model input):")
    for group, items in FEATURE_CATALOG.items():
        if group == "reddit" and not include_reddit:
            continue
        print(f"  {group}:")
        for name, column, source in items:
            print(f"    - {name} | {column} | {source}")
    print()


def load_universe_symbols() -> list[str] | None:
    """Load symbols from universe.csv if it exists."""
    if not UNIVERSE_FILE.exists():
        return None

    symbols = []
    with open(UNIVERSE_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ticker" in row:
                symbols.append(row["ticker"])

    return symbols if symbols else None


async def column_exists(conn: asyncpg.Connection, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
        )
        """,
        table_name,
        column_name,
    )
    return bool(exists)


async def get_available_columns(conn: asyncpg.Connection, table_name: str) -> set[str]:
    """Get all available columns in a table."""
    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        """,
        table_name,
    )
    return {r["column_name"] for r in rows}


# ============================================================================
# EXPLICIT FEATURE SPECIFICATION (TFT-safe, no leakage)
# ============================================================================
# IMPORTANT: target_return is NOT included - it's the prediction target!
# These are the input features fed to the model.

DESIRED_FEATURE_COLUMNS = [
    # Core price/volume features
    "price",
    "volume",
    "returns_1",
    "returns_5",
    "returns_15",
    "volatility_5",
    "volatility_15",
    # Technical indicators
    "sma_5",
    "sma_20",
    "ema_5",
    "ema_20",
    "rsi_14",
    "volume_zscore",
    # Time features
    "minute_of_day",
    "hour_of_day",
    "day_of_week",
    # Calendar features
    "is_weekend",
    "day_of_month",
    "week_of_year",
    "month_of_year",
    "is_month_end",
    # Reddit current features
    "reddit_mentions_count",
    "reddit_sentiment_mean",
    "reddit_sentiment_weighted",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
    # Reddit lag features (lag 1, 3, 5 days)
    "reddit_mentions_lag1",
    "reddit_mentions_lag3",
    "reddit_mentions_lag5",
    "reddit_sentiment_mean_lag1",
    "reddit_sentiment_mean_lag3",
    "reddit_sentiment_mean_lag5",
    "reddit_sentiment_weighted_lag1",
    "reddit_sentiment_weighted_lag3",
    "reddit_sentiment_weighted_lag5",
    "reddit_positive_ratio_lag1",
    "reddit_positive_ratio_lag3",
    "reddit_positive_ratio_lag5",
    "reddit_negative_ratio_lag1",
    "reddit_negative_ratio_lag3",
    "reddit_negative_ratio_lag5",
    # Market benchmark context
    "market_return_1",
    "market_volatility_5",
]

# Columns that must NEVER be input features (leakage risk)
FORBIDDEN_FEATURES = {
    "target_return",  # This is the prediction target!
    "ticker",         # Identifier, not a feature
    "ts",             # Timestamp, not a numeric feature
    "time_idx",       # Internal indexing
    "split",          # Train/val split marker
    "idempotency_key",
    "created_at",
}


async def select_available_features(
    conn: asyncpg.Connection,
    desired: list[str],
    table: str = "features_tft",
) -> tuple[list[str], list[str]]:
    """Select features that exist in DB, return (available, missing)."""
    available_cols = await get_available_columns(conn, table)
    
    selected = []
    missing = []
    
    for col in desired:
        if col in FORBIDDEN_FEATURES:
            print(f"  ⚠️  Skipping forbidden feature: {col}")
            continue
        if col in available_cols:
            selected.append(col)
        else:
            missing.append(col)
    
    return selected, missing


# Quantile settings — must match ONNXInferenceEngine expectations
NUM_HORIZONS = 3          # 30, 45, 60 min
NUM_QUANTILES = 3         # q10, q50, q90
QUANTILE_VALUES = [0.1, 0.5, 0.9]
HORIZON_SCALE = [1.0, 1.225, 1.414]  # sqrt(30/30), sqrt(45/30), sqrt(60/30)
NUM_OUTPUTS = NUM_HORIZONS * NUM_QUANTILES + 1  # 10 total


class SimpleTFTModel(nn.Module):
    """Simplified Temporal Fusion Transformer for time series prediction.

    This is a baseline model that captures the key TFT concepts:
    - Variable selection network
    - LSTM encoder for temporal dependencies
    - Self-attention for long-range patterns
    - Gated residual connections

    Outputs multi-horizon quantile predictions compatible with
    ONNXInferenceEngine (3 horizons × 3 quantiles + 1 confidence = 10).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        attention_heads: int = DEFAULT_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        num_horizons: int = NUM_HORIZONS,
        num_quantiles: int = NUM_QUANTILES,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles
        self.num_outputs = num_horizons * num_quantiles + 1

        # Variable selection (simplified)
        self.variable_selection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        # Temporal encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # Quantile output heads — one per horizon
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_quantiles)
            for _ in range(num_horizons)
        ])

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_size]

        Returns:
            predictions: [batch, num_outputs]
                Layout: [h30_q10, h30_q50, h30_q90, h45_q10, ..., h60_q90, confidence]
        """
        # Variable selection
        x = self.variable_selection(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Self-attention on last hidden states
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Gated residual
        grn_out = self.grn(attn_out[:, -1, :])  # Take last timestep
        gate = self.gate(attn_out[:, -1, :])
        hidden = gate * grn_out + (1 - gate) * lstm_out[:, -1, :]

        # Quantile predictions for each horizon
        quantiles = [head(hidden) for head in self.quantile_heads]
        flat_quantiles = torch.cat(quantiles, dim=-1)  # [batch, num_horizons * num_quantiles]

        # Confidence
        confidence = self.confidence_head(hidden)  # [batch, 1]

        return torch.cat([flat_quantiles, confidence], dim=-1)  # [batch, num_outputs]


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression.

    Supports multi-horizon training from a single target by scaling
    predictions across horizons using sqrt-time ratio.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        horizon_scales: list[float] | None = None,
    ):
        super().__init__()
        self.quantiles = quantiles or QUANTILE_VALUES
        self.horizon_scales = horizon_scales or HORIZON_SCALE
        self.num_quantiles = len(self.quantiles)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            predictions: [batch, num_horizons * num_quantiles + 1]
            targets: [batch, 1] — single-horizon target (horizon=30min)

        Returns:
            Scalar loss.
        """
        # Strip confidence column
        quantile_preds = predictions[:, :-1]  # [batch, num_horizons * num_quantiles]
        num_horizons = len(self.horizon_scales)
        quantile_preds = quantile_preds.view(-1, num_horizons, self.num_quantiles)

        losses = []
        for h_idx, scale in enumerate(self.horizon_scales):
            # Scale target for longer horizons: E[r_t] ∝ sqrt(t)
            scaled_target = targets * scale  # [batch, 1]
            for q_idx, q in enumerate(self.quantiles):
                pred_q = quantile_preds[:, h_idx, q_idx].unsqueeze(-1)  # [batch, 1]
                errors = scaled_target - pred_q
                loss_q = torch.max(q * errors, (q - 1) * errors)
                losses.append(loss_q.mean())

        return torch.stack(losses).mean()


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series training.

    Args:
        features: [N, num_features]
        targets: [N]
        seq_length: Number of timesteps per sequence

    Returns:
        X: [N-seq_length, seq_length, num_features]
        y: [N-seq_length, 1]
    """
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y).reshape(-1, 1)


async def load_data(
    conn: asyncpg.Connection,
    split: str,
    feature_columns: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features from database.

    Feature columns are explicitly provided to guarantee deterministic training.
    Optional date filters to focus on specific time ranges.
    """
    from datetime import datetime
    
    feature_list = ", ".join(feature_columns)
    
    # Build date filter clause
    date_clauses = []
    params: list = [split]
    param_idx = 2
    
    if start_date:
        date_clauses.append(f"AND ts >= ${param_idx}")
        params.append(datetime.fromisoformat(start_date))
        param_idx += 1
    
    if end_date:
        date_clauses.append(f"AND ts < ${param_idx}")
        params.append(datetime.fromisoformat(end_date))
        param_idx += 1
    
    date_filter = " ".join(date_clauses)
    
    query = f"""
        SELECT {feature_list}, target_return
        FROM features_tft
        WHERE split = $1
          AND target_return IS NOT NULL
          AND returns_1 IS NOT NULL
          AND volatility_5 IS NOT NULL
          {date_filter}
        ORDER BY ticker, time_idx
    """

    rows = await conn.fetch(query, *params)

    if not rows:
        return np.array([]), np.array([])

    # Convert to numpy
    features = []
    targets = []

    for row in rows:
        feat = []
        for col in feature_columns:
            value = row[col]
            if col == "rsi_14":
                feat.append(float(value) if value is not None else 50.0)
            elif col == "minute_of_day":
                feat.append(float(value) / 390.0 if value is not None else 0.0)
            elif col == "hour_of_day":
                feat.append(float(value) / 23.0 if value is not None else 0.0)
            elif col == "day_of_week":
                feat.append(float(value) / 6.0 if value is not None else 0.0)
            else:
                feat.append(float(value) if value is not None else 0.0)
        features.append(feat)
        targets.append(float(row["target_return"]))

    return np.array(features), np.array(targets)


def normalize_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Z-score normalize features using training statistics."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero

    train_norm = (train_features - mean) / std
    val_norm = (val_features - mean) / std

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return train_norm, val_norm, stats


def train_model(
    model: SimpleTFTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict:
    """Train the TFT model with quantile loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = QuantileLoss()

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    patience_counter = 0
    patience_limit = 10

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Logging
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"  Epoch {epoch + 1:3d}/{max_epochs}: "
              f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1],
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }


def export_onnx(
    model: SimpleTFTModel,
    seq_length: int,
    save_path: Path,
) -> None:
    """Export model to ONNX format.

    Output shape: [batch, num_outputs] where num_outputs = 10
    Layout: [h30_q10, h30_q50, h30_q90, h45_q10, ..., h60_q90, confidence]
    """
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, seq_length, model.input_size)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["prediction"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        },
    )

    print(f"  ✅ ONNX model exported: {save_path}")
    print(f"     Output shape: [batch, {model.num_outputs}]  (9 quantiles + 1 confidence)")


def update_registry(
    run_id: str,
    model_path: Path,
    metrics: dict,
    hyperparams: dict,
) -> None:
    """Update model registry with new model."""
    # Load existing registry
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {"models": [], "current_green": None}

    # Add new model entry
    entry = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": str(model_path.relative_to(PROJECT_ROOT)),
        "status": "candidate",  # Needs promotion to "green"
        "metrics": {
            "val_loss": metrics["best_val_loss"],
            "train_loss": metrics["final_train_loss"],
            "epochs": metrics["epochs_trained"],
        },
        "hyperparams": hyperparams,
    }

    registry["models"].append(entry)

    # Write back
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"  ✅ Registry updated: {REGISTRY_PATH}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train TFT baseline model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS,
        help=f"Maximum training epochs (default: {DEFAULT_MAX_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE,
        help=f"Hidden layer size (default: {DEFAULT_HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--attention-heads", type=int, default=DEFAULT_ATTENTION_HEADS,
        help=f"Number of attention heads (default: {DEFAULT_ATTENTION_HEADS})",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--seq-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Sequence length (default: {DEFAULT_SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate data + show ticker count, don't train",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date for training data (YYYY-MM-DD). Filters features_tft by ts >= start",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date for training data (YYYY-MM-DD). Filters features_tft by ts < end",
    )
    parser.add_argument(
        "--no-reddit", action="store_true",
        help="Exclude all reddit_* features from model input (train without Reddit data)",
    )

    args = parser.parse_args()

    # Configure logging (centralized — keeps correlation_id + contextvars)
    from src.core.logging import configure_logging
    configure_logging("INFO", console=True)

    # Load universe symbols
    universe_symbols = load_universe_symbols()

    print()
    print("=" * 60)
    print("TFT BASELINE TRAINER")
    print("=" * 60)

    # Show universe info
    if not universe_symbols:
        print("❌ ERROR: No universe.csv found.")
        print("   Run: make universe N=500 DAYS=180 MODE=daily")
        return 1

    universe_symbols = sorted(universe_symbols)
    print(f"  Universe:        {len(universe_symbols)} symbols (from universe.csv)")
    print(f"  First 20:        {', '.join(universe_symbols[:20])}")

    low_universe = len(universe_symbols) < 3
    if len(universe_symbols) < MIN_TICKERS_WARNING:
        print("⚠️  WARNING: Universe has fewer than 50 tickers.")

    print()
    print(f"  Hidden size:     {args.hidden_size}")
    print(f"  Attention heads: {args.attention_heads}")
    print(f"  Dropout:         {DEFAULT_DROPOUT}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Max epochs:      {args.max_epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Sequence length: {args.seq_length}")
    print()

    # Device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")
    print()

    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        missing_reddit = []
        for col in REQUIRED_REDDIT_COLUMNS:
            if not await column_exists(conn, "features_tft", col):
                missing_reddit.append(col)

        if missing_reddit:
            raise RuntimeError(
                "Missing required reddit columns in features_tft: "
                + ", ".join(missing_reddit)
            )

        print("  Reddit features: enabled")

        date_range = await conn.fetchrow(
            """
            SELECT MIN(ts)::date AS min_day, MAX(ts)::date AS max_day
            FROM market_raw_day
            WHERE ticker = ANY($1)
            """,
            universe_symbols,
        )
        if not date_range or not date_range["min_day"]:
            date_range = await conn.fetchrow(
                """
                SELECT MIN(ts)::date AS min_day, MAX(ts)::date AS max_day
                FROM market_raw_minute
                WHERE ticker = ANY($1)
                """,
                universe_symbols,
            )
        if date_range and date_range["min_day"]:
            print(f"  Market data:     {date_range['min_day']} to {date_range['max_day']}")
        else:
            print("  Market data:     not found for selected tickers")

        if low_universe:
            print("❌ ERROR: Universe has fewer than 3 tickers. Exiting.")
            return 1

        # Check how many unique tickers we have in features
        ticker_count = await conn.fetchval("SELECT COUNT(DISTINCT ticker) FROM features_tft")
        feature_count = await conn.fetchval("SELECT COUNT(*) FROM features_tft")

        print(f"  Unique tickers in DB: {ticker_count}")
        print(f"  Total features:       {feature_count:,}")

        if feature_count == 0:
            print("❌ ERROR: No data in features_tft!")
            print("   Run: python scripts/build_features.py first")
            return 1

        print()
        print("Target definition:")
        print("  target_return = log(close[t+horizon] / close[t])")
        print(f"  horizon = {DEFAULT_HORIZON} minutes (set by build_features --horizon)")
        print("  split = first 80% train, last 20% val per ticker (time-ordered)")
        print("  leakage: time-based split, ordered by ticker, time_idx")
        print()

        print_feature_catalog(True)

        # ================================================================
        # AUTO-DETECT AVAILABLE FEATURES FROM DATABASE
        # ================================================================
        print("Detecting available features in features_tft...")
        feature_columns, missing_columns = await select_available_features(
            conn, DESIRED_FEATURE_COLUMNS, "features_tft"
        )

        # Safety check: target_return must NOT be in features
        if "target_return" in feature_columns:
            print("  ❌ CRITICAL: target_return was in feature list (leakage risk)!")
            print("     Removing it automatically.")
            feature_columns.remove("target_return")

        # Handle --no-reddit flag: remove all reddit_* columns
        if args.no_reddit:
            reddit_cols_to_remove = [c for c in feature_columns if c.startswith("reddit_")]
            if reddit_cols_to_remove:
                print()
                print(f"🚫 --no-reddit flag: Removing {len(reddit_cols_to_remove)} Reddit features:")
                for col in reddit_cols_to_remove:
                    print(f"     - {col}")
                    feature_columns.remove(col)
            else:
                print("  (--no-reddit: No Reddit features to remove)")

        print()
        print(f"✅ Selected {len(feature_columns)} input features (from {len(DESIRED_FEATURE_COLUMNS)} desired):")
        
        # Group features by category for display
        core_cols = [c for c in feature_columns if c in ("price", "volume", "returns_1", "returns_5", "returns_15", "volatility_5", "volatility_15")]
        tech_cols = [c for c in feature_columns if c in ("sma_5", "sma_20", "ema_5", "ema_20", "rsi_14", "volume_zscore")]
        time_cols = [c for c in feature_columns if c in ("minute_of_day", "hour_of_day", "day_of_week")]
        calendar_cols = [c for c in feature_columns if c in ("is_weekend", "day_of_month", "week_of_year", "month_of_year", "is_month_end")]
        reddit_current = [c for c in feature_columns if c.startswith("reddit_") and "lag" not in c]
        reddit_lags = [c for c in feature_columns if c.startswith("reddit_") and "lag" in c]
        market_cols = [c for c in feature_columns if c.startswith("market_")]

        print(f"  Core price/volume:    {len(core_cols)} cols")
        print(f"  Technical indicators: {len(tech_cols)} cols")
        print(f"  Time features:        {len(time_cols)} cols")
        print(f"  Calendar features:    {len(calendar_cols)} cols")
        print(f"  Reddit current:       {len(reddit_current)} cols")
        print(f"  Reddit lags:          {len(reddit_lags)} cols")
        print(f"  Market benchmark:     {len(market_cols)} cols")

        if missing_columns:
            print()
            print(f"⚠️  Missing {len(missing_columns)} columns (will be ignored):")
            for col in missing_columns:
                print(f"    - {col}")

        print()
        print("Full feature list:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i:2}. {col}")

        total_train_rows = await conn.fetchval(
            """
            SELECT COUNT(*) FROM features_tft
            WHERE split = 'train'
              AND target_return IS NOT NULL
              AND returns_1 IS NOT NULL
              AND volatility_5 IS NOT NULL
            """
        )
        reddit_nonzero = await conn.fetchval(
            """
            SELECT COUNT(*) FROM features_tft
            WHERE split = 'train'
              AND target_return IS NOT NULL
              AND returns_1 IS NOT NULL
              AND volatility_5 IS NOT NULL
              AND reddit_mentions_count > 0
            """
        )
        coverage = (reddit_nonzero / total_train_rows * 100.0) if total_train_rows else 0.0
        print()
        print(f"  Total training rows:         {total_train_rows:,}")
        print(f"  Rows with reddit_mentions_count > 0: {reddit_nonzero:,}")
        print(f"  Reddit coverage:             {coverage:.2f}%")

        sample_rows = await conn.fetch(
            """
            SELECT ticker, ts, reddit_mentions_count,
                   reddit_sentiment_mean, reddit_sentiment_weighted,
                   reddit_positive_ratio, reddit_negative_ratio,
                   is_weekend, day_of_month, market_return_1
            FROM features_tft
            WHERE split = 'train'
              AND reddit_mentions_count > 0
            ORDER BY ts
            LIMIT 5
            """
        )
        print("\n  Sample reddit-positive rows (train split):")
        if sample_rows:
            for row in sample_rows:
                print(
                    f"    {row['ticker']} {row['ts']} "
                    f"mentions={row['reddit_mentions_count']} "
                    f"sent_mean={row['reddit_sentiment_mean']:.3f} "
                    f"weekend={row['is_weekend']} dom={row['day_of_month']} "
                    f"mkt_ret={row['market_return_1']:.4f}"
                )
        else:
            print("    (none)")

        # Guardrail: warn if ticker count is low
        if ticker_count < MIN_TICKERS_WARNING:
            print()
            print("⚠️  WARNING: Training on less than 50 tickers in features!")
            print(f"   Current: {ticker_count} tickers")
            print("   For production quality, use at least 100+ tickers.")
            print("   Run: make universe N=500 && make load-market-data && make build-features")

        # Show sample tickers
        sample_tickers = await conn.fetch(
            "SELECT DISTINCT ticker FROM features_tft ORDER BY ticker LIMIT 10"
        )
        if sample_tickers:
            tickers_str = ", ".join([r["ticker"] for r in sample_tickers])
            print(f"  Sample tickers:       {tickers_str}{'...' if ticker_count > 10 else ''}")

        # Load data
        print()
        if args.start or args.end:
            print(f"Date filter: {args.start or 'beginning'} to {args.end or 'end'}")
        print("Loading training data...")
        train_features, train_targets = await load_data(conn, "train", feature_columns, args.start, args.end)
        print(f"  Train samples: {len(train_features):,}")

        print("Loading validation data...")
        val_features, val_targets = await load_data(conn, "val", feature_columns, args.start, args.end)
        print(f"  Val samples:   {len(val_features):,}")

        if len(train_features) == 0:
            print("❌ ERROR: No training data!")
            return 1

        if len(val_features) == 0:
            print("⚠️  Warning: No validation data, using last 20% of train")
            split_idx = int(0.8 * len(train_features))
            val_features = train_features[split_idx:]
            val_targets = train_targets[split_idx:]
            train_features = train_features[:split_idx]
            train_targets = train_targets[:split_idx]

        if args.dry_run:
            print()
            print("=" * 60)
            print("DRY RUN SUMMARY")
            print("=" * 60)
            print(f"  ✅ Data validation passed")
            print(f"  Tickers:            {ticker_count}")
            print(f"  Features:           {feature_count:,}")
            print(f"  Training samples:   {len(train_features):,}")
            print(f"  Validation samples: {len(val_features):,}")
            print(f"  Feature dimensions: {train_features.shape}")
            print(f"  Target range:       [{train_targets.min():.4f}, {train_targets.max():.4f}]")
            print()
            if ticker_count < MIN_TICKERS_WARNING:
                print("  ⚠️  Scale warning: Run 'make universe N=500' for production")
            else:
                print("  ✅ Scale check passed: ready for production training")
            
            # Print verification SQL commands
            print()
            print("=" * 60)
            print("VERIFICATION QUERIES")
            print("=" * 60)
            print("Run these to verify Reddit/lag features are populated:")
            print()
            print("# 1. Check non-null counts for Reddit columns:")
            print("""PGPASSWORD=awet psql -h localhost -p 5433 -U awet -d awet -c "
SELECT
  ticker,
  COUNT(*) AS rows,
  SUM(CASE WHEN reddit_mentions_count > 0 THEN 1 ELSE 0 END) AS reddit_rows,
  SUM(CASE WHEN reddit_mentions_lag1 > 0 THEN 1 ELSE 0 END) AS lag1_rows,
  SUM(CASE WHEN reddit_sentiment_mean_lag3 != 0 THEN 1 ELSE 0 END) AS lag3_rows
FROM features_tft
WHERE ticker='AAPL'
GROUP BY ticker;
""")
            print()
            print("# 2. Check calendar feature distribution:")
            print("""PGPASSWORD=awet psql -h localhost -p 5433 -U awet -d awet -c "
SELECT
  is_weekend,
  COUNT(*) AS cnt,
  ROUND(AVG(day_of_month)::numeric, 1) AS avg_dom,
  ROUND(AVG(week_of_year)::numeric, 1) AS avg_woy
FROM features_tft
GROUP BY is_weekend;
""")
            print()
            print("# 3. Check market benchmark alignment:")
            print("""PGPASSWORD=awet psql -h localhost -p 5433 -U awet -d awet -c "
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN market_return_1 != 0 THEN 1 ELSE 0 END) AS with_mkt_return,
  ROUND(AVG(market_return_1)::numeric, 6) AS avg_mkt_ret,
  ROUND(AVG(market_volatility_5)::numeric, 6) AS avg_mkt_vol
FROM features_tft;
""")
            return 0

        # Normalize
        print()
        print("Normalizing features...")
        train_norm, val_norm, norm_stats = normalize_features(train_features, val_features)

        # Create sequences
        print("Creating sequences...")
        print("  Building X_train via create_sequences(train_norm, train_targets, seq_length)")
        X_train, y_train = create_sequences(train_norm, train_targets, args.seq_length)
        print("  Building X_val via create_sequences(val_norm, val_targets, seq_length)")
        X_val, y_val = create_sequences(val_norm, val_targets, args.seq_length)

        print(f"  Train sequences: {len(X_train):,}")
        print(f"  Val sequences:   {len(X_val):,}")
        print(f"  Num tickers:     {ticker_count}")
        print(f"  Num features:    {X_train.shape[2]}")
        print(f"  Timesteps/seq:   {args.seq_length}")

        if len(X_train) < 100:
            print("❌ ERROR: Not enough sequences for training!")
            return 1

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Create model
        input_size = X_train.shape[2]
        model = SimpleTFTModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            attention_heads=args.attention_heads,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")
        print()

        # Train
        print("Training...")
        print("-" * 60)
        metrics = train_model(
            model, train_loader, val_loader,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            device=device,
        )
        print("-" * 60)

        # Create output directory
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        output_dir = MODELS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = output_dir / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "hyperparams": {
                "input_size": input_size,
                "hidden_size": args.hidden_size,
                "attention_heads": args.attention_heads,
                "seq_length": args.seq_length,
                "num_horizons": NUM_HORIZONS,
                "num_quantiles": NUM_QUANTILES,
                "num_outputs": NUM_OUTPUTS,
            },
            "feature_columns": feature_columns,
            "norm_stats": norm_stats,
            "metrics": metrics,
        }, model_path)
        print(f"  ✅ PyTorch model saved: {model_path}")

        # Save feature metadata alongside ONNX for inference engine
        feature_meta = {
            "feature_columns": feature_columns,
            "norm_stats": norm_stats,
            "input_size": input_size,
            "seq_length": args.seq_length,
            "num_horizons": NUM_HORIZONS,
            "num_quantiles": NUM_QUANTILES,
            "num_outputs": NUM_OUTPUTS,
            "horizons": [30, 45, 60],
            "quantiles": QUANTILE_VALUES,
        }
        meta_path = output_dir / "feature_meta.json"
        with open(meta_path, "w") as f:
            json.dump(feature_meta, f, indent=2)
        print(f"  ✅ Feature metadata saved: {meta_path}")

        # Export ONNX
        model.cpu()
        onnx_path = output_dir / "model.onnx"
        export_onnx(model, args.seq_length, onnx_path)

        # Update registry
        update_registry(
            run_id=run_id,
            model_path=output_dir,
            metrics=metrics,
            hyperparams={
                "input_dim": input_size,
                "hidden_size": args.hidden_size,
                "attention_heads": args.attention_heads,
                "seq_length": args.seq_length,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_horizons": NUM_HORIZONS,
                "num_quantiles": NUM_QUANTILES,
                "num_outputs": NUM_OUTPUTS,
                "lookback_window": args.seq_length,
                "horizons": [30, 45, 60],
                "num_features": input_size,
            },
        )

        # Summary
        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Run ID:     {run_id}")
        print(f"  Val Loss:   {metrics['best_val_loss']:.6f}")
        print(f"  Epochs:     {metrics['epochs_trained']}")
        print(f"  Model:      {onnx_path}")
        print()
        print("  To promote this model to production:")
        print(f"    python -m execution.promote_model {run_id}")
        print()

        return 0

    finally:
        await conn.close()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        sys.exit(1)
