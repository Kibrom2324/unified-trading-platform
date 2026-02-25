"""TFT module for training and inference."""

from __future__ import annotations

from src.ml.tft.dataset import TFTDataset, TFTDataModule, DatasetConfig
from src.ml.tft.model import TFTConfig

# Conditionally import torch-dependent classes
try:
    from src.ml.tft.model import TFTLightning, TemporalFusionTransformerCore, QuantileLoss
except ImportError:
    TFTLightning = None
    TemporalFusionTransformerCore = None
    QuantileLoss = None

__all__ = [
    "TFTDataset",
    "TFTDataModule",
    "DatasetConfig",
    "TFTLightning",
    "TFTConfig",
    "TemporalFusionTransformerCore",
    "QuantileLoss",
]
