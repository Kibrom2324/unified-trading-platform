"""Temporal Fusion Transformer (TFT) model implementation.

.. deprecated::
    This is a legacy TFT model only used by ``src/ml/train_tft.py``.
    The canonical training pipeline is ``scripts/train_tft_baseline.py``
    which trains ``SimpleTFTModel``, exports to ONNX, saves
    ``feature_meta.json``, and registers in the model registry.
    Production inference uses ``ONNXInferenceEngine`` (``src/ml/onnx_engine.py``).

A simplified TFT architecture for time series forecasting with
quantile outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("tft_model")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("pytorch_not_installed", hint="pip install torch")


if TORCH_AVAILABLE:

    class GatedLinearUnit(nn.Module):
        """GLU activation for TFT."""

        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim * 2)
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.fc(x)
            return out[..., :self.output_dim] * torch.sigmoid(out[..., self.output_dim:])


    class GatedResidualNetwork(nn.Module):
        """GRN block for TFT."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.glu = GatedLinearUnit(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_dim)
            self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.skip(x) if self.skip else x
            h = F.elu(self.fc1(x))
            h = self.dropout(h)
            h = F.elu(self.fc2(h))
            h = self.dropout(h)
            h = self.glu(h)
            return self.layer_norm(h + residual)


    class TemporalFusionTransformer(nn.Module):
        """Simplified TFT for quantile forecasting."""

        def __init__(
            self,
            input_dim: int = 15,
            hidden_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            num_horizons: int = 3,
            num_quantiles: int = 3,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_horizons = num_horizons
            self.num_quantiles = num_quantiles
            self.input_embedding = GatedResidualNetwork(
                input_dim, hidden_dim, hidden_dim, dropout
            )
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_grn = GatedResidualNetwork(
                hidden_dim, hidden_dim, hidden_dim, dropout
            )
            self.quantile_heads = nn.ModuleList([
                nn.Linear(hidden_dim, num_quantiles)
                for _ in range(num_horizons)
            ])
            self.confidence_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, seq_len, input_dim)

            Returns:
                Dict with quantile predictions and confidence
            """
            batch_size = x.size(0)
            embedded = self.input_embedding(x)
            lstm_out, (h_n, _) = self.lstm(embedded)
            transformer_out = self.transformer(lstm_out)
            final_hidden = transformer_out[:, -1, :]
            processed = self.output_grn(final_hidden)
            quantiles = []
            for head in self.quantile_heads:
                q = head(processed)
                quantiles.append(q)
            quantiles_tensor = torch.stack(quantiles, dim=1)
            confidence = torch.sigmoid(self.confidence_head(processed))

            return {
                "quantiles": quantiles_tensor,
                "confidence": confidence.squeeze(-1),
                "hidden": processed,
            }

        def predict(self, x: torch.Tensor) -> np.ndarray:
            """Generate predictions as numpy array."""
            self.eval()
            with torch.no_grad():
                out = self.forward(x)
                quantiles = out["quantiles"].cpu().numpy()
                confidence = out["confidence"].cpu().numpy()
            result = np.concatenate([
                quantiles.reshape(quantiles.shape[0], -1),
                confidence.reshape(-1, 1)
            ], axis=1)
            return result


    class QuantileLoss(nn.Module):
        """Quantile loss for TFT training."""

        def __init__(self, quantiles: list[float] | None = None) -> None:
            super().__init__()
            self.quantiles = quantiles or [0.1, 0.5, 0.9]

        def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            """Compute quantile loss.

            Args:
                predictions: (batch, num_horizons, num_quantiles)
                targets: (batch, num_horizons)
            """
            losses = []
            for i, q in enumerate(self.quantiles):
                pred_q = predictions[:, :, i]
                errors = targets - pred_q
                loss_q = torch.max(q * errors, (q - 1) * errors)
                losses.append(loss_q.mean())
            return torch.stack(losses).mean()

else:
    class TemporalFusionTransformer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")

    class QuantileLoss:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")
