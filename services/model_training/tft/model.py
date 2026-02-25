"""TFT Model with PyTorch Lightning.

Implements Temporal Fusion Transformer with:
- Gated Residual Networks (GRN)
- Variable Selection Networks
- Multi-horizon quantile outputs
- Confidence score head
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger("tft_model")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import lightning.pytorch as pl
        LIGHTNING_AVAILABLE = True
    except ImportError:
        LIGHTNING_AVAILABLE = False
        pl = None


@dataclass
class TFTConfig:
    """TFT model configuration."""
    input_dim: int = 15
    hidden_dim: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    dropout: float = 0.1
    lookback_window: int = 120
    horizons: list[int] = field(default_factory=lambda: [30, 45, 60])
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


if TORCH_AVAILABLE:
    
    class GatedLinearUnit(nn.Module):
        """GLU activation: sigmoid(Wx + b) * (Vx + c)."""
        
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim * 2)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.fc(x)
            a, b = out.chunk(2, dim=-1)
            return a * torch.sigmoid(b)
    
    
    class GatedResidualNetwork(nn.Module):
        """GRN with skip connection and layer norm."""
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int | None = None,
            dropout: float = 0.1,
            context_dim: int | None = None,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim or input_dim
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.elu = nn.ELU()
            
            self.context_projection = None
            if context_dim:
                self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)
            
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.glu = GatedLinearUnit(hidden_dim, self.output_dim)
            self.layer_norm = nn.LayerNorm(self.output_dim)
            self.dropout = nn.Dropout(dropout)
            
            self.skip_projection = None
            if input_dim != self.output_dim:
                self.skip_projection = nn.Linear(input_dim, self.output_dim)
        
        def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
        ) -> torch.Tensor:
            residual = x
            if self.skip_projection:
                residual = self.skip_projection(residual)
            
            out = self.fc1(x)
            out = self.elu(out)
            
            if self.context_projection and context is not None:
                out = out + self.context_projection(context)
            
            out = self.fc2(out)
            out = self.elu(out)
            out = self.dropout(out)
            out = self.glu(out)
            
            return self.layer_norm(out + residual)
    
    
    class VariableSelectionNetwork(nn.Module):
        """Variable selection for interpretability."""
        
        def __init__(
            self,
            input_dim: int,
            num_features: int,
            hidden_dim: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.num_features = num_features
            
            # Per-feature GRNs
            self.feature_grns = nn.ModuleList([
                GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_features)
            ])
            
            # Softmax weights for feature selection
            self.weight_grn = GatedResidualNetwork(
                num_features * hidden_dim,
                hidden_dim,
                num_features,
                dropout,
            )
            
            self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.
            
            Args:
                x: (batch, seq_len, num_features, input_dim) or (batch, seq_len, num_features)
            
            Returns:
                (output, weights) - transformed output and feature weights
            """
            batch_size, seq_len = x.shape[:2]
            
            # Handle different input shapes
            if x.dim() == 3:
                # (batch, seq, num_features) -> need to expand
                x = x.unsqueeze(-1)  # (batch, seq, num_features, 1)
            
            # Process each feature
            processed = []
            for i, grn in enumerate(self.feature_grns):
                feat = x[:, :, i, :]  # (batch, seq, input_dim)
                processed.append(grn(feat))  # (batch, seq, hidden_dim)
            
            # Stack and compute weights
            stacked = torch.stack(processed, dim=2)  # (batch, seq, num_features, hidden_dim)
            flat = stacked.reshape(batch_size, seq_len, -1)  # (batch, seq, num_features * hidden_dim)
            
            weights = F.softmax(self.weight_grn(flat), dim=-1)  # (batch, seq, num_features)
            
            # Weighted sum
            weighted = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # (batch, seq, hidden_dim)
            output = self.output_grn(weighted)
            
            return output, weights
    
    
    class TemporalFusionTransformerCore(nn.Module):
        """Core TFT architecture."""
        
        def __init__(self, config: TFTConfig) -> None:
            super().__init__()
            self.config = config
            
            # Input embedding
            self.input_embedding = nn.Linear(config.input_dim, config.hidden_dim)
            
            # Variable selection
            self.vsn = VariableSelectionNetwork(
                input_dim=1,
                num_features=config.input_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            
            # LSTM encoder
            self.lstm_encoder = nn.LSTM(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=False,
            )
            
            # Post-LSTM GRN
            self.post_lstm_grn = GatedResidualNetwork(
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.dropout,
            )
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_encoder_layers,
            )
            
            # Output heads
            num_horizons = len(config.horizons)
            num_quantiles = len(config.quantiles)
            
            # Quantile heads: one output per (horizon, quantile)
            self.quantile_heads = nn.ModuleList([
                nn.Sequential(
                    GatedResidualNetwork(config.hidden_dim, config.hidden_dim, config.hidden_dim, config.dropout),
                    nn.Linear(config.hidden_dim, num_quantiles),
                )
                for _ in range(num_horizons)
            ])
            
            # Confidence head
            self.confidence_head = nn.Sequential(
                GatedResidualNetwork(config.hidden_dim, config.hidden_dim, config.hidden_dim // 2, config.dropout),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Forward pass.
            
            Args:
                x: (batch, seq_len, input_dim) features
                
            Returns:
                Dict with 'quantiles' (batch, num_horizons, num_quantiles) and 'confidence' (batch,)
            """
            batch_size = x.shape[0]
            
            # Variable selection - input shape (batch, seq, features)
            selected, weights = self.vsn(x)  # (batch, seq, hidden)
            
            # LSTM encoding
            lstm_out, _ = self.lstm_encoder(selected)
            
            # Post-LSTM GRN
            lstm_processed = self.post_lstm_grn(lstm_out)
            
            # Transformer
            transformed = self.transformer(lstm_processed)
            
            # Take the last timestep for prediction
            final_state = transformed[:, -1, :]  # (batch, hidden)
            
            # Compute quantile predictions for each horizon
            quantile_outputs = []
            for head in self.quantile_heads:
                q_out = head(final_state)  # (batch, num_quantiles)
                quantile_outputs.append(q_out)
            
            quantiles = torch.stack(quantile_outputs, dim=1)  # (batch, num_horizons, num_quantiles)
            
            # Confidence score
            confidence = self.confidence_head(final_state).squeeze(-1)  # (batch,)
            
            return {
                "quantiles": quantiles,
                "confidence": confidence,
                "attention_weights": weights,
            }
    
    
    class QuantileLoss(nn.Module):
        """Pinball loss for quantile regression."""
        
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
                targets: (batch, num_horizons) - actual returns
            """
            # Expand targets for broadcasting
            targets = targets.unsqueeze(-1)  # (batch, num_horizons, 1)
            
            losses = []
            for i, q in enumerate(self.quantiles):
                pred_q = predictions[:, :, i:i+1]  # (batch, num_horizons, 1)
                errors = targets - pred_q
                loss_q = torch.max(q * errors, (q - 1) * errors)
                losses.append(loss_q.mean())
            
            return torch.stack(losses).mean()


if TORCH_AVAILABLE and LIGHTNING_AVAILABLE:
    
    class TFTLightning(pl.LightningModule):
        """PyTorch Lightning wrapper for TFT."""
        
        def __init__(self, config: TFTConfig | None = None) -> None:
            super().__init__()
            self.config = config or TFTConfig()
            self.save_hyperparameters()
            
            self.model = TemporalFusionTransformerCore(self.config)
            self.criterion = QuantileLoss(self.config.quantiles)
        
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.model(x)
        
        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            x, y = batch
            out = self.model(x)
            loss = self.criterion(out["quantiles"], y)
            self.log("train_loss", loss, prog_bar=True)
            return loss
        
        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            x, y = batch
            out = self.model(x)
            loss = self.criterion(out["quantiles"], y)
            self.log("val_loss", loss, prog_bar=True)
            return loss
        
        def configure_optimizers(self) -> dict[str, Any]:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }


if TORCH_AVAILABLE:
    class ONNXWrapper(nn.Module):
        """Wrapper for ONNX export with flattened output."""
        
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward with flattened output for ONNX.
            
            Returns:
                (batch, num_horizons * num_quantiles + 1) - quantiles + confidence
            """
            out = self.model(x)
            batch_size = x.size(0)
            
            quantiles = out["quantiles"]  # (batch, num_horizons, num_quantiles)
            confidence = out["confidence"]  # (batch,)
            
            # Flatten quantiles and append confidence
            flat_quantiles = quantiles.reshape(batch_size, -1)  # (batch, num_horizons * num_quantiles)
            confidence = confidence.unsqueeze(-1)  # (batch, 1)
            
            return torch.cat([flat_quantiles, confidence], dim=-1)


def export_to_onnx(
    model: Any,
    output_path: str,
    lookback_window: int = 120,
    input_dim: int = 15,
    opset_version: int = 14,
) -> str:
    """Export TFT model to ONNX format.
    
    Args:
        model: TFT model (TFTLightning or TemporalFusionTransformerCore)
        output_path: Path for ONNX file
        lookback_window: Input sequence length
        input_dim: Number of input features
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX file
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for ONNX export")
    
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Get the core model
    if hasattr(model, "model"):
        core_model = model.model
    else:
        core_model = model
    
    core_model.eval()
    wrapper = ONNXWrapper(core_model)
    
    # Dummy input
    dummy_input = torch.randn(1, lookback_window, input_dim)
    
    # Export
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
        opset_version=opset_version,
    )
    
    logger.info("onnx_exported", path=output_path)
    return output_path
