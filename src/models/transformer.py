"""
Fuzzy-Transformer model for stock price forecasting.
Includes:
  - PositionalEncoding
  - TransformerModel (Encoder-only, multi-step output)
  - sequence creation utilities
  - training / evaluation loop
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ──────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        return x + self.pe[: x.size(0)]


class TransformerModel(nn.Module):
    """
    Transformer Encoder for multi-step time-series forecasting.

    Args:
        input_dim:           Number of input features per time step.
        d_model:             Internal embedding dimension.
        nhead:               Number of attention heads (must divide d_model).
        num_encoder_layers:  Number of Transformer encoder layers.
        dim_feedforward:     FFN hidden size inside each encoder layer.
        output_horizon:      Number of future steps to predict.
        target_dim:          Number of target variables (typically 1).
        dropout:             Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        output_horizon: int = 1,
        target_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_horizon = output_horizon
        self.target_dim = target_dim

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.output_linear = nn.Linear(d_model, output_horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, input_dim)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x_perm = x.permute(1, 0, 2)          # (seq, batch, d_model)
        x_perm = self.pos_encoder(x_perm)
        x = x_perm.permute(1, 0, 2)          # (batch, seq, d_model)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]                       # last time step
        x = self.output_linear(x)
        return x.view(-1, self.output_horizon, self.target_dim)


# ──────────────────────────────────────────────
# Sequence utilities
# ──────────────────────────────────────────────

def create_sequences(
    X: np.ndarray, Y: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X, Y) sequence pairs.

    Args:
        X: Feature array of shape (N, n_features).
        Y: Target array of shape (N, n_targets).
        lookback: Input window length.
        horizon: Number of future steps to predict.

    Returns:
        X_seq: (n_samples, lookback, n_features)
        Y_seq: (n_samples, horizon, n_targets)
    """
    xs, ys = [], []
    for i in range(len(X) - lookback - horizon + 1):
        xs.append(X[i : i + lookback])
        ys.append(Y[i + lookback : i + lookback + horizon])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_dataloaders(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray,   Y_val: np.ndarray,
    X_test: np.ndarray,  Y_test: np.ndarray,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convert numpy arrays into PyTorch DataLoaders."""
    def _loader(X, Y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _loader(X_train, Y_train, shuffle=True),
        _loader(X_val, Y_val, shuffle=False),
        _loader(X_test, Y_test, shuffle=False),
    )


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_transformer(
    model: TransformerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: torch.device | None = None,
) -> tuple[list[float], list[float]]:
    """
    Train the Transformer model and return (train_losses, val_losses).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for X_b, Y_b in tqdm(train_loader, desc=f"Epoch {epoch:02d} Train"):
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), Y_b)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_b, Y_b in tqdm(val_loader, desc=f"Epoch {epoch:02d} Val  "):
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                val_batch_losses.append(criterion(model(X_b), Y_b).item())
        val_losses.append(float(np.mean(val_batch_losses)))

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_losses[-1]:.6f} | val_loss={val_losses[-1]:.6f}"
        )

    return train_losses, val_losses


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def evaluate(
    model: TransformerModel,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the test set.

    Returns:
        (all_preds, all_actuals) — numpy arrays in scaled space.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_b, Y_b in tqdm(test_loader, desc="Evaluating"):
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            preds.append(model(X_b).cpu().numpy())
            actuals.append(Y_b.cpu().numpy())
    return np.concatenate(preds), np.concatenate(actuals)
