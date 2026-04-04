"""
Model 2: Residue MLP — mean-pooled node features → MLP → class.

Uses the full feature vector (including ESM-2, physicochemical, etc.)
but discards all graph topology.
"""

import torch
import torch.nn as nn


class ResidueMLP(nn.Module):
    """MLP operating on mean-pooled per-residue features (no topology)."""

    def __init__(self, in_ch: int, hidden: int, n_classes: int,
                 n_layers: int = 3, dropout: float = 0.4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_ch if i == 0 else hidden
            layers.extend([
                nn.Linear(d_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
