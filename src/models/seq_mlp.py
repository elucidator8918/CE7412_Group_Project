"""
Model 1: Sequence MLP — amino acid composition → MLP → class.

The simplest baseline: no structural information, only the 20-dim
normalized amino acid frequency vector.
"""

import torch
import torch.nn as nn


class SeqMLP(nn.Module):
    """MLP operating on amino acid composition (no structure)."""

    def __init__(self, n_classes: int, in_dim: int = 20, hidden: int = 256,
                 n_layers: int = 3, dropout: float = 0.4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
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
