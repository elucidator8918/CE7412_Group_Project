"""
Model 3: GATv2 — protein contact graph → GATv2 message passing → global pool → class.

Key improvements over vanilla GCN:
  - Anisotropic message passing via learned attention weights (GATv2Conv)
  - Edge feature integration (distance, sequence separation)
  - Multi-pool readout (mean + max + attention)
  - Pre-norm residual connections with LayerNorm
  - GELU activation (smoother gradients than ReLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .modules import ClassifierHead, MultiPoolReadout


class GATModel(nn.Module):
    """GATv2-based protein graph classifier with edge features."""

    def __init__(self, in_ch: int, hidden: int, n_classes: int,
                 edge_dim: int = 0, n_layers: int = 4, heads: int = 4,
                 dropout: float = 0.3, attn_dropout: float = 0.1,
                 pool_strategy: str = "multi"):
        super().__init__()
        self.n_layers = n_layers
        self.pool_strategy = pool_strategy

        # Input projection
        self.input_proj = nn.Linear(in_ch, hidden)
        self.input_ln = nn.LayerNorm(hidden)

        # GATv2 layers with pre-norm residual connections
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.ffs = nn.ModuleList()  # feed-forward after attention

        for i in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden,
                    out_channels=hidden // heads,
                    heads=heads,
                    concat=True,
                    dropout=attn_dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                    add_self_loops=True,
                    share_weights=False,
                )
            )
            self.lns.append(nn.LayerNorm(hidden))
            # Feed-forward block (like transformer)
            self.ffs.append(nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
                nn.Dropout(dropout),
            ))

        self.drop = nn.Dropout(dropout)

        # Pooling
        if pool_strategy == "multi":
            self.pool = MultiPoolReadout(hidden)
        else:
            self.pool = None

        # Classifier
        self.clf = ClassifierHead(hidden, n_classes, dropout)

    def _encode(self, data):
        """Run GATv2 encoder, return node embeddings and batch."""
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        # Input projection
        x = self.drop(F.gelu(self.input_ln(self.input_proj(x))))

        # GATv2 message passing with pre-norm residuals
        for conv, ln, ff in zip(self.convs, self.lns, self.ffs):
            # Attention block
            residual = x
            x_norm = ln(x)
            if edge_attr is not None:
                x_attn = conv(x_norm, ei, edge_attr=edge_attr)
            else:
                x_attn = conv(x_norm, ei)
            x = residual + self.drop(x_attn)

            # Feed-forward block
            x = x + ff(x)

        return x, batch

    def forward(self, data):
        x, batch = self._encode(data)

        if self.pool is not None:
            z = self.pool(x, batch)
        else:
            z = global_mean_pool(x, batch)

        return self.clf(z)

    @torch.no_grad()
    def embed(self, data):
        """Extract graph-level embeddings for visualization."""
        x, batch = self._encode(data)
        if self.pool is not None:
            z = self.pool(x, batch)
        else:
            z = global_mean_pool(x, batch)
        return z.cpu().numpy()
