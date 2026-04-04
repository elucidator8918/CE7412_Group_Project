"""
Shared neural network modules for protein graph classification.

Components:
  - GlobalClusterAttention: multi-head attention from global query to blob keys
  - FeatureWiseGateFusion: gated residual fusion z = LN(g + β ⊙ Wc*)
  - MultiPoolReadout: combined mean + max + attention-weighted global pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention


# ============================================================================
# Multi-pool readout
# ============================================================================

class MultiPoolReadout(nn.Module):
    """Combined global pooling: mean + max + learned attention.

    Produces a 3h-dimensional graph-level embedding from node features.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )
        self.attn_pool = GlobalAttention(gate_nn=self.attn_gate)
        self.proj = nn.Linear(hidden * 3, hidden)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x, batch):
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z_attn = self.attn_pool(x, batch)
        z = torch.cat([z_mean, z_max, z_attn], dim=-1)
        return self.ln(self.proj(z))


# ============================================================================
# BioBlobs-inspired cluster attention
# ============================================================================

class GlobalClusterAttention(nn.Module):
    """Multi-head attention: global embedding queries blob embeddings.

    Adapted from BioBlobs (Wang & Oliver, 2025). The global protein embedding
    acts as a single query attending to K blob (cluster) embeddings as keys/values.
    """

    def __init__(self, dim: int, heads: int = 4, drop_rate: float = 0.1):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        self.heads = heads
        self.d_head = dim // heads
        self.scale = self.d_head ** 0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, g, C, mask=None):
        """
        Args:
            g: [B, D] global embedding (query)
            C: [B, K, D] blob embeddings (keys/values)
            mask: [B, K] boolean mask for valid blobs

        Returns:
            pooled: [B, D] attention-weighted summary
        """
        B, K, D = C.shape
        H, d = self.heads, self.d_head

        q = self.q(g).view(B, H, d)                       # [B, H, d]
        k = self.k(C).view(B, K, H, d).transpose(1, 2)    # [B, H, K, d]
        v = self.v(C).view(B, K, H, d).transpose(1, 2)    # [B, H, K, d]

        scores = (q.unsqueeze(2) * k).sum(-1) / self.scale  # [B, H, K]

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        pooled = (attn.unsqueeze(-1) * v).sum(2)  # [B, H, d]
        pooled = pooled.reshape(B, D)
        return self.out(pooled)


# ============================================================================
# Feature-wise gated fusion
# ============================================================================

class FeatureWiseGateFusion(nn.Module):
    """Gated residual fusion from BioBlobs:
       z = LN(g + σ(MLP([g; c*])) ⊙ W_f c*)

    Learns element-wise gates that control how much of the cluster
    attention summary (c*) is fused with the global embedding (g).
    """

    def __init__(self, dim: int, hidden: int = 0, drop_rate: float = 0.1):
        super().__init__()
        if hidden <= 0:
            hidden = max(32, dim // 2)
        self.beta_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden, dim),
        )
        self.proj_c = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

        # Initialize gate bias to zero → starts at 0.5 sigmoid
        nn.init.zeros_(self.beta_mlp[-1].bias)

    def forward(self, g, c_star):
        """
        Args:
            g: [B, D] global embedding
            c_star: [B, D] cluster attention summary

        Returns:
            fused: [B, D]
        """
        beta = torch.sigmoid(self.beta_mlp(torch.cat([g, c_star], dim=-1)))
        fused = self.ln(g + beta * self.proj_c(c_star))
        return fused


# ============================================================================
# MLP Classifier Head
# ============================================================================

class ClassifierHead(nn.Module):
    """3-layer MLP classifier: h → 2h → h → n_classes."""

    def __init__(self, hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
