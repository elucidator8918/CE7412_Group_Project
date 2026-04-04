"""
Model 4: SoftBlobGAT — GATv2 encoder + differentiable blob pooling + cluster attention.

This is the project's novel contribution: a lightweight adaptive graph
partitioning method inspired by BioBlobs (Wang & Oliver, 2025) that
replaces the computationally expensive GVP + VQ-codebook pipeline with:

  1. GATv2 encoder with edge features (instead of vanilla GCN or GVP)
  2. Gumbel-softmax blob assignment (learnable soft partitions)
  3. GlobalClusterAttention: global embedding queries blob embeddings
  4. FeatureWiseGateFusion: gated residual fusion of global + cluster info
  5. Multi-pool global embedding as attention query

The architecture directly tests the central hypothesis: can adaptive
protein substructure partitioning improve classification without the
heavy machinery of full BioBlobs?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from .modules import (
    ClassifierHead,
    FeatureWiseGateFusion,
    GlobalClusterAttention,
    MultiPoolReadout,
)


class SoftBlobGAT(nn.Module):
    """SoftBlobGCN reimagined with GATv2 backbone and richer pooling."""

    def __init__(
        self,
        in_ch: int,
        hidden: int,
        n_classes: int,
        edge_dim: int = 0,
        n_blobs: int = 8,
        n_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
        blob_mlp_layers: int = 2,
        cluster_attn_heads: int = 4,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        pool_strategy: str = "multi",
    ):
        super().__init__()
        self.n_blobs = n_blobs
        self.tau_start = tau_start
        self.tau_end = tau_end
        self._current_tau = tau_start

        # ── GATv2 encoder ──────────────────────────────────────────────
        self.input_proj = nn.Linear(in_ch, hidden)
        self.input_ln = nn.LayerNorm(hidden)

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.ffs = nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    hidden, hidden // heads, heads=heads,
                    concat=True, dropout=attn_dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                    add_self_loops=True, share_weights=False,
                )
            )
            self.lns.append(nn.LayerNorm(hidden))
            self.ffs.append(nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
                nn.Dropout(dropout),
            ))

        self.drop = nn.Dropout(dropout)

        # ── Blob assignment head ───────────────────────────────────────
        self.blob_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, n_blobs),
        )

        # ── Blob refinement MLP ────────────────────────────────────────
        blob_layers = []
        for _ in range(blob_mlp_layers):
            blob_layers.extend([
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ])
        self.blob_mlp = nn.Sequential(*blob_layers)
        self.blob_ln = nn.LayerNorm(hidden)

        # ── Global embedding (multi-pool as query) ─────────────────────
        if pool_strategy == "multi":
            self.global_pool = MultiPoolReadout(hidden)
        else:
            self.global_pool = None

        # ── Cluster attention + gated fusion ───────────────────────────
        self.cluster_attn = GlobalClusterAttention(
            dim=hidden, heads=cluster_attn_heads, drop_rate=dropout * 0.5
        )
        self.gate_fusion = FeatureWiseGateFusion(
            dim=hidden, drop_rate=dropout * 0.5
        )

        # ── Classifier ─────────────────────────────────────────────────
        self.clf = ClassifierHead(hidden, n_classes, dropout)

    def set_tau(self, epoch: int, total_epochs: int):
        """Anneal Gumbel-softmax temperature from tau_start to tau_end."""
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        self._current_tau = self.tau_start + (self.tau_end - self.tau_start) * progress

    def _encode(self, data):
        """GATv2 message passing with pre-norm residuals."""
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        x = self.drop(F.gelu(self.input_ln(self.input_proj(x))))

        for conv, ln, ff in zip(self.convs, self.lns, self.ffs):
            residual = x
            x_norm = ln(x)
            x_attn = conv(x_norm, ei, edge_attr=edge_attr) if edge_attr is not None else conv(x_norm, ei)
            x = residual + self.drop(x_attn)
            x = x + ff(x)

        return x, batch

    def forward(self, data, return_blobs=False):
        x, batch = self._encode(data)

        # ── Soft blob assignment via Gumbel-softmax ────────────────────
        logits = self.blob_head(x)
        assign = F.gumbel_softmax(logits, tau=self._current_tau, hard=False, dim=-1)

        # ── Global embedding (query for cluster attention) ─────────────
        if self.global_pool is not None:
            global_emb = self.global_pool(x, batch)
        else:
            global_emb = global_mean_pool(x, batch)

        # ── Aggregate residues into blob embeddings per graph ──────────
        batch_ids = torch.unique(batch, sorted=True)
        B = batch_ids.shape[0]
        blob_list = []

        for b_id in batch_ids:
            mask = (batch == b_id)
            x_b = x[mask]        # [N_b, hidden]
            a_b = assign[mask]    # [N_b, K]

            # Weighted average: blob_k = sum(a_ik * h_i) / sum(a_ik)
            weights = a_b.T / (a_b.T.sum(dim=1, keepdim=True) + 1e-8)  # [K, N_b]
            blobs = weights @ x_b                                        # [K, hidden]
            blobs = self.blob_ln(self.blob_mlp(blobs))
            blob_list.append(blobs)

        blob_tensor = torch.stack(blob_list, dim=0)  # [B, K, hidden]

        # ── Cluster attention: global queries blobs ────────────────────
        cluster_summary = self.cluster_attn(global_emb, blob_tensor)
        fused = self.gate_fusion(global_emb, cluster_summary)

        out = self.clf(fused)

        if return_blobs:
            return out, assign
        return out

    @torch.no_grad()
    def embed(self, data):
        """Extract graph-level embeddings for visualization."""
        x, batch = self._encode(data)

        if self.global_pool is not None:
            global_emb = self.global_pool(x, batch)
        else:
            global_emb = global_mean_pool(x, batch)

        batch_ids = torch.unique(batch, sorted=True)
        blob_list = []
        for b_id in batch_ids:
            mask = (batch == b_id)
            x_b = x[mask]
            a_b = F.softmax(self.blob_head(x_b), dim=-1)
            weights = a_b.T / (a_b.T.sum(dim=1, keepdim=True) + 1e-8)
            blobs = weights @ x_b
            blobs = self.blob_ln(self.blob_mlp(blobs))
            blob_list.append(blobs)

        blob_tensor = torch.stack(blob_list, dim=0)
        cluster_summary = self.cluster_attn(global_emb, blob_tensor)
        fused = self.gate_fusion(global_emb, cluster_summary)
        return fused.cpu().numpy()
