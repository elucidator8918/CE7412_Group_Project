"""
Model 5: GIN — Graph Isomorphism Network for enzyme classification.

GIN (Xu et al., ICLR 2019) is provably the most powerful message-passing
GNN under the Weisfeiler-Leman graph isomorphism test. It uses a simple
MLP-based aggregation that is strictly more expressive than GCN and GAT
for distinguishing graph structures — while using FEWER parameters.

This implementation:
  - Uses GINEConv (GIN with edge features) from PyG
  - Keeps the model small (~200K-400K params) to avoid overfitting
  - Uses JumpingKnowledge (concatenation of all layer outputs)
  - Combines mean + max pooling for graph-level readout
  - No heavy transformer-style feed-forward blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, JumpingKnowledge


class GINModel(nn.Module):
    """Lightweight GIN with edge features and jumping knowledge."""

    def __init__(self, in_ch: int, hidden: int, n_classes: int,
                 edge_dim: int = 0, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_ch, hidden)

        # GIN layers — each uses a 2-layer MLP as the update function
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            if edge_dim > 0:
                self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            else:
                self.convs.append(GINEConv(nn=mlp, edge_dim=1))  # dummy
            self.bns.append(nn.BatchNorm1d(hidden))

        # Edge projection (if edge_dim doesn't match, GINEConv needs it)
        self.edge_proj = nn.Linear(edge_dim, edge_dim) if edge_dim > 0 else None
        self._dummy_edge = edge_dim == 0

        self.drop = nn.Dropout(dropout)

        # Jumping Knowledge — concatenate outputs from all layers
        self.jk = JumpingKnowledge(mode="cat", channels=hidden, num_layers=n_layers)

        # Classifier: JK outputs hidden*n_layers, pool gives 2x (mean+max)
        jk_dim = hidden * n_layers
        pool_dim = jk_dim * 2  # mean + max
        self.clf = nn.Sequential(
            nn.Linear(pool_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def _get_edge_attr(self, data):
        """Get edge attributes, creating dummy ones if needed."""
        if self._dummy_edge:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        ea = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        if ea is None:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        return ea

    def _encode(self, data):
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = self._get_edge_attr(data)

        x = self.input_proj(x)

        # Message passing with jumping knowledge collection
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)
            layer_outputs.append(x)

        # Jumping knowledge aggregation
        x = self.jk(layer_outputs)  # [N, hidden * n_layers]

        return x, batch

    def forward(self, data):
        x, batch = self._encode(data)

        # Dual pooling
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=-1)

        return self.clf(z)

    @torch.no_grad()
    def embed(self, data):
        """Extract embeddings for visualization."""
        x, batch = self._encode(data)
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=-1)
        return z.cpu().numpy()


class SoftBlobGIN(nn.Module):
    """SoftBlob pooling with GIN backbone — lightweight adaptive partitioning."""

    def __init__(self, in_ch: int, hidden: int, n_classes: int,
                 edge_dim: int = 0, n_blobs: int = 8, n_layers: int = 3,
                 dropout: float = 0.3, tau_start: float = 1.0, tau_end: float = 0.1):
        super().__init__()
        self.n_blobs = n_blobs
        self.tau_start = tau_start
        self.tau_end = tau_end
        self._current_tau = tau_start

        # Input projection
        self.input_proj = nn.Linear(in_ch, hidden)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self._dummy_edge = edge_dim == 0

        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            ed = edge_dim if edge_dim > 0 else 1
            self.convs.append(GINEConv(nn=mlp, edge_dim=ed))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.drop = nn.Dropout(dropout)

        # Blob assignment
        self.blob_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_blobs),
        )

        # Blob refinement
        self.blob_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.blob_ln = nn.LayerNorm(hidden)

        # Classifier: blob max-pool + global mean-pool → concat → classify
        self.clf = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def set_tau(self, epoch, total_epochs):
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        self._current_tau = self.tau_start + (self.tau_end - self.tau_start) * progress

    def _get_edge_attr(self, data):
        if self._dummy_edge:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        ea = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        if ea is None:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        return ea

    def _encode(self, data):
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = self._get_edge_attr(data)

        x = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)

        return x, batch

    def forward(self, data, return_blobs=False):
        x, batch = self._encode(data)

        # Soft blob assignment
        logits = self.blob_head(x)
        assign = F.gumbel_softmax(logits, tau=self._current_tau, hard=False, dim=-1)

        # Global embedding
        global_emb = global_mean_pool(x, batch)

        # Blob aggregation
        batch_ids = torch.unique(batch, sorted=True)
        blob_list = []
        for b_id in batch_ids:
            mask = (batch == b_id)
            x_b = x[mask]
            a_b = assign[mask]
            weights = a_b.T / (a_b.T.sum(dim=1, keepdim=True) + 1e-8)
            blobs = weights @ x_b
            blobs = self.blob_ln(self.blob_mlp(blobs))
            blob_list.append(blobs.max(dim=0).values)  # max-pool over blobs

        blob_emb = torch.stack(blob_list, dim=0)

        # Combine global + blob
        z = torch.cat([global_emb, blob_emb], dim=-1)
        out = self.clf(z)

        if return_blobs:
            return out, assign
        return out

    @torch.no_grad()
    def embed(self, data):
        x, batch = self._encode(data)
        logits = self.blob_head(x)
        assign = F.softmax(logits, dim=-1)
        global_emb = global_mean_pool(x, batch)

        batch_ids = torch.unique(batch, sorted=True)
        blob_list = []
        for b_id in batch_ids:
            mask = (batch == b_id)
            x_b = x[mask]
            a_b = assign[mask]
            weights = a_b.T / (a_b.T.sum(dim=1, keepdim=True) + 1e-8)
            blobs = weights @ x_b
            blobs = self.blob_ln(self.blob_mlp(blobs))
            blob_list.append(blobs.max(dim=0).values)

        blob_emb = torch.stack(blob_list, dim=0)
        z = torch.cat([global_emb, blob_emb], dim=-1)
        return z.cpu().numpy()