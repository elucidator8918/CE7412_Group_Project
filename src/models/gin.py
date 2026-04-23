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


class GINRegressor(nn.Module):
    """GIN for graph-level regression (e.g. LigandAffinityTask).

    Same encoder as GINModel, but outputs a single scalar per graph.
    """

    def __init__(self, in_ch: int, hidden: int,
                 edge_dim: int = 0, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_ch, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(), nn.Linear(hidden, hidden),
            )
            ed = edge_dim if edge_dim > 0 else 1
            self.convs.append(GINEConv(nn=mlp, edge_dim=ed))
            self.bns.append(nn.BatchNorm1d(hidden))
        self._dummy_edge = edge_dim == 0
        self.drop = nn.Dropout(dropout)
        self.jk = JumpingKnowledge(mode="cat", channels=hidden, num_layers=n_layers)
        jk_dim = hidden * n_layers
        pool_dim = jk_dim * 2
        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def _get_edge_attr(self, data):
        if self._dummy_edge:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        ea = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        return ea if ea is not None else torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)

    def _encode(self, data):
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = self._get_edge_attr(data)
        x = self.input_proj(x)
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei, edge_attr=edge_attr)
            x = bn(x); x = F.relu(x); x = self.drop(x)
            layer_outputs.append(x)
        x = self.jk(layer_outputs)
        return x, batch

    def forward(self, data):
        x, batch = self._encode(data)
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=-1)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def embed(self, data):
        x, batch = self._encode(data)
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=-1)
        return z.cpu().numpy()


class GINMultiLabel(nn.Module):
    """GIN for graph-level multi-label classification (e.g. GeneOntologyTask).

    Same encoder as GINModel, sigmoid output instead of softmax.
    """

    def __init__(self, in_ch: int, hidden: int, n_classes: int,
                 edge_dim: int = 0, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_ch, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(), nn.Linear(hidden, hidden),
            )
            ed = edge_dim if edge_dim > 0 else 1
            self.convs.append(GINEConv(nn=mlp, edge_dim=ed))
            self.bns.append(nn.BatchNorm1d(hidden))
        self._dummy_edge = edge_dim == 0
        self.drop = nn.Dropout(dropout)
        self.jk = JumpingKnowledge(mode="cat", channels=hidden, num_layers=n_layers)
        jk_dim = hidden * n_layers
        pool_dim = jk_dim * 2
        self.clf = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def _get_edge_attr(self, data):
        if self._dummy_edge:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        ea = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        return ea if ea is not None else torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)

    def _encode(self, data):
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = self._get_edge_attr(data)
        x = self.input_proj(x)
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei, edge_attr=edge_attr)
            x = bn(x); x = F.relu(x); x = self.drop(x)
            layer_outputs.append(x)
        x = self.jk(layer_outputs)
        return x, batch

    def forward(self, data):
        x, batch = self._encode(data)
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=-1)
        return self.clf(z)  # raw logits — apply sigmoid in loss/eval

    @torch.no_grad()
    def embed(self, data):
        x, batch = self._encode(data)
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=-1)
        return z.cpu().numpy()


class GINNodeClassifier(nn.Module):
    """GIN for node-level binary classification (e.g. BindingSiteDetection).

    Same encoder as GINModel, but returns per-node predictions instead of
    graph-level pooling. Uses JK output directly for node classification.
    """

    def __init__(self, in_ch: int, hidden: int,
                 edge_dim: int = 0, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_ch, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(), nn.Linear(hidden, hidden),
            )
            ed = edge_dim if edge_dim > 0 else 1
            self.convs.append(GINEConv(nn=mlp, edge_dim=ed))
            self.bns.append(nn.BatchNorm1d(hidden))
        self._dummy_edge = edge_dim == 0
        self.drop = nn.Dropout(dropout)
        self.jk = JumpingKnowledge(mode="cat", channels=hidden, num_layers=n_layers)
        jk_dim = hidden * n_layers
        self.node_head = nn.Sequential(
            nn.Linear(jk_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def _get_edge_attr(self, data):
        if self._dummy_edge:
            return torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)
        ea = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        return ea if ea is not None else torch.ones(data.edge_index.shape[1], 1, device=data.edge_index.device)

    def _encode(self, data):
        x, ei, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = self._get_edge_attr(data)
        x = self.input_proj(x)
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei, edge_attr=edge_attr)
            x = bn(x); x = F.relu(x); x = self.drop(x)
            layer_outputs.append(x)
        x = self.jk(layer_outputs)
        return x, batch

    def forward(self, data):
        x, batch = self._encode(data)
        return self.node_head(x).squeeze(-1)  # [total_nodes] logits

    @torch.no_grad()
    def embed(self, data):
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

    def forward_internal(self, data):
        """Returns the [mean, max] combined embedding before the classifier head."""
        x, batch = self._encode(data)
        assign = F.gumbel_softmax(self.blob_head(x), tau=self._current_tau, hard=False, dim=-1)
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
            blob_list.append(blobs.max(dim=0).values)  # max-pool over blobs

        blob_emb = torch.stack(blob_list, dim=0)
        return torch.cat([global_emb, blob_emb], dim=-1)

    def forward(self, data, return_blobs=False):
        z = self.forward_internal(data)
        out = self.clf(z)
        if return_blobs:
            # Re-calculate assign for return if needed, or refactor
            x, _ = self._encode(data)
            assign = F.softmax(self.blob_head(x), dim=-1)
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


class SoftBlobGINRegressor(SoftBlobGIN):
    """SoftBlobGIN for graph-level regression."""
    def __init__(self, *args, **kwargs):
        kwargs['n_classes'] = 1
        super().__init__(*args, **kwargs)
        self.clf[-1] = nn.Linear(self.clf[-1].in_features, 1)

    def forward(self, data):
        return super().forward(data).squeeze(-1)


class SoftBlobGINMultiLabel(SoftBlobGIN):
    """SoftBlobGIN for graph-level multi-label classification."""
    pass  # Same architecture, just raw logits (sigmoid in loss)


class SoftBlobGINNodeClassifier(nn.Module):
    """Bypasses blob pooling for node-level tasks using the GIN backbone."""
    def __init__(self, in_ch, hidden, edge_dim=0, n_layers=3, dropout=0.3):
        super().__init__()
        # Re-use the encoder part of SoftBlobGIN logic via a simple GIN wrapper
        # or just use the GINNodeClassifier already defined above.
        # Minimal change: just use GINNodeClassifier as it is architecturaly 
        # identical to the backbone of SoftBlobGIN.
        self.model = GINNodeClassifier(in_ch, hidden, edge_dim, n_layers, dropout)

    def forward(self, data):
        return self.model(data)

    def set_tau(self, *args, **kwargs):
        pass  # No temperature needed for node classification


class SoftBlobGINSiamese(nn.Module):
    """Wrapper for pairwise tasks (Similarity/Search).

    Encodes two graphs independently, computes similarity between embeddings.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Check if encoder has clf[0] to get input features
        if hasattr(encoder, "clf"):
            pool_dim = encoder.clf[0].in_features
        elif hasattr(encoder, "head"):
             pool_dim = encoder.head[0].in_features
             
        else:
            # Fallback for base SoftBlobGIN
            hidden = encoder.blob_mlp[0].in_features
            pool_dim = hidden * 2
            
        self.head = nn.Sequential(
            nn.Linear(pool_dim * 4, pool_dim),
            nn.ReLU(),
            nn.Linear(pool_dim, 1)
        )

    def forward(self, data1, data2=None):
        if data2 is None:
            # Handle single PairBatch object
            data2 = data1.b2
            data1 = data1.b1
            
        z1 = self.encoder.forward_internal(data1)
        z2 = self.encoder.forward_internal(data2)
        # Combined feature vector for the similarity head
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        return self.head(combined).squeeze(-1)

    def set_tau(self, epoch, total_epochs):
        if hasattr(self.encoder, "set_tau"):
            self.encoder.set_tau(epoch, total_epochs)


class SoftBlobGINPPI(nn.Module):
    """Wrapper for Protein-Protein Interface (PPI) task.

    Encodes Protein A (node-level) and Protein B (global-level).
    Concatenates B's global embedding to every node in A.
    """
    def __init__(self, node_encoder, graph_encoder, proj_dim=128):
        super().__init__()
        self.node_encoder = node_encoder
        self.graph_encoder = graph_encoder
        
        # Robustly extract hidden dimension from various encoder types
        node_model = self.node_encoder.model if hasattr(self.node_encoder, "model") else self.node_encoder
        if hasattr(node_model, "node_head"):
             node_hidden = node_model.node_head[0].in_features
        elif hasattr(node_model, "classifier"):
             node_hidden = node_model.classifier[0].in_features
        else:
             node_hidden = 1024 # Fallback assuming n_layers=4, hidden=256
             
        if hasattr(self.graph_encoder, "clf"):
             graph_hidden = self.graph_encoder.clf[0].in_features
        else:
             graph_hidden = 512 # Fallback assuming hidden=256 * 2 (mean pool + blob pool)
             
        self.node_proj = nn.Sequential(
            nn.Linear(node_hidden + graph_hidden, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, data1, data2=None):
        if data2 is None:
            # Handle single PairBatch object
            data2 = data1.b2
            data1 = data1.b1
            
        # data1: Protein A (nodes), data2: Protein B (context)
        
        # SoftBlobGINNodeClassifier has the actual encoder in its .model attribute
        node_model = self.node_encoder.model if hasattr(self.node_encoder, "model") else self.node_encoder
        
        h1, _ = node_model._encode(data1) # [N1, node_hidden]
        h2, _ = node_model._encode(data2) # [N2, node_hidden]
        
        z1 = self.graph_encoder.forward_internal(data1) # [Batch, graph_hidden]
        z2 = self.graph_encoder.forward_internal(data2) # [Batch, graph_hidden]
        
        # Expand global context
        z1_ext = z1[data2.batch] # [N2, graph_hidden]
        z2_ext = z2[data1.batch] # [N1, graph_hidden]
        
        # Project into interactive space
        q1 = self.node_proj(torch.cat([h1, z2_ext], dim=-1)) # [N1, proj_dim]
        k2 = self.node_proj(torch.cat([h2, z1_ext], dim=-1)) # [N2, proj_dim]
        
        # Since we use PyG Batch, N1 and N2 are sums over the batch.
        # We need to perform block-wise dot product.
        # For simplicity in 'crude approximation', we can use a loop or batch matmul if we pad.
        # But even better: we can just do outer product and mask it, or use a list of matrices.
        
        # List-based approach to handle varying N1, N2 in a batch
        batch1 = data1.batch
        batch2 = data2.batch
        u1 = torch.unique(batch1)
        
        out_list = []
        for b_id in u1:
            q1_b = q1[batch1 == b_id]
            k2_b = k2[batch2 == b_id]
            out_list.append(torch.matmul(q1_b, k2_b.T))
            
        return out_list

    def set_tau(self, epoch, total_epochs):
        if hasattr(self.graph_encoder, "set_tau"):
            self.graph_encoder.set_tau(epoch, total_epochs)