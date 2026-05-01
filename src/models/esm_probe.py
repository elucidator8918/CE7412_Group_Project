"""
ESM-2 + Linear Probe baseline.

Frozen ESM-2 per-residue embeddings → mean-pool per protein → single linear layer.

This establishes the sequence-only ceiling: the maximum performance achievable
from ESM-2 without any structural information. The GNN's marginal gains over
this baseline directly measure the value of 3-D structure.

All models accept PyG Data/Batch objects where data.x = [N, esm_dim] holds
the pre-computed, pre-frozen ESM-2 per-residue embeddings (ESM-2 is never
called at model forward time — it was extracted offline and cached).

Task variants mirror the SoftBlobGIN task split exactly, with a single linear
head replacing the GNN + MLP stack:

  ESMProbeClassifier     — graph_multiclass   (ProteinFamily, StructuralClass)
  ESMProbeMultiLabel     — graph_multilabel   (EnzymeClass, GeneOntology)
  ESMProbeRegressor      — graph_regression   (LigandAffinity)
  ESMProbeNodeClassifier — node_binary        (BindingSiteDetection)
  ESMProbePPI            — node_binary pair   (ProteinProteinInterface)
  ESMProbeSiamese        — pair_regression/   (StructureSimilarity, StructureSearch)
                           pair_retrieval
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class ESMProbeClassifier(nn.Module):
    """Linear probe for graph-level multiclass classification."""

    def __init__(self, esm_dim: int = 1280, n_classes: int = 7):
        super().__init__()
        self.linear = nn.Linear(esm_dim, n_classes)

    def forward(self, data):
        z = global_mean_pool(data.x.float(), data.batch)  # [B, esm_dim]
        return self.linear(z)                              # [B, n_classes]


class ESMProbeMultiLabel(nn.Module):
    """Linear probe for graph-level multi-label classification."""

    def __init__(self, esm_dim: int = 1280, n_classes: int = 100):
        super().__init__()
        self.linear = nn.Linear(esm_dim, n_classes)

    def forward(self, data):
        z = global_mean_pool(data.x.float(), data.batch)
        return self.linear(z)  # raw logits — sigmoid applied in loss/eval


class ESMProbeRegressor(nn.Module):
    """Linear probe for graph-level regression."""

    def __init__(self, esm_dim: int = 1280):
        super().__init__()
        self.linear = nn.Linear(esm_dim, 1)

    def forward(self, data):
        z = global_mean_pool(data.x.float(), data.batch)
        return self.linear(z).squeeze(-1)  # [B]


class ESMProbeNodeClassifier(nn.Module):
    """Linear probe for node-level binary classification (BindingSiteDetection)."""

    def __init__(self, esm_dim: int = 1280):
        super().__init__()
        self.linear = nn.Linear(esm_dim, 1)

    def forward(self, data):
        return self.linear(data.x.float()).squeeze(-1)  # [N] logits


class ESMProbePPI(nn.Module):
    """Linear probe for Protein-Protein Interface node prediction.

    For each residue in protein A, concatenates:
      - its own per-residue ESM-2 embedding
      - mean-pooled ESM-2 of protein B (global cross-protein context)
    → single linear layer → binary logit per residue of A.
    """

    def __init__(self, esm_dim: int = 1280):
        super().__init__()
        self.linear = nn.Linear(esm_dim * 2, 1)

    def forward(self, data1, data2=None):
        if data2 is None:
            data2 = data1.b2
            data1 = data1.b1
        z_b = global_mean_pool(data2.x.float(), data2.batch)  # [B, esm_dim]
        # Broadcast protein-B context to every residue of protein A
        z_b_per_node = z_b[data1.batch]                        # [N_A, esm_dim]
        cat = torch.cat([data1.x.float(), z_b_per_node], dim=-1)
        return self.linear(cat).squeeze(-1)                    # [N_A] logits


class ESMProbeSiamese(nn.Module):
    """Linear probe for protein-pair tasks (StructureSimilarity, StructureSearch).

    Uses the same 4-way feature vector as SoftBlobGINSiamese:
      [z1 | z2 | |z1-z2| | z1*z2] → Linear(esm_dim*4, 1)
    This matches the SoftBlobGIN head exactly so differences are due to the
    encoder (ESM mean-pool vs GNN) and not the similarity head.
    """

    def __init__(self, esm_dim: int = 1280):
        super().__init__()
        self.linear = nn.Linear(esm_dim * 4, 1)

    def forward(self, data1, data2=None):
        if data2 is None:
            data2 = data1.b2
            data1 = data1.b1
        z1 = global_mean_pool(data1.x.float(), data1.batch)  # [B, esm_dim]
        z2 = global_mean_pool(data2.x.float(), data2.batch)
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        return self.linear(combined).squeeze(-1)              # [B]
