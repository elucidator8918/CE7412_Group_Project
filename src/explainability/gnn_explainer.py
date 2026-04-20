"""
GNNExplainer for graph classification (Ying et al., NeurIPS 2019).

Adapted for enzyme classification with GIN models. Learns:
  - Edge mask M ∈ R^E: importance of each edge in the contact graph
  - Feature mask F ∈ R^d: importance of each node feature dimension

Optimization objective (Eq. 5 from paper):
  min_M  -sum_c 1[y=c] * log P_Φ(Y=y | G=A_c⊙σ(M), X=X^F)
  + λ_size * ||σ(M)||_1
  + λ_ent * H(σ(M))
  + λ_feat_size * ||σ(F)||_1

For graph classification, the mask covers ALL edges in the graph since
the readout aggregates all node embeddings.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Container for a single-instance explanation."""
    # Input info
    protein_idx: int = -1
    true_label: int = -1
    predicted_label: int = -1
    predicted_prob: float = 0.0

    # Edge explanation
    edge_mask: np.ndarray = None        # [E] importance per edge
    edge_index: np.ndarray = None       # [2, E] edge indices

    # Feature explanation
    feature_mask: np.ndarray = None     # [d] importance per feature dim

    # Node-level importance (derived from edge mask)
    node_importance: np.ndarray = None  # [N] per-residue importance

    # Metadata
    n_nodes: int = 0
    n_edges: int = 0
    masked_pred_prob: float = 0.0       # model confidence on masked graph


class GNNExplainerGraph:
    """GNNExplainer for graph-level classification tasks.

    Learns edge and feature masks that maximize mutual information
    between the explanation subgraph and the model's prediction.

    Args:
        model: Trained GIN model with edge_weight support
        device: torch device
        epochs: optimization steps for mask learning
        lr: learning rate for mask parameters
        lambda_size: L1 penalty on edge mask (sparsity)
        lambda_ent: entropy penalty on edge mask (discreteness)
        lambda_feat_size: L1 penalty on feature mask
        lambda_feat_ent: entropy penalty on feature mask
        init_bias: initial bias for mask logits (0.0 = start at 0.5)
        temp: temperature for feature marginalization sampling
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epochs: int = 300,
        lr: float = 0.01,
        lambda_size: float = 0.07,
        lambda_ent: float = 0.1,
        lambda_feat_size: float = 0.01,
        lambda_feat_ent: float = 0.1,
        init_bias: float = 0.0,
        temp: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.lambda_size = lambda_size
        self.lambda_ent = lambda_ent
        self.lambda_feat_size = lambda_feat_size
        self.lambda_feat_ent = lambda_feat_ent
        self.init_bias = init_bias
        self.temp = temp

        self.model.eval()

    def explain(self, data: Data, target_class: int = None) -> ExplanationResult:
        """Generate explanation for a single graph's prediction.

        Args:
            data: PyG Data object (single graph, no batch dimension)
            target_class: class to explain (default: model's predicted class)

        Returns:
            ExplanationResult with edge and feature importance scores
        """
        data = data.clone().to(self.device)

        # Ensure batch attribute exists for single graph
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long,
                                     device=self.device)

        # Get model's original prediction
        self.model.eval()
        with torch.no_grad():
            original_logits = self.model(data)
            original_probs = F.softmax(original_logits, dim=1)
            pred_class = original_logits.argmax(dim=1).item()
            pred_prob = original_probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Initialize masks
        n_edges = data.edge_index.shape[1]
        feat_dim = data.x.shape[1]

        # Edge mask logits — initialized at init_bias (0.0 = start at σ=0.5)
        edge_mask_logits = nn.Parameter(
            torch.full((n_edges,), self.init_bias, device=self.device)
        )

        # Feature mask logits — initialized slightly positive so features
        # start mostly included, then get pruned
        feat_mask_logits = nn.Parameter(
            torch.full((feat_dim,), 1.0, device=self.device)
        )

        # Compute empirical feature marginal for marginalization trick
        # Z ~ empirical distribution of node features in this graph
        feat_marginal = data.x.float().mean(dim=0).detach()

        optimizer = torch.optim.Adam([edge_mask_logits, feat_mask_logits],
                                     lr=self.lr)

        # Precompute node degree for normalization
        ei_cpu = data.edge_index.cpu()
        node_degree = torch.zeros(data.num_nodes, device=self.device)
        node_degree.scatter_add_(0, data.edge_index[0],
                                 torch.ones(n_edges, device=self.device))
        node_degree = node_degree.clamp(min=1.0)

        # Optimization loop
        best_loss = float('inf')
        best_edge_mask = None
        best_feat_mask = None

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Compute masks via sigmoid
            edge_mask = torch.sigmoid(edge_mask_logits)
            feat_mask = torch.sigmoid(feat_mask_logits)

            # Apply feature marginalization trick (Section 4.2):
            # X = Z + (X_S - Z) ⊙ F
            # When F=1: use original feature. When F=0: use marginal.
            x_masked = feat_marginal + (data.x.float() - feat_marginal) * feat_mask

            # Create masked data
            masked_data = Data(
                x=x_masked,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
                num_nodes=data.num_nodes,
            )

            # Forward pass with edge weights
            logits = self.model(masked_data, edge_weight=edge_mask)
            log_probs = F.log_softmax(logits, dim=1)

            # Prediction loss: cross-entropy for target class (Eq. 5)
            pred_loss = -log_probs[0, target_class]

            # Regularization: edge mask size (L1 sparsity)
            # Normalize by number of edges so lambda is graph-size-independent
            size_loss = self.lambda_size * (edge_mask.sum() / n_edges)

            # Regularization: edge mask entropy (encourage discreteness)
            edge_ent = -edge_mask * torch.log(edge_mask + 1e-8) \
                       - (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
            ent_loss = self.lambda_ent * edge_ent.mean()

            # Regularization: feature mask size (normalized)
            feat_size_loss = self.lambda_feat_size * (feat_mask.sum() / feat_dim)

            # Regularization: feature mask entropy
            feat_ent = -feat_mask * torch.log(feat_mask + 1e-8) \
                       - (1 - feat_mask) * torch.log(1 - feat_mask + 1e-8)
            feat_ent_loss = self.lambda_feat_ent * feat_ent.mean()

            # Total loss
            loss = pred_loss + size_loss + ent_loss + feat_size_loss + feat_ent_loss
            loss.backward()
            optimizer.step()

            # Track best (lowest pred_loss while maintaining sparsity)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_edge_mask = edge_mask.detach().clone()
                best_feat_mask = feat_mask.detach().clone()

        # Use best masks found during optimization
        final_edge_mask = best_edge_mask.cpu().numpy()
        final_feat_mask = best_feat_mask.cpu().numpy()

        # Compute node importance from edge mask:
        # node importance = mean of incident edge importances
        with torch.no_grad():
            node_imp_t = torch.zeros(data.num_nodes, device=self.device)
            node_imp_t.scatter_add_(0, data.edge_index[0], best_edge_mask)
            node_imp_t.scatter_add_(0, data.edge_index[1], best_edge_mask)
            node_imp = (node_imp_t / (2.0 * node_degree)).cpu().numpy()

            # Get masked prediction confidence
            x_m = feat_marginal + (data.x.float() - feat_marginal) * best_feat_mask
            masked_data = Data(
                x=x_m, edge_index=data.edge_index,
                edge_attr=data.edge_attr, batch=data.batch,
                num_nodes=data.num_nodes,
            )
            masked_logits = self.model(masked_data, edge_weight=best_edge_mask)
            masked_prob = F.softmax(masked_logits, dim=1)[0, target_class].item()

        return ExplanationResult(
            predicted_label=pred_class,
            true_label=int(data.y.item()) if data.y is not None else -1,
            predicted_prob=pred_prob,
            edge_mask=final_edge_mask,
            edge_index=data.edge_index.cpu().numpy(),
            feature_mask=final_feat_mask,
            node_importance=node_imp,
            n_nodes=data.num_nodes,
            n_edges=n_edges,
            masked_pred_prob=masked_prob,
        )

    def explain_batch(self, graphs: list, target_classes: list = None,
                      verbose: bool = True) -> list:
        """Generate explanations for multiple graphs.

        Args:
            graphs: list of PyG Data objects
            target_classes: list of target classes (None = use predicted)
            verbose: show progress

        Returns:
            list of ExplanationResult
        """
        results = []
        n = len(graphs)

        for i, data in enumerate(graphs):
            tc = target_classes[i] if target_classes else None
            result = self.explain(data, target_class=tc)
            result.protein_idx = i
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Explained {i+1}/{n} graphs")

        return results
