"""
Quantitative evaluation metrics for GNN explanations.

Metrics:
  - Fidelity+ (sufficiency): model accuracy using ONLY top-K important edges
  - Fidelity- (necessity): accuracy DROP when REMOVING top-K important edges
  - Sparsity: fraction of edges/features retained in explanation
  - Stability: consistency of explanations across similar proteins

These metrics are essential for NeurIPS-level evaluation — visualizations
alone are insufficient.

References:
  - Pope et al., "Explainability Methods for GNNs", CVPR 2019
  - Yuan et al., "Explainability in GNNs: A Taxonomic Survey", TPAMI 2022
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class FidelityResult:
    """Fidelity evaluation at a single sparsity level."""
    sparsity: float           # fraction of edges retained
    fidelity_plus: float      # accuracy with only top-K edges
    fidelity_minus: float     # accuracy drop when removing top-K edges
    prob_fidelity_plus: float  # prob difference (sufficiency)
    prob_fidelity_minus: float  # prob difference (necessity)


def compute_fidelity_metrics(
    model,
    graphs: list,
    edge_masks: list,
    device: torch.device,
    sparsity_levels: list = None,
) -> list:
    """Compute fidelity metrics across multiple sparsity levels.

    Args:
        model: trained GIN model
        graphs: list of PyG Data objects
        edge_masks: list of numpy arrays [E_i] with edge importance scores
        device: torch device
        sparsity_levels: list of fractions (e.g., [0.1, 0.2, 0.3, 0.5])

    Returns:
        list of FidelityResult, one per sparsity level
    """
    if sparsity_levels is None:
        sparsity_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    model.eval()
    results = []

    for sparsity in sparsity_levels:
        fid_plus_correct = 0
        fid_minus_correct = 0
        prob_plus_sum = 0.0
        prob_minus_sum = 0.0
        n_total = 0

        for data, emask in zip(graphs, edge_masks):
            data = data.clone().to(device)
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long,
                                         device=device)

            n_edges = data.edge_index.shape[1]
            if n_edges == 0:
                continue

            # Original prediction
            with torch.no_grad():
                orig_logits = model(data)
                orig_pred = orig_logits.argmax(dim=1).item()
                orig_prob = F.softmax(orig_logits, dim=1)[0, orig_pred].item()

            # Top-K edges by importance
            k = max(1, int(sparsity * n_edges))
            top_k_idx = np.argsort(emask)[-k:]  # highest importance
            bottom_idx = np.argsort(emask)[:-k]  # lowest importance

            # Fidelity+ : keep only top-K edges
            top_k_weight = torch.zeros(n_edges, device=device)
            top_k_weight[top_k_idx] = 1.0
            with torch.no_grad():
                plus_logits = model(data, edge_weight=top_k_weight)
                plus_pred = plus_logits.argmax(dim=1).item()
                plus_prob = F.softmax(plus_logits, dim=1)[0, orig_pred].item()

            # Fidelity- : remove top-K edges (keep the rest)
            minus_weight = torch.ones(n_edges, device=device)
            minus_weight[top_k_idx] = 0.0
            with torch.no_grad():
                minus_logits = model(data, edge_weight=minus_weight)
                minus_pred = minus_logits.argmax(dim=1).item()
                minus_prob = F.softmax(minus_logits, dim=1)[0, orig_pred].item()

            # Accumulate
            fid_plus_correct += int(plus_pred == orig_pred)
            fid_minus_correct += int(minus_pred == orig_pred)
            prob_plus_sum += (orig_prob - plus_prob)  # lower = better explanation
            prob_minus_sum += (orig_prob - minus_prob)  # higher = better explanation
            n_total += 1

        if n_total == 0:
            continue

        results.append(FidelityResult(
            sparsity=sparsity,
            fidelity_plus=fid_plus_correct / n_total,
            fidelity_minus=1.0 - (fid_minus_correct / n_total),
            prob_fidelity_plus=1.0 - (prob_plus_sum / n_total),
            prob_fidelity_minus=prob_minus_sum / n_total,
        ))

    return results


def compute_sparsity(edge_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute sparsity of an edge mask.

    Returns fraction of edges with importance below threshold.
    """
    return float((edge_mask < threshold).sum()) / len(edge_mask)


def compute_stability(explanations: list, labels: np.ndarray,
                      n_classes: int = 7) -> dict:
    """Compute explanation stability within each class.

    Measures how consistent explanations are for proteins of the same class
    using cosine similarity of feature masks.

    Args:
        explanations: list of ExplanationResult objects
        labels: true labels for each explained protein
        n_classes: number of classes

    Returns:
        dict with per-class mean cosine similarity of feature masks
    """
    from itertools import combinations

    stability = {}
    for c in range(n_classes):
        class_idx = [i for i, l in enumerate(labels) if l == c]
        if len(class_idx) < 2:
            stability[c] = float('nan')
            continue

        # Pairwise cosine similarity of feature masks
        masks = [explanations[i].feature_mask for i in class_idx]
        sims = []
        for i, j in combinations(range(len(masks)), 2):
            m1 = masks[i] / (np.linalg.norm(masks[i]) + 1e-8)
            m2 = masks[j] / (np.linalg.norm(masks[j]) + 1e-8)
            sims.append(float(np.dot(m1, m2)))

        stability[c] = float(np.mean(sims))

    return stability
