"""
Quantitative evaluation metrics for GNN explanations.

Metrics:
  - Fidelity+ (sufficiency): model accuracy using ONLY top-K important edges
  - Fidelity- (necessity): accuracy DROP when REMOVING top-K important edges
  - Sparsity: fraction of edges/features retained in explanation
  - Stability: consistency of explanations across similar proteins
  - Unfaithfulness: KL divergence between original and perturbed predictions
  - Characterization score: AUROC of edge mask as binary classifier

References:
  - Pope et al., "Explainability Methods for GNNs", CVPR 2019
  - Yuan et al., "Explainability in GNNs: A Taxonomic Survey", TPAMI 2022
  - PyG torch_geometric.explain.metric API
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
    """
    from itertools import combinations

    stability = {}
    for c in range(n_classes):
        class_idx = [i for i, l in enumerate(labels) if l == c]
        if len(class_idx) < 2:
            stability[c] = float('nan')
            continue

        masks = [explanations[i].feature_mask for i in class_idx]
        sims = []
        for i, j in combinations(range(len(masks)), 2):
            m1 = masks[i] / (np.linalg.norm(masks[i]) + 1e-8)
            m2 = masks[j] / (np.linalg.norm(masks[j]) + 1e-8)
            sims.append(float(np.dot(m1, m2)))

        stability[c] = float(np.mean(sims))

    return stability


# ============================================================================
# PyG-compatible explanation metrics
# ============================================================================

def compute_unfaithfulness(model, graphs: list, edge_masks: list,
                           device: torch.device,
                           n_perturbations: int = 10,
                           seed: int = 42) -> float:
    """Unfaithfulness metric (aligned with PyG's unfaithfulness).

    Randomly perturbs the edge mask and measures how well the mask's
    importance scores predict the change in model output. Lower = more faithful.

    For each graph, samples binary perturbation masks, computes the model
    output difference, and correlates it with the explanation's predicted
    importance difference. Unfaithfulness = 1 - correlation.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    kl_divs = []

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
            orig_probs = F.softmax(orig_logits, dim=1)

        # Perturb: sample random subsets of edges to keep
        for _ in range(n_perturbations):
            # Random binary mask
            keep_prob = rng.uniform(0.1, 0.9)
            rand_mask = torch.tensor(
                rng.random(n_edges) < keep_prob, dtype=torch.float32,
                device=device
            )

            with torch.no_grad():
                pert_logits = model(data, edge_weight=rand_mask)
                pert_probs = F.softmax(pert_logits, dim=1)

            # KL divergence between original and perturbed
            kl = F.kl_div(pert_probs.log(), orig_probs, reduction='batchmean')

            # Expected importance of removed edges
            removed_importance = ((1 - rand_mask.cpu().numpy()) * emask).sum()

            kl_divs.append((kl.item(), removed_importance))

    if not kl_divs:
        return float('nan')

    # Unfaithfulness = 1 - Spearman correlation between
    # removed_importance and actual KL divergence
    from scipy.stats import spearmanr
    kls = [x[0] for x in kl_divs]
    imps = [x[1] for x in kl_divs]
    corr, _ = spearmanr(kls, imps)
    if np.isnan(corr):
        return float('nan')
    return 1.0 - corr


def compute_characterization_score(model, graphs: list, edge_masks: list,
                                    device: torch.device,
                                    threshold: float = 0.5) -> float:
    """Characterization score (aligned with PyG's characterization_score).

    Treats the edge mask as a binary classifier: edges with mask > threshold
    are "positive" (important). Measures how well this binary classification
    predicts whether removing the edge changes the model's prediction.

    Returns AUROC of edge_mask as predictor of prediction-changing edges.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_mask_vals = []
    all_labels = []  # 1 if removing edge changes prediction, 0 otherwise

    for data, emask in zip(graphs, edge_masks):
        data = data.clone().to(device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long,
                                     device=device)

        n_edges = data.edge_index.shape[1]
        if n_edges == 0 or n_edges > 2000:  # skip very large graphs for speed
            continue

        with torch.no_grad():
            orig_pred = model(data).argmax(dim=1).item()

        # Sample edges to test (not all — too expensive)
        n_sample = min(50, n_edges)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_edges, size=n_sample, replace=False)

        for idx in sample_idx:
            # Remove single edge
            weight = torch.ones(n_edges, device=device)
            weight[idx] = 0.0
            with torch.no_grad():
                new_pred = model(data, edge_weight=weight).argmax(dim=1).item()

            all_mask_vals.append(emask[idx])
            all_labels.append(1 if new_pred != orig_pred else 0)

    if not all_labels or sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        return float('nan')

    return roc_auc_score(all_labels, all_mask_vals)


def compute_pyg_metrics(model, graphs: list, edge_masks: list,
                        device: torch.device) -> dict:
    """Compute all PyG-aligned explanation metrics.

    Returns dict with:
      - unfaithfulness: lower = more faithful explanation
      - characterization_score: AUROC of mask as edge importance predictor
    """
    logger.info("  Computing unfaithfulness...")
    unfaith = compute_unfaithfulness(model, graphs, edge_masks, device)

    logger.info("  Computing characterization score...")
    char_score = compute_characterization_score(model, graphs, edge_masks, device)

    return {
        "unfaithfulness": unfaith,
        "characterization_score": char_score,
    }
