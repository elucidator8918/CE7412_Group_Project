"""
SoftBlobGIN interpretability analysis.

Extracts and analyzes blob assignments from the trained SoftBlobGIN model:
  - Which residues are grouped into which blobs
  - Per-class blob patterns (do certain EC classes use blobs differently?)
  - Blob spatial coherence (are blobs spatially contiguous in 3D?)
  - Blob-level feature profiles (what AA/physicochemical properties per blob?)
  - Blob importance (which blobs contribute most to the prediction?)

This is the built-in interpretability of SoftBlobGIN — no post-hoc
optimization needed, unlike GNNExplainer.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
EC_NAMES = [
    "Oxidoreductase", "Transferase", "Hydrolase",
    "Lyase", "Isomerase", "Ligase", "Translocase"
]


@dataclass
class BlobResult:
    """Blob analysis for a single protein."""
    protein_idx: int = -1
    true_label: int = -1
    predicted_label: int = -1
    predicted_prob: float = 0.0
    n_nodes: int = 0
    n_blobs: int = 0

    # Blob assignments: [N, K] soft assignment matrix
    assignments: np.ndarray = None

    # Hard assignment: [N] blob index per residue
    hard_assignments: np.ndarray = None

    # Per-blob statistics
    blob_sizes: np.ndarray = None          # [K] number of residues per blob
    blob_spatial_radius: np.ndarray = None  # [K] mean pairwise Cα distance within blob
    blob_aa_composition: np.ndarray = None  # [K, 20] AA composition per blob
    blob_mean_sasa: np.ndarray = None       # [K] mean SASA per blob
    blob_seq_span: np.ndarray = None        # [K] sequence span (max_pos - min_pos) per blob


def extract_blob_assignments(model, data: Data, device: torch.device) -> BlobResult:
    """Extract blob assignments from a trained SoftBlobGIN for a single graph."""
    model.eval()
    data = data.clone().to(device)
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        logits, assign = model(data, return_blobs=True)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        prob = probs[0, pred].item()

    # Soft assignments [N, K]
    assign_np = assign.cpu().numpy()
    # Hard assignments
    hard = assign_np.argmax(axis=1)

    n_nodes = data.num_nodes
    n_blobs = assign_np.shape[1]

    result = BlobResult(
        true_label=int(data.y.item()) if data.y is not None else -1,
        predicted_label=pred,
        predicted_prob=prob,
        n_nodes=n_nodes,
        n_blobs=n_blobs,
        assignments=assign_np,
        hard_assignments=hard,
    )

    # Blob sizes
    result.blob_sizes = np.array([(hard == k).sum() for k in range(n_blobs)])

    # AA composition per blob [K, 20]
    onehot = data.x[:n_nodes, :20].cpu().numpy()
    blob_aa = np.zeros((n_blobs, 20))
    for k in range(n_blobs):
        mask = hard == k
        if mask.sum() > 0:
            blob_aa[k] = onehot[mask].mean(axis=0)
    result.blob_aa_composition = blob_aa

    # SASA per blob
    sasa = data.x[:n_nodes, 30].cpu().numpy()  # SASA at index 30
    blob_sasa = np.zeros(n_blobs)
    for k in range(n_blobs):
        mask = hard == k
        if mask.sum() > 0:
            blob_sasa[k] = sasa[mask].mean()
    result.blob_mean_sasa = blob_sasa

    # Spatial radius per blob (if coords available)
    if hasattr(data, 'coords') and data.coords is not None:
        coords = data.coords[:n_nodes].cpu().numpy()
        blob_radius = np.zeros(n_blobs)
        for k in range(n_blobs):
            mask = hard == k
            if mask.sum() > 1:
                c = coords[mask]
                dists = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
                blob_radius[k] = dists[np.triu_indices(len(c), k=1)].mean()
        result.blob_spatial_radius = blob_radius

    # Sequence span per blob
    blob_span = np.zeros(n_blobs)
    for k in range(n_blobs):
        positions = np.where(hard == k)[0]
        if len(positions) > 0:
            blob_span[k] = positions.max() - positions.min()
    result.blob_seq_span = blob_span

    return result


def extract_blob_batch(model, graphs: list, device: torch.device,
                       verbose: bool = True) -> list:
    """Extract blob assignments for multiple graphs."""
    results = []
    for i, data in enumerate(graphs):
        result = extract_blob_assignments(model, data, device)
        result.protein_idx = i
        results.append(result)
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Blob analysis {i+1}/{len(graphs)} graphs")
    return results


def compute_blob_importance(model, data: Data, device: torch.device,
                            n_blobs: int) -> np.ndarray:
    """Compute per-blob importance scores (analogous to π_t in BioBlobs).

    Two complementary methods:
    1. Max-pool contribution: which blob "wins" the max-pool most often
       across hidden dimensions — this is the direct π_t analog.
    2. Ablation: mask each blob's features and measure confidence drop.

    Returns [K] array combining both signals.
    """
    model.eval()
    data = data.clone().to(device)
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        orig_logits = model(data)
        orig_prob = F.softmax(orig_logits, dim=1)
        pred_class = orig_logits.argmax(dim=1).item()
        orig_conf = orig_prob[0, pred_class].item()

        # Get blob assignments and embeddings
        x, batch = model._encode(data)
        logits_blob = model.blob_head(x)
        assign = F.softmax(logits_blob, dim=-1)  # use softmax at inference
        hard = assign.argmax(dim=1).cpu().numpy()

        # Compute blob embeddings
        mask = (batch == 0)
        x_b = x[mask]
        a_b = assign[mask]
        weights = a_b.T / (a_b.T.sum(dim=1, keepdim=True) + 1e-8)
        blobs = weights @ x_b  # [K, hidden]
        blobs = model.blob_ln(model.blob_mlp(blobs))  # [K, hidden]

        # Method 1: Max-pool contribution (π_t analog)
        # For each hidden dimension, which blob provides the max value?
        # π_t[k] = fraction of hidden dims where blob k is the argmax
        blob_argmax = blobs.argmax(dim=0)  # [hidden] — which blob wins each dim
        pi_t = torch.zeros(n_blobs, device=device)
        for k in range(n_blobs):
            pi_t[k] = (blob_argmax == k).float().mean()
        pi_t = pi_t.cpu().numpy()

    # Method 2: Ablation importance
    ablation_imp = np.zeros(n_blobs)
    for k in range(n_blobs):
        blob_mask = hard == k
        if blob_mask.sum() == 0:
            continue

        masked_data = data.clone()
        x_masked = masked_data.x.float().clone()
        x_masked[blob_mask] = 0.0
        masked_data.x = x_masked

        with torch.no_grad():
            masked_logits = model(masked_data)
            masked_prob = F.softmax(masked_logits, dim=1)
            masked_conf = masked_prob[0, pred_class].item()

        ablation_imp[k] = orig_conf - masked_conf

    # Combine: use π_t as primary (it's the direct BioBlobs analog),
    # ablation as secondary signal
    return pi_t, ablation_imp


def aggregate_blob_stats(blob_results: list, labels: np.ndarray,
                         n_classes: int = 7) -> dict:
    """Aggregate blob statistics per EC class."""
    by_class = defaultdict(list)
    for br, label in zip(blob_results, labels):
        by_class[int(label)].append(br)

    stats = {}
    for c in range(n_classes):
        results = by_class.get(c, [])
        if not results:
            stats[c] = None
            continue

        n_blobs = results[0].n_blobs
        stats[c] = {
            "n_samples": len(results),
            "mean_blob_sizes": np.mean([r.blob_sizes for r in results], axis=0),
            "mean_blob_sasa": np.mean([r.blob_mean_sasa for r in results], axis=0),
            "mean_blob_radius": np.mean(
                [r.blob_spatial_radius for r in results if r.blob_spatial_radius is not None],
                axis=0
            ) if any(r.blob_spatial_radius is not None for r in results) else None,
            "mean_blob_seq_span": np.mean([r.blob_seq_span for r in results], axis=0),
            "mean_aa_composition": np.mean([r.blob_aa_composition for r in results], axis=0),
        }

    return stats


# ============================================================================
# Blob-level biological validation
# ============================================================================

CATALYTIC_AA = set("HCSDEKRY")


def compute_blob_aa_enrichment(blob_results: list, labels: np.ndarray,
                                n_classes: int = 7) -> dict:
    """Per-blob AA enrichment relative to whole-protein background.

    Returns dict[class_id] -> [K, 20] log2 enrichment matrix.
    """
    by_class = defaultdict(list)
    for br, label in zip(blob_results, labels):
        by_class[int(label)].append(br)

    enrichment = {}
    for c in range(n_classes):
        results = by_class.get(c, [])
        if not results:
            enrichment[c] = None
            continue

        n_blobs = results[0].n_blobs
        # Mean AA composition per blob across all proteins in this class
        blob_aa = np.mean([r.blob_aa_composition for r in results], axis=0)  # [K, 20]
        # Background: mean AA composition across all residues
        bg_aa = blob_aa.mean(axis=0, keepdims=True)  # [1, 20]

        blob_freq = blob_aa / (blob_aa.sum(axis=1, keepdims=True) + 1e-8)
        bg_freq = bg_aa / (bg_aa.sum(axis=1, keepdims=True) + 1e-8)
        enrichment[c] = np.log2((blob_freq + 1e-6) / (bg_freq + 1e-6))

    return enrichment


def compute_blob_sasa_profiles(blob_results: list, labels: np.ndarray,
                                n_classes: int = 7) -> dict:
    """Per-blob mean SASA per class.

    Returns dict[class_id] -> [K] mean SASA per blob.
    """
    by_class = defaultdict(list)
    for br, label in zip(blob_results, labels):
        by_class[int(label)].append(br)

    profiles = {}
    for c in range(n_classes):
        results = by_class.get(c, [])
        if not results:
            profiles[c] = None
            continue
        profiles[c] = np.mean([r.blob_mean_sasa for r in results], axis=0)

    return profiles


def compute_gin_blob_overlap(gnnexp_results: list, blob_results: list,
                              top_frac: float = 0.2) -> list:
    """Compute overlap between GIN's GNNExplainer important residues and
    SoftBlobGIN's most important blob.

    For each protein, finds which blob has the highest importance, then
    computes Jaccard overlap with GNNExplainer's top-K residues.

    Returns list of (jaccard, blob_id) per protein.
    """
    overlaps = []
    for gnn_r, blob_r in zip(gnnexp_results, blob_results):
        if gnn_r.node_importance is None or blob_r.hard_assignments is None:
            overlaps.append((float('nan'), -1))
            continue

        n = min(len(gnn_r.node_importance), len(blob_r.hard_assignments))
        if n < 3:
            overlaps.append((float('nan'), -1))
            continue

        # GNNExplainer top-K residues
        k = max(1, int(top_frac * n))
        gnn_top = set(np.argsort(gnn_r.node_importance[:n])[-k:])

        # Most important blob's residues (by size — largest non-dominant blob)
        # Find the blob with most residues excluding the dominant one
        sizes = blob_r.blob_sizes.copy()
        dominant = sizes.argmax()
        sizes[dominant] = 0
        best_blob = sizes.argmax() if sizes.max() > 0 else dominant

        blob_residues = set(np.where(blob_r.hard_assignments[:n] == best_blob)[0])

        if not blob_residues or not gnn_top:
            overlaps.append((0.0, best_blob))
            continue

        jaccard = len(gnn_top & blob_residues) / len(gnn_top | blob_residues)
        overlaps.append((jaccard, best_blob))

    return overlaps
