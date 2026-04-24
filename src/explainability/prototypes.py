"""
Multi-instance explanations via class prototypes (Section 4.3 of GNNExplainer).

For each EC class, aggregates individual explanations to produce:
  - Feature prototype: which feature dimensions consistently matter for this class
  - Residue position importance: which sequence positions are consistently important
  - Edge pattern summary: common contact patterns in explanations

This answers: "What does the model look for when predicting EC class X?"
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassPrototype:
    """Aggregated explanation for a single EC class."""
    class_id: int
    class_name: str = ""
    n_samples: int = 0

    # Aggregated feature importance (mean across instances)
    feature_importance: np.ndarray = None  # [d]
    feature_importance_std: np.ndarray = None  # [d]

    # Top-K most important features
    top_features: list = field(default_factory=list)  # [(idx, importance, name)]

    # Residue position importance distribution
    # Normalized to [0, 1] sequence position since proteins have variable length
    position_importance_hist: np.ndarray = None  # [n_bins]

    # Mean sparsity of explanations
    mean_sparsity: float = 0.0

    # Mean confidence on masked graph
    mean_masked_confidence: float = 0.0


EC_NAMES = [
    "Oxidoreductase", "Transferase", "Hydrolase",
    "Lyase", "Isomerase", "Ligase", "Translocase"
]

# Feature group definitions for interpretability
FEATURE_GROUPS = {
    "One-hot AA": (0, 20),
    "Physicochemical": (20, 30),
    "SASA/RSA": (30, 32),
    "ESM-2": (32, 1312),
    "Degree": (1312, 1313),
    "Positional": (1313, 1318),
}


class ClassPrototypes:
    """Builds per-class explanation prototypes from individual explanations.

    Args:
        n_classes: number of EC classes
        n_position_bins: bins for residue position histogram
        feature_groups: dict mapping group name to (start, end) indices
    """

    def __init__(self, n_classes: int = 7, n_position_bins: int = 20,
                 feature_groups: dict = None):
        self.n_classes = n_classes
        self.n_position_bins = n_position_bins
        self.feature_groups = feature_groups or FEATURE_GROUPS

    def build(self, explanations: list, labels: np.ndarray) -> list:
        """Build class prototypes from individual explanations.

        Args:
            explanations: list of ExplanationResult objects
            labels: true class labels for each explanation

        Returns:
            list of ClassPrototype, one per class
        """
        # Group explanations by class
        by_class = defaultdict(list)
        for exp, label in zip(explanations, labels):
            by_class[int(label)].append(exp)

        prototypes = []
        for c in range(self.n_classes):
            exps = by_class.get(c, [])
            proto = self._build_single_prototype(c, exps)
            prototypes.append(proto)

        return prototypes

    def _build_single_prototype(self, class_id: int,
                                explanations: list) -> ClassPrototype:
        """Build prototype for a single class."""
        proto = ClassPrototype(
            class_id=class_id,
            class_name=EC_NAMES[class_id] if class_id < len(EC_NAMES) else f"EC{class_id+1}",
            n_samples=len(explanations),
        )

        if not explanations:
            return proto

        # Aggregate feature masks
        feat_masks = np.array([e.feature_mask for e in explanations
                               if e.feature_mask is not None])
        if len(feat_masks) > 0:
            proto.feature_importance = feat_masks.mean(axis=0)
            proto.feature_importance_std = feat_masks.std(axis=0)

            # Top features by group
            proto.top_features = self._get_top_features_by_group(
                proto.feature_importance
            )

        # Residue position importance histogram
        position_hists = []
        for exp in explanations:
            if exp.node_importance is not None and len(exp.node_importance) > 0:
                hist = self._node_importance_to_position_hist(
                    exp.node_importance, self.n_position_bins
                )
                position_hists.append(hist)

        if position_hists:
            proto.position_importance_hist = np.mean(position_hists, axis=0)

        # Mean sparsity and confidence
        sparsities = []
        confidences = []
        for exp in explanations:
            if exp.edge_mask is not None:
                sparsities.append(float((exp.edge_mask < 0.5).sum()) / len(exp.edge_mask))
            confidences.append(exp.masked_pred_prob)

        proto.mean_sparsity = float(np.mean(sparsities)) if sparsities else 0.0
        proto.mean_masked_confidence = float(np.mean(confidences)) if confidences else 0.0

        return proto

    def _node_importance_to_position_hist(self, node_imp: np.ndarray,
                                          n_bins: int) -> np.ndarray:
        """Convert variable-length node importance to fixed-size position histogram."""
        n = len(node_imp)
        # Normalize positions to [0, 1]
        positions = np.linspace(0, 1, n)
        hist, _ = np.histogram(positions, bins=n_bins, weights=node_imp,
                               range=(0, 1))
        # Normalize by bin count
        counts, _ = np.histogram(positions, bins=n_bins, range=(0, 1))
        counts = np.maximum(counts, 1)
        return hist / counts

    def _get_top_features_by_group(self, feat_importance: np.ndarray,
                                   top_k_per_group: int = 5) -> list:
        """Get top-K most important features within each feature group."""
        results = []
        for group_name, (start, end) in self.feature_groups.items():
            group_imp = feat_importance[start:end]
            mean_imp = float(group_imp.mean())
            results.append((group_name, mean_imp))
        # Sort by importance
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def to_dict(self, prototypes: list) -> dict:
        """Convert prototypes to serializable dict."""
        out = {}
        for proto in prototypes:
            out[proto.class_name] = {
                "class_id": proto.class_id,
                "n_samples": proto.n_samples,
                "mean_sparsity": proto.mean_sparsity,
                "mean_masked_confidence": proto.mean_masked_confidence,
                "feature_group_importance": proto.top_features,
                "position_importance": proto.position_importance_hist.tolist()
                    if proto.position_importance_hist is not None else None,
            }
        return out
