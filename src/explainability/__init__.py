"""
Explainability methods for GNN-based enzyme classification.

Methods:
  - GNNExplainer: optimization-based edge + feature mask learning
  - IntegratedGradients: attribution-based node/edge importance
  - Multi-instance class prototypes
  - Quantitative fidelity evaluation
"""

from .gnn_explainer import GNNExplainerGraph
from .integrated_gradients import IntegratedGradientsExplainer
from .metrics import compute_fidelity_metrics
from .prototypes import ClassPrototypes
