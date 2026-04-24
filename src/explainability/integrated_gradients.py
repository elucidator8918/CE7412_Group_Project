"""
Integrated Gradients for GNN graph classification.

Computes feature attributions by integrating gradients along a straight-line
path from a baseline (zero features) to the actual input. This provides a
complementary, optimization-free explanation method.

For edge importance, we use gradient of the output w.r.t. edge features,
aggregated across feature dimensions.

Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class IGResult:
    """Container for Integrated Gradients explanation."""
    protein_idx: int = -1
    true_label: int = -1
    predicted_label: int = -1

    # Attributions
    node_attributions: np.ndarray = None   # [N] per-node importance
    feature_attributions: np.ndarray = None  # [d] per-feature importance
    edge_attributions: np.ndarray = None   # [E] per-edge importance

    n_nodes: int = 0
    n_edges: int = 0


class IntegratedGradientsExplainer:
    """Integrated Gradients for GIN graph classification.

    Args:
        model: Trained GIN model
        device: torch device
        n_steps: number of interpolation steps (higher = more accurate)
        baseline_type: "zero" (zero features) or "mean" (dataset mean)
    """

    def __init__(self, model, device, n_steps: int = 50,
                 baseline_type: str = "zero"):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.model.eval()

    def explain(self, data: Data, target_class: int = None,
                dataset_mean: torch.Tensor = None) -> IGResult:
        """Compute Integrated Gradients for a single graph.

        Args:
            data: PyG Data object (single graph)
            target_class: class to explain (default: predicted class)
            dataset_mean: mean node features across dataset (for baseline)

        Returns:
            IGResult with node, feature, and edge attributions
        """
        data = data.clone().to(self.device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long,
                                     device=self.device)

        # Get predicted class
        with torch.no_grad():
            logits = self.model(data)
            pred_class = logits.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class

        # Define baseline
        x_input = data.x.float()
        if self.baseline_type == "mean" and dataset_mean is not None:
            x_baseline = dataset_mean.to(self.device).expand_as(x_input)
        else:
            x_baseline = torch.zeros_like(x_input)

        # Integrated Gradients: integrate gradient along path from baseline to input
        # IG(x) = (x - x') * integral_0^1 (dF/dx at x' + alpha*(x-x')) d_alpha
        scaled_grads = []
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            x_interp = x_baseline + alpha * (x_input - x_baseline)
            x_interp = x_interp.detach().requires_grad_(True)

            interp_data = Data(
                x=x_interp,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
                num_nodes=data.num_nodes,
            )

            logits = self.model(interp_data)
            score = logits[0, target_class]
            score.backward()

            scaled_grads.append(x_interp.grad.detach().clone())

        # Trapezoidal approximation of the integral
        grads = torch.stack(scaled_grads, dim=0)  # [steps+1, N, d]
        avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0

        # IG attribution = (input - baseline) * avg_gradient
        attributions = (x_input - x_baseline).detach() * avg_grads  # [N, d]

        # Per-node importance: sum absolute attributions across features
        node_attr = attributions.abs().sum(dim=1).cpu().numpy()

        # Per-feature importance: sum absolute attributions across nodes
        feat_attr = attributions.abs().sum(dim=0).cpu().numpy()

        # Edge importance via gradient w.r.t. edge features
        edge_attr = self._compute_edge_attributions(data, target_class)

        return IGResult(
            predicted_label=pred_class,
            true_label=int(data.y.item()) if data.y is not None else -1,
            node_attributions=node_attr,
            feature_attributions=feat_attr,
            edge_attributions=edge_attr,
            n_nodes=data.num_nodes,
            n_edges=data.edge_index.shape[1],
        )

    def _compute_edge_attributions(self, data: Data,
                                   target_class: int) -> np.ndarray:
        """Compute edge importance via gradient of output w.r.t. edge_attr."""
        if data.edge_attr is None:
            return np.ones(data.edge_index.shape[1])

        edge_attr = data.edge_attr.float().detach().requires_grad_(True)
        ea_data = Data(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            batch=data.batch,
            num_nodes=data.num_nodes,
        )

        logits = self.model(ea_data)
        score = logits[0, target_class]
        score.backward()

        # Per-edge importance = L2 norm of gradient across edge feature dims
        edge_grad = edge_attr.grad.detach()
        edge_imp = edge_grad.norm(dim=1).cpu().numpy()
        return edge_imp

    def explain_batch(self, graphs: list, target_classes: list = None,
                      dataset_mean: torch.Tensor = None,
                      verbose: bool = True) -> list:
        """Compute IG explanations for multiple graphs."""
        results = []
        n = len(graphs)

        for i, data in enumerate(graphs):
            tc = target_classes[i] if target_classes else None
            result = self.explain(data, target_class=tc,
                                  dataset_mean=dataset_mean)
            result.protein_idx = i
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  IG explained {i+1}/{n} graphs")

        return results
