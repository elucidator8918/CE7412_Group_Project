"""
Graph augmentation transforms for training-time regularization.

Augmentations:
  - EdgeDrop: randomly remove a fraction of edges
  - FeatureMask: randomly zero-out node feature dimensions
  - Combined: apply both sequentially
"""

import torch
from torch_geometric.data import Data


class EdgeDrop:
    """Randomly drop edges from the graph during training."""

    def __init__(self, drop_rate: float = 0.1):
        self.drop_rate = drop_rate

    def __call__(self, data: Data) -> Data:
        if self.drop_rate <= 0:
            return data
        E = data.edge_index.shape[1]
        mask = torch.rand(E) > self.drop_rate
        data.edge_index = data.edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        return data


class FeatureMask:
    """Randomly zero-out node feature dimensions."""

    def __init__(self, mask_rate: float = 0.1):
        self.mask_rate = mask_rate

    def __call__(self, data: Data) -> Data:
        if self.mask_rate <= 0:
            return data
        mask = torch.rand_like(data.x) > self.mask_rate
        data.x = data.x * mask.float()
        return data


class GraphAugmentation:
    """Combined graph augmentation: edge drop + feature mask."""

    def __init__(self, edge_drop_rate: float = 0.1, feature_mask_rate: float = 0.1,
                 enabled: bool = True):
        self.enabled = enabled
        self.edge_drop = EdgeDrop(edge_drop_rate) if edge_drop_rate > 0 else None
        self.feature_mask = FeatureMask(feature_mask_rate) if feature_mask_rate > 0 else None

    def __call__(self, data: Data) -> Data:
        if not self.enabled:
            return data
        if self.edge_drop is not None:
            data = self.edge_drop(data)
        if self.feature_mask is not None:
            data = self.feature_mask(data)
        return data
