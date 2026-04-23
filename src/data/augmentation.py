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

    def __call__(self, data) -> Data:
        if self.drop_rate <= 0:
            return data
        
        # Handle PairBatch
        if hasattr(data, 'b1') and hasattr(data, 'b2'):
             self.__call__(data.b1)
             self.__call__(data.b2)
             return data
        
        if hasattr(data, 'edge_index1'):
            # PairData: apply to both
            data.edge_index1, data.edge_attr1 = self._apply(data.edge_index1, data.edge_attr1)
            data.edge_index2, data.edge_attr2 = self._apply(data.edge_index2, data.edge_attr2)
        elif hasattr(data, 'edge_index'):
            data.edge_index, data.edge_attr = self._apply(data.edge_index, data.edge_attr)
        return data

    def _apply(self, edge_index, edge_attr):
        if edge_index is None or edge_index.shape[1] == 0:
            return edge_index, edge_attr
        E = edge_index.shape[1]
        mask = torch.rand(E, device=edge_index.device) > self.drop_rate
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        return edge_index, edge_attr


class FeatureMask:
    """Randomly zero-out node feature dimensions."""

    def __init__(self, mask_rate: float = 0.1):
        self.mask_rate = mask_rate

    def __call__(self, data) -> Data:
        if self.mask_rate <= 0:
            return data
        
        # Handle PairBatch
        if hasattr(data, 'b1') and hasattr(data, 'b2'):
             self.__call__(data.b1)
             self.__call__(data.b2)
             return data

        if hasattr(data, 'x1'):
            data.x1 = self._apply(data.x1)
            data.x2 = self._apply(data.x2)
        elif hasattr(data, 'x'):
            data.x = self._apply(data.x)
        return data

    def _apply(self, x):
        if x is None: return x
        mask = torch.rand_like(x) > self.mask_rate
        return x * mask.float()


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
