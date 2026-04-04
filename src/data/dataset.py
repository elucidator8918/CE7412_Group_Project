"""
Dataset loading and preparation for ProteinShake enzyme classification.

Handles:
  - Loading EnzymeClassTask from ProteinShake
  - Converting to PyG Data objects with rich features
  - ESM-2 embedding extraction and caching
  - Balanced subsampling and split construction
  - Multiple protein representations (sequence, residue, graph)
"""

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from tqdm import tqdm

from .features import (
    AA_DIM,
    ESM2Extractor,
    compute_edge_dim,
    compute_edge_features,
    compute_feat_dim,
    compute_node_features,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core dataset class
# ============================================================================

class EnzymeDataset:
    """Full pipeline from ProteinShake to ready-to-train data loaders."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = cfg["dataset"]
        self.feat_cfg = cfg["features"]
        self.train_cfg = cfg["training"]

        self.feat_dim = compute_feat_dim(self.feat_cfg)
        self.edge_dim = compute_edge_dim(self.feat_cfg)
        self.n_classes = self.data_cfg["n_classes"]

        logger.info(f"Feature dim: {self.feat_dim}, Edge dim: {self.edge_dim}")

        # Will be populated by prepare()
        self.all_graphs = []
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []

        # Loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_graphs = []
        self.val_graphs = []
        self.test_graphs = []

        # Non-graph representations
        self.seq_loaders = {}
        self.res_loaders = {}

    def prepare(self):
        """Run the full data preparation pipeline."""
        self._load_proteinshake()
        self._build_graphs()
        self._create_splits()
        self._build_loaders()
        self._build_flat_representations()
        return self

    def _load_proteinshake(self):
        """Load ProteinShake EnzymeClassTask and raw PyG graphs."""
        from proteinshake.tasks import EnzymeClassTask

        root = self.cfg["paths"]["data_root"]
        eps = self.data_cfg["eps_default"]
        threshold = self.data_cfg["split_similarity_threshold"]

        logger.info(f"Loading EnzymeClassTask (root={root}, eps={eps}Å, θ={threshold})")
        self.task = EnzymeClassTask(
            root=root, split="random",
            split_similarity_threshold=threshold
        )
        logger.info("Building residue contact graphs...")
        self.raw_pyg = list(self.task.dataset.to_graph(eps=eps).pyg())
        logger.info(f"  Loaded {len(self.raw_pyg)} raw protein graphs")

        # Get split indices from ProteinShake
        self._ps_train_idx = self.task.train_index
        self._ps_val_idx = self.task.val_index
        self._ps_test_idx = self.task.test_index

    def _build_graphs(self):
        """Convert raw ProteinShake graphs to rich PyG Data objects."""
        esm_extractor = None
        if self.feat_cfg.get("use_esm2", False):
            esm_extractor = ESM2Extractor(
                model_name=self.feat_cfg.get("esm2_model", "esm2_t6_8M_UR50D"),
                layer=self.feat_cfg.get("esm2_layer", 6),
                cache_dir=self.cfg["paths"]["esm_cache"],
            )

        logger.info("Building graph features...")
        graphs = []
        skipped = 0

        for i, (g, prot) in enumerate(tqdm(self.raw_pyg, desc="Building graphs")):
            # Extract EC label
            ec_key = str(prot["protein"]["EC"]).split(".")[0]
            label = self.task.token_map.get(ec_key, -1)
            if label == -1:
                skipped += 1
                graphs.append(None)
                continue

            # Residue sequence (string of 1-letter codes)
            sequence = prot["protein"].get("sequence", "")
            res_types = list(sequence)
            n_nodes = len(res_types)
            if n_nodes == 0:
                skipped += 1
                graphs.append(None)
                continue

            # Edge index
            edge_index = g.edge_index if hasattr(g, "edge_index") else torch.zeros(2, 0, dtype=torch.long)

            # Coordinates from residue dict
            residue_data = prot.get("residue", {})
            coords = None
            if "x" in residue_data and "y" in residue_data and "z" in residue_data:
                cx = torch.tensor(residue_data["x"], dtype=torch.float32)
                cy = torch.tensor(residue_data["y"], dtype=torch.float32)
                cz = torch.tensor(residue_data["z"], dtype=torch.float32)
                coords = torch.stack([cx, cy, cz], dim=1)  # [N, 3]

            # SASA and RSA from residue dict
            sasa = None
            rsa = None
            if "SASA" in residue_data:
                sasa = torch.tensor(residue_data["SASA"], dtype=torch.float32)
            if "RSA" in residue_data:
                rsa = torch.tensor(residue_data["RSA"], dtype=torch.float32)

            # ESM-2 embeddings
            esm_emb = None
            if esm_extractor is not None:
                pid = str(prot["protein"].get("ID", f"prot_{i}"))
                try:
                    esm_emb = esm_extractor.extract(pid, sequence)
                except Exception as e:
                    logger.warning(f"ESM-2 failed for {pid}: {e}")

            # Build node features
            x = compute_node_features(
                residue_types=res_types,
                edge_index=edge_index,
                coords=coords,
                sasa=sasa,
                rsa=rsa,
                esm_embeddings=esm_emb,
                cfg=self.feat_cfg,
            )

            # Build edge features
            edge_attr = None
            if self.edge_dim > 0 and coords is not None:
                edge_attr = compute_edge_features(
                    edge_index=edge_index,
                    coords=coords,
                    n_nodes=n_nodes,
                    cfg=self.feat_cfg,
                )

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.long),
                num_nodes=n_nodes,
            )

            if coords is not None:
                data.coords = coords

            graphs.append(data)

        self.all_graphs = graphs
        logger.info(f"  Built {sum(1 for g in graphs if g is not None)} graphs, skipped {skipped}")

    def _create_splits(self):
        """Create balanced train/val/test splits."""
        cap = self.data_cfg.get("max_per_class", 0)

        def _filter_valid(indices):
            return [i for i in indices if i < len(self.all_graphs) and self.all_graphs[i] is not None]

        raw_train = _filter_valid(self._ps_train_idx)
        raw_val = _filter_valid(self._ps_val_idx)
        raw_test = _filter_valid(self._ps_test_idx)

        if cap > 0:
            self.train_idx = _subsample_balanced(raw_train, self.all_graphs, cap)
            self.val_idx = _subsample_balanced(raw_val, self.all_graphs, max(1, cap // 5))
            self.test_idx = _subsample_balanced(raw_test, self.all_graphs, max(1, cap // 5))
        else:
            self.train_idx = raw_train
            self.val_idx = raw_val
            self.test_idx = raw_test

        logger.info(f"  Splits: train={len(self.train_idx)}, val={len(self.val_idx)}, test={len(self.test_idx)}")

        # Log class distribution
        for name, idx in [("train", self.train_idx), ("val", self.val_idx), ("test", self.test_idx)]:
            dist = Counter(int(self.all_graphs[i].y.item()) for i in idx)
            dist_str = " | ".join(f"EC{c+1}:{dist.get(c,0)}" for c in range(self.n_classes))
            logger.info(f"    {name}: {dist_str}")

    def _build_loaders(self):
        """Build PyG DataLoaders for graph-based models."""
        bs = self.train_cfg["batch_size"]

        self.train_graphs = [self.all_graphs[i] for i in self.train_idx]
        self.val_graphs = [self.all_graphs[i] for i in self.val_idx]
        self.test_graphs = [self.all_graphs[i] for i in self.test_idx]

        self.train_loader = PyGLoader(self.train_graphs, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = PyGLoader(self.val_graphs, batch_size=bs, num_workers=4, pin_memory=True)
        self.test_loader = PyGLoader(self.test_graphs, batch_size=bs, num_workers=4, pin_memory=True)

    def _build_flat_representations(self):
        """Build non-graph representations for baseline MLPs."""
        bs = self.train_cfg["batch_size"]

        # Residue MLP: mean-pool node features (full feature vector)
        def _to_meanpool(glist):
            X = torch.stack([g.x.float().mean(dim=0) for g in glist])
            Y = torch.tensor([int(g.y.item()) for g in glist], dtype=torch.long)
            return X, Y

        Xr_tr, Yr_tr = _to_meanpool(self.train_graphs)
        Xr_va, Yr_va = _to_meanpool(self.val_graphs)
        Xr_te, Yr_te = _to_meanpool(self.test_graphs)

        self.res_loaders = {
            "train": TorchLoader(TensorDataset(Xr_tr, Yr_tr), batch_size=bs, shuffle=True),
            "val":   TorchLoader(TensorDataset(Xr_va, Yr_va), batch_size=bs),
            "test":  TorchLoader(TensorDataset(Xr_te, Yr_te), batch_size=bs),
        }

        # Sequence MLP: AA composition (20-dim normalized frequency)
        def _to_aa_comp(glist):
            X_list = []
            for g in glist:
                feat = g.x.float()
                comp = feat[:, :AA_DIM].sum(dim=0)
                X_list.append(comp / (comp.sum() + 1e-8))
            X = torch.stack(X_list)
            Y = torch.tensor([int(g.y.item()) for g in glist], dtype=torch.long)
            return X, Y

        Xs_tr, Ys_tr = _to_aa_comp(self.train_graphs)
        Xs_va, Ys_va = _to_aa_comp(self.val_graphs)
        Xs_te, Ys_te = _to_aa_comp(self.test_graphs)

        self.seq_loaders = {
            "train": TorchLoader(TensorDataset(Xs_tr, Ys_tr), batch_size=bs, shuffle=True),
            "val":   TorchLoader(TensorDataset(Xs_va, Ys_va), batch_size=bs),
            "test":  TorchLoader(TensorDataset(Xs_te, Ys_te), batch_size=bs),
        }

    def get_class_weights(self):
        """Compute inverse-frequency class weights for loss balancing."""
        counts = Counter(int(self.all_graphs[i].y.item()) for i in self.train_idx)
        total = sum(counts.values())
        weights = torch.tensor(
            [total / (self.n_classes * counts.get(c, 1)) for c in range(self.n_classes)],
            dtype=torch.float32
        )
        return weights

    def build_graphs_at_eps(self, eps: float):
        """Rebuild graph loaders at a different contact radius (for ablation)."""
        logger.info(f"  Rebuilding graphs at eps={eps} Å ...")
        raw = list(self.task.dataset.to_graph(eps=eps).pyg())

        # Load ESM-2 extractor if needed (reads from cache, no model needed)
        esm_extractor = None
        if self.feat_cfg.get("use_esm2", False):
            esm_extractor = ESM2Extractor(
                model_name=self.feat_cfg.get("esm2_model", "esm2_t6_8M_UR50D"),
                layer=self.feat_cfg.get("esm2_layer", 6),
                cache_dir=self.cfg["paths"]["esm_cache"],
            )

        graphs = []
        for i, (g, prot) in enumerate(raw):
            ec_key = str(prot["protein"]["EC"]).split(".")[0]
            label = self.task.token_map.get(ec_key, -1)
            if label == -1:
                graphs.append(None)
                continue

            sequence = prot["protein"].get("sequence", "")
            res_types = list(sequence)
            n_nodes = len(res_types)
            if n_nodes == 0:
                graphs.append(None)
                continue

            edge_index = g.edge_index

            residue_data = prot.get("residue", {})
            coords = None
            if "x" in residue_data and "y" in residue_data and "z" in residue_data:
                cx = torch.tensor(residue_data["x"], dtype=torch.float32)
                cy = torch.tensor(residue_data["y"], dtype=torch.float32)
                cz = torch.tensor(residue_data["z"], dtype=torch.float32)
                coords = torch.stack([cx, cy, cz], dim=1)

            sasa = torch.tensor(residue_data["SASA"], dtype=torch.float32) if "SASA" in residue_data else None
            rsa = torch.tensor(residue_data["RSA"], dtype=torch.float32) if "RSA" in residue_data else None

            # Load ESM-2 embeddings from cache
            esm_emb = None
            if esm_extractor is not None:
                pid = str(prot["protein"].get("ID", f"prot_{i}"))
                try:
                    esm_emb = esm_extractor.extract(pid, sequence)
                except Exception:
                    pass

            x = compute_node_features(
                residue_types=res_types,
                edge_index=edge_index,
                coords=coords,
                sasa=sasa,
                rsa=rsa,
                esm_embeddings=esm_emb,
                cfg=self.feat_cfg,
            )

            edge_attr = None
            if self.edge_dim > 0 and coords is not None:
                edge_attr = compute_edge_features(
                    edge_index=edge_index, coords=coords,
                    n_nodes=n_nodes, cfg=self.feat_cfg,
                )

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=torch.tensor(label, dtype=torch.long), num_nodes=n_nodes)
            if coords is not None:
                data.coords = coords
            graphs.append(data)

        bs = self.train_cfg["batch_size"]
        train_g = [graphs[i] for i in self.train_idx if graphs[i] is not None]
        val_g = [graphs[i] for i in self.val_idx if graphs[i] is not None]
        test_g = [graphs[i] for i in self.test_idx if graphs[i] is not None]

        return (
            PyGLoader(train_g, batch_size=bs, shuffle=True),
            PyGLoader(val_g, batch_size=bs),
            PyGLoader(test_g, batch_size=bs),
        )


# ============================================================================
# Helpers
# ============================================================================

def _subsample_balanced(indices, graphs, cap, seed=42):
    """Subsample to at most `cap` per class, preserving balance."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i in indices:
        by_class[int(graphs[i].y.item())].append(i)
    out = []
    for cls in sorted(by_class):
        pool = by_class[cls][:]
        rng.shuffle(pool)
        out.extend(pool[:cap])
    rng.shuffle(out)
    return out
