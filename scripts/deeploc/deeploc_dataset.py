"""
DeepLoc-2.1 Dataset — Converts protein sequences into PyG graphs for
subcellular localization prediction.

Since DeepLoc is sequence-only (no 3D coordinates), we construct a
sequential k-NN graph: each residue connects to its k nearest neighbors
in sequence position.  Edge features encode sequence separation; node
features use the full pipeline (one-hot, physicochemical, ESM-2, etc.).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

# Reuse existing feature engineering — no modifications needed
from src.data.features import (
    compute_node_features,
    compute_edge_features,
    compute_feat_dim,
    compute_edge_dim,
    ESM2Extractor,
)

logger = logging.getLogger(__name__)

# The 10 DeepLoc subcellular localization classes (order matters for label vectors)
LOCALIZATION_CLASSES = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]
N_CLASSES = len(LOCALIZATION_CLASSES)

MEMBRANE_CLASSES = [
    "Peripheral",
    "Transmembrane",
    "LipidAnchor",
    "Soluble",
]

def build_sequential_graph(n_nodes: int, k: int = 10) -> torch.Tensor:
    """Build a sequential k-nearest-neighbor graph for a protein chain.

    Each residue i is connected to residues in [i-k//2, i+k//2] (bounded).
    All edges are bidirectional.

    Args:
        n_nodes: Number of residues.
        k: Neighbourhood radius (total neighbors ≈ k).

    Returns:
        edge_index: [2, E] LongTensor of edges.
    """
    if n_nodes <= 1:
        return torch.zeros(2, 0, dtype=torch.long)

    half_k = k // 2
    src, dst = [], []
    for i in range(n_nodes):
        lo = max(0, i - half_k)
        hi = min(n_nodes, i + half_k + 1)
        for j in range(lo, hi):
            if j != i:
                src.append(i)
                dst.append(j)

    edge_index = torch.from_numpy(np.array([src, dst], dtype=np.int64))
    return edge_index





class DeepLocDataset(torch.utils.data.Dataset):
    """PyG-compatible dataset for DeepLoc subcellular localization.

    Reads a CSV with columns: ACC, Sequence, and 10 localization labels.
    Constructs a sequential graph per protein with full node/edge features.
    """

    def __init__(
        self,
        csv_path: str,
        feat_cfg: dict,
        esm_extractor: ESM2Extractor = None,
        max_seq_len: int = 2000,
        k_neighbors: int = 10,
        label_columns: list = None,
        id_column: str = None,
        seq_column: str = "Sequence",
        membrane_csv_path: str = None,
    ):
        """
        Args:
            csv_path: Path to the CSV file.
            feat_cfg: Feature config dict (same format as benchmark.yaml features).
            esm_extractor: Pre-initialized ESM2Extractor (with RAM cache).
            max_seq_len: Truncate sequences longer than this.
            k_neighbors: k for sequential graph construction.
            label_columns: Override list of label column names.
            id_column: Column name for the protein ID.
            seq_column: Column name for the amino acid sequence.
            membrane_csv_path: Path to the Swissprot Membrane dataset CSV for MTL.
        """
        super().__init__()
        self.feat_cfg = feat_cfg
        self.esm_extractor = esm_extractor
        self.max_seq_len = max_seq_len
        self.k_neighbors = k_neighbors
        self.label_columns = label_columns or LOCALIZATION_CLASSES
        self.membrane_columns = MEMBRANE_CLASSES

        # Read CSV
        self.df = pd.read_csv(csv_path)

        # Auto-detect ID column
        if id_column is not None:
            self.id_col = id_column
        elif "ACC" in self.df.columns:
            self.id_col = "ACC"
        elif "ACC\n" in self.df.columns:
            # Handle the quirky column name in the SwissProt CSV
            self.id_col = "ACC\n"
        elif "sid" in self.df.columns:
            self.id_col = "sid"
        else:
            self.id_col = self.df.columns[1]  # Fallback

        self.seq_col = seq_column

        # Use the master LOCALIZATION_CLASSES list for indexing to ensure alignment with model outputs.
        # This list remains fixed at 10 classes, even if the CSV provides only a subset (e.g. HPA).
        self.label_columns = LOCALIZATION_CLASSES
        
        # Validate that at least some expected labels are found (for warning only)
        available = set(self.df.columns)
        found = [c for c in self.label_columns if c in available]
        if not found and not label_columns:
             logger.warning(f"No localization label columns found in CSV. Expected subset of {LOCALIZATION_CLASSES}")

        # Merge Membrane data if provided
        if membrane_csv_path and os.path.exists(membrane_csv_path):
            mem_df = pd.read_csv(membrane_csv_path)
            # Find the ID column in Mem DF that matches our ID col
            mem_id_col = None
            if self.id_col in mem_df.columns: mem_id_col = self.id_col
            elif "ACC" in mem_df.columns: mem_id_col = "ACC"
            elif "ACC\n" in mem_df.columns: mem_id_col = "ACC\n"
            
            if mem_id_col:
                # Merge specifically the membrane columns
                mem_cols = [mem_id_col] + [c for c in self.membrane_columns if c in mem_df.columns]
                mem_df_sub = mem_df[mem_cols].drop_duplicates(subset=[mem_id_col])
                
                # Merge into self.df
                self.df = pd.merge(self.df, mem_df_sub, left_on=self.id_col, right_on=mem_id_col, how="left")
                logger.info(f"Merged membrane labels from {membrane_csv_path}.")
            else:
                logger.warning("Could not find matching ID column in membrane CSV.")
        else:
            # If not provided, we just add NaN columns
            for c in self.membrane_columns:
                if c not in self.df.columns:
                    self.df[c] = np.nan

        logger.info(
            f"DeepLocDataset: {len(self.df)} proteins, "
            f"{len(self.label_columns)} location labels, 4 membrane classes, "
            f"max_seq_len={max_seq_len}, k={k_neighbors}"
        )
        self._total_calls = 0

        self._project_root = Path(__file__).parents[2]




    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        pid = str(row[self.id_col])

        # Truncate long sequences
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]

        n = len(seq)
        if n == 0:
            return None

        residue_types = list(seq)

        # 1. Build sequential graph
        edge_index = build_sequential_graph(n, self.k_neighbors)
        coords = None



        # 2. Get ESM-2 embeddings (from disk cache)
        esm_emb = None
        if self.esm_extractor is not None and self.feat_cfg.get("use_esm2", False):
            try:
                esm_emb = self.esm_extractor.extract(pid, seq)
                # Immediately evict from RAM cache to prevent memory explosion during training
                self.esm_extractor.clear_ram_cache()
                if esm_emb is not None:
                    esm_emb = esm_emb.cpu()
            except Exception as e:
                logger.debug(f"ESM-2 extraction failed for {pid}: {e}")
                esm_emb = None

        # 3. Node features (reuse existing pipeline)
        x = compute_node_features(
            residue_types=residue_types,
            edge_index=edge_index,
            coords=coords,     # Pass the 3D structure if available
            sasa=None,
            rsa=None,
            esm_embeddings=esm_emb,
            cfg=self.feat_cfg,
        )

        # 4. Edge features (distance RBF if coords available)
        edge_attr = compute_edge_features(
            edge_index=edge_index,
            coords=coords,     # Pass the 3D structure if available
            n_nodes=n,
            cfg=self.feat_cfg,
        )

        # Location labels [1, 10]
        # Always use the master LOCALIZATION_CLASSES list (self.label_columns)
        # to ensure index alignment with the model's output neurons.
        y_loc_list = []
        for c in self.label_columns:
            val = row.get(c, 0.0)
            if pd.isna(val): val = 0.0
            y_loc_list.append(float(val))
            
        y_loc = torch.tensor(y_loc_list, dtype=torch.float32).unsqueeze(0)
        
        # Membrane labels: integer class index for CrossEntropyLoss [1]
        # Ignore index is -100
        mem_idx = -100
        for i, c in enumerate(self.membrane_columns):
            val = row.get(c, np.nan)
            if pd.notna(val) and float(val) == 1.0:
                mem_idx = i
                break
        y_mem = torch.tensor([mem_idx], dtype=torch.long)

        # Build PyG Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y_loc=y_loc, y_mem=y_mem, pid=pid, num_nodes=n)
        
        # Add global ESM context for the model's skip connection
        esm_dim = self.feat_cfg.get("esm2_dim", 320)
        if esm_emb is not None:
            data.global_x_esm = esm_emb.mean(dim=0, keepdim=True)
        else:
            # Always provide a zero vector to avoid batching/attribute errors
            data.global_x_esm = torch.zeros(1, esm_dim)
            
        return data


def preload_esm_for_deeploc(
    df: pd.DataFrame,
    esm_extractor: ESM2Extractor,
    id_col: str,
    seq_col: str = "Sequence",
    max_seq_len: int = 1000,
    batch_size: int = 4,
):
    """Pre-extract ESM-2 embeddings to DISK for all proteins in the DataFrame.

    Unlike the ProteinShake RAM-cache approach, this function:
    1. Skips proteins that already have disk-cached embeddings
    2. Extracts one protein at a time
    3. Immediately evicts each embedding from RAM after disk write
    This keeps RAM usage minimal (~2.5GB for the ESM model only).
    """
    # Collect proteins that need extraction
    uncached = []
    for _, row in df.iterrows():
        pid = str(row[id_col])
        seq = str(row[seq_col])
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        cache_file = esm_extractor._cache_path(pid)
        if not cache_file.exists():
            uncached.append((pid, seq))

    total = len(df)
    n_cached = total - len(uncached)
    logger.info(
        f"ESM-2 disk cache: {n_cached}/{total} already cached, "
        f"{len(uncached)} to extract"
    )

    if not uncached:
        logger.info("All proteins already cached on disk. Skipping extraction.")
        return

    # Load model and extract one-by-one, evicting RAM after each
    esm_extractor._load_model()
    for i, (pid, seq) in enumerate(uncached):
        esm_extractor.extract(pid, seq)
        # Immediately evict from RAM — we only need disk cache
        esm_extractor.clear_ram_cache()

        if (i + 1) % 200 == 0:
            logger.info(f"  Extracted {i + 1}/{len(uncached)} proteins to disk...")

    esm_extractor.unload_model()
    esm_extractor.clear_ram_cache()  # Final cleanup
    logger.info(f"ESM-2 extraction complete. {len(uncached)} proteins cached to disk.")




