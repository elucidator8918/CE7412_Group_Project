"""
Feature engineering for protein graphs.

Produces rich node and edge features from raw ProteinShake data:
  - Amino acid one-hot encoding (20 dim)
  - Physicochemical properties (10 dim)
  - AlphaFold pLDDT confidence score (1 dim)
  - ESM-2 per-residue embeddings (320 dim)
  - Node degree (1 dim)
  - Positional encoding: linear + sin/cos (5 dim)
  - Edge: RBF-encoded Ca-Ca distance + sequence separation
"""

import math
import os
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Amino acid properties
# ============================================================================

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_DIM = len(AA_ORDER)

# Physicochemical property table (normalized to [0, 1])
# Sources: Kyte-Doolittle hydrophobicity, charge at pH 7, molecular weight,
#          van der Waals volume, polarity (Grantham), flexibility (Vihinen),
#          surface accessibility, helix propensity, sheet propensity, turn propensity
_RAW_PHYSICO = {
    #        hydro   charge  MW      volume  polar   flex    access  helix   sheet   turn
    "A": [  1.800,  0.000,  89.1,   67.0,   8.1,    0.360,  1.181,  1.450,  0.970,  0.660],
    "C": [  2.500,  0.000, 121.2,   86.0,   5.5,    0.350,  1.461,  0.770,  1.300,  1.190],
    "D": [ -3.500, -1.000, 133.1,   91.0,  13.0,    0.510,  1.587,  0.980,  0.800,  1.460],
    "E": [ -3.500, -1.000, 147.1,  109.0,  12.3,    0.500,  1.862,  1.530,  0.370,  0.740],
    "F": [  2.800,  0.000, 165.2,  135.0,   5.2,    0.310,  2.228,  1.120,  1.280,  0.600],
    "G": [ -0.400,  0.000,  75.0,   48.0,   9.0,    0.540,  0.881,  0.530,  0.810,  1.560],
    "H": [ -3.200,  0.500, 155.2,  118.0,  10.4,    0.320,  2.025,  1.240,  0.710,  0.950],
    "I": [  4.500,  0.000, 131.2,  124.0,   5.2,    0.460,  1.810,  1.000,  1.600,  0.470],
    "K": [ -3.900,  1.000, 146.2,  135.0,  11.3,    0.470,  2.258,  1.070,  0.740,  1.010],
    "L": [  3.800,  0.000, 131.2,  124.0,   4.9,    0.400,  1.931,  1.340,  1.220,  0.590],
    "M": [  1.900,  0.000, 149.2,  124.0,   5.7,    0.300,  2.034,  1.200,  1.670,  0.390],
    "N": [ -3.500,  0.000, 132.1,   96.0,  11.6,    0.460,  1.655,  0.730,  0.650,  1.560],
    "P": [ -1.600,  0.000, 115.1,   90.0,   8.0,    0.510,  1.468,  0.590,  0.620,  1.520],
    "Q": [ -3.500,  0.000, 146.2,  114.0,  10.5,    0.490,  1.932,  1.170,  1.230,  0.980],
    "R": [ -4.500,  1.000, 174.2,  148.0,  10.5,    0.470,  2.560,  0.790,  0.900,  0.950],
    "S": [ -0.800,  0.000, 105.1,   73.0,   9.2,    0.510,  1.298,  0.790,  0.720,  1.430],
    "T": [ -0.700,  0.000, 119.1,   93.0,   8.6,    0.440,  1.525,  0.820,  1.200,  0.960],
    "V": [  4.200,  0.000, 117.1,  105.0,   5.9,    0.390,  1.645,  1.140,  1.650,  0.500],
    "W": [ -0.900,  0.000, 204.2,  163.0,   5.4,    0.310,  2.663,  1.140,  1.190,  0.960],
    "Y": [ -1.300,  0.000, 181.2,  141.0,   6.2,    0.420,  2.368,  0.610,  1.290,  1.140],
}

# Precompute normalized physicochemical tensor
def _build_physico_table():
    """Build [20, 10] normalized physicochemical property tensor."""
    raw = np.array([_RAW_PHYSICO[aa] for aa in AA_ORDER], dtype=np.float32)
    # Min-max normalize each property to [0, 1]
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (raw - mins) / ranges
    return torch.from_numpy(normed)

PHYSICO_TABLE = _build_physico_table()  # [20, 10]
PHYSICO_DIM = PHYSICO_TABLE.shape[1]


# ============================================================================
# ESM-2 embedding extraction
# ============================================================================

class ESM2Extractor:
    """Extract per-residue embeddings from ESM-2 protein language model.

    Embeddings are cached to disk to avoid recomputation.
    Uses the smallest ESM-2 model (8M params) by default — runs on CPU in seconds.
    """

    def __init__(self, model_name="esm2_t6_8M_UR50D", layer=6, cache_dir="./data/esm_cache",
                 device=None):
        self.model_name = model_name
        self.layer = layer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._ram_cache = {}

    def _load_model(self):
        """Lazy-load ESM-2 model on first use."""
        if self._model is not None:
            return
        import esm
        logger.info(f"Loading ESM-2 model: {self.model_name}")
        self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(
            self.model_name
        )
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info(f"ESM-2 loaded on {self.device}")

    def _cache_path(self, protein_id: str) -> Path:
        return self.cache_dir / f"{protein_id}.pt"

    def extract(self, protein_id: str, sequence: str) -> torch.Tensor:
        """Extract per-residue embeddings [L, esm_dim] for a single protein.

        Caches result to disk. Returns tensor on CPU.
        """
        if protein_id in self._ram_cache:
            return self._ram_cache[protein_id]

        cache_file = self._cache_path(protein_id)
        if cache_file.exists():
            emb = torch.load(cache_file, map_location="cpu", weights_only=True)
            self._ram_cache[protein_id] = emb
            return emb

        self._load_model()
        
        # Truncate very long sequences (ESM-2 limit ~1024 for small model)
        max_len = 1022
        seq = sequence[:max_len]

        _, _, tokens = self._batch_converter([("protein", seq)])
        tokens = tokens.to(self.device)

        with torch.no_grad():
            result = self._model(tokens, repr_layers=[self.layer], return_contacts=False)

        # Remove BOS and EOS tokens
        embeddings = result["representations"][self.layer][0, 1:len(seq)+1].cpu()

        torch.save(embeddings, cache_file)
        self._ram_cache[protein_id] = embeddings
        return embeddings

    def unload_model(self):
        """Free GPU memory by deleting the model."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("ESM-2 model unloaded from memory.")

    def clear_ram_cache(self):
        """Clear the in-memory cache of embeddings."""
        self._ram_cache = {}
        logger.debug("ESM-2 RAM cache cleared.")

    def clear_disk_cache(self):
        """Delete all cached .pt files from disk."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ESM-2 disk cache cleared: {self.cache_dir}")

    def extract_batch(self, proteins: list, batch_size: int = 8) -> dict:
        """Extract embeddings for a batch of (id, sequence) tuples."""
        results = {}
        uncached = []

        for pid, seq in proteins:
            cache_file = self._cache_path(pid)
            if cache_file.exists():
                results[pid] = torch.load(cache_file, map_location="cpu", weights_only=True)
            else:
                uncached.append((pid, seq))

        if uncached:
            self._load_model()
            logger.info(f"Extracting ESM-2 for {len(uncached)} uncached proteins...")
            for i in range(0, len(uncached), batch_size):
                batch = uncached[i:i + batch_size]
                for pid, seq in batch:
                    results[pid] = self.extract(pid, seq)

        return results


# ============================================================================
# Node feature builder
# ============================================================================

def compute_node_features(
    residue_types: list,
    edge_index: torch.Tensor,
    coords: torch.Tensor = None,
    sasa: torch.Tensor = None,
    rsa: torch.Tensor = None,
    esm_embeddings: torch.Tensor = None,
    cfg=None,
):
    """Build rich per-residue feature matrix.

    Args:
        residue_types: list of 1-letter amino acid codes (length N)
        edge_index: [2, E] edge indices
        coords: [N, 3] Ca coordinates (optional)
        sasa: [N] solvent accessible surface area (optional)
        rsa: [N] relative solvent accessibility (optional)
        esm_embeddings: [N, esm_dim] ESM-2 embeddings (optional)
        cfg: features config dict

    Returns:
        x: [N, feat_dim] node feature tensor
    """
    n = len(residue_types)
    parts = []

    # 1) One-hot amino acid encoding (20 dim)
    if cfg is None or cfg.get("use_onehot", True):
        idx = [AA_TO_IDX.get(r, 0) for r in residue_types]
        onehot = torch.zeros(n, AA_DIM)
        onehot[torch.arange(n), torch.tensor(idx)] = 1.0
        parts.append(onehot)

    # 2) Physicochemical properties (10 dim)
    if cfg is None or cfg.get("use_physicochemical", True):
        idx = torch.tensor([AA_TO_IDX.get(r, 0) for r in residue_types])
        physico = PHYSICO_TABLE[idx]  # [N, 10]
        parts.append(physico)

    # 3) SASA + RSA solvent accessibility (2 dim)
    if cfg is None or cfg.get("use_sasa", True):
        if sasa is not None:
            sasa_t = sasa.float().unsqueeze(1) if sasa.dim() == 1 else sasa.float()
            sasa_norm = sasa_t / (sasa_t.max().clamp(min=1.0))  # normalize
            parts.append(sasa_norm)
        else:
            parts.append(torch.zeros(n, 1))
        if rsa is not None:
            rsa_t = rsa.float().unsqueeze(1) if rsa.dim() == 1 else rsa.float()
            parts.append(rsa_t)  # RSA is already in [0, 1]
        else:
            parts.append(torch.zeros(n, 1))

    # 4) ESM-2 embeddings (320 or 1280 dim)
    if cfg is None or cfg.get("use_esm2", True):
        esm_dim = cfg.get("esm2_dim", 320) if cfg else 320
        if esm_embeddings is not None:
            # Pad/truncate to match residue count if needed
            if esm_embeddings.shape[0] < n:
                pad = torch.zeros(n - esm_embeddings.shape[0], esm_embeddings.shape[1])
                esm_embeddings = torch.cat([esm_embeddings, pad], dim=0)
            elif esm_embeddings.shape[0] > n:
                esm_embeddings = esm_embeddings[:n]
            parts.append(esm_embeddings.float())
        else:
            # ESM expected but missing - pad with zeros to maintain dimension consistency
            parts.append(torch.zeros(n, esm_dim))

    # 5) Node degree (1 dim)
    if cfg is None or cfg.get("use_degree", True):
        if edge_index is not None and edge_index.shape[1] > 0:
            degrees = torch.zeros(n, dtype=torch.float32)
            degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
            max_deg = degrees.max().clamp(min=1.0)
            degree_feat = (degrees / max_deg).unsqueeze(1)
        else:
            degree_feat = torch.zeros(n, 1)
        parts.append(degree_feat)

    # 6) Positional encoding (5 dim: linear + 2 sin/cos pairs)
    if cfg is None or cfg.get("use_positional", True):
        if n > 1:
            t = torch.linspace(0, 1, n).unsqueeze(1)
        else:
            t = torch.zeros(n, 1)
        freq1_sin = torch.sin(t * math.pi)
        freq1_cos = torch.cos(t * math.pi)
        freq2_sin = torch.sin(t * 2 * math.pi)
        freq2_cos = torch.cos(t * 2 * math.pi)
        parts.append(torch.cat([t, freq1_sin, freq1_cos, freq2_sin, freq2_cos], dim=1))

    x = torch.cat(parts, dim=1)
    return x


def compute_feat_dim(cfg=None):
    """Compute the expected node feature dimensionality from config."""
    dim = 0
    if cfg is None or cfg.get("use_onehot", True):
        dim += AA_DIM  # 20
    if cfg is None or cfg.get("use_physicochemical", True):
        dim += PHYSICO_DIM  # 10
    if cfg is None or cfg.get("use_sasa", True):
        dim += 2  # SASA + RSA
    if cfg is None or cfg.get("use_esm2", True):
        dim += cfg.get("esm2_dim", 320) if cfg else 320
    if cfg is None or cfg.get("use_degree", True):
        dim += 1
    if cfg is None or cfg.get("use_positional", True):
        dim += 5
    return dim


# ============================================================================
# Edge feature builder
# ============================================================================

def compute_edge_features(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    n_nodes: int,
    cfg=None,
):
    """Build per-edge feature matrix.

    Args:
        edge_index: [2, E] edge indices
        coords: [N, 3] Ca coordinates
        n_nodes: number of residues
        cfg: features config dict

    Returns:
        edge_attr: [E, edge_dim] edge feature tensor
    """
    parts = []
    E = edge_index.shape[1]

    # 1) RBF-encoded Euclidean distance
    if cfg is None or cfg.get("use_edge_distance", True):
        if coords is not None:
            src_coords = coords[edge_index[0]]
            dst_coords = coords[edge_index[1]]
            dist = torch.norm(src_coords - dst_coords, dim=1)  # [E]

            n_centers = cfg.get("rbf_centers", 16) if cfg else 16
            sigma = cfg.get("rbf_sigma", 0.5) if cfg else 0.5
            centers = torch.linspace(0, 15.0, n_centers)  # 0 to 15 angstrom
            rbf = torch.exp(-0.5 * ((dist.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)
            parts.append(rbf)
        else:
            n_centers = cfg.get("rbf_centers", 16) if cfg else 16
            parts.append(torch.zeros(E, n_centers))

    # 2) Sequence separation (normalized)
    if cfg is None or cfg.get("use_edge_seqsep", True):
        seq_sep = torch.abs(edge_index[0] - edge_index[1]).float()
        max_sep = max(seq_sep.max().item(), 1.0)
        seq_sep_norm = (seq_sep / max_sep).unsqueeze(1)
        # Also log-scaled separation (captures long-range vs short-range)
        seq_sep_log = torch.log1p(seq_sep).unsqueeze(1) / math.log(max(n_nodes, 2))
        parts.append(torch.cat([seq_sep_norm, seq_sep_log], dim=1))

    if not parts:
        return None

    edge_attr = torch.cat(parts, dim=1)
    return edge_attr


def compute_edge_dim(cfg=None):
    """Compute the expected edge feature dimensionality from config."""
    dim = 0
    if cfg is None or cfg.get("use_edge_distance", True):
        dim += cfg.get("rbf_centers", 16) if cfg else 16
    if cfg is None or cfg.get("use_edge_seqsep", True):
        dim += 2  # normalized + log-scaled
    return dim
