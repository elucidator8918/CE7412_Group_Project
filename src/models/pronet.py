"""
ProNet: Dilated 1D CNN for protein sequence-based prediction.

Faithfully adapted from ProNet/src/pronet.py for the ProteinShake benchmark.
Preserves the original architecture (channels, kernel widths, dilation schedule,
grouped convolutions, skip-connection accumulation) while adding:

  - Variable-length support via sequence padding + global average pooling,
    replacing the original fixed-length flatten → Linear(L*600, n_classes) head.
  - Per-task wrappers that accept PyG Data/Batch objects from BenchmarkDataset.
  - Node-level output for BindingSiteDetection / PPI.

Original hyperparameters kept verbatim:
  L  = 64             (channel width)
  W  = [11]*8+[21]*4+[41]*4   (kernel sizes for 16 residual units)
  AR = [1]*4+[4]*4+[10]*4+[25]*4 (dilation rates)
  CARDINALITY_ITEM = 16       (for grouped convolutions: C = L // 16 = 4)

Input to CNN: [B, 20, max_seq_len] — one-hot amino acid encoding, padded.
Proteins are truncated at max_seq_len; residues beyond it are not predicted.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Original ProNet constants ────────────────────────────────────────────────
_L               = 64
_INPUT_CHANNELS  = 20
_CARDINALITY     = 16                         # groups = L // CARDINALITY = 4
_W  = np.array([11]*8 + [21]*4 + [41]*4)     # kernel widths, 16 residual units
_AR = np.array([1]*4  + [4]*4  + [10]*4 + [25]*4)  # dilation rates


# ── Building blocks (unchanged from ProNet/src/pronet.py) ───────────────────

class _ResidualUnit(nn.Module):
    def __init__(self, l, w, ar, bot_mul=1):
        super().__init__()
        bot_channels = int(round(l * bot_mul))
        C = bot_channels // _CARDINALITY
        self.bn1   = nn.BatchNorm1d(l)
        self.bn2   = nn.BatchNorm1d(l)
        self.relu  = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2, groups=C)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2, groups=C)

    def forward(self, x, y):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        return x + x2, y          # residual update; skip y passes through


class _Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y   # accumulate skip; x passes through


# ── CNN backbone ─────────────────────────────────────────────────────────────

class ProNetEncoder(nn.Module):
    """ProNet dilated-CNN backbone.

    Forward returns per-position features [B, L, T] (same spatial resolution
    as the input). Callers can either pool (graph tasks) or index (node tasks).
    """

    def __init__(self, input_channels=_INPUT_CHANNELS, L=_L,
                 W=_W, AR=_AR):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, L, 1)
        self.skip1 = _Skip(L)

        blocks = []
        for i, (w, r) in enumerate(zip(W, AR)):
            blocks.append(_ResidualUnit(L, int(w), int(r)))
            if (i + 1) % 4 == 0:
                blocks.append(_Skip(L))
        if (len(W) + 1) % 4 != 0:
            blocks.append(_Skip(L))
        self.blocks = nn.ModuleList(blocks)

        self.last_conv = nn.Conv1d(L, L, 1)
        self.output_dim = L

    def forward(self, x):
        """x: [B, input_channels, T] → [B, L, T] per-position features."""
        x, skip = self.skip1(self.conv1(x), 0)
        for blk in self.blocks:
            x, skip = blk(x, skip)
        return self.last_conv(skip)   # [B, L, T]


# ── Shared base: handles PyG Batch → padded tensor conversion ────────────────

class _ProNetBase(nn.Module):
    """Common helper shared by all task-specific ProNet wrappers.

    BenchmarkDataset stores per-residue features in data.x [N_total, 20].
    We extract per-protein slices via data.ptr, pad/truncate to max_seq_len,
    and stack into [B, 20, max_seq_len] before passing to the CNN.
    """

    def __init__(self, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = ProNetEncoder(input_channels, L, W, AR)

    def _to_padded(self, data):
        """data.x [N_total, C] + data.ptr → [B, C, max_seq_len]."""
        ptr    = data.ptr.cpu()
        B      = len(ptr) - 1
        device = data.x.device
        C      = data.x.shape[1]
        T      = self.max_seq_len

        out = data.x.new_zeros(B, C, T)
        for g in range(B):
            s, e = int(ptr[g]), int(ptr[g + 1])
            lng  = e - s
            take = min(lng, T)
            out[g, :, :take] = data.x[s:s + take].T.float()
        return out            # [B, C, T]

    def _lengths(self, data):
        ptr = data.ptr.cpu()
        return [int(ptr[g + 1]) - int(ptr[g]) for g in range(len(ptr) - 1)]

    def _encode_graph(self, data):
        """→ [B, L] via global average pool."""
        x   = self._to_padded(data)            # [B, C, T]
        out = self.encoder(x)                  # [B, L, T]
        return out.mean(dim=-1)                # [B, L]

    def _encode_nodes(self, data):
        """→ [N_total, L] per-residue features (truncated residues padded to 0)."""
        ptr    = data.ptr.cpu()
        B      = len(ptr) - 1
        T      = self.max_seq_len

        x      = self._to_padded(data)                      # [B, C, T]
        out    = self.encoder(x).permute(0, 2, 1)           # [B, T, L]

        node_feats = []
        for g in range(B):
            s, e  = int(ptr[g]), int(ptr[g + 1])
            lng   = e - s
            take  = min(lng, T)
            real  = out[g, :take]                            # [take, L]
            if lng > T:
                # Residues beyond max_seq_len get zero features (no prediction)
                pad  = real.new_zeros(lng - T, out.shape[-1])
                real = torch.cat([real, pad], dim=0)
            node_feats.append(real)
        return torch.cat(node_feats, dim=0)                  # [N_total, L]


# ── Task-specific wrappers ───────────────────────────────────────────────────

class ProNetClassifier(_ProNetBase):
    """ProNet for graph-level multiclass classification (ProteinFamily, StructuralClass)."""

    def __init__(self, n_classes, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__(max_seq_len, L, input_channels, W, AR)
        self.head = nn.Linear(L, n_classes)

    def forward(self, data):
        z = self._encode_graph(data)   # [B, L]
        return self.head(z)            # [B, n_classes]


class ProNetMultiLabel(_ProNetBase):
    """ProNet for graph-level multi-label classification (EnzymeClass, GeneOntology)."""

    def __init__(self, n_classes, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__(max_seq_len, L, input_channels, W, AR)
        self.head = nn.Linear(L, n_classes)

    def forward(self, data):
        z = self._encode_graph(data)
        return self.head(z)            # raw logits — sigmoid in loss/eval


class ProNetRegressor(_ProNetBase):
    """ProNet for graph-level regression (LigandAffinity)."""

    def __init__(self, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__(max_seq_len, L, input_channels, W, AR)
        self.head = nn.Linear(L, 1)

    def forward(self, data):
        z = self._encode_graph(data)
        return self.head(z).squeeze(-1)   # [B]


class ProNetNodeClassifier(_ProNetBase):
    """ProNet for node-level binary classification (BindingSiteDetection).

    Uses per-position CNN features directly, without global pooling.
    Residues truncated beyond max_seq_len receive logit=0 (probability=0.5).
    """

    def __init__(self, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__(max_seq_len, L, input_channels, W, AR)
        self.head = nn.Linear(L, 1)

    def forward(self, data):
        h = self._encode_nodes(data)           # [N_total, L]
        return self.head(h).squeeze(-1)        # [N_total] logits


class ProNetPPI(_ProNetBase):
    """ProNet for Protein-Protein Interface prediction.

    Per-residue of protein A concatenated with mean-pooled protein B context
    → single linear layer → binary logit per residue of A.
    Mirrors ESMProbePPI's head design for a fair structural comparison.
    """

    def __init__(self, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__(max_seq_len, L, input_channels, W, AR)
        self.head = nn.Linear(L * 2, 1)

    def forward(self, data1, data2=None):
        if data2 is None:
            data2 = data1.b2
            data1 = data1.b1

        h_a  = self._encode_nodes(data1)          # [N_A, L]
        z_b  = self._encode_graph(data2)           # [B, L]
        # Broadcast protein-B context to every residue of A
        z_b_per_node = z_b[data1.batch]            # [N_A, L]
        cat = torch.cat([h_a, z_b_per_node], dim=-1)
        return self.head(cat).squeeze(-1)          # [N_A] logits


class ProNetSiamese(_ProNetBase):
    """ProNet for protein-pair tasks (StructureSimilarity, StructureSearch).

    Same 4-way feature vector as SoftBlobGINSiamese and ESMProbeSiamese:
      [z1 | z2 | |z1-z2| | z1*z2] → Linear
    """

    def __init__(self, max_seq_len=1024, L=_L,
                 input_channels=_INPUT_CHANNELS, W=_W, AR=_AR):
        super().__init__(max_seq_len, L, input_channels, W, AR)
        self.head = nn.Linear(L * 4, 1)

    def forward(self, data1, data2=None):
        if data2 is None:
            data2 = data1.b2
            data1 = data1.b1

        z1 = self._encode_graph(data1)              # [B, L]
        z2 = self._encode_graph(data2)
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        return self.head(combined).squeeze(-1)      # [B]
