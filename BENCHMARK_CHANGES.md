# Benchmark Integration Summary

This document summarizes the changes made to the SoftBlobGIN pipeline to support the full [ProteinShake benchmark suite](https://proteinshake.ai/).

## 1. Training Pipeline Changes

### Data Loading & Collation
- **File**: `src/data/benchmark_dataset.py`
- **Key Modification**: Added `PairBatch` class and `pair_collate` function.
- **Change**: Tasks involving protein pairs (Similarity, Search, PPI) now use a custom collator that handles dual-graph batches. Specifically for PPI, interaction matrices are returned as a **list of tensors** to accommodate variable residue counts ($N_1 \times N_2$) within a single batch.

### Training & Evaluation Loops
- **Files**: `src/training/trainer.py` & `scripts/benchmark.py`
- **Key Modification**: Support for `PairBatch` and heterogeneous output handling.
- **Line-for-Line logic (Abstracted)**:
```python
# Before
out = self.model(batch)
loss = criterion(out, batch.y.long())

# After
if hasattr(batch, 'b1'):
    out = self.model(batch.b1, batch.b2)
else:
    out = self.model(batch)

if isinstance(batch.y, list):
    # Sum/Average loss across variable-sized interaction matrices
    loss = sum(criterion(o, t) for o, t in zip(out, batch.y)) / len(batch.y)
```

---

## 2. Model Architecture Changes

The core **SoftBlobGIN** backbone remains identical to the Enzyme Classification baseline, preserving the **Novelty** (Gumbel-Softmax Blob Pooling, GINEConv, and ESM-2 features). We implemented task-specific wrappers to adapt this backbone.

### [NEW] SoftBlobGINSiamese
- **Task**: `StructureSimilarityTask`, `StructureSearchTask`
- **Architecture**:
    1.  Encodes Protein A and Protein B using the **same** shared SoftBlobGIN backbone.
    2.  Extracts "Blob-Enhanced" global embeddings (Global Mean + Global Blob Max).
    3.  Concatenates features: $[z1, z2, |z1-z2|, z1 \cdot z2]$.
    4.  Passes through a 2-layer MLP head to produce a scalar score.

### [NEW] SoftBlobGINPPI
- **Task**: `ProteinProteinInterfaceTask`
- **Architecture**:
    1.  **Node Encoder**: Standard GIN backbone extracts per-residue features ($h_1, h_2$).
    2.  **Graph Encoder**: The **SoftBlobGIN** backbone extracts a global structural summary of the partner protein ($z_2, z_1$).
    3.  **Partner Context**: Concatenates partner summary to each residue: $\tilde{h}_1 = [h_1, z_2], \tilde{h}_2 = [h_2, z_1]$.
    4.  **Interaction Head**: Computes interaction matrix $M = Q(\tilde{h}_1) \cdot K(\tilde{h}_2)^T$.

---

## 3. Tasks, Heads, and Losses

| Task Type | Examples | Output Head | Loss Function | Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Multiclass** | Enzyme Class | GIN Linear Head | Focal Loss ($\gamma=1.0$) | Accuracy |
| **Multilabel** | Gene Ontology | GIN Linear Head | Binary Focal Loss | Fmax |
| **Multiclass** | Structural Class | GIN Linear Head | Focal Loss ($\gamma=1.0$) | Accuracy |
| **Pair Regression** | Ligand Affinity | Siamese MLP | Mean Squared Error (MSE) | MSE |
| **Pair Retrieval** | S. Similarity | Siamese MLP | Focal Loss (Binary) | Hit@10 |
| **Node Binary** | Binding Site | GIN Node Head | Focal Loss (Binary) | F1-Score |
| **PPI Interaction** | Interface | Bilinear Dot-Product | Focal Loss (Avg/Matrix) | Accuracy |

---

## 4. Hyperparameter Synchronization
The `configs/benchmark.yaml` has been fully synchronized with the `full_power.yaml` settings to ensure the benchmark reflects the model's maximum performance:
- **Features**: ESM-2 (650M), Physically-informed nodes, RBF-encoded edges.
- **Optimization**: LR `1e-3`, Cosine Warmup, Weight Decay `1e-4`.
- **Regularization**: Dropout `0.3`, Edge Drop `0.05`.

> [!IMPORTANT]
> The implementation has been verified to be **bioinformatically correct**, ensuring that residue-level predictions respect variable sequence lengths and that structural similarities are learned in a shared, blob-aware latent space.
