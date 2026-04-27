"""
DeepLoc-2.1 Training Script — Train SoftBlobGIN on subcellular localization.

Uses the exact same model architecture and features as the ProteinShake benchmark.
Only the dataset loading and classifier head are adapted for DeepLoc.

Usage:
    python scripts/deeploc/train_deeploc.py --config configs/deeploc.yaml
    python scripts/deeploc/train_deeploc.py --config configs/deeploc.yaml --quick
"""

import argparse
import logging
import os
import sys
import time
import shutil
from pathlib import Path

# Fix PyTorch OpenMP thread explosion on high-core HPC nodes
# This prevents workers from causing massive RAM bloat via malloc arenas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
# Safety pin for main process too
torch.set_num_threads(4) 

import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader as PyGLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.deeploc.deeploc_dataset import (
    DeepLocDataset,
    LOCALIZATION_CLASSES,
    N_CLASSES,
    preload_esm_for_deeploc,
)
from scripts.deeploc.evaluate_deeploc import evaluate_deeploc
from src.data.features import ESM2Extractor, compute_feat_dim, compute_edge_dim
from src.models.gin import SoftBlobGIN
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str):
    """Setup file + console logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "deeploc_train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


class SoftBlobGIN_MTL(SoftBlobGIN):
    """SoftBlobGIN with Gated Fusion + Dual Read-Out Heads."""
    def __init__(self, in_ch, hidden, n_loc_classes=10, n_mem_classes=4, esm_skip_dim=0, **kwargs):
        super().__init__(in_ch, hidden, n_classes=n_loc_classes, **kwargs)
        
        self.esm_skip_dim = esm_skip_dim
        # Graph embedding dim is hidden * 2 (global_mean + blob_max)
        self.graph_dim = hidden * 2
        
        if esm_skip_dim > 0:
            # Gated fusion to learn how to weigh global vs local context
            self.esm_proj = nn.Linear(esm_skip_dim, self.graph_dim)
            self.gate = nn.Sequential(
                nn.Linear(self.graph_dim * 2, 1),
                nn.Sigmoid()
            )
        
        self.clf_loc = nn.Sequential(
            nn.Linear(self.graph_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.5)),
            nn.Linear(hidden, n_loc_classes),
        )
        self.clf_mem = nn.Sequential(
            nn.Linear(self.graph_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.5)),
            nn.Linear(hidden, n_mem_classes),
        )
        del self.clf

    def forward(self, data, return_blobs=False):
        z_graph = self.forward_internal(data) # [B, hidden*2]
        
        if self.esm_skip_dim > 0:
            if hasattr(data, "global_x_esm"):
                z_esm = data.global_x_esm
            else:
                z_esm = torch.zeros(z_graph.shape[0], self.esm_skip_dim, device=z_graph.device)
            
            # Gated residual fusion
            z_esm_proj = F.relu(self.esm_proj(z_esm))
            # learnable gate: how much ESM vs how much GIN
            g = self.gate(torch.cat([z_graph, z_esm_proj], dim=-1))
            z = (1 - g) * z_graph + g * z_esm_proj
        else:
            z = z_graph
            
        out_loc = self.clf_loc(z)
        out_mem = self.clf_mem(z)
        out = [out_loc, out_mem]
        
        if return_blobs:
            x, _ = self._encode(data)
            assign = F.softmax(self.blob_head(x), dim=-1)
            return out, assign
        return out


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for Multi-label Binary Classification."""
    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits
        # targets: [B, C] floats
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss) # prob of correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DeepLocMTLLoss(nn.Module):
    """Joint Loss for Subcellular Location (Multi-Label) + Membrane Type (Multi-Class)."""
    def __init__(self, lambda_mem=0.2, pos_weight=None):
        super().__init__()
        # Use Focal Loss with class weights for imbalanced localization
        self.criterion_loc = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=2.0)
        self.criterion_mem = nn.CrossEntropyLoss(ignore_index=-100)
        self.lambda_mem = lambda_mem
        self.is_mtl = True  # Signal to Trainer to pass lists directly
        
    def forward(self, out_list, y):
        out_loc, out_mem = out_list[0], out_list[1]
        
        # If y is Batch, use y.y_loc. If y is list, it's the old way (fallback)
        if hasattr(y, "y_loc"):
            y_loc = y.y_loc
            y_mem = y.y_mem
        else:
            y_loc, y_mem = y[0], y[1]
            
        if y_mem.dim() == 2 and y_mem.shape[1] == 1:
            y_mem = y_mem.squeeze(1)
            
        if y_loc.dim() != out_loc.dim():
            y_loc = y_loc.view(out_loc.shape)
            
        loss_loc = self.criterion_loc(out_loc, y_loc)
        loss_mem = self.criterion_mem(out_mem, y_mem)
        return loss_loc + self.lambda_mem * loss_mem


def build_model(feat_dim: int, edge_dim: int, n_classes: int, model_cfg: dict, feat_cfg: dict, device: torch.device):
    """Build SoftBlobGIN_MTL with dual heads."""
    model = SoftBlobGIN_MTL(
        in_ch=feat_dim,
        hidden=model_cfg.get("hidden", 256),
        n_loc_classes=N_CLASSES,
        n_mem_classes=4,
        edge_dim=edge_dim,
        n_layers=model_cfg.get("n_layers", 3),
        n_blobs=model_cfg.get("n_blobs", 8),
        dropout=model_cfg.get("dropout", 0.5),
        tau_start=model_cfg.get("tau_start", 1.0),
        tau_end=model_cfg.get("tau_end", 0.1),
        esm_skip_dim=feat_cfg.get("esm2_dim", 1280) if (feat_cfg.get("use_esm2") and model_cfg.get("use_global_skip", True)) else 0,
    ).to(device)
    return model


def collate_filter(batch):
    """Remove None entries from the batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def main():
    parser = argparse.ArgumentParser(description="DeepLoc-2.1 SoftBlobGIN Training")
    parser.add_argument("--config", type=str, default="configs/deeploc.yaml")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 epoch, small subset for pipeline verification",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fold", type=int, default=None,
                        help="Override validation partition (0-4) for cross-validation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup
    logger_inst = setup_logging(cfg["paths"]["log_dir"])
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    logger.info("=" * 70)
    logger.info("DeepLoc-2.1 SoftBlobGIN Training")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # Paths
    data_root = Path(cfg["paths"]["data_root"])
    feat_cfg = cfg.get("features", {})
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    val_partition = args.fold if args.fold is not None else ds_cfg.get("val_partition", 4)
    max_seq_len = ds_cfg.get("max_seq_len", 2000)
    k_neighbors = ds_cfg.get("k_neighbors", 10)

    # ------------------------------------------------------------------
    # 1. ESM-2 — Disk-only caching (no RAM preload to save memory)
    # ------------------------------------------------------------------
    esm_extractor = None
    if feat_cfg.get("use_esm2", False):
        cache_dir = cfg["paths"].get("esm_cache", "./data/esm_cache_650M/DeepLoc")
        esm_extractor = ESM2Extractor(
            model_name=feat_cfg.get("esm2_model", "esm2_t33_650M_UR50D"),
            layer=feat_cfg.get("esm2_layer", 33),
            cache_dir=cache_dir,
        )

        # Pre-extract embeddings to DISK (not RAM) for all proteins
        # This avoids ~80GB RAM usage for 28k proteins
        train_csv = data_root / ds_cfg.get("train_csv", "Swissprot_Train_Validation_dataset.csv")
        df_all = pd.read_csv(train_csv)

        # Detect ID column
        if "ACC" in df_all.columns:
            id_col = "ACC"
        elif "ACC\n" in df_all.columns:
            id_col = "ACC\n"
        else:
            id_col = df_all.columns[1]

        # Also include test set
        test_csv_path = data_root / ds_cfg.get("test_csv", "hpa_testset.csv")
        if test_csv_path.exists():
            df_test = pd.read_csv(test_csv_path)
            test_id_col = "sid" if "sid" in df_test.columns else df_test.columns[0]
            df_test_renamed = df_test.rename(columns={test_id_col: id_col})
            if "Sequence" not in df_test_renamed.columns and "fasta" in df_test_renamed.columns:
                df_test_renamed = df_test_renamed.rename(columns={"fasta": "Sequence"})
            df_combined = pd.concat([df_all, df_test_renamed], ignore_index=True)
        else:
            df_combined = df_all

        if not args.quick:
            # Extract to disk only — don't keep in RAM
            preload_esm_for_deeploc(
                df_combined, esm_extractor, id_col,
                max_seq_len=max_seq_len, batch_size=4,
            )
        else:
            preload_esm_for_deeploc(
                df_combined.head(20), esm_extractor, id_col,
                max_seq_len=max_seq_len, batch_size=4,
            )

        # Clear the RAM cache — rely on disk cache during training
        esm_extractor.clear_ram_cache()
        logger.info("Cleared ESM-2 RAM cache — using disk-only mode for training.")

    # ------------------------------------------------------------------
    # 2. Dataset Construction
    # ------------------------------------------------------------------
    train_csv_path = data_root / ds_cfg.get("train_csv", "Swissprot_Train_Validation_dataset.csv")
    df_full = pd.read_csv(train_csv_path)

    # Split by Partition column
    train_mask = df_full["Partition"] != val_partition
    val_mask = df_full["Partition"] == val_partition

    df_train = df_full[train_mask].reset_index(drop=True)
    df_val = df_full[val_mask].reset_index(drop=True)

    if args.quick:
        df_train = df_train.head(10)
        df_val = df_val.head(5)

    # Save split DataFrames to temp CSVs for the Dataset class
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    train_tmp = os.path.join(cfg["paths"]["output_dir"], "_train_split.csv")
    val_tmp = os.path.join(cfg["paths"]["output_dir"], "_val_split.csv")
    df_train.to_csv(train_tmp, index=False)
    df_val.to_csv(val_tmp, index=False)

    membrane_csv = data_root / "Swissprot_Membrane_Train_Validation_dataset.csv"

    ds_kwargs = {
        "feat_cfg": feat_cfg,
        "esm_extractor": esm_extractor,
        "max_seq_len": max_seq_len,
        "k_neighbors": k_neighbors,
        "membrane_csv_path": str(membrane_csv) if membrane_csv.exists() else None,
    }

    train_ds = DeepLocDataset(csv_path=train_tmp, id_column=id_col, **ds_kwargs)
    val_ds = DeepLocDataset(csv_path=val_tmp, id_column=id_col, **ds_kwargs)

    # Test set (HPA)
    test_csv_path = data_root / ds_cfg.get("test_csv", "hpa_testset.csv")
    test_ds = None
    if test_csv_path.exists():
        # HPA has different column names
        hpa_df = pd.read_csv(test_csv_path)
        hpa_label_cols = [c for c in LOCALIZATION_CLASSES if c in hpa_df.columns]

        # Handle "fasta" column → "Sequence"
        seq_col = "Sequence"
        if "Sequence" not in hpa_df.columns and "fasta" in hpa_df.columns:
            seq_col = "fasta"

        test_ds = DeepLocDataset(
            csv_path=str(test_csv_path),
            label_columns=hpa_label_cols,
            id_column="sid" if "sid" in hpa_df.columns else None,
            seq_column=seq_col,
            **ds_kwargs
        )
        if args.quick:
            test_ds.df = test_ds.df.head(5)

    # DataLoaders
    bs = train_cfg.get("batch_size", 32)
    nw = train_cfg.get("num_workers", 4) if not args.quick else 0

    train_loader = PyGLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=False,
        persistent_workers=nw > 0,
        collate_fn=collate_filter if nw == 0 else None,
    )
    val_loader = PyGLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=False,
        persistent_workers=nw > 0,
        collate_fn=collate_filter if nw == 0 else None,
    )

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds) if test_ds else 'N/A'}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    feat_dim = compute_feat_dim(feat_cfg)
    edge_dim = compute_edge_dim(feat_cfg)
    model = build_model(feat_dim, edge_dim, N_CLASSES, model_cfg, feat_cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__}, params={n_params:,}, feat_dim={feat_dim}, edge_dim={edge_dim}")

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    # Compute class weights for Focal Loss
    label_counts = df_train[LOCALIZATION_CLASSES].sum(axis=0).values
    pos_weight = torch.tensor((len(df_train) - label_counts) / (label_counts + 1), dtype=torch.float32).to(device)
    logger.info(f"Class frequencies: {label_counts / len(df_train)}")
    
    criterion = DeepLocMTLLoss(lambda_mem=0.2, pos_weight=pos_weight)

    # Build config for Trainer
    trainer_cfg = dict(train_cfg)
    if args.quick:
        trainer_cfg["epochs"] = 1
        trainer_cfg["patience"] = 1

    # Build full config for Trainer (it expects cfg["training"] and cfg["paths"])
    trainer_full_cfg = {
        "training": trainer_cfg,
        "paths": cfg["paths"],
    }

    trainer = Trainer(
        model, trainer_full_cfg, device,
        is_pyg=True,
        model_name="SoftBlobGIN_DeepLoc",
    )


    logger.info("Starting training...")
    t_start = time.time()
    history, t_elapsed = trainer.train(train_loader, val_loader, criterion)
    logger.info(f"Training complete in {t_elapsed:.1f}s ({time.time() - t_start:.1f}s total)")

    # Save checkpoint
    ckpt_dir = cfg["paths"].get("checkpoint_dir", "./outputs_deeploc/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "deeploc_softblobgin_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------
    # 5. Evaluation (with Threshold Tuning on Validation)
    # ------------------------------------------------------------------
    logger.info("Evaluating on Validation Set (tuning thresholds)...")
    val_metrics = evaluate_deeploc(model, val_loader, device, LOCALIZATION_CLASSES)
    val_thresholds = val_metrics.get("best_thresholds")
    
    logger.info("Validation Results:")
    for k, v in val_metrics.items():
        if k != "best_thresholds":
            logger.info(f"  {k}: {v:.4f}")

    if test_ds is not None:
        logger.info("Evaluating on HPA Test Set (applying val thresholds)...")
        test_loader = PyGLoader(
            test_ds, batch_size=bs, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        
        # Only evaluate on classes present in HPA but keep the same threshold indexing
        metrics = evaluate_deeploc(model, test_loader, device, LOCALIZATION_CLASSES, thresholds=val_thresholds)
        
        # Filter metrics to only show HPA-present classes for clarity
        hpa_present = [c for c in LOCALIZATION_CLASSES if c in hpa_label_cols]
        # Only report metrics for classes actually present in HPA to avoid deflating macro-averages
        logger.info("=" * 50)
        logger.info(f"HPA Test Set Results (Classes present: {len(hpa_present)}/10):")
        for k, v in metrics.items():
            if k == "best_thresholds": continue
            # Check if this metric belongs to an absent class
            is_absent = any((c in k and c not in hpa_present) for c in LOCALIZATION_CLASSES)
            if not is_absent:
                logger.info(f"  {k}: {v:.4f}")

        # Save results
        results_path = os.path.join(cfg["paths"]["output_dir"], "deeploc_results.csv")
        pd.DataFrame([metrics]).to_csv(results_path, index=False)
        logger.info(f"Results saved: {results_path}")
    else:
        logger.info("No test set available — skipping evaluation.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
