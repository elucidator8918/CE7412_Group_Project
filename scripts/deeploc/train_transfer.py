"""
DeepLoc Transfer Learning Script — Transfer weights from ProteinShake benchmarks.
Available sources: BindingSiteDetectionTask, ProteinFamilyTask, GeneOntologyTask.

Usage:
    python scripts/deeploc/train_transfer.py --config configs/deeploc.yaml --source ProteinFamilyTask
"""

import argparse
import logging
import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.deeploc.train_deeploc import (
    SoftBlobGIN_MTL, DeepLocMTLLoss, setup_logging,
    DeepLocDataset, PyGLoader, preload_esm_for_deeploc,
    LOCALIZATION_CLASSES, N_CLASSES
)
from src.data.features import compute_feat_dim, compute_edge_dim, ESM2Extractor
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

def load_transfer_weights(model, checkpoint_path, freeze=True):
    """Load weights from a benchmark checkpoint into the DeepLoc model."""
    logger.info(f"Loading transfer weights from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Strip 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    model_dict = model.state_dict()
    
    # Filter out classifier weights and any keys not in the target model
    transfer_dict = {k: v for k, v in state_dict.items() if k in model_dict and "clf" not in k}
    
    if len(transfer_dict) == 0:
        logger.warning("  WARNING: No keys were transferred! Check checkpoint compatibility.")
    else:
        logger.info(f"  Transferred {len(transfer_dict)} keys (GIN backbone + Blob head).")
    
    model_dict.update(transfer_dict)
    model.load_state_dict(model_dict)
    
    if freeze:
        logger.info("  Freezing transferred layers...")
        for name, param in model.named_parameters():
            if name in transfer_dict:
                param.requires_grad = False
                
    return len(transfer_dict)

def main():
    parser = argparse.ArgumentParser(description="DeepLoc Transfer Learning")
    parser.add_argument("--config", type=str, default="configs/deeploc.yaml")
    parser.add_argument("--source", type=str, choices=["BindingSiteDetectionTask", "ProteinFamilyTask", "GeneOntologyTask"], default="ProteinFamilyTask")
    parser.add_argument("--freeze", action="store_true", default=True, help="Freeze transferred backbone")
    parser.add_argument("--quick", action="store_true", help="Run on tiny subset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg["paths"]["log_dir"]
    setup_logging(log_dir)
    logger.info("=" * 70)
    logger.info(f"DeepLoc Transfer Learning (Source: {args.source})")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset & Loaders
    logger.info("Preparing datasets...")
    data_root = Path(cfg["paths"]["data_root"])
    ds_cfg = cfg.get("dataset", {})
    feat_cfg = cfg.get("features", {})
    
    val_partition = ds_cfg.get("val_partition", 4)
    train_csv_path = data_root / ds_cfg.get("train_csv", "Swissprot_Train_Validation_dataset.csv")
    df_full = pd.read_csv(train_csv_path)

    # Detect ID column
    if "ACC" in df_full.columns: id_col = "ACC"
    elif "ACC\n" in df_full.columns: id_col = "ACC\n"
    else: id_col = df_full.columns[1]

    train_mask = df_full["Partition"] != val_partition
    val_mask = df_full["Partition"] == val_partition
    df_train = df_full[train_mask].reset_index(drop=True)
    df_val = df_full[val_mask].reset_index(drop=True)

    if args.quick:
        df_train = df_train.head(10)
        df_val = df_val.head(5)

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    train_tmp = os.path.join(cfg["paths"]["output_dir"], "_train_transfer_split.csv")
    val_tmp = os.path.join(cfg["paths"]["output_dir"], "_val_transfer_split.csv")
    df_train.to_csv(train_tmp, index=False)
    df_val.to_csv(val_tmp, index=False)

    esm_extractor = None
    if feat_cfg.get("use_esm2", False):
        esm_extractor = ESM2Extractor(
            model_name=feat_cfg.get("esm2_model", "esm2_t33_650M_UR50D"),
            layer=feat_cfg.get("esm2_layer", 33),
            cache_dir=cfg["paths"].get("esm_cache", "./data/esm_cache_650M/DeepLoc"),
        )

    ds_kwargs = {
        "feat_cfg": feat_cfg,
        "esm_extractor": esm_extractor,
        "max_seq_len": ds_cfg.get("max_seq_len", 2000),
        "k_neighbors": ds_cfg.get("k_neighbors", 10),
    }

    train_ds = DeepLocDataset(csv_path=train_tmp, id_column=id_col, **ds_kwargs)
    val_ds = DeepLocDataset(csv_path=val_tmp, id_column=id_col, **ds_kwargs)

    train_loader = PyGLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = PyGLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    # 2. Model Initialization
    in_ch = compute_feat_dim(cfg["features"])
    edge_dim = compute_edge_dim(cfg["features"])
    hidden = cfg["model"]["hidden"]
    # Force ESM skip to 0 to ensure no ESM injection at the output
    esm_skip_dim = 0

    model = SoftBlobGIN_MTL(
        in_ch=in_ch,
        hidden=hidden,
        n_loc_classes=N_CLASSES,
        n_mem_classes=4,
        edge_dim=edge_dim,
        n_layers=cfg["model"]["n_layers"],
        n_blobs=cfg["model"]["n_blobs"],
        dropout=cfg["model"]["dropout"],
        tau_start=cfg["model"]["tau_start"],
        tau_end=cfg["model"]["tau_end"],
        esm_skip_dim=esm_skip_dim
    )

    # 3. Weight Transfer
    cp_name = f"GIN_{args.source}_best.pt"
    cp_path = os.path.join(cfg["paths"]["checkpoint_dir"].replace("outputs_deeploc", "outputs_benchmark"), cp_name)
    # Correcting path if the above heuristic fails
    if not os.path.exists(cp_path):
        cp_path = f"outputs_benchmark/checkpoints/{cp_name}"
        
    load_transfer_weights(model, cp_path, freeze=args.freeze)
    model.to(device)

    # 4. Loss & Trainer
    # Compute class weights for Focal Loss
    label_counts = train_ds.df[LOCALIZATION_CLASSES].sum(axis=0).values
    neg_counts = len(train_ds.df) - label_counts
    pos_weight = torch.tensor(neg_counts / (label_counts + 1e-6), dtype=torch.float32).to(device)
    
    criterion = DeepLocMTLLoss(pos_weight=pos_weight)
    
    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        model_name=f"DeepLoc_Transfer_{args.source}"
    )

    # 5. Execute Training
    logger.info("Starting Transfer Learning Training...")
    trainer.train(train_loader, val_loader, criterion=criterion)

if __name__ == "__main__":
    main()
