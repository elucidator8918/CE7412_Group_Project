"""
DeepLoc-2.1 Final Result Reporter — Load checkpoint and print detailed metrics.
"""

import argparse
import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from torch_geometric.loader import DataLoader as PyGLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.deeploc.deeploc_dataset import DeepLocDataset, LOCALIZATION_CLASSES, MEMBRANE_CLASSES
from scripts.deeploc.evaluate_deeploc import evaluate_deeploc
from scripts.deeploc.train_deeploc import SoftBlobGIN_MTL
from src.data.features import ESM2Extractor, compute_feat_dim, compute_edge_dim

def main():
    parser = argparse.ArgumentParser(description="DeepLoc-2.1 Result Reporter")
    parser.add_argument("--config", type=str, default="configs/deeploc.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs_deeploc/checkpoints/deeploc_softblobgin_best.pt")
    parser.add_argument("--quick", action="store_true", help="Run on tiny subset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(cfg["paths"]["data_root"])
    ds_cfg = cfg.get("dataset", {})
    feat_cfg = cfg.get("features", {})
    val_partition = ds_cfg.get("val_partition", 4)

    # 1. Load Data
    print(f"Loading data from partition {val_partition}...")
    train_csv_path = data_root / ds_cfg.get("train_csv", "Swissprot_Train_Validation_dataset.csv")
    df_full = pd.read_csv(train_csv_path)
    
    # Detect ID column
    if "ACC" in df_full.columns: id_col = "ACC"
    elif "ACC\n" in df_full.columns: id_col = "ACC\n"
    else: id_col = df_full.columns[1]

    val_df = df_full[df_full["Partition"] == val_partition].reset_index(drop=True)
    if args.quick: val_df = val_df.head(10)
    
    # Save temp split
    val_tmp = "outputs_deeploc/_val_eval_tmp.csv"
    val_df.to_csv(val_tmp, index=False)

    esm_extractor = None
    if feat_cfg.get("use_esm2", False):
        esm_extractor = ESM2Extractor(
            model_name=feat_cfg.get("esm2_model", "esm2_t33_650M_UR50D"),
            layer=feat_cfg.get("esm2_layer", 33),
            cache_dir=cfg["paths"].get("esm_cache", "./data/esm_cache_650M/DeepLoc"),
        )

    val_ds = DeepLocDataset(
        csv_path=val_tmp, 
        id_column=id_col,
        feat_cfg=feat_cfg,
        esm_extractor=esm_extractor,
        max_seq_len=ds_cfg.get("max_seq_len", 2000),
        k_neighbors=ds_cfg.get("k_neighbors", 10),
        membrane_csv_path=str(data_root / "Swissprot_Membrane_Train_Validation_dataset.csv")
    )
    val_loader = PyGLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    in_ch = compute_feat_dim(feat_cfg)
    edge_dim = compute_edge_dim(feat_cfg)
    esm_skip_dim = feat_cfg.get("esm2_dim", 1280) if (feat_cfg.get("use_esm2") and cfg["model"].get("use_global_skip", True)) else 0

    model_kwargs = cfg["model"].copy()
    model_type = model_kwargs.pop("type", "soft_blob_gin")
    hidden = model_kwargs.pop("hidden", 256)

    model = SoftBlobGIN_MTL(
        in_ch=in_ch,
        hidden=hidden,
        n_loc_classes=len(LOCALIZATION_CLASSES),
        n_mem_classes=len(MEMBRANE_CLASSES),
        edge_dim=edge_dim,
        esm_skip_dim=esm_skip_dim,
        n_layers=cfg["model"]["n_layers"],
        n_blobs=cfg["model"]["n_blobs"],
        dropout=cfg["model"]["dropout"],
        tau_start=cfg["model"]["tau_start"],
        tau_end=cfg["model"]["tau_end"]
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # 3. Evaluate
    print("Evaluating on Validation Set...")
    metrics = evaluate_deeploc(model, val_loader, device, LOCALIZATION_CLASSES)

    # 4. Print Results
    print("\n" + "="*50)
    print("DEEPLOC-2.1 FINAL EVALUATION RESULTS (VALIDATION)")
    print("="*50)
    
    summary_data = [
        ["Fmax", f"{metrics['fmax']:.4f}"],
        ["AUROC (Macro)", f"{metrics['auroc_macro']:.4f}"],
        ["MCC (Macro)", f"{metrics['mcc_macro']:.4f}"],
        ["Hamming Accuracy", f"{metrics['hamming_accuracy']:.4f}"],
        ["F1 (Micro)", f"{metrics['f1_micro']:.4f}"],
        ["F1 (Macro)", f"{metrics['f1_macro']:.4f}"],
    ]
    print("\nOVERALL METRICS:")
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid"))

    class_data = []
    for cls in LOCALIZATION_CLASSES:
        auc_val = metrics.get(f'auroc_{cls}', "N/A")
        mcc_val = metrics.get(f'mcc_{cls}', "N/A")
        
        class_data.append([
            cls,
            f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val,
            f"{mcc_val:.4f}" if isinstance(mcc_val, float) else mcc_val,
            f"{metrics['best_thresholds'][LOCALIZATION_CLASSES.index(cls)]:.2f}"
        ])
    
    print("\nPER-CLASS METRICS:")
    print(tabulate(class_data, headers=["Location", "AUROC", "MCC", "Threshold"], tablefmt="grid"))

    if "membrane_accuracy" in metrics:
        print("\nMEMBRANE CLASSIFICATION (MTL Head):")
        mem_data = [
            ["Accuracy", f"{metrics['membrane_accuracy']:.4f}"],
            ["MCC", f"{metrics['membrane_mcc']:.4f}"],
            ["F1 (Macro)", f"{metrics['membrane_f1_macro']:.4f}"]
        ]
        print(tabulate(mem_data, headers=["Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()
