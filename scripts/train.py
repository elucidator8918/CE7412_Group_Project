#!/usr/bin/env python3
"""
CE7412 Enzyme Classification — Main Training Script

Trains all four models (Seq MLP, Residue MLP, GAT, SoftBlobGAT),
evaluates them, generates all figures, and saves results.

Usage:
    python scripts/train.py                          # default config
    python scripts/train.py --config configs/my.yaml # custom config
    python scripts/train.py --no-esm                 # skip ESM-2 (faster)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EnzymeDataset
from src.data.features import compute_edge_dim, compute_feat_dim
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.models.gat import GATModel
from src.models.residue_mlp import ResidueMLP
from src.models.seq_mlp import SeqMLP
from src.models.soft_blob_gat import SoftBlobGAT
from src.training.losses import build_criterion
from src.training.trainer import (
    Trainer,
    get_embeddings,
    predict_plain,
    predict_pyg,
)
from src.visualization.plots import (
    plot_aa_composition,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_embeddings,
    plot_model_comparison_bar,
    plot_roc_curves,
    plot_training_curves,
)


def setup_logging(log_dir):
    """Configure logging to console + file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "training.log", mode="w"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path=None):
    """Load YAML config, with CLI overrides."""
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    cfg_path = Path(path) if path else default_path

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def build_tensorboard_writer(log_dir):
    """Create TensorBoard writer if available."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path(log_dir) / "tensorboard"))
        return writer
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description="CE7412 Enzyme Classification Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--no-esm", action="store_true", help="Skip ESM-2 embeddings")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--quick", action="store_true", help="Quick run (50 epochs, 250/class)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    if args.no_esm:
        cfg["features"]["use_esm2"] = False
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.quick:
        cfg["training"]["epochs"] = 50
        cfg["dataset"]["max_per_class"] = 250
        cfg["training"]["patience"] = 15

    # Setup
    set_seed(cfg["seed"])
    logger = setup_logging(cfg["paths"]["log_dir"])
    logger.info("=" * 70)
    logger.info("CE7412 Enzyme Classification — Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # Output directories
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tb_writer = build_tensorboard_writer(cfg["paths"]["log_dir"])

    # ══════════════════════════════════════════════════════════════════════
    # 1. DATA PREPARATION
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: Data Preparation")
    logger.info("=" * 50)

    dataset = EnzymeDataset(cfg).prepare()

    feat_dim = dataset.feat_dim
    edge_dim = dataset.edge_dim
    n_classes = dataset.n_classes

    logger.info(f"Feature dim: {feat_dim}, Edge dim: {edge_dim}, Classes: {n_classes}")

    # Dataset figures
    plot_class_distribution(
        dataset.train_graphs, dataset.val_graphs, dataset.test_graphs,
        n_classes, fig_dir
    )
    plot_aa_composition(dataset.train_graphs, n_classes, fig_dir)

    # Class weights for focal loss
    class_weights = dataset.get_class_weights().to(device)
    logger.info(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. MODEL TRAINING
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Model Training")
    logger.info("=" * 50)

    criterion = build_criterion(cfg["training"], class_weights)
    # Seq MLP is too weak for focal loss — use plain CE with label smoothing
    seq_criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"].get("label_smoothing", 0.1))
    all_results = {}
    all_histories = {}

    # ── Model 1: Sequence MLP ──────────────────────────────────────────
    logger.info("\n--- [1/4] Sequence MLP (AA composition only) ---")
    seq_cfg = cfg["models"]["seq_mlp"]
    seq_model = SeqMLP(
        n_classes=n_classes, in_dim=20,
        hidden=seq_cfg["hidden"], n_layers=seq_cfg["n_layers"],
        dropout=seq_cfg["dropout"]
    )
    trainer = Trainer(seq_model, cfg, device, is_pyg=False,
                      model_name="SeqMLP", tb_writer=tb_writer)
    hist, t_elapsed = trainer.train(
        dataset.seq_loaders["train"], dataset.seq_loaders["val"], seq_criterion
    )
    yt, yp, ypr = predict_plain(seq_model, dataset.seq_loaders["test"], device)
    metrics = compute_metrics(yt, yp, ypr, n_classes)
    all_results["Seq MLP"] = {
        "metrics": metrics, "y_true": yt, "y_pred": yp, "y_prob": ypr,
        "train_time": t_elapsed,
        "n_params": sum(p.numel() for p in seq_model.parameters()),
    }
    all_histories["Seq MLP"] = hist
    logger.info(f"  Seq MLP: acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")

    # ── Model 2: Residue MLP ──────────────────────────────────────────
    logger.info("\n--- [2/4] Residue MLP (mean-pooled features, no topology) ---")
    res_cfg = cfg["models"]["residue_mlp"]
    res_model = ResidueMLP(
        in_ch=feat_dim, hidden=res_cfg["hidden"], n_classes=n_classes,
        n_layers=res_cfg["n_layers"], dropout=res_cfg["dropout"]
    )
    trainer = Trainer(res_model, cfg, device, is_pyg=False,
                      model_name="ResidueMLP", tb_writer=tb_writer)
    hist, t_elapsed = trainer.train(
        dataset.res_loaders["train"], dataset.res_loaders["val"], criterion
    )
    yt, yp, ypr = predict_plain(res_model, dataset.res_loaders["test"], device)
    metrics = compute_metrics(yt, yp, ypr, n_classes)
    all_results["Residue MLP"] = {
        "metrics": metrics, "y_true": yt, "y_pred": yp, "y_prob": ypr,
        "train_time": t_elapsed,
        "n_params": sum(p.numel() for p in res_model.parameters()),
    }
    all_histories["Residue MLP"] = hist
    logger.info(f"  Residue MLP: acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")

    # ── Model 3: Graph model (GIN or GAT based on config) ──────────────
    graph_model = None
    graph_name = None

    if "gin" in cfg["models"]:
        from src.models.gin import GINModel
        logger.info("\n--- [3/4] GIN (Graph Isomorphism Network + edge features) ---")
        gin_cfg = cfg["models"]["gin"]
        graph_model = GINModel(
            in_ch=feat_dim, hidden=gin_cfg["hidden"], n_classes=n_classes,
            edge_dim=edge_dim, n_layers=gin_cfg["n_layers"],
            dropout=gin_cfg["dropout"],
        )
        graph_name = "GIN"
    elif "gat" in cfg["models"]:
        logger.info("\n--- [3/4] GAT (GATv2 + edge features + multi-pool) ---")
        gat_cfg = cfg["models"]["gat"]
        graph_model = GATModel(
            in_ch=feat_dim, hidden=gat_cfg["hidden"], n_classes=n_classes,
            edge_dim=edge_dim, n_layers=gat_cfg["n_layers"], heads=gat_cfg["heads"],
            dropout=gat_cfg["dropout"], attn_dropout=gat_cfg["attn_dropout"],
            pool_strategy=gat_cfg["pool_strategy"],
        )
        graph_name = "GAT"

    if graph_model is not None:
        trainer = Trainer(graph_model, cfg, device, is_pyg=True,
                          model_name=graph_name, tb_writer=tb_writer)
        hist, t_elapsed = trainer.train(
            dataset.train_loader, dataset.val_loader, criterion
        )
        yt, yp, ypr = predict_pyg(graph_model, dataset.test_loader, device)
        metrics = compute_metrics(yt, yp, ypr, n_classes)
        all_results[graph_name] = {
            "metrics": metrics, "y_true": yt, "y_pred": yp, "y_prob": ypr,
            "train_time": t_elapsed,
            "n_params": sum(p.numel() for p in graph_model.parameters()),
        }
        all_histories[graph_name] = hist
        logger.info(f"  {graph_name}: acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")

    # ── Model 4: Blob model (SoftBlobGIN or SoftBlobGAT based on config) ─
    blob_model = None
    blob_name = None

    if "soft_blob_gin" in cfg["models"]:
        from src.models.gin import SoftBlobGIN
        logger.info("\n--- [4/4] SoftBlobGIN (adaptive blob pooling + GIN backbone) ---")
        sb_cfg = cfg["models"]["soft_blob_gin"]
        blob_model = SoftBlobGIN(
            in_ch=feat_dim, hidden=sb_cfg["hidden"], n_classes=n_classes,
            edge_dim=edge_dim, n_blobs=sb_cfg["n_blobs"],
            n_layers=sb_cfg["n_layers"], dropout=sb_cfg["dropout"],
            tau_start=sb_cfg["tau_start"], tau_end=sb_cfg["tau_end"],
        )
        blob_name = "SoftBlobGIN"
    elif "soft_blob_gat" in cfg["models"]:
        logger.info("\n--- [4/4] SoftBlobGAT (adaptive blob pooling + cluster attention) ---")
        sb_cfg = cfg["models"]["soft_blob_gat"]
        blob_model = SoftBlobGAT(
            in_ch=feat_dim, hidden=sb_cfg["hidden"], n_classes=n_classes,
            edge_dim=edge_dim, n_blobs=sb_cfg["n_blobs"],
            n_layers=sb_cfg["n_layers"], heads=sb_cfg["heads"],
            dropout=sb_cfg["dropout"], attn_dropout=sb_cfg["attn_dropout"],
            blob_mlp_layers=sb_cfg["blob_mlp_layers"],
            cluster_attn_heads=sb_cfg["cluster_attn_heads"],
            tau_start=sb_cfg["tau_start"], tau_end=sb_cfg["tau_end"],
            pool_strategy=sb_cfg.get("pool_strategy", "multi"),
        )
        blob_name = "SoftBlobGAT"

    if blob_model is not None:
        trainer = Trainer(blob_model, cfg, device, is_pyg=True,
                          model_name=blob_name, tb_writer=tb_writer)
        hist, t_elapsed = trainer.train(
            dataset.train_loader, dataset.val_loader, criterion
        )
        yt, yp, ypr = predict_pyg(blob_model, dataset.test_loader, device)
        metrics = compute_metrics(yt, yp, ypr, n_classes)
        all_results[blob_name] = {
            "metrics": metrics, "y_true": yt, "y_pred": yp, "y_prob": ypr,
            "train_time": t_elapsed,
            "n_params": sum(p.numel() for p in blob_model.parameters()),
        }
        all_histories[blob_name] = hist
        logger.info(f"  {blob_name}: acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. EVALUATION & VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: Evaluation & Visualization")
    logger.info("=" * 50)

    # Summary table
    table = format_metrics_table(all_results, list(all_results.keys()), n_classes)
    logger.info("\n" + table)

    # Save metrics to CSV
    rows = []
    for mname, res in all_results.items():
        m = res["metrics"]
        rows.append({
            "Model": mname,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Macro AUROC": f"{m['macro_auroc']:.4f}",
            "Params": res["n_params"],
            "Train Time (s)": f"{res['train_time']:.0f}",
        })
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)

    # Save per-class metrics
    for mname, res in all_results.items():
        m = res["metrics"]
        ec_names = ["Oxidoreductase", "Transferase", "Hydrolase",
                    "Lyase", "Isomerase", "Ligase", "Translocase"]
        per_class = pd.DataFrame({
            "Class": ec_names,
            "Precision": m["per_class_precision"],
            "Recall": m["per_class_recall"],
            "F1": m["per_class_f1"],
            "Support": m["support"],
        })
        safe_name = mname.replace(" ", "_").lower()
        per_class.to_csv(out_dir / f"per_class_metrics_{safe_name}.csv", index=False)

    # ── Generate all figures ───────────────────────────────────────────
    logger.info("Generating figures...")

    # Training curves
    plot_training_curves(all_histories, fig_dir)

    # Confusion matrices
    plot_confusion_matrices(all_results, n_classes, fig_dir)

    # ROC curves
    plot_roc_curves(all_results, n_classes, fig_dir, highlight="SoftBlobGAT", baseline="GAT")

    # Model comparison bar
    plot_model_comparison_bar(all_results, fig_dir)

    # Embeddings (only for graph models)
    logger.info("Extracting embeddings for visualization...")
    all_emb = {}
    graph_models = {}
    if graph_model is not None:
        graph_models[graph_name] = graph_model
    if blob_model is not None:
        graph_models[blob_name] = blob_model
    for mname, model in graph_models.items():
        emb, lab = get_embeddings(model, dataset.test_loader, device)
        all_emb[mname] = (emb, lab)
    if all_emb:
        plot_embeddings(all_emb, n_classes, fig_dir)

    # ROC curves — use the best graph models
    model_names = list(all_results.keys())
    if len(model_names) >= 2:
        plot_roc_curves(all_results, n_classes, fig_dir,
                        highlight=model_names[-1], baseline=model_names[-2])

    # ══════════════════════════════════════════════════════════════════════
    # 4. DONE
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 50)
    logger.info("COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Figures saved to: {fig_dir}")
    logger.info(f"Results saved to: {out_dir}")
    if tb_writer:
        logger.info(f"TensorBoard: tensorboard --logdir={cfg['paths']['log_dir']}/tensorboard")
        tb_writer.close()

    # Final summary
    logger.info("\nFinal Results:")
    logger.info(table)


if __name__ == "__main__":
    main()