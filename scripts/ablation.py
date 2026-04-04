#!/usr/bin/env python3
"""
CE7412 Enzyme Classification — Ablation Studies

Runs systematic ablations to understand the contribution of each component:
  1. Contact radius ε: {4, 8, 12} Å
  2. Blob count K: {3, 5, 8, 12}
  3. Feature sets: onehot → +physico → +esm2 → full

Usage:
    python scripts/ablation.py
    python scripts/ablation.py --config configs/default.yaml
    python scripts/ablation.py --ablation eps       # run only eps ablation
    python scripts/ablation.py --ablation blobs     # run only blob ablation
    python scripts/ablation.py --ablation features  # run only feature ablation
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EnzymeDataset
from src.data.features import compute_edge_dim, compute_feat_dim
from src.evaluation.metrics import compute_metrics
from src.training.losses import build_criterion
from src.training.trainer import Trainer, predict_pyg
from src.visualization.plots import (
    plot_ablation_blobs,
    plot_ablation_eps,
    plot_ablation_features,
)


def _make_graph_model(cfg, feat_dim, n_classes, edge_dim):
    """Create the right graph model based on config."""
    if "gin" in cfg["models"]:
        from src.models.gin import GINModel
        c = cfg["models"]["gin"]
        return GINModel(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                        edge_dim=edge_dim, n_layers=c["n_layers"], dropout=c["dropout"]), "GIN"
    else:
        from src.models.gat import GATModel
        c = cfg["models"]["gat"]
        return GATModel(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                        edge_dim=edge_dim, n_layers=c["n_layers"], heads=c["heads"],
                        dropout=c["dropout"], attn_dropout=c["attn_dropout"],
                        pool_strategy=c["pool_strategy"]), "GAT"


def _make_blob_model(cfg, feat_dim, n_classes, edge_dim, n_blobs_override=None):
    """Create the right blob model based on config."""
    if "soft_blob_gin" in cfg["models"]:
        from src.models.gin import SoftBlobGIN
        c = cfg["models"]["soft_blob_gin"]
        K = n_blobs_override or c["n_blobs"]
        return SoftBlobGIN(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                           edge_dim=edge_dim, n_blobs=K, n_layers=c["n_layers"],
                           dropout=c["dropout"], tau_start=c["tau_start"],
                           tau_end=c["tau_end"]), "SoftBlobGIN"
    else:
        from src.models.soft_blob_gat import SoftBlobGAT
        c = cfg["models"]["soft_blob_gat"]
        K = n_blobs_override or c["n_blobs"]
        return SoftBlobGAT(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                           edge_dim=edge_dim, n_blobs=K, n_layers=c["n_layers"],
                           heads=c["heads"], dropout=c["dropout"],
                           attn_dropout=c["attn_dropout"],
                           blob_mlp_layers=c["blob_mlp_layers"],
                           cluster_attn_heads=c["cluster_attn_heads"],
                           tau_start=c["tau_start"], tau_end=c["tau_end"]), "SoftBlobGAT"


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "ablation.log", mode="w"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "eps", "blobs", "features"])
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(cfg["paths"]["log_dir"])

    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reduce epochs for ablations
    cfg["training"]["epochs"] = min(cfg["training"]["epochs"], 150)
    cfg["training"]["patience"] = min(cfg["training"]["patience"], 25)

    # Load base dataset
    logger.info("Loading dataset...")
    dataset = EnzymeDataset(cfg).prepare()
    class_weights = dataset.get_class_weights().to(device)
    criterion = build_criterion(cfg["training"], class_weights)

    # ══════════════════════════════════════════════════════════════════════
    # Ablation 1: Contact radius epsilon
    # ══════════════════════════════════════════════════════════════════════
    if args.ablation in ("all", "eps"):
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION 1: Contact radius epsilon")
        logger.info("=" * 50)

        eps_values = cfg["ablations"]["eps_values"]
        eps_results = []

        for eps in eps_values:
            logger.info(f"\n  eps = {eps} Å")
            set_seed(cfg["seed"])

            train_ld, val_ld, test_ld = dataset.build_graphs_at_eps(eps)

            model, mtype = _make_graph_model(cfg, dataset.feat_dim, dataset.n_classes, dataset.edge_dim)
            trainer = Trainer(model, cfg, device, is_pyg=True,
                              model_name=f"{mtype}_eps{eps}")
            _, elapsed = trainer.train(train_ld, val_ld, criterion)

            yt, yp, ypr = predict_pyg(model, test_ld, device)
            m = compute_metrics(yt, yp, ypr, dataset.n_classes)

            eps_results.append({
                "eps": eps,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "macro_auroc": m["macro_auroc"],
                "time_s": elapsed,
            })
            logger.info(f"    acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

        df_eps = pd.DataFrame(eps_results)
        df_eps.to_csv(out_dir / "ablation_eps.csv", index=False)
        plot_ablation_eps(df_eps, fig_dir)
        logger.info(f"\n{df_eps.to_string(index=False)}")

    # ══════════════════════════════════════════════════════════════════════
    # Ablation 2: Blob count K
    # ══════════════════════════════════════════════════════════════════════
    if args.ablation in ("all", "blobs"):
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION 2: Blob count K")
        logger.info("=" * 50)

        blob_counts = cfg["ablations"]["blob_counts"]
        blob_results = []

        for K in blob_counts:
            logger.info(f"\n  K = {K} blobs")
            set_seed(cfg["seed"])

            model, mtype = _make_blob_model(cfg, dataset.feat_dim, dataset.n_classes,
                                            dataset.edge_dim, n_blobs_override=K)
            trainer = Trainer(model, cfg, device, is_pyg=True,
                              model_name=f"{mtype}_K{K}")
            _, elapsed = trainer.train(
                dataset.train_loader, dataset.val_loader, criterion
            )

            yt, yp, ypr = predict_pyg(model, dataset.test_loader, device)
            m = compute_metrics(yt, yp, ypr, dataset.n_classes)

            blob_results.append({
                "n_blobs": K,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "macro_auroc": m["macro_auroc"],
                "time_s": elapsed,
            })
            logger.info(f"    acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

        df_blobs = pd.DataFrame(blob_results)
        df_blobs.to_csv(out_dir / "ablation_blobs.csv", index=False)
        plot_ablation_blobs(df_blobs, fig_dir)
        logger.info(f"\n{df_blobs.to_string(index=False)}")

    # ══════════════════════════════════════════════════════════════════════
    # Ablation 3: Feature sets
    # ══════════════════════════════════════════════════════════════════════
    if args.ablation in ("all", "features"):
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION 3: Feature sets")
        logger.info("=" * 50)

        feature_configs = {
            "onehot_only": {
                "use_onehot": True, "use_physicochemical": False,
                "use_sasa": False, "use_esm2": False,
                "use_degree": False, "use_positional": False,
                "use_edge_distance": False, "use_edge_seqsep": False,
            },
            "onehot+physico": {
                "use_onehot": True, "use_physicochemical": True,
                "use_sasa": True, "use_esm2": False,
                "use_degree": True, "use_positional": True,
                "use_edge_distance": True, "use_edge_seqsep": True,
            },
            "onehot+physico+esm2": {
                "use_onehot": True, "use_physicochemical": True,
                "use_sasa": True, "use_esm2": True,
                "use_degree": True, "use_positional": True,
                "use_edge_distance": True, "use_edge_seqsep": True,
            },
            "full": cfg["features"],  # whatever default is
        }

        feat_results = []
        for fname, feat_cfg in feature_configs.items():
            logger.info(f"\n  Feature set: {fname}")
            set_seed(cfg["seed"])

            # Rebuild dataset with modified features
            cfg_copy = {**cfg}
            cfg_copy["features"] = {**cfg["features"], **feat_cfg}
            ds = EnzymeDataset(cfg_copy).prepare()

            model, mtype = _make_graph_model(cfg_copy, ds.feat_dim, ds.n_classes, ds.edge_dim)
            trainer = Trainer(model, cfg_copy, device, is_pyg=True,
                              model_name=f"{mtype}_{fname}")
            _, elapsed = trainer.train(ds.train_loader, ds.val_loader, criterion)

            yt, yp, ypr = predict_pyg(model, ds.test_loader, device)
            m = compute_metrics(yt, yp, ypr, ds.n_classes)

            feat_results.append({
                "feature_set": fname,
                "feat_dim": ds.feat_dim,
                "edge_dim": ds.edge_dim,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "macro_auroc": m["macro_auroc"],
                "time_s": elapsed,
            })
            logger.info(f"    dim={ds.feat_dim}  acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

        df_feat = pd.DataFrame(feat_results)
        df_feat.to_csv(out_dir / "ablation_features.csv", index=False)
        plot_ablation_features(df_feat, fig_dir)
        logger.info(f"\n{df_feat.to_string(index=False)}")

    logger.info("\nAblation studies complete.")


if __name__ == "__main__":
    main()