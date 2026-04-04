#!/usr/bin/env python3
"""
CE7412 Enzyme Classification — Seed Ensemble

Trains the best model (SoftBlobGAT) across multiple random seeds,
then averages softmax outputs for the final prediction.

Usage:
    python scripts/ensemble.py
    python scripts/ensemble.py --model gat --n-seeds 5
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
from src.evaluation.metrics import compute_metrics
from src.training.losses import build_criterion
from src.training.trainer import Trainer, predict_pyg


def _make_model(model_type, cfg, feat_dim, n_classes, edge_dim):
    """Create model based on type and config."""
    if model_type == "soft_blob_gin":
        from src.models.gin import SoftBlobGIN
        c = cfg["models"]["soft_blob_gin"]
        return SoftBlobGIN(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                           edge_dim=edge_dim, n_blobs=c["n_blobs"], n_layers=c["n_layers"],
                           dropout=c["dropout"], tau_start=c["tau_start"],
                           tau_end=c["tau_end"])
    elif model_type == "gin":
        from src.models.gin import GINModel
        c = cfg["models"]["gin"]
        return GINModel(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                        edge_dim=edge_dim, n_layers=c["n_layers"], dropout=c["dropout"])
    elif model_type == "soft_blob_gat":
        from src.models.soft_blob_gat import SoftBlobGAT
        c = cfg["models"]["soft_blob_gat"]
        return SoftBlobGAT(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                           edge_dim=edge_dim, n_blobs=c["n_blobs"], n_layers=c["n_layers"],
                           heads=c["heads"], dropout=c["dropout"],
                           attn_dropout=c["attn_dropout"],
                           blob_mlp_layers=c["blob_mlp_layers"],
                           cluster_attn_heads=c["cluster_attn_heads"],
                           tau_start=c["tau_start"], tau_end=c["tau_end"],
                           pool_strategy=c.get("pool_strategy", "multi"))
    else:  # gat
        from src.models.gat import GATModel
        c = cfg["models"]["gat"]
        return GATModel(in_ch=feat_dim, hidden=c["hidden"], n_classes=n_classes,
                        edge_dim=edge_dim, n_layers=c["n_layers"], heads=c["heads"],
                        dropout=c["dropout"], attn_dropout=c["attn_dropout"],
                        pool_strategy=c.get("pool_strategy", "multi"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="auto",
                        choices=["auto", "gin", "soft_blob_gin", "gat", "soft_blob_gat"],
                        help="Model to ensemble (auto picks best from config)")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or cfg["ensemble"]["seeds"][:args.n_seeds]
    logger.info(f"Ensemble: {len(seeds)} seeds = {seeds}")

    # Determine which model to use
    model_type = args.model
    if model_type == "auto":
        if "soft_blob_gin" in cfg["models"]:
            model_type = "soft_blob_gin"
        elif "soft_blob_gat" in cfg["models"]:
            model_type = "soft_blob_gat"
        elif "gin" in cfg["models"]:
            model_type = "gin"
        else:
            model_type = "gat"

    logger.info(f"Model: {model_type}, Device: {device}")

    # ── Load dataset with first seed ───────────────────────────────────
    set_seed(seeds[0])
    dataset = EnzymeDataset(cfg).prepare()
    class_weights = dataset.get_class_weights().to(device)
    criterion = build_criterion(cfg["training"], class_weights)

    logger.info(f"Ensemble model: {model_type}")

    # ── Train one model per seed ───────────────────────────────────────
    all_probs = []
    all_true = None
    individual_metrics = []

    for i, seed in enumerate(seeds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Seed {i+1}/{len(seeds)}: {seed}")
        logger.info(f"{'='*50}")
        set_seed(seed)

        model = _make_model(model_type, cfg, dataset.feat_dim,
                            dataset.n_classes, dataset.edge_dim)
        name = f"{model_type}_seed{seed}"

        trainer = Trainer(model, cfg, device, is_pyg=True, model_name=name)
        trainer.train(dataset.train_loader, dataset.val_loader, criterion)

        yt, yp, ypr = predict_pyg(model, dataset.test_loader, device)
        all_probs.append(ypr)
        all_true = yt

        m = compute_metrics(yt, yp, ypr, dataset.n_classes)
        individual_metrics.append({
            "seed": seed,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_auroc": m["macro_auroc"],
        })
        logger.info(f"  Seed {seed}: acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

    # ── Ensemble prediction (average softmax) ──────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info("ENSEMBLE RESULTS")
    logger.info(f"{'='*50}")

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_pred = np.argmax(ensemble_probs, axis=1)
    ensemble_m = compute_metrics(all_true, ensemble_pred, ensemble_probs, dataset.n_classes)

    logger.info(f"  Ensemble: acc={ensemble_m['accuracy']:.4f}  "
                f"F1={ensemble_m['macro_f1']:.4f}  "
                f"AUROC={ensemble_m['macro_auroc']:.4f}")

    # Compare with individual models
    df_ind = pd.DataFrame(individual_metrics)
    logger.info(f"\nIndividual models:")
    logger.info(df_ind.to_string(index=False))
    logger.info(f"\nMean individual: acc={df_ind['accuracy'].mean():.4f}  "
                f"F1={df_ind['macro_f1'].mean():.4f}")
    logger.info(f"Ensemble:        acc={ensemble_m['accuracy']:.4f}  "
                f"F1={ensemble_m['macro_f1']:.4f}")
    logger.info(f"Ensemble gain:   acc=+{ensemble_m['accuracy'] - df_ind['accuracy'].mean():.4f}  "
                f"F1=+{ensemble_m['macro_f1'] - df_ind['macro_f1'].mean():.4f}")

    # Save
    df_ind.to_csv(out_dir / "ensemble_individual.csv", index=False)
    ensemble_row = {
        "method": "ensemble",
        "n_seeds": len(seeds),
        "accuracy": ensemble_m["accuracy"],
        "macro_f1": ensemble_m["macro_f1"],
        "macro_auroc": ensemble_m["macro_auroc"],
    }
    pd.DataFrame([ensemble_row]).to_csv(out_dir / "ensemble_result.csv", index=False)
    logger.info(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()