#!/usr/bin/env python3
"""
ProteinShake Benchmark — ProNet (dilated 1D CNN) baseline across all tasks.

ProNet operates on one-hot amino acid sequences alone (no 3-D structure).
Compares directly against SoftBlobGIN (benchmark.py) to isolate how much of
GNN performance is attributable to structure vs. raw sequence information.

Architecture: faithful adaptation of ProNet/src/pronet.py using global average
pooling (replacing original fixed-length flatten) for ProteinShake's
variable-length proteins. Training follows ProNet/src/2_train.py:
AdamW (lr=1e-3), cosine warmup scheduler, batch_size=64.

Usage:
    python scripts/benchmark_pronet.py --config configs/benchmark_pronet.yaml
    python scripts/benchmark_pronet.py --config configs/benchmark_pronet.yaml --tasks EnzymeClassTask
    python scripts/benchmark_pronet.py --config configs/benchmark_pronet.yaml --quick
"""

import argparse
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.benchmark_dataset import BenchmarkDataset, TASK_TYPE_MAP
from src.data.features import compute_feat_dim, compute_edge_dim
from src.models.pronet import (
    ProNetClassifier, ProNetMultiLabel, ProNetRegressor,
    ProNetNodeClassifier, ProNetPPI, ProNetSiamese,
)
from src.training.trainer import Trainer
from src.training.losses import build_criterion

from proteinshake.tasks import (
    EnzymeClassTask, GeneOntologyTask, ProteinFamilyTask,
    LigandAffinityTask, BindingSiteDetectionTask,
    ProteinProteinInterfaceTask, StructuralClassTask,
    StructureSimilarityTask, StructureSearchTask,
)

TASK_CLASSES = {
    "EnzymeClassTask": EnzymeClassTask,
    "GeneOntologyTask": GeneOntologyTask,
    "ProteinFamilyTask": ProteinFamilyTask,
    "StructuralClassTask": StructuralClassTask,
    "LigandAffinityTask": LigandAffinityTask,
    "BindingSiteDetectionTask": BindingSiteDetectionTask,
    "ProteinProteinInterfaceTask": ProteinProteinInterfaceTask,
    "StructureSimilarityTask": StructureSimilarityTask,
    "StructureSearchTask": StructureSearchTask,
}

# Primary evaluation metric per task — identical to benchmark.py for comparison.
TASK_METRICS = {
    "EnzymeClassTask": "accuracy",   # multiclass (7 EC classes, single label) — official ProteinShake metric
    "GeneOntologyTask": "fmax",
    "ProteinFamilyTask": "accuracy",
    "StructuralClassTask": "accuracy",
    "LigandAffinityTask": "r2",
    "BindingSiteDetectionTask": "mcc",
    "ProteinProteinInterfaceTask": "auroc",
    "StructureSimilarityTask": "spearman",
    "StructureSearchTask": "precision_at_k",
}


# ---------------------------------------------------------------------------
# Logging & reproducibility
# ---------------------------------------------------------------------------

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "benchmark_pronet.log", mode="w"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_f1_max(y_true, y_prob):
    """Maximum F1 over all classification thresholds (Fmax metric)."""
    from sklearn.metrics import precision_recall_fscore_support
    thresholds = np.linspace(0, 1, 101)
    f1s = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        _, _, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        f1s.append(f)
    return float(max(f1s))


def compute_auprc_micro(y_true, y_prob):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_prob, average="micro"))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(task_name, task_type, pronet_cfg):
    """Return the appropriate ProNet variant for this task."""
    max_seq_len = pronet_cfg.get("max_seq_len", 1024)
    L            = pronet_cfg.get("L", 64)
    input_ch     = pronet_cfg.get("input_channels", 20)

    common = dict(max_seq_len=max_seq_len, L=L, input_channels=input_ch)

    if task_type == "graph_multiclass":
        n_classes = pronet_cfg["n_classes"]
        return ProNetClassifier(n_classes=n_classes, **common)
    elif task_type == "graph_multilabel":
        n_classes = pronet_cfg["n_classes"]
        return ProNetMultiLabel(n_classes=n_classes, **common)
    elif task_type == "graph_regression":
        return ProNetRegressor(**common)
    elif task_name == "ProteinProteinInterfaceTask":
        return ProNetPPI(**common)
    elif task_type == "node_binary":
        return ProNetNodeClassifier(**common)
    elif "pair" in str(task_type) or task_name in ("StructureSimilarityTask", "StructureSearchTask"):
        return ProNetSiamese(**common)
    else:
        n_classes = pronet_cfg.get("n_classes", 2)
        return ProNetClassifier(n_classes=n_classes, **common)


def build_loss(task_type):
    if task_type == "graph_multiclass":
        return nn.CrossEntropyLoss()
    elif task_type == "graph_multilabel":
        return nn.BCEWithLogitsLoss()
    elif task_type in ("graph_regression", "pair_regression"):
        return nn.MSELoss()
    elif task_type in ("node_binary", "pair_retrieval"):
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Evaluation — mirrors benchmark_gearnet.py for consistent metric reporting
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_task(model, loader, task_type, device, task_obj=None):
    """Evaluate model and return metrics dict."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    if task_type == "graph_multiclass":
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            all_true.extend(batch.y.long().cpu().numpy())
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())
        y_true, y_pred = np.array(all_true), np.array(all_pred)
        acc = float((y_true == y_pred).mean())
        metrics = {"accuracy": acc}
        if task_obj is not None:
            try:
                official = task_obj.evaluate(task_obj.test_targets, y_pred)
                metrics.update({k: float(v) for k, v in official.items()})
            except Exception:
                pass
        return metrics

    elif task_type == "graph_multilabel":
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits)
            all_true.append(batch.y.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
        y_true = np.vstack(all_true)
        y_prob = np.vstack(all_prob)
        y_pred = (y_prob > 0.5).astype(int)
        metrics = {
            "accuracy": float((y_true == y_pred).all(axis=1).mean()),
            "fmax":     compute_f1_max(y_true, y_prob),
            "auprc":    compute_auprc_micro(y_true, y_prob),
        }
        if task_obj is not None:
            try:
                official = task_obj.evaluate(task_obj.test_targets, y_pred)
                for k, v in official.items():
                    if k not in metrics:
                        metrics[k] = float(v)
            except Exception:
                pass
        return metrics

    elif task_type == "graph_regression" or "pair" in str(task_type):
        for batch in loader:
            batch = batch.to(device)
            if hasattr(batch, "b1"):
                pred = model(batch.b1.to(device), batch.b2.to(device))
                all_true.extend(batch.y.cpu().numpy())
            else:
                pred = model(batch)
                all_true.extend(batch.y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
        y_true, y_pred = np.array(all_true), np.array(all_pred)
        if task_obj is not None:
            try:
                official = task_obj.evaluate(task_obj.test_targets, y_pred)
                return {k: float(v) for k, v in official.items()}
            except Exception:
                pass
        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr
        return {
            "r2":       float(r2_score(y_true, y_pred)),
            "mse":      float(mean_squared_error(y_true, y_pred)),
            "spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0,
        }

    elif task_type == "node_binary":
        for batch in loader:
            batch = batch.to(device)
            if hasattr(batch, "b1"):
                logits = model(batch.b1.to(device), batch.b2.to(device))
                y = batch.y
            else:
                logits = model(batch)
                y = batch.y
            probs = torch.sigmoid(logits)
            all_true.extend(y.cpu().numpy().flatten())
            all_pred.extend((probs > 0.5).long().cpu().numpy().flatten())
            all_prob.extend(probs.cpu().numpy().flatten())
        y_true, y_pred = np.array(all_true), np.array(all_pred)
        y_prob = np.array(all_prob)
        from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc":      float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        }
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = 0.0
        return metrics

    return {}


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate_task(task_name, task_class, cfg, pronet_cfg, device, logger, quick=False):
    """Full pipeline for one task: load data → train ProNet → evaluate."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TASK: {task_name} (ProNet baseline)")
    logger.info(f"{'='*60}")

    task_type = TASK_TYPE_MAP.get(task_name, "graph_multiclass")
    t_start = time.time()

    try:
        import requests
        try:
            dataset = BenchmarkDataset(task_class, task_name, cfg, dummy=quick)
            dataset.prepare()
        except requests.exceptions.RequestException as e:
            logger.warning(f"  Skipping {task_name}: download failed: {e}")
            return {"task": task_name, "status": "skipped", "reason": "download_failed",
                    "task_type": task_type, "time_s": 0}

        if dataset.train_loader is None:
            return {"task": task_name, "status": "failed", "reason": "no training data",
                    "task_type": task_type, "time_s": 0}

        n_classes = dataset.n_classes
        pronet_cfg_copy = dict(pronet_cfg)
        pronet_cfg_copy["n_classes"] = n_classes

        model = build_model(task_name, task_type, pronet_cfg_copy)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"  Model: ProNet ({task_type}), "
            f"input_channels={pronet_cfg.get('input_channels', 20)}, "
            f"L={pronet_cfg.get('L', 64)}, n_classes={n_classes}, params={n_params:,}"
        )

        criterion = build_loss(task_type)

        cfg_copy = dict(cfg)
        cfg_copy["training"] = dict(cfg["training"])
        for k, v in cfg.get("task_overrides", {}).get(task_name, {}).items():
            cfg_copy["training"][k] = v
        if quick:
            cfg_copy["training"]["epochs"] = 1
            cfg_copy["training"]["patience"] = 1

        trainer = Trainer(model, cfg_copy, device, is_pyg=True,
                          model_name=f"ProNet_{task_name}")
        _, t_elapsed = trainer.train(dataset.train_loader, dataset.val_loader, criterion)

        metrics = evaluate_task(model, dataset.test_loader, task_type, device, dataset.task)
        total_time = time.time() - t_start

        result = {
            "task":         task_name,
            "status":       "completed",
            "task_type":    task_type,
            "n_classes":    n_classes,
            "n_params":     n_params,
            "train_time_s": t_elapsed,
            "total_time_s": total_time,
            **metrics,
        }

        primary = TASK_METRICS.get(task_name, "accuracy")
        logger.info(f"  RESULT: {primary}={metrics.get(primary, 'N/A')}, time={total_time:.0f}s")
        return result

    except Exception as e:
        total_time = time.time() - t_start
        logger.error(f"  FAILED: {task_name}: {e}")
        logger.error(traceback.format_exc())
        return {"task": task_name, "status": "failed", "reason": str(e),
                "task_type": task_type, "time_s": total_time}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ProteinShake ProNet Benchmark")
    parser.add_argument("--config", type=str, default="configs/benchmark_pronet.yaml")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--dummy", "--quick", action="store_true", dest="quick")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    set_seed(cfg["seed"])
    logger = setup_logging(cfg["paths"]["log_dir"])

    logger.info("=" * 70)
    logger.info("ProteinShake — ProNet (dilated 1D CNN) Benchmark")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"ProNet config: {cfg['pronet']}")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    tasks_to_run = args.tasks or cfg.get("benchmark_tasks", list(TASK_CLASSES.keys()))
    pronet_cfg   = cfg["pronet"]
    all_results  = []

    # Resume logic — skip tasks that already completed successfully.
    res_path = out_dir / "benchmark_pronet_results.csv"
    completed_tasks = set()
    if res_path.exists():
        try:
            existing_df = pd.read_csv(res_path)
            all_results = existing_df.to_dict("records")
            completed_tasks = set(
                existing_df[existing_df["status"] == "completed"]["task"].tolist()
            )
            logger.info(f"Resuming: {len(completed_tasks)} completed tasks found")
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}")

    for task_name in tasks_to_run:
        if task_name in completed_tasks:
            logger.info(f"Skipping {task_name} (already completed)")
            continue
        if task_name not in TASK_CLASSES:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue

        set_seed(cfg["seed"])
        result = train_and_evaluate_task(
            task_name, TASK_CLASSES[task_name], cfg, pronet_cfg, device, logger,
            quick=args.quick,
        )
        all_results.append(result)

        df = pd.DataFrame(all_results)
        df.to_csv(res_path, index=False)
        logger.info(f"  Intermediate results saved to {res_path}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PRONET BENCHMARK COMPLETE")
    logger.info("=" * 70)

    df = pd.DataFrame(all_results)
    df.to_csv(res_path, index=False)

    summary_cols = ["task", "status", "task_type", "n_classes"]
    for m in ["fmax", "auprc", "accuracy", "r2", "mcc", "auroc", "spearman"]:
        if m in df.columns:
            summary_cols.append(m)
    summary_cols.extend(["n_params", "total_time_s"])
    available = [c for c in summary_cols if c in df.columns]
    logger.info(f"\n{df[available].to_string(index=False)}")
    logger.info(f"\nResults saved to: {res_path}")


if __name__ == "__main__":
    main()
