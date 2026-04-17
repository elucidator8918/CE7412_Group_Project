#!/usr/bin/env python3
"""
ProteinShake Benchmark — GIN across all tasks.

Uses the SAME GIN configuration as full_power.yaml (hidden=256, n_layers=4,
dropout=0.3, ESM-2 650M 1280-dim, full feature set).

Usage:
    python scripts/benchmark.py --config configs/benchmark.yaml
    python scripts/benchmark.py --config configs/benchmark.yaml --tasks EnzymeClassTask StructuralClassTask
    python scripts/benchmark.py --config configs/benchmark.yaml --quick
"""

import argparse
import json
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
from src.data.features import compute_edge_dim, compute_feat_dim
from src.models.gin import (
    GINModel, GINRegressor, GINMultiLabel, GINNodeClassifier,
    SoftBlobGIN, SoftBlobGINRegressor, SoftBlobGINMultiLabel, SoftBlobGINNodeClassifier,
    SoftBlobGINSiamese, SoftBlobGINPPI
)
from src.training.trainer import Trainer, predict_pyg
from src.training.losses import build_criterion

# Task imports
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

TASK_METRICS = {
    "EnzymeClassTask": "accuracy",
    "GeneOntologyTask": "fmax",
    "ProteinFamilyTask": "accuracy",
    "StructuralClassTask": "accuracy",
    "LigandAffinityTask": "r2",
    "BindingSiteDetectionTask": "mcc",
    "ProteinProteinInterfaceTask": "auroc",
    "StructureSimilarityTask": "spearman",
    "StructureSearchTask": "precision_at_k",
}


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "benchmark.log", mode="w"),
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


def build_model(task_name, task_type, feat_dim, edge_dim, n_classes, gin_cfg, model_type="gin"):
    """Build the appropriate model variant for a given task type."""
    hidden = gin_cfg["hidden"]
    n_layers = gin_cfg["n_layers"]
    dropout = gin_cfg["dropout"]
    
    # SoftBlob specific
    n_blobs = gin_cfg.get("n_blobs", 8)

    if model_type == "soft_blob_gin":
        if task_type == "graph_multiclass":
            return SoftBlobGIN(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                               edge_dim=edge_dim, n_layers=n_layers, dropout=dropout, n_blobs=n_blobs)
        elif task_type == "graph_multilabel":
            return SoftBlobGINMultiLabel(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                                         edge_dim=edge_dim, n_layers=n_layers, dropout=dropout, n_blobs=n_blobs)
        elif task_type == "graph_regression":
            return SoftBlobGINRegressor(in_ch=feat_dim, hidden=hidden,
                                        edge_dim=edge_dim, n_layers=n_layers, dropout=dropout, n_blobs=n_blobs)
        elif task_name == "ProteinProteinInterfaceTask":
            node_encoder = SoftBlobGINNodeClassifier(in_ch=feat_dim, hidden=hidden,
                                                     edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)
            graph_encoder = SoftBlobGIN(in_ch=feat_dim, hidden=hidden, n_classes=1,
                                        edge_dim=edge_dim, n_layers=n_layers, dropout=dropout, n_blobs=n_blobs)
            return SoftBlobGINPPI(node_encoder=node_encoder, graph_encoder=graph_encoder)
        elif task_type == "node_binary":
            return SoftBlobGINNodeClassifier(in_ch=feat_dim, hidden=hidden,
                                             edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)
        elif "pair" in str(task_type) or task_name in ["StructureSimilarityTask", "StructureSearchTask"]:
            encoder = SoftBlobGIN(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                                  edge_dim=edge_dim, n_layers=n_layers, dropout=dropout, n_blobs=n_blobs)
            return SoftBlobGINSiamese(encoder=encoder)
        else:
            return SoftBlobGIN(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                               edge_dim=edge_dim, n_layers=n_layers, dropout=dropout, n_blobs=n_blobs)
    else:
        # Standard GIN
        if task_type == "graph_multiclass":
            return GINModel(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                            edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)
        elif task_type == "graph_multilabel":
            return GINMultiLabel(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                                 edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)
        elif task_type == "graph_regression":
            return GINRegressor(in_ch=feat_dim, hidden=hidden,
                                edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)
        elif task_type == "node_binary":
            return GINNodeClassifier(in_ch=feat_dim, hidden=hidden,
                                      edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)
        else:
            return GINModel(in_ch=feat_dim, hidden=hidden, n_classes=n_classes,
                            edge_dim=edge_dim, n_layers=n_layers, dropout=dropout)


def build_loss(task_type, n_classes, cfg, class_weights=None):
    """Build appropriate loss for each task type."""
    if task_type == "graph_multiclass":
        return build_criterion(cfg["training"], class_weights)
    elif task_type == "graph_multilabel":
        return nn.BCEWithLogitsLoss()
    elif task_type in ("graph_regression", "pair_regression"):
        return nn.MSELoss()
    elif task_type in ("node_binary", "pair_retrieval"):
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate_task(model, loader, task_type, device, task_obj=None):
    """Evaluate model and return metrics dict."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    if task_type == "graph_multiclass":
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_true.extend(batch.y.long().cpu().numpy())
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        y_prob = np.vstack(all_prob)
        acc = (y_true == y_pred).mean()

        # Try task.evaluate for official metrics
        metrics = {"accuracy": float(acc)}
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
            probs = torch.sigmoid(logits)
            all_true.append(batch.y.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
        y_true = np.vstack(all_true)
        y_prob = np.vstack(all_prob)
        y_pred = (y_prob > 0.5).astype(int)

        metrics = {}
        if task_obj is not None:
            try:
                official = task_obj.evaluate(task_obj.test_targets, y_pred)
                metrics.update({k: float(v) for k, v in official.items()})
            except Exception as e:
                # Try with probabilities
                try:
                    official = task_obj.evaluate(task_obj.test_targets, y_prob)
                    metrics.update({k: float(v) for k, v in official.items()})
                except Exception:
                    pass
        if not metrics:
            from sklearn.metrics import f1_score
            metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return metrics

    elif task_type == "graph_regression" or "pair" in str(task_type):
        for batch in loader:
            batch = batch.to(device)
            # Handle PairBatch or Data
            if hasattr(batch, 'b1'):
                pred = model(batch.b1, batch.b2)
            else:
                pred = model(batch)
                
            all_true.extend(batch.y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        
        # Check for matching shapes (for evaluate call)
        if task_obj is not None:
             try:
                 # Flatten targets if they are provided as such, or let task handle it
                 official = task_obj.evaluate(task_obj.test_targets, y_pred)
                 metrics = {k: float(v) for k, v in official.items()}
                 return metrics
             except Exception:
                 pass

        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import pearsonr, spearmanr
        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0,
            "spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0,
        }
        return metrics

    elif task_type == "node_binary":
        for batch in loader:
            batch = batch.to(device)
            if hasattr(batch, 'b1'):
                logits = model(batch.b1, batch.b2)
            else:
                logits = model(batch)
            
            y = batch.y
            if isinstance(logits, list):
                # Handle list of outputs (e.g. PPI)
                for l_i, y_i in zip(logits, y):
                    probs = torch.sigmoid(l_i)
                    all_true.extend(y_i.cpu().numpy().flatten())
                    all_pred.extend((probs > 0.5).long().cpu().numpy().flatten())
                    all_prob.extend(probs.cpu().numpy().flatten())
            else:
                probs = torch.sigmoid(logits)
                all_true.extend(y.cpu().numpy().flatten())
                all_pred.extend((probs > 0.5).long().cpu().numpy().flatten())
                all_prob.extend(probs.cpu().numpy().flatten())
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        y_prob = np.array(all_prob)

        from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        }
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = 0.0
        return metrics

    elif task_type in ("pair_regression", "pair_retrieval", "protein_pair", "residue_pair") or "pair" in str(task_type):
        # We need to compute predictions from the Siamese/PPI model
        for batch in loader:
            batch = batch.to(device)
            # Use same logic as trainer to construct d1, d2
            if hasattr(batch, 'x1'):
                from torch_geometric.data import Data
                d1 = Data(x=batch.x1, edge_index=batch.edge_index1, edge_attr=batch.edge_attr1, batch=batch.x1_batch if hasattr(batch, 'x1_batch') else None)
                d2 = Data(x=batch.x2, edge_index=batch.edge_index2, edge_attr=batch.edge_attr2, batch=batch.x2_batch if hasattr(batch, 'x2_batch') else None)
                out = model(d1, d2)
            else:
                out = model(batch)
            
            all_true.append(batch.y.cpu().numpy())
            all_pred.append(out.detach().cpu().numpy())
        
        y_true = all_true
        y_pred = all_pred
        
        if task_obj is not None:
            try:
                # Many pair tasks expect a list of predictions matching the indices
                return task_obj.evaluate(task_obj.test_targets, y_pred)
            except Exception as e:
                logger.warning(f"  Official eval failed for {task_type}: {e}")
                return {"eval_error": 1.0}
        return {"status": "completed"}

    return {}


def train_and_evaluate_task(task_name, task_class, cfg, gin_cfg, device, logger, quick=False):
    """Full pipeline for one task: load data -> train GIN -> evaluate."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TASK: {task_name}")
    logger.info(f"{'='*60}")

    task_type = TASK_TYPE_MAP.get(task_name, "graph_multiclass")
    t_start = time.time()

    # Skip logic removed as tasks are now supported
    pass

    try:
        # 1. Load dataset
        import requests
        try:
            dataset = BenchmarkDataset(task_class, task_name, cfg, dummy=quick)
            dataset.prepare()
        except requests.exceptions.RequestException as e:
            logger.warning(f"  Skipping {task_name}: Dataset download failed (upstream error): {e}")
            return {"task": task_name, "status": "skipped", "reason": "download_failed", "task_type": task_type, "time_s": 0}

        if dataset.train_loader is None:
            return {"task": task_name, "status": "failed", "reason": "no training data",
                    "task_type": task_type, "time_s": 0}

        feat_dim = dataset.feat_dim
        edge_dim = dataset.edge_dim
        n_classes = dataset.n_classes

        # 2. Build model
        model_type = cfg.get("model_type", "gin")
        model = build_model(task_name, task_type, feat_dim, edge_dim, n_classes, gin_cfg, model_type=model_type)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model: {model_type} ({task_type}), params={n_params:,}")

        # 3. Build loss
        criterion = build_loss(task_type, n_classes, cfg)

        # 4. Override epochs for dummy/quick mode
        cfg_copy = dict(cfg)
        cfg_copy["training"] = dict(cfg["training"])
        if quick:
            cfg_copy["training"]["epochs"] = 1
            cfg_copy["training"]["patience"] = 1
            cfg_copy["training"]["batch_size"] = min(2, cfg_copy["training"].get("batch_size", 2))

        # 5. Train
        # Use custom training loop for non-standard task types
        if task_type in ("graph_multiclass", "graph_multilabel"):
            trainer = Trainer(model, cfg_copy, device, is_pyg=True,
                              model_name=f"GIN_{task_name}")
            hist, t_elapsed = trainer.train(
                dataset.train_loader, dataset.val_loader, criterion
            )
        elif task_type == "graph_regression":
            trainer = Trainer(model, cfg_copy, device, is_pyg=True,
                              model_name=f"GIN_{task_name}")
            hist, t_elapsed = trainer.train(
                dataset.train_loader, dataset.val_loader, criterion
            )
        elif task_type == "node_binary":
            trainer = Trainer(model, cfg_copy, device, is_pyg=True,
                              model_name=f"GIN_{task_name}")
            hist, t_elapsed = trainer.train(
                dataset.train_loader, dataset.val_loader, criterion
            )
        else:
            trainer = Trainer(model, cfg_copy, device, is_pyg=True,
                              model_name=f"GIN_{task_name}")
            hist, t_elapsed = trainer.train(
                dataset.train_loader, dataset.val_loader, criterion
            )

        # 6. Evaluate
        metrics = evaluate_task(model, dataset.test_loader, task_type, device, dataset.task)
        total_time = time.time() - t_start

        result = {
            "task": task_name,
            "status": "completed",
            "task_type": task_type,
            "n_classes": n_classes,
            "n_params": n_params,
            "train_time_s": t_elapsed,
            "total_time_s": total_time,
            **metrics,
        }

        # Log primary metric
        primary = TASK_METRICS.get(task_name, "accuracy")
        primary_val = metrics.get(primary, "N/A")
        logger.info(f"  RESULT: {primary}={primary_val}, time={total_time:.0f}s")

        return result

    except Exception as e:
        total_time = time.time() - t_start
        logger.error(f"  FAILED: {task_name}: {e}")
        logger.error(traceback.format_exc())
        return {"task": task_name, "status": "failed", "reason": str(e),
                "task_type": task_type, "time_s": total_time}


def main():
    parser = argparse.ArgumentParser(description="ProteinShake GIN Benchmark")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to run (default: all)")
    parser.add_argument("--dummy", "--quick", action="store_true", dest="quick",
                        help="Dummy mode: limits datasets to 10 items, epochs to 1 for pipeline verification")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg["seed"])
    logger = setup_logging(cfg["paths"]["log_dir"])

    logger.info("=" * 70)
    logger.info("ProteinShake GIN Benchmark")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"GIN config: {cfg['gin']}")

    # Output dirs
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Determine tasks to run
    tasks_to_run = args.tasks or cfg.get("benchmark_tasks", list(TASK_CLASSES.keys()))

    gin_cfg = cfg["gin"]
    all_results = []
    
    # Resume logic: Load existing results
    res_path = out_dir / "benchmark_results.csv"
    completed_tasks = set()
    if res_path.exists():
        try:
            existing_df = pd.read_csv(res_path)
            all_results = existing_df.to_dict('records')
            completed_tasks = set(existing_df[existing_df['status'] == 'completed']['task'].tolist())
            logger.info(f"Resuming benchmark: Found {len(completed_tasks)} completed tasks in {res_path}")
        except Exception as e:
            logger.warning(f"Could not read existing results for resume: {e}")

    for task_name in tasks_to_run:
        if task_name in completed_tasks:
            logger.info(f"Skipping {task_name} (already completed in results.csv)")
            continue
            
        if task_name not in TASK_CLASSES:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue

        task_class = TASK_CLASSES[task_name]
        set_seed(cfg["seed"])  # Reset seed per task for reproducibility

        result = train_and_evaluate_task(
            task_name, task_class, cfg, gin_cfg, device, logger, quick=args.quick
        )
        all_results.append(result)

        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / "benchmark_results.csv", index=False)
        logger.info(f"  Intermediate results saved to {out_dir / 'benchmark_results.csv'}")

        # Free GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)

    # Print summary table
    summary_cols = ["task", "status", "task_type", "n_classes"]
    for metric in ["accuracy", "fmax", "r2", "mcc", "auroc", "spearman", "f1_micro"]:
        if metric in df.columns:
            summary_cols.append(metric)
    summary_cols.extend(["n_params", "total_time_s"])

    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df[available_cols].to_string(index=False)
    logger.info(f"\n{summary}")

    logger.info(f"\nResults saved to: {out_dir / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
