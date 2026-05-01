#!/usr/bin/env python3
"""
ProteinShake Benchmark — GearNet baseline across all tasks.

Evaluates GearNet (Zhang et al., ICLR 2023) on the same ProteinShake benchmarks
and splits as scripts/benchmark.py, for direct comparison with our SoftBlobGIN.

Usage:
    python scripts/benchmark_gearnet.py --config configs/benchmark_gearnet.yaml
    python scripts/benchmark_gearnet.py --config configs/benchmark_gearnet.yaml --tasks EnzymeClassTask
    python scripts/benchmark_gearnet.py --config configs/benchmark_gearnet.yaml --quick
"""

import argparse, json, logging, os, random, sys, time, traceback
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
from src.models.gearnet import (
    GearNet, GearNetClassifier, GearNetMultiLabel, GearNetRegressor,
    GearNetNodeClassifier, GearNetSiamese, GearNetGraphBuilder,
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

TASK_METRICS = {
    "EnzymeClassTask": "accuracy",  # multiclass (7 EC classes, single label) — official ProteinShake metric
    "GeneOntologyTask": "fmax",
    "ProteinFamilyTask": "accuracy",
    "StructuralClassTask": "accuracy",
    "LigandAffinityTask": "r2",
    "BindingSiteDetectionTask": "mcc",
    "ProteinProteinInterfaceTask": "auroc",
    "StructureSimilarityTask": "spearman",
    "StructureSearchTask": "precision_at_k",
}

def compute_f1_max(y_true, y_prob):
    """Compute F1-max metric (maximum F1 over all thresholds)."""
    from sklearn.metrics import precision_recall_fscore_support
    thresholds = np.linspace(0, 1, 101)
    f1s = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        f1s.append(f)
    return float(max(f1s))

def compute_auprc_micro(y_true, y_prob):
    """Compute micro-averaged AUPRC."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_prob, average="micro"))


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "benchmark_gearnet.log", mode="w"),
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
# GearNet graph construction hook — applied per-batch on GPU
# ---------------------------------------------------------------------------

class GearNetEdgeBuilder:
    """Builds GearNet-style 7-relation edges, operating per-protein within a batch.

    Edge type mapping (num_relation=7, max_seq_dist=2):
      0: sequential i→i+1      4: spatial radius (forward)
      1: sequential i→i-1      5: KNN (forward)
      2: sequential i→i+2      6: reverse of spatial+KNN
      3: sequential i→i-2
    """

    def __init__(self, num_relation=7, max_seq_dist=2, radius=10.0, knn_k=10, min_distance=5):
        self.num_relation = num_relation
        self.max_seq_dist = max_seq_dist
        self.radius = radius
        self.knn_k = knn_k
        self.min_distance = min_distance

    def augment_data(self, data):
        """Build multi-relational edges for a PyG Data or Batch object.

        Must operate per-protein to avoid creating edges that cross protein
        boundaries in a batched graph. Uses data.ptr for protein boundaries.
        """
        device = data.x.device if data.x is not None else 'cpu'
        num_seq_types = 2 * self.max_seq_dist  # 4

        # Protein boundaries: ptr[g]..ptr[g+1]-1 are the nodes of protein g.
        if hasattr(data, 'ptr') and data.ptr is not None:
            ptr = data.ptr.cpu()
        else:
            ptr = torch.tensor([0, data.num_nodes])

        all_src, all_dst, all_type = [], [], []
        # Collect spatial+KNN edges so we can add their reverse as type 6.
        spatial_knn_src, spatial_knn_dst = [], []

        coords = getattr(data, 'coords', None)

        for g in range(len(ptr) - 1):
            start, end = int(ptr[g]), int(ptr[g + 1])
            lng = end - start
            if lng == 0:
                continue

            # ── Sequential edges (within this protein only) ────────────────
            for d in range(1, self.max_seq_dist + 1):
                if lng <= d:
                    continue
                s = torch.arange(start, end - d, device=device)
                t = torch.arange(start + d, end, device=device)
                all_src.extend([s, t])
                all_dst.extend([t, s])
                all_type.extend([
                    torch.full((len(s),), 2 * (d - 1),     dtype=torch.long, device=device),
                    torch.full((len(t),), 2 * (d - 1) + 1, dtype=torch.long, device=device),
                ])

            if lng < 2:
                continue

            # ── Spatial + KNN edges from 3-D coordinates ──────────────────
            if coords is not None:
                c = coords[start:end].cpu()
                dm = torch.cdist(c, c)  # [lng, lng] — within this protein only
                sd = torch.abs(
                    torch.arange(lng).unsqueeze(0) - torch.arange(lng).unsqueeze(1)
                )

                # Spatial radius edges (type 4): seq-distant pairs within radius
                mask = (dm <= self.radius) & (sd >= self.min_distance)
                sp, dp = mask.nonzero(as_tuple=True)
                sp, dp = (sp + start).to(device), (dp + start).to(device)
                if len(sp) > 0:
                    all_src.append(sp); all_dst.append(dp)
                    all_type.append(torch.full((len(sp),), num_seq_types, dtype=torch.long, device=device))
                    spatial_knn_src.append(sp); spatial_knn_dst.append(dp)

                # KNN edges (type 5)
                dm_masked = dm.clone()
                dm_masked[sd < self.min_distance] = float('inf')
                dm_masked.fill_diagonal_(float('inf'))
                k = min(self.knn_k, lng - 1)
                if k > 0:
                    _, knn_idx = dm_masked.topk(k, dim=1, largest=False)
                    ks = torch.arange(lng).unsqueeze(1).expand(-1, k).reshape(-1)
                    kd = knn_idx.reshape(-1)
                    valid = dm_masked[ks, kd] < float('inf')
                    ks = (ks[valid] + start).to(device)
                    kd = (kd[valid] + start).to(device)
                    if len(ks) > 0:
                        all_src.append(ks); all_dst.append(kd)
                        all_type.append(torch.full((len(ks),), num_seq_types + 1, dtype=torch.long, device=device))
                        spatial_knn_src.append(ks); spatial_knn_dst.append(kd)

        # ── Reverse of spatial+KNN → type 6 ──────────────────────────────
        # Type 6 is intentional: the official GearNet 7-relation design has a
        # dedicated reverse-spatial/KNN relation separate from forward types 4/5.
        if spatial_knn_src:
            rev_s = torch.cat(spatial_knn_dst)
            rev_d = torch.cat(spatial_knn_src)
            all_src.append(rev_s); all_dst.append(rev_d)
            all_type.append(torch.full((len(rev_s),), num_seq_types + 2, dtype=torch.long, device=device))

        if all_src:
            edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
            edge_type = torch.cat(all_type)
        else:
            edge_index = torch.zeros(2, 1, dtype=torch.long, device=device)
            edge_type = torch.zeros(1, dtype=torch.long, device=device)

        data.edge_index = edge_index
        data.edge_type = edge_type
        data.edge_attr = None
        return data


def _make_edge_builder(cfg):
    """Build a GearNetEdgeBuilder from the YAML config."""
    g = cfg.get("gearnet", {})
    return GearNetEdgeBuilder(
        num_relation=g.get("num_relation", 7),
        max_seq_dist=g.get("max_seq_dist", 2),
        radius=g.get("radius", 10.0),
        knn_k=g.get("knn_k", 10),
        min_distance=g.get("min_distance", 5),
    )


# Placeholder; replaced with config-driven instance in main() and train_and_evaluate_task().
_edge_builder = GearNetEdgeBuilder()


def build_gearnet_model(task_name, task_type, feat_dim, n_classes, gearnet_cfg):
    """Build the appropriate GearNet variant for a given task."""
    hidden_dims = gearnet_cfg.get("hidden_dims", [512, 512, 512, 512, 512, 512])
    num_relation = gearnet_cfg.get("num_relation", 7)
    batch_norm = gearnet_cfg.get("batch_norm", True)
    concat_hidden = gearnet_cfg.get("concat_hidden", True)
    short_cut = gearnet_cfg.get("short_cut", True)
    readout = gearnet_cfg.get("readout", "sum")
    dropout = gearnet_cfg.get("dropout", 0.0)
    num_mlp_layer = gearnet_cfg.get("num_mlp_layer", 3)

    common = dict(
        input_dim=feat_dim, hidden_dims=hidden_dims, num_relation=num_relation,
        batch_norm=batch_norm, concat_hidden=concat_hidden, short_cut=short_cut,
        readout=readout, dropout=dropout, num_mlp_layer=num_mlp_layer,
    )

    if task_type == "graph_multiclass":
        return GearNetClassifier(n_classes=n_classes, **common)
    elif task_type == "graph_multilabel":
        return GearNetMultiLabel(n_classes=n_classes, **common)
    elif task_type == "graph_regression":
        return GearNetRegressor(**common)
    elif task_name == "ProteinProteinInterfaceTask":
        # PPI is node_binary but requires protein-B context; GearNet has no dedicated PPI head.
        # Disabled in config (OOM). If re-enabled, a proper PPI wrapper must be added first.
        raise NotImplementedError(
            "GearNet PPI: no protein-B context head implemented. "
            "Use GearNetNodeClassifier only after adding a PPI-specific wrapper."
        )
    elif task_type == "node_binary":
        return GearNetNodeClassifier(
            input_dim=feat_dim, hidden_dims=hidden_dims, num_relation=num_relation,
            batch_norm=batch_norm, concat_hidden=concat_hidden, short_cut=short_cut,
            dropout=dropout,
        )
    elif task_name in ("StructureSimilarityTask", "StructureSearchTask"):
        encoder = GearNetClassifier(n_classes=n_classes, **common)
        return GearNetSiamese(encoder)
    else:
        return GearNetClassifier(n_classes=n_classes, **common)


def build_loss(task_type, n_classes, cfg):
    """Build appropriate loss for each task type."""
    if task_type == "graph_multiclass":
        return nn.CrossEntropyLoss()
    elif task_type == "graph_multilabel":
        return nn.BCEWithLogitsLoss()
    elif task_type in ("graph_regression", "pair_regression"):
        return nn.MSELoss()
    elif task_type in ("node_binary", "pair_retrieval"):
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


class GearNetBatchWrapper:
    """Wraps a DataLoader to add GearNet edge construction on-the-fly."""
    def __init__(self, loader, edge_builder):
        self.loader = loader
        self.edge_builder = edge_builder

    def __iter__(self):
        for batch in self.loader:
            batch = self.edge_builder.augment_data(batch)
            yield batch

    def __len__(self):
        return len(self.loader)


@torch.no_grad()
def evaluate_task(model, loader, task_type, device, task_obj=None):
    """Evaluate model and return metrics dict."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for batch in loader:
        batch = _edge_builder.augment_data(batch)
        batch = batch.to(device)

        if task_type == "graph_multiclass":
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_true.extend(batch.y.long().cpu().numpy())
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())

        elif task_type == "graph_multilabel":
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_true.append(batch.y.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

        elif task_type == "graph_regression":
            pred = model(batch)
            all_true.extend(batch.y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

        elif task_type == "node_binary":
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_true.extend(batch.y.cpu().numpy().flatten())
            all_pred.extend((probs > 0.5).long().cpu().numpy().flatten())
            all_prob.extend(probs.cpu().numpy().flatten())

    # Compute metrics
    if task_type == "graph_multiclass":
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
        y_true = np.vstack(all_true)
        y_prob = np.vstack(all_prob)
        y_pred = (y_prob > 0.5).astype(int)
        metrics = {}
        
        # Calculate standard metrics
        metrics["accuracy"] = float((y_true == y_pred).all(axis=1).mean()) # subset accuracy
        metrics["fmax"] = compute_f1_max(y_true, y_prob)
        metrics["auprc"] = compute_auprc_micro(y_true, y_prob)
        
        # If task object is available, try to get official metrics
        if task_obj is not None:
            try:
                official = task_obj.evaluate(task_obj.test_targets, y_pred)
                for k, v in official.items():
                    if k not in metrics: metrics[k] = float(v)
            except Exception:
                try:
                    official = task_obj.evaluate(task_obj.test_targets, y_prob)
                    for k, v in official.items():
                        if k not in metrics: metrics[k] = float(v)
                except Exception:
                    pass
        return metrics

    elif task_type == "graph_regression":
        y_true, y_pred = np.array(all_true), np.array(all_pred)
        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0,
        }

    elif task_type == "node_binary":
        y_true, y_pred = np.array(all_true), np.array(all_pred)
        y_prob_arr = np.array(all_prob)
        from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        }
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob_arr))
        except Exception:
            metrics["auroc"] = 0.0
        return metrics

    return {}


class GearNetTrainerAdapter:
    """Wraps the existing Trainer to inject GearNet edge construction."""

    def __init__(self, model, cfg, device, model_name="GearNet"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_cfg = cfg["training"]

        # Build optimizer
        lr = self.train_cfg.get("lr", 1e-4)
        wd = self.train_cfg.get("weight_decay", 0.0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        # Build scheduler
        sched_type = self.train_cfg.get("scheduler", "reduce_on_plateau")
        if sched_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.train_cfg.get("scheduler_factor", 0.6),
                patience=self.train_cfg.get("scheduler_patience", 5),
            )
            self._sched_is_plateau = True
        else:
            from src.training.trainer import CosineWarmupScheduler
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=self.train_cfg.get("warmup_epochs", 10),
                total_epochs=self.train_cfg["epochs"],
            )
            self._sched_is_plateau = False

        self.grad_clip = self.train_cfg.get("gradient_clip", 1.0)
        ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def train(self, train_loader, val_loader, criterion):
        from copy import deepcopy
        logger = logging.getLogger(__name__)

        epochs = self.train_cfg["epochs"]
        patience = self.train_cfg.get("patience", 30)
        min_delta = self.train_cfg.get("min_delta", 1e-4)

        best_val_loss = float("inf")
        best_state = deepcopy(self.model.state_dict())
        stale = 0
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            total_loss = total_samples = 0
            for batch in train_loader:
                batch = _edge_builder.augment_data(batch)
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch)

                y = batch.y
                if y is not None and len(y.shape) > 1 and y.shape == out.shape:
                    loss = criterion(out, y)
                elif y is not None and out.shape == y.shape:
                    loss = criterion(out, y)
                else:
                    loss = criterion(out, y.long())

                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                total_samples += batch.num_graphs

            train_loss = total_loss / max(total_samples, 1)

            # Validate
            self.model.eval()
            val_loss_sum = val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = _edge_builder.augment_data(batch)
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    y = batch.y
                    if y is not None and len(y.shape) > 1 and y.shape == out.shape:
                        loss = criterion(out, y)
                    elif y is not None and out.shape == y.shape:
                        loss = criterion(out, y)
                    else:
                        loss = criterion(out, y.long())
                    val_loss_sum += loss.item() * batch.num_graphs
                    val_samples += batch.num_graphs

            val_loss = val_loss_sum / max(val_samples, 1)

            # Schedule
            if self._sched_is_plateau:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(epoch)

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1

            if epoch % 10 == 0 or epoch == 1 or stale == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                marker = " *" if stale == 0 else ""
                logger.info(
                    f"  [{self.model_name}] ep {epoch:3d}/{epochs}  "
                    f"loss={train_loss:.4f}/{val_loss:.4f}  lr={lr:.2e}{marker}"
                )

            if stale >= patience:
                logger.info(f"  [{self.model_name}] Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t_start
        self.model.load_state_dict(best_state)

        ckpt_path = self.ckpt_dir / f"{self.model_name}_best.pt"
        torch.save(best_state, ckpt_path)
        logger.info(f"  Saved best checkpoint: {ckpt_path}")

        return {}, elapsed


def train_and_evaluate_task(task_name, task_class, cfg, gearnet_cfg, device, logger, quick=False):
    """Full pipeline for one task: load data -> train GearNet -> evaluate."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TASK: {task_name} (GearNet baseline)")
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

        feat_dim = dataset.feat_dim
        n_classes = dataset.n_classes

        # Override input_dim based on actual feature dim
        gearnet_cfg_copy = dict(gearnet_cfg)
        gearnet_cfg_copy["input_dim"] = feat_dim

        # Build model
        model = build_gearnet_model(task_name, task_type, feat_dim, n_classes, gearnet_cfg_copy)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model: GearNet ({task_type}), feat_dim={feat_dim}, params={n_params:,}")

        # Build loss
        criterion = build_loss(task_type, n_classes, cfg)

        # Apply task-specific overrides
        cfg_copy = dict(cfg)
        cfg_copy["training"] = dict(cfg["training"])
        overrides = cfg.get("task_overrides", {}).get(task_name, {})
        for k, v in overrides.items():
            cfg_copy["training"][k] = v

        if quick:
            cfg_copy["training"]["epochs"] = 1
            cfg_copy["training"]["patience"] = 1

        # Train
        trainer = GearNetTrainerAdapter(model, cfg_copy, device, model_name=f"GearNet_{task_name}")
        _, t_elapsed = trainer.train(dataset.train_loader, dataset.val_loader, criterion)

        # Evaluate
        metrics = evaluate_task(model, dataset.test_loader, task_type, device, dataset.task)
        total_time = time.time() - t_start

        result = {
            "task": task_name, "status": "completed", "task_type": task_type,
            "n_classes": n_classes, "n_params": n_params,
            "train_time_s": t_elapsed, "total_time_s": total_time,
            **metrics,
        }

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
    parser = argparse.ArgumentParser(description="ProteinShake GearNet Benchmark")
    parser.add_argument("--config", type=str, default="configs/benchmark_gearnet.yaml")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--dummy", "--quick", action="store_true", dest="quick")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg["seed"])
    logger = setup_logging(cfg["paths"]["log_dir"])

    # Replace the module-level placeholder with a config-driven instance so all
    # training and evaluation loops use the correct radius/knn_k/min_distance.
    global _edge_builder
    _edge_builder = _make_edge_builder(cfg)

    logger.info("=" * 70)
    logger.info("ProteinShake GearNet Benchmark (train from scratch)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"GearNet config: {cfg['gearnet']}")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    tasks_to_run = args.tasks or cfg.get("benchmark_tasks", list(TASK_CLASSES.keys()))
    gearnet_cfg = cfg["gearnet"]
    all_results = []

    # Resume logic
    res_path = out_dir / "benchmark_gearnet_results.csv"
    completed_tasks = set()
    if res_path.exists():
        try:
            existing_df = pd.read_csv(res_path)
            all_results = existing_df.to_dict('records')
            completed_tasks = set(existing_df[existing_df['status'] == 'completed']['task'].tolist())
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

        task_class = TASK_CLASSES[task_name]
        set_seed(cfg["seed"])

        result = train_and_evaluate_task(
            task_name, task_class, cfg, gearnet_cfg, device, logger, quick=args.quick
        )
        all_results.append(result)

        df = pd.DataFrame(all_results)
        df.to_csv(res_path, index=False)
        logger.info(f"  Intermediate results saved to {res_path}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("GEARNET BENCHMARK COMPLETE")
    logger.info("=" * 70)

    df = pd.DataFrame(all_results)
    df.to_csv(res_path, index=False)

    summary_cols = ["task", "status", "task_type", "n_classes"]
    for m in ["fmax", "auprc", "accuracy", "r2", "mcc", "auroc", "spearman"]:
        if m in df.columns:
            summary_cols.append(m)
    summary_cols.extend(["n_params", "total_time_s"])
    available_cols = [c for c in summary_cols if c in df.columns]
    logger.info(f"\n{df[available_cols].to_string(index=False)}")
    logger.info(f"\nResults saved to: {res_path}")


if __name__ == "__main__":
    main()
