"""
DeepLoc-2.1 Evaluation — Compute per-class and overall metrics.

Standard DeepLoc metric: Fmax (maximum F1 over a sweep of probability thresholds).
Also computes per-class AUROC, F1, and MCC.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def compute_fmax(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Compute Fmax and the optimal thresholds per class.

    Args:
        y_true: [N, C] binary label matrix
        y_prob: [N, C] predicted probability matrix

    Returns:
        fmax: scalar
        best_thresholds: [C] array of thresholds that maximized F1
    """
    n_classes = y_true.shape[1]
    best_thresholds = np.full(n_classes, 0.5)
    
    # Protein-centric Fmax sweep (as per CAFA/DeepLoc)
    # This sweep finds a GLOBAL threshold that maximizes the protein-centric F1
    best_f1_global = 0.0
    best_thresh_global = 0.5
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        if y_pred.sum() == 0: continue
        
        tp = (y_pred * y_true).sum(axis=1)
        fp = (y_pred * (1 - y_true)).sum(axis=1)
        fn = ((1 - y_pred) * y_true).sum(axis=1)
        
        pre = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * pre * rec / (pre + rec + 1e-8)
        score = f1.mean()
        
        if score > best_f1_global:
            best_f1_global = score
            best_thresh_global = threshold
            
    # Also find optimal PER-CLASS thresholds to maximize class-wise F1 / MCC
    # This is often better for imbalanced datasets
    for i in range(n_classes):
        if y_true[:, i].sum() == 0: continue
        best_f1_c = 0.0
        for t in np.arange(0.01, 0.99, 0.01):
            y_pred_c = (y_prob[:, i] >= t).astype(int)
            f1_c = f1_score(y_true[:, i], y_pred_c, zero_division=0)
            if f1_c > best_f1_c:
                best_f1_c = f1_c
                best_thresholds[i] = t
                
    return float(best_f1_global), best_thresholds


@torch.no_grad()
def evaluate_deeploc(
    model,
    loader,
    device,
    label_names: list,
    thresholds: np.ndarray = None,
) -> dict:
    """Run evaluation and compute DeepLoc metrics.

    Args:
        model: Trained SoftBlobGIN_MTL model.
        loader: PyG DataLoader for the evaluation set.
        device: torch device.
        label_names: Names of the localization classes.
        thresholds: Fixed per-class thresholds. If None, computes optimal ones.

    Returns:
        Dictionary of metrics.
    """
    model.eval()
    all_true_loc = []
    all_prob_loc = []
    all_true_mem = []
    all_pred_mem = []

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)
        out = model(batch)
        
        # Handle MTL [loc, mem]
        if isinstance(out, (list, tuple)):
            logits_loc = out[0]
            logits_mem = out[1]
        else:
            logits_loc = out
            logits_mem = None
            
        probs_loc = torch.sigmoid(logits_loc)
        
        # Targets are also [loc, mem]
        y_loc = batch.y_loc
        y_mem = batch.y_mem if hasattr(batch, 'y_mem') else None
            
        all_true_loc.append(y_loc.cpu().numpy())
        all_prob_loc.append(probs_loc.cpu().numpy())
        
        if logits_mem is not None and y_mem is not None:
            preds_mem = torch.argmax(logits_mem, dim=-1)
            all_true_mem.append(y_mem.cpu().numpy())
            all_pred_mem.append(preds_mem.cpu().numpy())

    if not all_true_loc:
        logger.warning("No valid batches in evaluation set.")
        return {"fmax": 0.0}

    y_true = np.vstack(all_true_loc)
    y_prob = np.vstack(all_prob_loc)

    metrics = {}

    # 1. Fmax and Threshold Tuning
    fmax, best_thresholds = compute_fmax(y_true, y_prob)
    metrics["fmax"] = fmax
    
    # If thresholds were provided (from validation set), use them.
    if thresholds is not None:
        eval_thresholds = thresholds
    else:
        eval_thresholds = best_thresholds
    
    metrics["best_thresholds"] = eval_thresholds

    # 2. Per-class Metrics (AUROC, MCC)
    aurocs = []
    mccs = []
    for i, name in enumerate(label_names):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            # AUROC
            try:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                metrics[f"auroc_{name}"] = auc
                aurocs.append(auc)
            except ValueError: pass
            
            # MCC (standard for imbalanced bio datasets)
            y_pred_i = (y_prob[:, i] >= eval_thresholds[i]).astype(int)
            mcc = matthews_corrcoef(y_true[:, i], y_pred_i)
            metrics[f"mcc_{name}"] = mcc
            mccs.append(mcc)

    if aurocs:
        metrics["auroc_macro"] = np.mean(aurocs)
    if mccs:
        metrics["mcc_macro"] = np.mean(mccs)

    # 3. Overall Localization metrics
    y_pred = (y_prob >= eval_thresholds).astype(int)
    metrics["hamming_accuracy"] = float((y_pred == y_true).mean())
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # 4. Membrane Task Metrics (Multi-Class)
    if all_true_mem:
        y_true_mem = np.concatenate(all_true_mem).flatten()
        y_pred_mem = np.concatenate(all_pred_mem).flatten()
        
        # Filter out ignore_index (-100)
        mask = y_true_mem != -100
        if mask.any():
            y_tm = y_true_mem[mask]
            y_pm = y_pred_mem[mask]
            metrics["membrane_accuracy"] = float((y_tm == y_pm).mean())
            metrics["membrane_mcc"] = float(matthews_corrcoef(y_tm, y_pm))
            metrics["membrane_f1_macro"] = float(f1_score(y_tm, y_pm, average="macro", zero_division=0))

    return metrics
