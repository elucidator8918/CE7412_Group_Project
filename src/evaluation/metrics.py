"""
Evaluation metrics for multi-class enzyme classification.

Computes:
  - Accuracy, Macro-F1, Macro-AUROC
  - Per-class precision, recall, F1
  - Normalized confusion matrix
  - Silhouette score on embeddings
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_prob, n_classes=7):
    """Compute all evaluation metrics for a model.

    Returns:
        dict with keys: accuracy, macro_f1, macro_auroc, per_class_f1,
                        per_class_precision, per_class_recall, confusion_matrix,
                        confusion_matrix_normalized
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec, rec, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # AUROC
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        auroc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auroc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Per-class ROC data
    roc_data = {}
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            roc_data[c] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    except Exception:
        pass

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_auroc": auroc,
        "per_class_f1": f1_per,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "support": support,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "roc_data": roc_data,
    }


def compute_silhouette(embeddings, labels):
    """Compute silhouette score on 2D embeddings."""
    try:
        if len(set(labels)) < 2:
            return float("nan")
        return silhouette_score(embeddings, labels)
    except Exception:
        return float("nan")


def format_metrics_table(results: dict, model_names: list, n_classes: int = 7):
    """Format metrics into a printable comparison table."""
    header = f"{'Model':<22} {'Acc':>7} {'F1':>7} {'AUROC':>7} {'Params':>10} {'Time(s)':>8}"
    lines = [header, "-" * len(header)]

    for name in model_names:
        r = results[name]
        m = r["metrics"]
        lines.append(
            f"{name:<22} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
            f"{m['macro_auroc']:>7.4f} {r.get('n_params', 0):>10,} "
            f"{r.get('train_time', 0):>8.0f}"
        )

    return "\n".join(lines)
