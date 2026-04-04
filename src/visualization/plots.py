"""
Visualization suite for enzyme classification project.

Generates all figures required for the CE7412 project report:
  - Dataset statistics and class distributions
  - Training curves (loss + accuracy)
  - Confusion matrices (raw + normalized)
  - Per-class ROC curves
  - Model comparison bar charts
  - t-SNE and UMAP embedding visualizations
  - Ablation study plots
  - Blob assignment heatmaps
  - Feature importance analysis
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

# ── Style configuration ────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DAA51B"]
EC_SHORT = ["EC 1", "EC 2", "EC 3", "EC 4", "EC 5", "EC 6", "EC 7"]
EC_FULL = ["Oxidoreductase", "Transferase", "Hydrolase", "Lyase",
           "Isomerase", "Ligase", "Translocase"]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def save_fig(fig, path, name):
    """Save figure and log."""
    fpath = Path(path) / name
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fpath}")


# ============================================================================
# Dataset figures
# ============================================================================

def plot_class_distribution(train_graphs, val_graphs, test_graphs, n_classes, out_dir):
    """Figure: class distribution across splits."""
    from collections import Counter

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (sname, glist) in zip(axes, [("Train", train_graphs), ("Val", val_graphs), ("Test", test_graphs)]):
        counts = [Counter(int(g.y.item()) for g in glist).get(c, 0) for c in range(n_classes)]
        bars = ax.bar(EC_SHORT, counts, color=PALETTE, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{sname} (n={sum(counts)})", fontweight="bold")
        ax.set_xlabel("Enzyme Commission class")
        if ax is axes[0]:
            ax.set_ylabel("Count")
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                    ha="center", fontsize=8)
    fig.suptitle("Class Distribution Across Splits", fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_class_distribution.png")


def plot_aa_composition(train_graphs, n_classes, out_dir):
    """Figure: amino acid composition heatmap per EC class."""
    from collections import defaultdict
    AA_LETTERS = list("ACDEFGHIKLMNPQRSTVWY")

    comp_by_class = defaultdict(list)
    for g in train_graphs:
        cls = int(g.y.item())
        feat = g.x.float()
        comp = feat[:, :20].mean(dim=0).numpy()
        comp_by_class[cls].append(comp)

    mat = np.array([np.mean(comp_by_class[c], axis=0) for c in range(n_classes)])

    fig, ax = plt.subplots(figsize=(15, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(20))
    ax.set_xticklabels(AA_LETTERS, fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"EC {c+1} {EC_FULL[c]}" for c in range(n_classes)], fontsize=9)
    ax.set_xlabel("Amino acid")
    plt.colorbar(im, ax=ax, label="Mean frequency")
    for i in range(n_classes):
        for j in range(20):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    ax.set_title("Mean Amino-acid Composition per EC Class", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_aa_composition.png")


# ============================================================================
# Training figures
# ============================================================================

def plot_training_curves(all_histories: dict, out_dir):
    """Figure: loss and accuracy curves for all models."""
    n_models = len(all_histories)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    clr = {"train": "#4C72B0", "val": "#DD8452"}

    for col, (mname, hist) in enumerate(all_histories.items()):
        ep = range(1, len(hist["train_loss"]) + 1)

        # Loss
        ax = axes[0, col]
        ax.plot(ep, hist["train_loss"], color=clr["train"], lw=1.5, label="Train")
        ax.plot(ep, hist["val_loss"], color=clr["val"], lw=1.5, label="Val", ls="--")
        ax.set_title(mname, fontweight="bold", fontsize=10)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Accuracy
        ax = axes[1, col]
        ax.plot(ep, hist["train_acc"], color=clr["train"], lw=1.5, label="Train")
        ax.plot(ep, hist["val_acc"], color=clr["val"], lw=1.5, label="Val", ls="--")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Training and Validation Curves", fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_training_curves.png")


# ============================================================================
# Evaluation figures
# ============================================================================

def plot_confusion_matrices(all_results: dict, n_classes, out_dir):
    """Figure: normalized confusion matrices for all models."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    ticks = [f"EC{c+1}" for c in range(n_classes)]

    for ax, (mname, res) in zip(axes, all_results.items()):
        m = res["metrics"]
        cm = m["confusion_matrix"]
        cm_norm = m["confusion_matrix_normalized"]

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(ticks, fontsize=8)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        acc = m["accuracy"]
        ax.set_title(f"{mname}\n(acc={acc:.3f})", fontweight="bold", fontsize=9)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{int(cm[i, j])}\n{cm_norm[i, j]:.0%}",
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Recall")

    fig.suptitle("Confusion Matrices (row-normalised)", fontweight="bold", fontsize=12)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_confusion_matrices.png")


def plot_roc_curves(all_results: dict, n_classes, out_dir, highlight="SoftBlobGAT",
                    baseline="GAT"):
    """Figure: per-class ROC curves comparing two models."""
    n_cols = min(4, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i in range(n_classes):
        ax = axes[i]
        # Highlight model
        if highlight in all_results and i in all_results[highlight]["metrics"].get("roc_data", {}):
            roc = all_results[highlight]["metrics"]["roc_data"][i]
            ax.plot(roc["fpr"], roc["tpr"], color=PALETTE[i], lw=2.5,
                    label=f"{highlight} AUC={roc['auc']:.3f}")
            ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.1, color=PALETTE[i])

        # Baseline model
        if baseline in all_results and i in all_results[baseline]["metrics"].get("roc_data", {}):
            roc_b = all_results[baseline]["metrics"]["roc_data"][i]
            ax.plot(roc_b["fpr"], roc_b["tpr"], color="gray", lw=1.2, ls="--",
                    label=f"{baseline} AUC={roc_b['auc']:.3f}")

        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"EC {i+1} — {EC_FULL[i]}", fontweight="bold", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    # Hide unused axes
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Per-class ROC Curves ({highlight} vs {baseline})", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_roc_curves.png")


def plot_model_comparison_bar(all_results: dict, out_dir):
    """Figure: bar chart comparing model metrics."""
    models = list(all_results.keys())
    acc = [all_results[m]["metrics"]["accuracy"] for m in models]
    f1 = [all_results[m]["metrics"]["macro_f1"] for m in models]
    auroc = [all_results[m]["metrics"]["macro_auroc"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(models)), 5))
    bars1 = ax.bar(x - width, acc, width, label="Accuracy", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x, f1, width, label="Macro F1", color="#55A868", alpha=0.85)
    bars3 = ax.bar(x + width, auroc, width, label="Macro AUROC", color="#DD8452", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_title("Model Performance Comparison", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_model_comparison.png")


# ============================================================================
# Embedding visualization
# ============================================================================

def plot_embeddings(all_embeddings: dict, n_classes, out_dir, seed=42, max_points=500):
    """Figure: t-SNE and UMAP projections of graph embeddings."""
    n_models = len(all_embeddings)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 6 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, (mname, (emb, lab)) in enumerate(all_embeddings.items()):
        # Subsample for speed
        n_pts = min(len(emb), max_points)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(emb), n_pts, replace=False)
        e_sub, l_sub = emb[idx], lab[idx]

        # t-SNE
        ax = axes[row, 0]
        tsne_xy = TSNE(n_components=2, perplexity=min(30, n_pts - 1),
                       random_state=seed, max_iter=1000).fit_transform(e_sub)
        for c in range(n_classes):
            mask = l_sub == c
            ax.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=PALETTE[c],
                       label=f"EC {c+1}", s=35, alpha=0.7, edgecolors="white", linewidths=0.3)
        sil_tsne = _safe_silhouette(tsne_xy, l_sub)
        ax.set_title(f"t-SNE — {mname} (sil={sil_tsne:.3f})", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, title="EC class", title_fontsize=7)
        ax.grid(alpha=0.2)

        # UMAP
        ax = axes[row, 1]
        try:
            import umap
            umap_xy = umap.UMAP(n_components=2, random_state=seed,
                                n_neighbors=min(15, n_pts - 1), min_dist=0.1).fit_transform(e_sub)
            method = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            umap_xy = PCA(n_components=2, random_state=seed).fit_transform(e_sub)
            method = "PCA"

        for c in range(n_classes):
            mask = l_sub == c
            ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1], c=PALETTE[c],
                       label=f"EC {c+1}", s=35, alpha=0.7, edgecolors="white", linewidths=0.3)
        sil_umap = _safe_silhouette(umap_xy, l_sub)
        ax.set_title(f"{method} — {mname} (sil={sil_umap:.3f})", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7, title="EC class", title_fontsize=7)
        ax.grid(alpha=0.2)

    fig.suptitle("Embedding Space Visualisation (test set)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_embeddings.png")


def _safe_silhouette(X, labels):
    """Compute silhouette score, handling edge cases."""
    from sklearn.metrics import silhouette_score
    try:
        if len(set(labels)) < 2:
            return float("nan")
        return silhouette_score(X, labels)
    except Exception:
        return float("nan")


# ============================================================================
# Ablation plots
# ============================================================================

def plot_ablation_eps(results_df: pd.DataFrame, out_dir):
    """Figure: ablation on contact radius epsilon."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["eps"], results_df["accuracy"], "o-", color="#4C72B0",
            lw=2, markersize=8, label="Accuracy")
    ax.plot(results_df["eps"], results_df["macro_f1"], "s--", color="#55A868",
            lw=2, markersize=8, label="Macro F1")
    ax.set_xlabel("Contact radius ε (Å)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Ablation: Contact Radius", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_ablation_eps.png")


def plot_ablation_blobs(results_df: pd.DataFrame, out_dir):
    """Figure: ablation on blob count K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["n_blobs"], results_df["accuracy"], "o-", color="#4C72B0",
            lw=2, markersize=8, label="Accuracy")
    ax.plot(results_df["n_blobs"], results_df["macro_f1"], "s--", color="#55A868",
            lw=2, markersize=8, label="Macro F1")
    ax.set_xlabel("Number of blobs K")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Ablation: Blob Count", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_ablation_blobs.png")


def plot_ablation_features(results_df: pd.DataFrame, out_dir):
    """Figure: ablation on feature sets."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df["accuracy"], width, label="Accuracy",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, results_df["macro_f1"], width, label="Macro F1",
           color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["feature_set"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Ablation: Feature Sets", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_ablation_features.png")
