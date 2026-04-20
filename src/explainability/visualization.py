"""
Visualization for GNN explainability results.

Generates figures for:
  - Edge importance on protein contact graphs
  - Residue importance heatmaps along sequence
  - Feature group importance bar charts
  - Fidelity curves (fidelity+ and fidelity- vs sparsity)
  - Per-class prototype comparisons
  - GNNExplainer vs Integrated Gradients comparison
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DAA51B"]
EC_NAMES = ["EC1 Oxidoreductase", "EC2 Transferase", "EC3 Hydrolase",
            "EC4 Lyase", "EC5 Isomerase", "EC6 Ligase", "EC7 Translocase"]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "font.size": 10,
})


def save_fig(fig, path, name):
    fpath = Path(path) / name
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fpath}")


# ============================================================================
# Edge importance visualization
# ============================================================================

def plot_edge_importance_examples(explanations: list, graphs: list,
                                  n_per_class: int, n_classes: int,
                                  out_dir: str):
    """Plot protein contact graphs with edge importance coloring.

    Shows top examples per class with edges colored by GNNExplainer importance.
    """
    from collections import defaultdict
    import networkx as nx

    # Group by predicted class
    by_class = defaultdict(list)
    for i, exp in enumerate(explanations):
        by_class[exp.true_label].append((i, exp))

    n_cols = min(n_per_class, 3)
    n_rows = min(n_classes, 7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row in range(n_rows):
        examples = by_class.get(row, [])[:n_cols]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(examples):
                ax.set_visible(False)
                continue

            idx, exp = examples[col]
            ei = exp.edge_index
            emask = exp.edge_mask

            # Build networkx graph
            G = nx.Graph()
            G.add_nodes_from(range(exp.n_nodes))
            edge_weights = {}
            for e in range(ei.shape[1]):
                src, dst = int(ei[0, e]), int(ei[1, e])
                if src < dst:  # avoid duplicates for undirected
                    edge_weights[(src, dst)] = emask[e]
                    G.add_edge(src, dst, weight=emask[e])

            # Layout
            pos = nx.spring_layout(G, seed=42, k=2.0/np.sqrt(exp.n_nodes))

            # Draw nodes colored by importance
            node_colors = exp.node_importance
            node_colors = node_colors / (node_colors.max() + 1e-8)

            # Draw edges colored by importance
            edges = list(G.edges())
            if edges:
                weights = [G[u][v]['weight'] for u, v in edges]
                weights = np.array(weights)
                weights = weights / (weights.max() + 1e-8)

                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, edge_color=weights,
                    edge_cmap=plt.cm.Reds, width=1.5, alpha=0.7, ax=ax
                )

            nx.draw_networkx_nodes(
                G, pos, node_color=node_colors, cmap=plt.cm.YlOrRd,
                node_size=30, alpha=0.8, ax=ax
            )

            ax.set_title(
                f"EC{row+1} | pred={exp.predicted_label+1} | "
                f"conf={exp.predicted_prob:.2f}",
                fontsize=9, fontweight="bold"
            )
            ax.axis("off")

    fig.suptitle("Edge Importance (GNNExplainer) — Example Proteins",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_edge_importance_examples.png")


# ============================================================================
# Feature group importance
# ============================================================================

def plot_feature_group_importance(prototypes: list, out_dir: str):
    """Bar chart showing which feature groups matter most per EC class."""
    if not prototypes or prototypes[0].top_features is None:
        return

    groups = [name for name, _ in prototypes[0].top_features]
    n_groups = len(groups)
    n_classes = len(prototypes)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_groups)
    width = 0.8 / n_classes

    for i, proto in enumerate(prototypes):
        if proto.top_features is None:
            continue
        vals = [imp for _, imp in proto.top_features]
        offset = (i - n_classes / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=f"EC{i+1}",
               color=PALETTE[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Mean Feature Mask Value")
    ax.set_title("Feature Group Importance by EC Class (GNNExplainer)",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_feature_group_importance.png")


# ============================================================================
# Fidelity curves
# ============================================================================

def plot_fidelity_curves(fidelity_results: list, out_dir: str,
                         method_name: str = "GNNExplainer"):
    """Plot fidelity+ and fidelity- as a function of sparsity."""
    if not fidelity_results:
        return

    sparsities = [r.sparsity for r in fidelity_results]
    fid_plus = [r.fidelity_plus for r in fidelity_results]
    fid_minus = [r.fidelity_minus for r in fidelity_results]
    prob_plus = [r.prob_fidelity_plus for r in fidelity_results]
    prob_minus = [r.prob_fidelity_minus for r in fidelity_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy-based fidelity
    ax = axes[0]
    ax.plot(sparsities, fid_plus, "o-", color="#4C72B0", lw=2, markersize=7,
            label="Fidelity+ (sufficiency)")
    ax.plot(sparsities, fid_minus, "s--", color="#C44E52", lw=2, markersize=7,
            label="Fidelity- (necessity)")
    ax.set_xlabel("Fraction of edges retained")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy-based Fidelity", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Probability-based fidelity
    ax = axes[1]
    ax.plot(sparsities, prob_plus, "o-", color="#4C72B0", lw=2, markersize=7,
            label="Prob Fidelity+ (1 - Δp)")
    ax.plot(sparsities, prob_minus, "s--", color="#C44E52", lw=2, markersize=7,
            label="Prob Fidelity- (Δp on removal)")
    ax.set_xlabel("Fraction of edges retained")
    ax.set_ylabel("Score")
    ax.set_title("Probability-based Fidelity", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Explanation Fidelity — {method_name}", fontweight="bold",
                 fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_fidelity_curves.png")


# ============================================================================
# Residue position importance
# ============================================================================

def plot_position_importance(prototypes: list, out_dir: str):
    """Heatmap of residue position importance per EC class."""
    valid = [p for p in prototypes if p.position_importance_hist is not None]
    if not valid:
        return

    n_bins = len(valid[0].position_importance_hist)
    mat = np.array([p.position_importance_hist for p in valid])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   extent=[0, 1, len(valid) - 0.5, -0.5])
    ax.set_xlabel("Normalized sequence position (N-term → C-term)")
    ax.set_ylabel("EC Class")
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([f"EC{p.class_id+1} {p.class_name}" for p in valid],
                       fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean residue importance")
    ax.set_title("Residue Position Importance by EC Class", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_position_importance.png")


# ============================================================================
# Method comparison
# ============================================================================

def plot_method_comparison(gnnexp_results: list, ig_results: list,
                           out_dir: str):
    """Compare GNNExplainer vs Integrated Gradients node importance."""
    if not gnnexp_results or not ig_results:
        return

    # Compute correlation between methods for each protein
    correlations = []
    for gnn_r, ig_r in zip(gnnexp_results, ig_results):
        if gnn_r.node_importance is None or ig_r.node_attributions is None:
            continue
        n = min(len(gnn_r.node_importance), len(ig_r.node_attributions))
        if n < 3:
            continue
        gnn_imp = gnn_r.node_importance[:n]
        ig_imp = ig_r.node_attributions[:n]
        # Normalize
        gnn_imp = gnn_imp / (gnn_imp.max() + 1e-8)
        ig_imp = ig_imp / (ig_imp.max() + 1e-8)
        corr = np.corrcoef(gnn_imp, ig_imp)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Correlation histogram
    ax = axes[0]
    ax.hist(correlations, bins=20, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(correlations), color="#C44E52", lw=2, ls="--",
               label=f"Mean r={np.mean(correlations):.3f}")
    ax.set_xlabel("Pearson correlation (node importance)")
    ax.set_ylabel("Count")
    ax.set_title("GNNExplainer vs IG Agreement", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Scatter plot for a single example
    ax = axes[1]
    if gnnexp_results and ig_results:
        # Pick example with median correlation
        idx = len(correlations) // 2
        gnn_imp = gnnexp_results[idx].node_importance
        ig_imp = ig_results[idx].node_attributions
        n = min(len(gnn_imp), len(ig_imp))
        gnn_imp = gnn_imp[:n] / (gnn_imp[:n].max() + 1e-8)
        ig_imp = ig_imp[:n] / (ig_imp[:n].max() + 1e-8)
        ax.scatter(gnn_imp, ig_imp, alpha=0.5, s=20, color="#55A868")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("GNNExplainer (node importance)")
        ax.set_ylabel("Integrated Gradients (node importance)")
        ax.set_title(f"Example protein (n={n} residues)", fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle("Explainability Method Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_ig_vs_gnnexplainer.png")


# ============================================================================
# Class prototype comparison
# ============================================================================

def plot_class_prototypes(prototypes: list, out_dir: str):
    """Summary figure comparing prototypes across all EC classes."""
    valid = [p for p in prototypes if p.n_samples > 0]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Mean sparsity per class
    ax = axes[0]
    classes = [f"EC{p.class_id+1}" for p in valid]
    sparsities = [p.mean_sparsity for p in valid]
    ax.barh(classes, sparsities, color=[PALETTE[p.class_id] for p in valid],
            alpha=0.8)
    ax.set_xlabel("Mean explanation sparsity")
    ax.set_title("Explanation Sparsity", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: Mean masked confidence per class
    ax = axes[1]
    confs = [p.mean_masked_confidence for p in valid]
    ax.barh(classes, confs, color=[PALETTE[p.class_id] for p in valid],
            alpha=0.8)
    ax.set_xlabel("Mean confidence on masked graph")
    ax.set_title("Explanation Sufficiency", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 3: Feature group importance (stacked)
    ax = axes[2]
    if valid[0].top_features:
        groups = [name for name, _ in valid[0].top_features]
        bottom = np.zeros(len(valid))
        for g_idx, group in enumerate(groups):
            vals = []
            for p in valid:
                if p.top_features:
                    vals.append(p.top_features[g_idx][1])
                else:
                    vals.append(0)
            color = plt.cm.Set3(g_idx / len(groups))
            ax.barh(classes, vals, left=bottom, color=color, alpha=0.8,
                    label=group)
            bottom += np.array(vals)
        ax.set_xlabel("Cumulative feature importance")
        ax.set_title("Feature Groups", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Per-Class Explanation Prototypes", fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_class_prototypes.png")
