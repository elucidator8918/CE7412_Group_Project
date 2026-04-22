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

# ============================================================================
# Biological validation plots
# ============================================================================

# Known catalytic residues enriched in enzyme active sites
CATALYTIC_AA = set("HCSDEKRY")  # His, Cys, Ser, Asp, Glu, Lys, Arg, Tyr
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def plot_aa_enrichment(explanations: list, graphs: list, n_classes: int,
                      out_dir: str, top_frac: float = 0.2):
    """Amino acid enrichment in important vs unimportant residues.

    Compares AA composition of top-K important residues (by GNNExplainer)
    against background frequency. Catalytic residues (His, Cys, Ser, Asp, Glu)
    should be enriched in explanations if the model captures active site chemistry.
    """
    from collections import defaultdict

    class_top_aa = defaultdict(lambda: np.zeros(20))  # important residues
    class_bg_aa = defaultdict(lambda: np.zeros(20))   # all residues
    class_counts = defaultdict(int)

    for exp, graph in zip(explanations, graphs):
        if exp.node_importance is None:
            continue
        c = exp.true_label
        n = min(len(exp.node_importance), graph.x.shape[0])
        onehot = graph.x[:n, :20].numpy()  # [N, 20] one-hot

        # Top-K important residues
        k = max(1, int(top_frac * n))
        top_idx = np.argsort(exp.node_importance[:n])[-k:]

        class_top_aa[c] += onehot[top_idx].sum(axis=0)
        class_bg_aa[c] += onehot.sum(axis=0)
        class_counts[c] += 1

    # Compute enrichment ratio per class
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Panel 1: Enrichment heatmap (log2 ratio of top-K vs background)
    ax = axes[0]
    enrichment = np.zeros((n_classes, 20))
    for c in range(n_classes):
        top_freq = class_top_aa[c] / (class_top_aa[c].sum() + 1e-8)
        bg_freq = class_bg_aa[c] / (class_bg_aa[c].sum() + 1e-8)
        enrichment[c] = np.log2((top_freq + 1e-6) / (bg_freq + 1e-6))

    im = ax.imshow(enrichment, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(20))
    ax.set_xticklabels(AA_ORDER, fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"EC{c+1}" for c in range(n_classes)], fontsize=9)
    plt.colorbar(im, ax=ax, label="log₂(enrichment)")

    # Highlight known catalytic residues
    for j, aa in enumerate(AA_ORDER):
        if aa in CATALYTIC_AA:
            ax.text(j, -0.7, "★", ha="center", va="center", fontsize=8,
                    color="#C44E52")

    ax.set_title(f"AA Enrichment in Top-{int(top_frac*100)}% Important Residues "
                 f"(★ = known catalytic)", fontweight="bold")
    ax.set_xlabel("Amino acid")

    # Panel 2: Aggregated catalytic vs non-catalytic enrichment
    ax = axes[1]
    cat_idx = [i for i, aa in enumerate(AA_ORDER) if aa in CATALYTIC_AA]
    noncat_idx = [i for i, aa in enumerate(AA_ORDER) if aa not in CATALYTIC_AA]

    cat_enrich = [enrichment[c, cat_idx].mean() for c in range(n_classes)]
    noncat_enrich = [enrichment[c, noncat_idx].mean() for c in range(n_classes)]

    x = np.arange(n_classes)
    width = 0.35
    ax.bar(x - width/2, cat_enrich, width, label="Catalytic (H,C,S,D,E,K,R,Y)",
           color="#C44E52", alpha=0.8)
    ax.bar(x + width/2, noncat_enrich, width, label="Non-catalytic",
           color="#4C72B0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"EC{c+1}" for c in range(n_classes)])
    ax.set_ylabel("Mean log₂(enrichment)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Catalytic vs Non-catalytic Residue Enrichment", fontweight="bold")

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_aa_enrichment.png")


def plot_sasa_vs_importance(explanations: list, graphs: list, n_classes: int,
                           out_dir: str):
    """SASA distribution of important vs unimportant residues.

    Catalytic residues tend to reside in clefts with moderate SASA.
    If explanations are biologically meaningful, important residues should
    have lower SASA than surface-exposed residues.
    """
    from collections import defaultdict

    # SASA is at feature index 30 (after 20 one-hot + 10 physico)
    SASA_IDX = 30

    top_sasa = defaultdict(list)
    bot_sasa = defaultdict(list)

    for exp, graph in zip(explanations, graphs):
        if exp.node_importance is None:
            continue
        c = exp.true_label
        n = min(len(exp.node_importance), graph.x.shape[0])
        sasa = graph.x[:n, SASA_IDX].numpy()
        imp = exp.node_importance[:n]

        k = max(1, int(0.2 * n))
        top_idx = np.argsort(imp)[-k:]
        bot_idx = np.argsort(imp)[:k]

        top_sasa[c].extend(sasa[top_idx].tolist())
        bot_sasa[c].extend(sasa[bot_idx].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Violin plot per class
    ax = axes[0]
    data_rows = []
    for c in range(n_classes):
        for v in top_sasa.get(c, []):
            data_rows.append({"EC": f"EC{c+1}", "SASA": v, "Group": "Important (top 20%)"})
        for v in bot_sasa.get(c, []):
            data_rows.append({"EC": f"EC{c+1}", "SASA": v, "Group": "Unimportant (bottom 20%)"})

    if data_rows:
        import pandas as pd
        df = pd.DataFrame(data_rows)
        sns.boxplot(data=df, x="EC", y="SASA", hue="Group", ax=ax,
                    palette=["#C44E52", "#4C72B0"], fliersize=2)
        ax.set_ylabel("Normalized SASA")
        ax.set_title("Solvent Accessibility: Important vs Unimportant Residues",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Panel 2: Aggregated across all classes
    ax = axes[1]
    all_top = [v for vals in top_sasa.values() for v in vals]
    all_bot = [v for vals in bot_sasa.values() for v in vals]
    if all_top and all_bot:
        ax.hist(all_top, bins=30, alpha=0.7, color="#C44E52", density=True,
                label=f"Important (n={len(all_top)})")
        ax.hist(all_bot, bins=30, alpha=0.7, color="#4C72B0", density=True,
                label=f"Unimportant (n={len(all_bot)})")
        ax.axvline(np.mean(all_top), color="#C44E52", lw=2, ls="--")
        ax.axvline(np.mean(all_bot), color="#4C72B0", lw=2, ls="--")
        ax.set_xlabel("Normalized SASA")
        ax.set_ylabel("Density")
        ax.set_title("SASA Distribution (all classes)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_sasa_vs_importance.png")


def plot_spatial_clustering(explanations: list, graphs: list, n_classes: int,
                           out_dir: str, top_frac: float = 0.2):
    """Spatial clustering of important residues in 3D.

    Computes mean pairwise Cα distance among top-K important residues vs
    random subsets. If explanations are biologically meaningful, important
    residues should be spatially co-localized (near active site).
    """
    from collections import defaultdict

    real_dists = defaultdict(list)    # mean pairwise dist of top-K
    random_dists = defaultdict(list)  # mean pairwise dist of random-K

    rng = np.random.default_rng(42)

    for exp, graph in zip(explanations, graphs):
        if exp.node_importance is None or not hasattr(graph, 'coords'):
            continue
        c = exp.true_label
        coords = graph.coords.numpy()  # [N, 3]
        n = min(len(exp.node_importance), coords.shape[0])
        if n < 5:
            continue

        k = max(3, int(top_frac * n))
        top_idx = np.argsort(exp.node_importance[:n])[-k:]

        # Mean pairwise distance of top-K
        top_coords = coords[top_idx]
        dists = np.linalg.norm(top_coords[:, None] - top_coords[None, :], axis=-1)
        real_dists[c].append(dists[np.triu_indices(k, k=1)].mean())

        # Random baseline (10 samples)
        for _ in range(10):
            rand_idx = rng.choice(n, size=k, replace=False)
            rand_coords = coords[rand_idx]
            rdists = np.linalg.norm(rand_coords[:, None] - rand_coords[None, :], axis=-1)
            random_dists[c].append(rdists[np.triu_indices(k, k=1)].mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Per-class comparison
    ax = axes[0]
    classes = []
    real_means = []
    rand_means = []
    for c in range(n_classes):
        if real_dists[c]:
            classes.append(f"EC{c+1}")
            real_means.append(np.mean(real_dists[c]))
            rand_means.append(np.mean(random_dists[c]))

    if classes:
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, real_means, width, label="Important residues",
               color="#C44E52", alpha=0.8)
        ax.bar(x + width/2, rand_means, width, label="Random residues",
               color="#4C72B0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Mean pairwise Cα distance (Å)")
        ax.set_title("Spatial Clustering of Important Residues", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    # Panel 2: Ratio (lower = more clustered)
    ax = axes[1]
    if classes:
        ratios = [r / (d + 1e-8) for r, d in zip(real_means, rand_means)]
        colors = ["#55A868" if r < 1.0 else "#C44E52" for r in ratios]
        ax.bar(classes, ratios, color=colors, alpha=0.8)
        ax.axhline(1.0, color="black", lw=1, ls="--", label="Random baseline")
        ax.set_ylabel("Distance ratio (important / random)")
        ax.set_title("Clustering Ratio (< 1.0 = more clustered)", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_spatial_clustering.png")


def plot_contact_distance_distribution(explanations: list, graphs: list,
                                       n_classes: int, out_dir: str):
    """Cα-Cα distance distribution of important vs unimportant edges.

    Important contacts may correspond to specific distance ranges:
    short-range (< 6Å, backbone neighbors), medium (6-12Å, active site),
    or long-range (> 12Å, allosteric).
    """
    from collections import defaultdict

    imp_dists = defaultdict(list)   # distances of important edges
    unimp_dists = defaultdict(list) # distances of unimportant edges

    for exp, graph in zip(explanations, graphs):
        if exp.edge_mask is None or not hasattr(graph, 'coords'):
            continue
        c = exp.true_label
        coords = graph.coords.numpy()
        ei = exp.edge_index
        emask = exp.edge_mask

        for e in range(ei.shape[1]):
            src, dst = int(ei[0, e]), int(ei[1, e])
            if src >= coords.shape[0] or dst >= coords.shape[0]:
                continue
            d = np.linalg.norm(coords[src] - coords[dst])
            if emask[e] > 0.5:
                imp_dists[c].append(d)
            else:
                unimp_dists[c].append(d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Overlaid histograms (all classes)
    ax = axes[0]
    all_imp = [d for dists in imp_dists.values() for d in dists]
    all_unimp = [d for dists in unimp_dists.values() for d in dists]
    if all_imp and all_unimp:
        bins = np.linspace(0, 20, 40)
        ax.hist(all_imp, bins=bins, alpha=0.7, color="#C44E52", density=True,
                label=f"Important edges (n={len(all_imp)})")
        ax.hist(all_unimp, bins=bins, alpha=0.7, color="#4C72B0", density=True,
                label=f"Unimportant edges (n={len(all_unimp)})")
        ax.axvline(6.0, color="gray", ls=":", lw=1, label="6Å (backbone)")
        ax.axvline(12.0, color="gray", ls="--", lw=1, label="12Å (long-range)")
        ax.set_xlabel("Cα-Cα distance (Å)")
        ax.set_ylabel("Density")
        ax.set_title("Contact Distance: Important vs Unimportant Edges",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 2: Per-class mean distance
    ax = axes[1]
    classes = []
    imp_means = []
    unimp_means = []
    for c in range(n_classes):
        if imp_dists[c] and unimp_dists[c]:
            classes.append(f"EC{c+1}")
            imp_means.append(np.mean(imp_dists[c]))
            unimp_means.append(np.mean(unimp_dists[c]))

    if classes:
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, imp_means, width, label="Important",
               color="#C44E52", alpha=0.8)
        ax.bar(x + width/2, unimp_means, width, label="Unimportant",
               color="#4C72B0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Mean Cα-Cα distance (Å)")
        ax.set_title("Mean Contact Distance by EC Class", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_contact_distance_distribution.png")


def plot_sequence_separation_importance(explanations: list, graphs: list,
                                        n_classes: int, out_dir: str):
    """Sequence separation (|i-j|) of important vs unimportant edges.

    Short-range contacts (|i-j| < 5) = local backbone.
    Medium-range (5-20) = secondary structure.
    Long-range (> 20) = tertiary contacts, often functionally important.
    """
    from collections import defaultdict

    imp_sep = defaultdict(list)
    unimp_sep = defaultdict(list)

    for exp, graph in zip(explanations, graphs):
        if exp.edge_mask is None:
            continue
        c = exp.true_label
        ei = exp.edge_index
        emask = exp.edge_mask

        for e in range(ei.shape[1]):
            sep = abs(int(ei[0, e]) - int(ei[1, e]))
            if emask[e] > 0.5:
                imp_sep[c].append(sep)
            else:
                unimp_sep[c].append(sep)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Histogram
    ax = axes[0]
    all_imp = [s for seps in imp_sep.values() for s in seps]
    all_unimp = [s for seps in unimp_sep.values() for s in seps]
    if all_imp and all_unimp:
        bins = np.arange(0, 100, 2)
        ax.hist(all_imp, bins=bins, alpha=0.7, color="#C44E52", density=True,
                label=f"Important (n={len(all_imp)})")
        ax.hist(all_unimp, bins=bins, alpha=0.7, color="#4C72B0", density=True,
                label=f"Unimportant (n={len(all_unimp)})")
        ax.axvline(5, color="gray", ls=":", lw=1)
        ax.axvline(20, color="gray", ls="--", lw=1)
        ax.text(2, ax.get_ylim()[1]*0.9, "Local", fontsize=8, color="gray")
        ax.text(10, ax.get_ylim()[1]*0.9, "Secondary", fontsize=8, color="gray")
        ax.text(30, ax.get_ylim()[1]*0.9, "Tertiary", fontsize=8, color="gray")
        ax.set_xlabel("Sequence separation |i - j|")
        ax.set_ylabel("Density")
        ax.set_title("Sequence Separation of Important Contacts", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 2: Fraction of long-range contacts per class
    ax = axes[1]
    classes = []
    imp_lr_frac = []
    unimp_lr_frac = []
    for c in range(n_classes):
        if imp_sep[c] and unimp_sep[c]:
            classes.append(f"EC{c+1}")
            imp_arr = np.array(imp_sep[c])
            unimp_arr = np.array(unimp_sep[c])
            imp_lr_frac.append((imp_arr > 20).mean())
            unimp_lr_frac.append((unimp_arr > 20).mean())

    if classes:
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, imp_lr_frac, width, label="Important",
               color="#C44E52", alpha=0.8)
        ax.bar(x + width/2, unimp_lr_frac, width, label="Unimportant",
               color="#4C72B0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Fraction of long-range contacts (|i-j| > 20)")
        ax.set_title("Long-range Contact Enrichment", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_sequence_separation.png")


def plot_physicochemical_importance(explanations: list, graphs: list,
                                    n_classes: int, out_dir: str):
    """Physicochemical property importance per EC class.

    Shows which physicochemical properties (hydrophobicity, charge, volume, etc.)
    are most important for each enzyme class, derived from the feature mask.
    """
    from collections import defaultdict

    PHYSICO_NAMES = ["Hydrophobicity", "Charge", "Mol. Weight", "Volume",
                     "Polarity", "Flexibility", "Accessibility",
                     "Helix prop.", "Sheet prop.", "Turn prop."]
    PHYSICO_START = 20  # after one-hot
    PHYSICO_END = 30

    class_physico = defaultdict(list)

    for exp in explanations:
        if exp.feature_mask is None:
            continue
        c = exp.true_label
        physico_mask = exp.feature_mask[PHYSICO_START:PHYSICO_END]
        class_physico[c].append(physico_mask)

    fig, ax = plt.subplots(figsize=(12, 5))

    mat = np.zeros((n_classes, 10))
    for c in range(n_classes):
        if class_physico[c]:
            mat[c] = np.mean(class_physico[c], axis=0)

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(10))
    ax.set_xticklabels(PHYSICO_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"EC{c+1}" for c in range(n_classes)])
    plt.colorbar(im, ax=ax, label="Feature mask value")

    for i in range(n_classes):
        for j in range(10):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if mat[i,j] > mat.max()*0.6 else "black")

    ax.set_title("Physicochemical Property Importance by EC Class", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_physicochemical_importance.png")


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
