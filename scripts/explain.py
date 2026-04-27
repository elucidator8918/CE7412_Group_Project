#!/usr/bin/env python3
"""
CE7412 Enzyme Classification — Explainability Analysis

Runs GNNExplainer and Integrated Gradients on the trained GIN model,
computes fidelity metrics, builds per-class prototypes, and generates
all explainability figures.

Usage:
    python scripts/explain.py --config configs/full_power.yaml
    python scripts/explain.py --config configs/full_power.yaml --n-samples 100
    python scripts/explain.py --config configs/full_power.yaml --method gnnexplainer
    python scripts/explain.py --config configs/full_power.yaml --method ig
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EnzymeDataset
from src.explainability.gnn_explainer import GNNExplainerGraph
from src.explainability.integrated_gradients import IntegratedGradientsExplainer
from src.explainability.metrics import (
    compute_fidelity_metrics,
    compute_pyg_metrics,
    compute_sparsity,
    compute_stability,
)
from src.explainability.prototypes import ClassPrototypes
from src.explainability.visualization import (
    plot_aa_enrichment,
    plot_blob_aa_enrichment,
    plot_blob_assignments,
    plot_blob_importance,
    plot_blob_sasa_profiles,
    plot_blob_spatial_coherence,
    plot_blob_summary,
    plot_class_prototypes,
    plot_contact_distance_distribution,
    plot_edge_importance_examples,
    plot_fidelity_curves,
    plot_feature_group_importance,
    plot_gin_blob_overlap,
    plot_method_comparison,
    plot_physicochemical_importance,
    plot_position_importance,
    plot_sasa_vs_importance,
    plot_sequence_separation_importance,
    plot_spatial_clustering,
)
from src.models.gin import GINModel, SoftBlobGIN
from src.explainability.blob_analysis import (
    extract_blob_batch,
    compute_blob_importance,
    aggregate_blob_stats,
    compute_blob_aa_enrichment,
    compute_blob_sasa_profiles,
    compute_gin_blob_overlap,
)


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "explain.log", mode="w"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_samples(graphs: list, n_per_class: int, n_classes: int,
                   seed: int = 42) -> list:
    """Select stratified samples from test set for explanation."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, g in enumerate(graphs):
        by_class[int(g.y.item())].append(i)

    selected = []
    for c in range(n_classes):
        pool = by_class.get(c, [])
        rng.shuffle(pool)
        selected.extend(pool[:n_per_class])

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="CE7412 — GNN Explainability Analysis"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Total samples to explain (stratified by class)")
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "gnnexplainer", "ig"],
                        help="Which explainability method to run")
    parser.add_argument("--epochs", type=int, default=300,
                        help="GNNExplainer optimization epochs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to GIN checkpoint (default: auto-detect)")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "full_power.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(cfg["paths"]["log_dir"])

    # Output directory
    explain_dir = Path(cfg["paths"]["output_dir"]) / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = explain_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CE7412 — GNN Explainability Analysis")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Samples: {args.n_samples}")

    # ══════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA AND MODEL
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Loading dataset ---")
    dataset = EnzymeDataset(cfg).prepare()

    logger.info("\n--- Loading trained GIN model ---")
    gin_cfg = cfg["models"]["gin"]
    model = GINModel(
        in_ch=dataset.feat_dim, hidden=gin_cfg["hidden"],
        n_classes=dataset.n_classes, edge_dim=dataset.edge_dim,
        n_layers=gin_cfg["n_layers"], dropout=gin_cfg["dropout"],
    ).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint or str(
        Path(cfg["paths"]["checkpoint_dir"]) / "GIN_best.pt"
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"  Loaded checkpoint: {ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {n_params:,}")

    # Select samples for explanation (stratified)
    n_per_class = max(1, args.n_samples // dataset.n_classes)
    sample_idx = select_samples(dataset.test_graphs, n_per_class,
                                dataset.n_classes, seed=cfg["seed"])
    sample_graphs = [dataset.test_graphs[i] for i in sample_idx]
    sample_labels = np.array([int(g.y.item()) for g in sample_graphs])

    logger.info(f"  Selected {len(sample_graphs)} test proteins for explanation")
    for c in range(dataset.n_classes):
        n_c = (sample_labels == c).sum()
        logger.info(f"    EC{c+1}: {n_c} samples")

    # ══════════════════════════════════════════════════════════════════════
    # 2. GNNEXPLAINER
    # ══════════════════════════════════════════════════════════════════════
    gnnexp_results = None
    if args.method in ("all", "gnnexplainer"):
        logger.info("\n" + "=" * 50)
        logger.info("GNNExplainer — Optimization-based explanation")
        logger.info("=" * 50)

        explainer = GNNExplainerGraph(
            model=model,
            device=device,
            epochs=args.epochs,
            lr=0.01,
            lambda_size=0.07,
            lambda_ent=0.1,
            lambda_feat_size=0.01,
            lambda_feat_ent=0.1,
            init_bias=0.0,
        )

        gnnexp_results = explainer.explain_batch(sample_graphs, verbose=True)
        logger.info(f"  Explained {len(gnnexp_results)} proteins")

        # Summary statistics
        mean_sparsity = np.mean([
            compute_sparsity(r.edge_mask) for r in gnnexp_results
        ])
        mean_confidence = np.mean([r.masked_pred_prob for r in gnnexp_results])
        logger.info(f"  Mean sparsity: {mean_sparsity:.3f}")
        logger.info(f"  Mean masked confidence: {mean_confidence:.3f}")

        # Fidelity metrics
        logger.info("\n  Computing fidelity metrics...")
        edge_masks = [r.edge_mask for r in gnnexp_results]
        fidelity = compute_fidelity_metrics(
            model, sample_graphs, edge_masks, device,
            sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
        )

        logger.info("  Fidelity results:")
        fid_rows = []
        for fr in fidelity:
            logger.info(
                f"    sparsity={fr.sparsity:.2f}  "
                f"fid+={fr.fidelity_plus:.3f}  "
                f"fid-={fr.fidelity_minus:.3f}  "
                f"prob_fid+={fr.prob_fidelity_plus:.3f}  "
                f"prob_fid-={fr.prob_fidelity_minus:.3f}"
            )
            fid_rows.append({
                "sparsity": fr.sparsity,
                "fidelity_plus": fr.fidelity_plus,
                "fidelity_minus": fr.fidelity_minus,
                "prob_fidelity_plus": fr.prob_fidelity_plus,
                "prob_fidelity_minus": fr.prob_fidelity_minus,
            })
        pd.DataFrame(fid_rows).to_csv(
            explain_dir / "fidelity_metrics.csv", index=False
        )

        # Stability
        stability = compute_stability(gnnexp_results, sample_labels,
                                      dataset.n_classes)
        logger.info(f"  Stability (intra-class feature mask cosine sim):")
        for c, s in stability.items():
            logger.info(f"    EC{c+1}: {s:.3f}")

        # PyG-aligned metrics
        logger.info("\n  Computing PyG-aligned explanation metrics...")
        pyg_metrics = compute_pyg_metrics(
            model, sample_graphs, edge_masks, device
        )
        logger.info(f"  Unfaithfulness: {pyg_metrics['unfaithfulness']:.4f}")
        logger.info(f"  Characterization score: {pyg_metrics['characterization_score']:.4f}")

        # Save all metrics
        metrics_summary = {
            "mean_sparsity": mean_sparsity,
            "mean_masked_confidence": mean_confidence,
            **{f"stability_EC{c+1}": s for c, s in stability.items()},
            **pyg_metrics,
        }
        pd.DataFrame([metrics_summary]).to_csv(
            explain_dir / "explanation_metrics_summary.csv", index=False
        )

        # Visualizations
        logger.info("\n  Generating GNNExplainer figures...")
        plot_fidelity_curves(fidelity, fig_dir, method_name="GNNExplainer")
        plot_edge_importance_examples(
            gnnexp_results, sample_graphs, n_per_class,
            dataset.n_classes, fig_dir
        )

        # Class prototypes
        logger.info("  Building class prototypes...")
        proto_builder = ClassPrototypes(n_classes=dataset.n_classes)
        prototypes = proto_builder.build(gnnexp_results, sample_labels)
        plot_feature_group_importance(prototypes, fig_dir)
        plot_position_importance(prototypes, fig_dir)
        plot_class_prototypes(prototypes, fig_dir)

        # Save prototype data
        proto_dict = proto_builder.to_dict(prototypes)
        with open(explain_dir / "class_prototypes.json", "w") as f:
            json.dump(proto_dict, f, indent=2)

        # Biological validation plots
        logger.info("  Generating biological validation figures...")
        plot_aa_enrichment(gnnexp_results, sample_graphs,
                           dataset.n_classes, fig_dir)
        plot_sasa_vs_importance(gnnexp_results, sample_graphs,
                                dataset.n_classes, fig_dir)
        plot_spatial_clustering(gnnexp_results, sample_graphs,
                                dataset.n_classes, fig_dir)
        plot_contact_distance_distribution(gnnexp_results, sample_graphs,
                                           dataset.n_classes, fig_dir)
        plot_sequence_separation_importance(gnnexp_results, sample_graphs,
                                            dataset.n_classes, fig_dir)
        plot_physicochemical_importance(gnnexp_results, sample_graphs,
                                        dataset.n_classes, fig_dir)

        # Save per-sample edge importance summary
        edge_summary = []
        for r in gnnexp_results:
            edge_summary.append({
                "protein_idx": r.protein_idx,
                "true_label": r.true_label,
                "predicted_label": r.predicted_label,
                "predicted_prob": r.predicted_prob,
                "masked_pred_prob": r.masked_pred_prob,
                "n_nodes": r.n_nodes,
                "n_edges": r.n_edges,
                "mean_edge_importance": float(r.edge_mask.mean()),
                "sparsity": compute_sparsity(r.edge_mask),
            })
        pd.DataFrame(edge_summary).to_csv(
            explain_dir / "edge_importance_samples.csv", index=False
        )

    # ══════════════════════════════════════════════════════════════════════
    # 3. INTEGRATED GRADIENTS
    # ══════════════════════════════════════════════════════════════════════
    ig_results = None
    if args.method in ("all", "ig"):
        logger.info("\n" + "=" * 50)
        logger.info("Integrated Gradients — Attribution-based explanation")
        logger.info("=" * 50)

        # Compute dataset mean for baseline
        all_feats = torch.stack([g.x.float().mean(dim=0)
                                 for g in dataset.train_graphs])
        dataset_mean = all_feats.mean(dim=0)

        ig_explainer = IntegratedGradientsExplainer(
            model=model, device=device, n_steps=50, baseline_type="zero"
        )

        ig_results = ig_explainer.explain_batch(
            sample_graphs, dataset_mean=dataset_mean, verbose=True
        )
        logger.info(f"  IG explained {len(ig_results)} proteins")

        # Feature importance aggregation
        feat_attrs = np.array([r.feature_attributions for r in ig_results])
        mean_feat_attr = feat_attrs.mean(axis=0)

        # Save feature importance
        feat_df = pd.DataFrame({
            "feature_idx": range(len(mean_feat_attr)),
            "importance": mean_feat_attr,
        })
        feat_df.to_csv(explain_dir / "ig_feature_importance.csv", index=False)

        # Fidelity using IG edge attributions
        logger.info("  Computing IG fidelity metrics...")
        ig_edge_masks = [r.edge_attributions for r in ig_results]
        ig_fidelity = compute_fidelity_metrics(
            model, sample_graphs, ig_edge_masks, device,
            sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
        )

        ig_fid_rows = []
        for fr in ig_fidelity:
            logger.info(
                f"    sparsity={fr.sparsity:.2f}  "
                f"fid+={fr.fidelity_plus:.3f}  "
                f"fid-={fr.fidelity_minus:.3f}"
            )
            ig_fid_rows.append({
                "sparsity": fr.sparsity,
                "fidelity_plus": fr.fidelity_plus,
                "fidelity_minus": fr.fidelity_minus,
                "prob_fidelity_plus": fr.prob_fidelity_plus,
                "prob_fidelity_minus": fr.prob_fidelity_minus,
            })
        pd.DataFrame(ig_fid_rows).to_csv(
            explain_dir / "ig_fidelity_metrics.csv", index=False
        )

    # ══════════════════════════════════════════════════════════════════════
    # 4. METHOD COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    if gnnexp_results and ig_results:
        logger.info("\n" + "=" * 50)
        logger.info("Method Comparison — GNNExplainer vs Integrated Gradients")
        logger.info("=" * 50)

        plot_method_comparison(gnnexp_results, ig_results, fig_dir)

        # Compare fidelity
        logger.info("  Fidelity comparison (at sparsity=0.2):")
        gnn_fid_02 = next((f for f in fidelity if f.sparsity == 0.2), None)
        ig_fid_02 = next((f for f in ig_fidelity if f.sparsity == 0.2), None)
        if gnn_fid_02 and ig_fid_02:
            logger.info(
                f"    GNNExplainer: fid+={gnn_fid_02.fidelity_plus:.3f}  "
                f"fid-={gnn_fid_02.fidelity_minus:.3f}"
            )
            logger.info(
                f"    IG:           fid+={ig_fid_02.fidelity_plus:.3f}  "
                f"fid-={ig_fid_02.fidelity_minus:.3f}"
            )

    # ══════════════════════════════════════════════════════════════════════
    # 5. SOFTBLOBGIN INTERPRETABILITY
    # ══════════════════════════════════════════════════════════════════════
    blob_ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "SoftBlobGIN_best.pt"
    if blob_ckpt.exists() and "soft_blob_gin" in cfg["models"]:
        logger.info("\n" + "=" * 50)
        logger.info("SoftBlobGIN — Built-in blob interpretability")
        logger.info("=" * 50)

        sb_cfg = cfg["models"]["soft_blob_gin"]
        blob_model = SoftBlobGIN(
            in_ch=dataset.feat_dim, hidden=sb_cfg["hidden"],
            n_classes=dataset.n_classes, edge_dim=dataset.edge_dim,
            n_blobs=sb_cfg["n_blobs"], n_layers=sb_cfg["n_layers"],
            dropout=sb_cfg["dropout"],
            tau_start=sb_cfg["tau_start"], tau_end=sb_cfg["tau_end"],
        ).to(device)
        blob_state = torch.load(str(blob_ckpt), map_location=device, weights_only=True)
        blob_model.load_state_dict(blob_state)
        blob_model.eval()
        # Set tau to final (inference) value
        blob_model._current_tau = sb_cfg["tau_end"]
        logger.info(f"  Loaded SoftBlobGIN checkpoint: {blob_ckpt}")
        logger.info(f"  n_blobs={sb_cfg['n_blobs']}, tau={blob_model._current_tau}")

        # Extract blob assignments
        logger.info("  Extracting blob assignments...")
        blob_results = extract_blob_batch(blob_model, sample_graphs, device,
                                          verbose=True)

        # Compute blob importance (which blobs matter most per protein)
        logger.info("  Computing blob importance...")
        blob_importances = []
        for i, data in enumerate(sample_graphs):
            imp = compute_blob_importance(blob_model, data, device,
                                          n_blobs=sb_cfg["n_blobs"])
            blob_importances.append(imp)
            if (i + 1) % 20 == 0:
                logger.info(f"    Blob importance {i+1}/{len(sample_graphs)}")

        # Aggregate stats per class
        blob_stats = aggregate_blob_stats(blob_results, sample_labels,
                                          dataset.n_classes)

        # Log summary
        for c in range(dataset.n_classes):
            if blob_stats.get(c) and blob_stats[c] is not None:
                s = blob_stats[c]
                sizes_str = ", ".join(f"{v:.0f}" for v in s["mean_blob_sizes"])
                logger.info(f"  EC{c+1}: mean blob sizes = [{sizes_str}]")

        # Generate blob figures
        logger.info("  Generating SoftBlobGIN figures...")
        n_blobs = sb_cfg["n_blobs"]
        plot_blob_assignments(blob_results, dataset.n_classes, fig_dir)
        plot_blob_spatial_coherence(blob_results, dataset.n_classes, fig_dir)
        plot_blob_importance(blob_importances, sample_labels,
                             dataset.n_classes, n_blobs, fig_dir)
        plot_blob_summary(blob_results, blob_importances, sample_labels,
                          dataset.n_classes, n_blobs, fig_dir)

        # Blob-level biological validation
        logger.info("  Blob-level biological validation...")

        # Blob AA enrichment: which amino acids does each blob capture?
        blob_aa_enrich = compute_blob_aa_enrichment(
            blob_results, sample_labels, dataset.n_classes
        )
        plot_blob_aa_enrichment(blob_aa_enrich, dataset.n_classes, n_blobs, fig_dir)

        # Blob SASA profiles: which blobs are buried vs exposed?
        blob_sasa = compute_blob_sasa_profiles(
            blob_results, sample_labels, dataset.n_classes
        )
        plot_blob_sasa_profiles(blob_sasa, dataset.n_classes, n_blobs, fig_dir)

        # Log blob SASA summary
        for c in range(dataset.n_classes):
            if blob_sasa.get(c) is not None:
                sasa_str = ", ".join(f"{v:.3f}" for v in blob_sasa[c])
                logger.info(f"  EC{c+1} blob SASA: [{sasa_str}]")

        # Cross-model comparison: GIN GNNExplainer vs SoftBlobGIN blobs
        if gnnexp_results is not None:
            logger.info("  Cross-model comparison: GIN vs SoftBlobGIN...")
            overlaps = compute_gin_blob_overlap(gnnexp_results, blob_results)
            plot_gin_blob_overlap(overlaps, sample_labels,
                                  dataset.n_classes, fig_dir)

            mean_jaccard = np.nanmean([j for j, _ in overlaps])
            logger.info(f"  Mean Jaccard overlap (GIN important vs SoftBlobGIN top blob): {mean_jaccard:.3f}")

        # Save blob data
        blob_summary_rows = []
        for br, imp in zip(blob_results, blob_importances):
            blob_summary_rows.append({
                "protein_idx": br.protein_idx,
                "true_label": br.true_label,
                "predicted_label": br.predicted_label,
                "predicted_prob": br.predicted_prob,
                "n_nodes": br.n_nodes,
                "blob_sizes": br.blob_sizes.tolist(),
                "blob_importance": imp.tolist(),
            })
        with open(explain_dir / "softblobgin_blob_analysis.json", "w") as f:
            json.dump(blob_summary_rows, f, indent=2)

        logger.info(f"  SoftBlobGIN analysis complete.")
    else:
        logger.info("\n  SoftBlobGIN checkpoint not found, skipping blob analysis.")

    # ══════════════════════════════════════════════════════════════════════
    # 6. SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {explain_dir}")
    logger.info(f"Figures saved to: {fig_dir}")
    logger.info("\nGenerated files:")
    for f in sorted(explain_dir.rglob("*")):
        if f.is_file():
            logger.info(f"  {f.relative_to(explain_dir)}")


if __name__ == "__main__":
    main()
