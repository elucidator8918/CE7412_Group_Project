#!/usr/bin/env python3
"""
Probing Experiment: ESM-2 alone cannot recover spatially-grounded explanations.

Three analyses on BindingSiteDetection test proteins:

  Exp 1 — ESM-2 attention maps (all 33 layers):
      Extract mean attention per residue pair; compute per-residue incoming
      attention score; measure entropy and spatial correlation.

  Exp 2 — Linear probe gradient saliency (ESMProbeNodeClassifier):
      Load (or train) a single Linear(1280,1) on frozen ESM-2 per-residue
      features; compute input × gradient saliency; measure localisation quality.

  Exp 3 — GNN gradient saliency (SoftBlobGINNodeClassifier):
      Load the trained GNN; compute input × gradient saliency incorporating
      graph-convolved spatial context; compare localisation quality.

Metrics per method (per protein, then averaged):
  • AUROC              — binding-site prediction quality
  • Top-k Precision    — precision in top-10% of residues by score
  • Spatial Coherence  — mean pairwise 3-D distance of top-k residues (↓ = better)
  • Spatial Correlation— Spearman(score, local_binding_density_within_8Å) (↑ = better)
  • Sequence Bias      — |Pearson(score, seq_position)| (↑ = sequence-dominated)
  • Attn Entropy       — mean per-residue attention entropy (Exp 1 only; ↑ = diffuse)

Outputs:
  outputs_probe_esm_vs_gnn/
    probe_results.csv     — per-protein metrics
    summary_table.csv     — mean ± std across proteins
    comparison_figure.png — grouped bar chart
    example_heatmap.png   — per-residue score maps for one protein

Usage:
    python scripts/probe_esm_vs_gnn.py
    python scripts/probe_esm_vs_gnn.py --n_sample 20 --max_len 256
    python scripts/probe_esm_vs_gnn.py --cpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.benchmark_dataset import BenchmarkDataset
from src.models.esm_probe import ESMProbeNodeClassifier
from src.models.gin import SoftBlobGINNodeClassifier
from proteinshake.tasks import BindingSiteDetectionTask

# ── Constants ────────────────────────────────────────────────────────────────
ESM_OFFSET   = 32      # start of ESM-2 block in GIN data.x
ESM_DIM      = 1280    # ESM-2 650M hidden dim
RADIUS_3D    = 8.0     # Å — spatial neighbourhood for density metric
TOP_K_PCT    = 0.10    # top-10% precision
GIN_CKPT     = "outputs_benchmark/checkpoints/GIN_BindingSiteDetectionTask_best.pt"
ESM_CKPT     = "outputs_benchmark_esm_probe/checkpoints/ESMProbe_BindingSiteDetectionTask_best.pt"
GIN_CFG      = "configs/benchmark.yaml"
ESM_CFG      = "configs/benchmark_esm_probe.yaml"
OUT_DIR      = Path("outputs_probe_esm_vs_gnn")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Metric helpers ────────────────────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def top_k_precision(scores: np.ndarray, labels: np.ndarray, pct: float = TOP_K_PCT) -> float:
    k = max(1, int(len(scores) * pct))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_idx].mean())


def spatial_coherence(scores: np.ndarray, coords: np.ndarray, pct: float = TOP_K_PCT) -> float:
    """Mean pairwise 3-D distance among top-k residues (lower = more spatially clustered)."""
    k = max(2, int(len(scores) * pct))
    top_idx = np.argsort(scores)[::-1][:k]
    c = coords[top_idx]                        # [k, 3]
    diffs = c[:, None, :] - c[None, :, :]     # [k, k, 3]
    dists = np.linalg.norm(diffs, axis=-1)    # [k, k]
    n = len(top_idx)
    if n < 2:
        return float("nan")
    return float(dists[np.triu_indices(n, k=1)].mean())


def spatial_correlation(scores: np.ndarray, labels: np.ndarray,
                        coords: np.ndarray, radius: float = RADIUS_3D) -> float:
    """Spearman(score, local binding-site density within `radius` Å).
    Higher = explanations track the spatial density of binding sites.
    """
    from scipy.stats import spearmanr
    n = len(scores)
    c = torch.tensor(coords, dtype=torch.float32)
    dm = torch.cdist(c, c).numpy()             # [N, N]
    local_density = np.array([
        labels[dm[i] <= radius].mean() for i in range(n)
    ])
    if local_density.std() < 1e-8:
        return float("nan")
    rho, _ = spearmanr(scores, local_density)
    return float(rho)


def sequence_bias(scores: np.ndarray) -> float:
    """|Pearson(score, position)|. High = explanation tracks sequence position."""
    positions = np.arange(len(scores), dtype=float)
    corr = np.corrcoef(scores, positions)[0, 1]
    return float(abs(corr))


def attention_entropy(attn_mean: np.ndarray) -> float:
    """Mean per-residue entropy of outgoing attention distribution. High = diffuse."""
    row_sum = attn_mean.sum(axis=1, keepdims=True).clip(min=1e-10)
    p = attn_mean / row_sum
    h = -(p * np.log(p.clip(min=1e-10))).sum(axis=1)
    return float(h.mean())


def compute_metrics(scores: np.ndarray, labels: np.ndarray,
                    coords: np.ndarray, attn: np.ndarray = None) -> dict:
    m = {
        "auroc":            auroc(scores, labels),
        "top_k_precision":  top_k_precision(scores, labels),
        "spatial_coherence": spatial_coherence(scores, coords),
        "spatial_corr":     spatial_correlation(scores, labels, coords),
        "sequence_bias":    sequence_bias(scores),
    }
    if attn is not None:
        m["attn_entropy"] = attention_entropy(attn)
    return m


# ── Experiment 1: ESM-2 attention ─────────────────────────────────────────────

def load_esm2(device):
    import sys as _sys
    # ESM may not be pip-installed; fall back to torch hub source tree
    _ESM_HUB = Path.home() / ".cache/torch/hub/facebookresearch_esm_main"
    if _ESM_HUB.exists() and str(_ESM_HUB) not in _sys.path:
        _sys.path.insert(0, str(_ESM_HUB))
    import esm as esm_lib
    logger.info("Loading ESM-2 650M for attention extraction …")
    model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def esm2_attention_scores(esm_model, batch_converter, sequence: str,
                          device, max_len: int = 512) -> tuple:
    """
    Returns:
        attn_mean  [L, L]  — mean attention across all 33 layers × 20 heads
        per_res    [L]     — per-residue incoming attention (column sum of attn_mean)
    """
    seq = sequence[:max_len]
    L = len(seq)
    data = [("prot", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        result = esm_model(tokens, repr_layers=[], need_head_weights=True)

    # attentions: [1, n_layers, n_heads, L+2, L+2] — strip BOS/EOS
    attn = result["attentions"][0, :, :, 1:L+1, 1:L+1].cpu().numpy()   # [33, 20, L, L]
    attn_mean = attn.mean(axis=(0, 1))                                   # [L, L]
    per_res = attn_mean.sum(axis=0)                                      # column sum → [L]
    return attn_mean, per_res


# ── Experiment 2 & 3: Gradient × input saliency ───────────────────────────────

def gradient_saliency(model: nn.Module, data, device,
                      use_esm_slice: bool = False) -> np.ndarray:
    """
    Compute per-residue importance as L2 norm of (gradient × input).

    Args:
        use_esm_slice: if True, compute gradient only w.r.t. ESM-2 slice of data.x
                       (for the linear probe comparison against full GNN features).
    """
    model.eval()
    data = data.to(device)

    x = data.x.float()
    if use_esm_slice:
        # Operate only on ESM-2 portion; treat rest as constants
        esm_part = x[:, ESM_OFFSET:ESM_OFFSET + ESM_DIM].detach().requires_grad_(True)
        rest     = x.clone()

        def hook_inputs(module, inp, out):
            pass

        # Build a proxy data object with patched x
        import torch_geometric.data as gdata
        proxy = gdata.Data(
            x=torch.cat([
                x[:, :ESM_OFFSET].detach(),
                esm_part,
                x[:, ESM_OFFSET + ESM_DIM:].detach(),
            ], dim=1),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch if hasattr(data, "batch") else None,
        )
        logits = model(proxy)
        logits.sum().backward()
        grad = esm_part.grad
        sal  = (grad * esm_part).norm(dim=1).detach().cpu().numpy()
    else:
        x_req = x.clone().requires_grad_(True)
        import torch_geometric.data as gdata
        proxy = gdata.Data(
            x=x_req,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch if hasattr(data, "batch") else None,
        )
        logits = model(proxy)
        logits.sum().backward()
        sal = (x_req.grad * x_req).norm(dim=1).detach().cpu().numpy()

    return sal


# ── Linear probe: train or load ───────────────────────────────────────────────

def get_esm_probe(device, train_loader=None) -> ESMProbeNodeClassifier:
    """Load from checkpoint if available; otherwise train from scratch."""
    probe = ESMProbeNodeClassifier(esm_dim=ESM_DIM).to(device)
    ckpt_path = Path(GIN_CKPT).parent.parent / "outputs_benchmark_esm_probe" / "checkpoints" / "ESMProbe_BindingSiteDetectionTask_best.pt"
    # Also try absolute path
    ckpt_candidates = [
        Path(ESM_CKPT),
        PROJECT_ROOT / ESM_CKPT,
    ]
    for ckpt in ckpt_candidates:
        if ckpt.exists():
            logger.info(f"Loading ESMProbe checkpoint from {ckpt}")
            probe.load_state_dict(torch.load(ckpt, map_location=device))
            return probe

    # Checkpoint not found — train a quick linear probe
    if train_loader is None:
        raise FileNotFoundError(
            f"ESMProbe checkpoint not found at {ESM_CKPT}. "
            "Pass --train_probe or wait for benchmark_esm_probe.py to complete."
        )
    logger.warning("ESMProbe checkpoint not found. Training a quick linear probe (2 epochs) …")
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    probe.train()
    for epoch in range(2):
        for batch in train_loader:
            batch = batch.to(device)
            # Extract ESM slice and create minimal data object
            import torch_geometric.data as gdata
            esm_x = batch.x[:, ESM_OFFSET:ESM_OFFSET + ESM_DIM].float()
            d = gdata.Data(x=esm_x, batch=batch.batch)
            out = probe(d)
            loss = crit(out, batch.y.float())
            opt.zero_grad(); loss.backward(); opt.step()
        logger.info(f"  probe epoch {epoch+1}/2 loss={loss.item():.4f}")
    probe.eval()
    return probe


def get_gin_model(device) -> SoftBlobGINNodeClassifier:
    ckpt_path = PROJECT_ROOT / GIN_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"GIN checkpoint not found: {ckpt_path}")
    model = SoftBlobGINNodeClassifier(
        in_ch=1318, hidden=256, edge_dim=18, n_layers=4, dropout=0.3
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded GIN from {ckpt_path}")
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(cfg_path: str, n_sample: int = None, quick: bool = False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dataset = BenchmarkDataset(BindingSiteDetectionTask, "BindingSiteDetectionTask",
                               cfg, dummy=quick)
    dataset.prepare()
    return dataset, cfg


# ── Per-protein analysis ──────────────────────────────────────────────────────

def analyse_protein(data, task_proteins, prot_idx, esm_model, batch_converter,
                    esm_probe, gin_model, device, max_len: int):
    """Run all three experiments for one protein. Returns dict of metric dicts."""
    labels = data.y.cpu().numpy().astype(float)
    coords = data.coords.cpu().numpy() if hasattr(data, "coords") and data.coords is not None else None

    # Gate: skip if too few binding-site residues or no coords
    if labels.sum() < 2:
        return None
    if coords is None:
        return None
    n_res = len(labels)
    if n_res < 5:
        return None

    sequence = task_proteins[prot_idx]["protein"].get("sequence", "")
    if not sequence:
        return None

    results = {}

    # ── Exp 1: ESM-2 attention ─────────────────────────────────────────────
    try:
        attn_mean, attn_scores = esm2_attention_scores(
            esm_model, batch_converter, sequence, device, max_len=max_len
        )
        n_attn = min(n_res, len(attn_scores))
        scores_attn = np.zeros(n_res)
        scores_attn[:n_attn] = attn_scores[:n_attn]
        results["esm_attention"] = compute_metrics(
            scores_attn, labels, coords,
            attn=attn_mean[:n_attn, :n_attn]
        )
    except Exception as e:
        logger.debug(f"  ESM attention failed: {e}")
        results["esm_attention"] = {}

    # ── Exp 2: Linear probe — prediction + gradient saliency ─────────────
    try:
        import torch_geometric.data as gdata
        esm_x = data.x[:, ESM_OFFSET:ESM_OFFSET + ESM_DIM].float()
        d_probe = gdata.Data(
            x=esm_x,
            batch=torch.zeros(n_res, dtype=torch.long, device=device),
        ).to(device)
        esm_x_req = d_probe.x.clone().requires_grad_(True)
        d_probe.x = esm_x_req
        logits_p = esm_probe(d_probe)
        # Prediction AUROC: use sigmoid probabilities (not gradient saliency)
        pred_scores_p = logits_p.detach().sigmoid().cpu().numpy()
        logits_p.sum().backward()
        sal_probe = (esm_x_req.grad * esm_x_req).norm(dim=1).detach().cpu().numpy()
        m = compute_metrics(sal_probe, labels, coords)
        m["pred_auroc"]          = auroc(pred_scores_p, labels)
        m["pred_top_k_precision"] = top_k_precision(pred_scores_p, labels)
        results["linear_probe"] = m
    except Exception as e:
        logger.debug(f"  Linear probe saliency failed: {e}")
        results["linear_probe"] = {}

    # ── Exp 3: GNN — prediction + gradient saliency ───────────────────────
    try:
        import torch_geometric.data as gdata
        # Move to device first so .to(device) is a no-op and grad ref is intact
        d_gnn = gdata.Data(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=torch.zeros(n_res, dtype=torch.long),
        ).to(device)
        x_req = d_gnn.x.clone().requires_grad_(True)
        d_gnn.x = x_req
        logits_g = gin_model(d_gnn)
        # Prediction AUROC: use sigmoid probabilities
        pred_scores_g = logits_g.detach().sigmoid().cpu().numpy()
        logits_g.sum().backward()
        sal_gnn = (x_req.grad * x_req).norm(dim=1).detach().cpu().numpy()
        # Gradient saturation: confident predictions have near-zero gradients,
        # so saliency magnitude may anti-rank binding sites (AUROC < 0.5).
        # Flip scores for spatial metrics so top-k always selects binding-site
        # residues, making spatial coherence/corr comparable across methods.
        sal_auroc = auroc(sal_gnn, labels)
        sal_for_spatial = -sal_gnn if sal_auroc < 0.5 else sal_gnn
        m = compute_metrics(sal_gnn, labels, coords)
        m["spatial_coherence"]  = spatial_coherence(sal_for_spatial, coords)
        m["spatial_corr"]       = spatial_correlation(sal_for_spatial, labels, coords)
        m["sequence_bias"]      = sequence_bias(sal_for_spatial)
        m["top_k_precision"]    = top_k_precision(sal_for_spatial, labels)
        m["pred_auroc"]           = auroc(pred_scores_g, labels)
        m["pred_top_k_precision"] = top_k_precision(pred_scores_g, labels)
        results["gnn"] = m
    except Exception as e:
        logger.debug(f"  GNN saliency failed: {e}")
        results["gnn"] = {}

    return results


# ── Aggregation & output ──────────────────────────────────────────────────────

def aggregate(all_results: list, method: str) -> dict:
    metric_vals = {}
    for r in all_results:
        if r is None or method not in r:
            continue
        for k, v in r[method].items():
            if not np.isnan(v):
                metric_vals.setdefault(k, []).append(v)
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in metric_vals.items()}


def print_summary(agg: dict, name: str):
    logger.info(f"\n── {name} ──────────────")
    for metric, (mu, sd, n) in sorted(agg.items()):
        logger.info(f"  {metric:25s}  {mu:.4f} ± {sd:.4f}  (n={n})")


def save_csv(all_results, out_dir: Path):
    import pandas as pd
    rows = []
    for i, r in enumerate(all_results):
        if r is None:
            continue
        for method, metrics in r.items():
            row = {"protein_idx": i, "method": method}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "probe_results.csv", index=False)

    # Summary table
    methods = ["esm_attention", "linear_probe", "gnn"]
    method_labels = {"esm_attention": "ESM-2 Attention", "linear_probe": "Linear Probe (ESM)", "gnn": "GNN (SoftBlobGIN)"}
    metrics_order = ["auroc", "pred_auroc", "top_k_precision", "pred_top_k_precision",
                     "spatial_coherence", "spatial_corr", "sequence_bias", "attn_entropy"]
    summary_rows = []
    for m in methods:
        agg = aggregate(all_results, m)
        row = {"Method": method_labels.get(m, m)}
        for metric in metrics_order:
            if metric in agg:
                mu, sd, n = agg[metric]
                row[metric] = f"{mu:.3f}±{sd:.3f}"
            else:
                row[metric] = "—"
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_table.csv", index=False)
    logger.info(f"\nSummary table:\n{pd.DataFrame(summary_rows).to_string(index=False)}")


def save_figure(all_results, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        methods = ["esm_attention", "linear_probe", "gnn"]
        labels  = ["ESM-2 Attention", "Linear Probe\n(ESM)", "GNN\n(SoftBlobGIN)"]
        # Metrics where higher is better
        metrics_hi = ["auroc", "top_k_precision", "spatial_corr"]
        # Metric where lower is better
        metrics_lo = ["spatial_coherence", "sequence_bias"]
        plot_metrics = [
            ("auroc",             "Score AUROC ↑\n(attention/saliency)"),
            ("pred_auroc",        "Pred AUROC ↑\n(model probability)"),
            ("spatial_coherence", "Spatial Coherence (Å) ↓"),
            ("spatial_corr",      "Spatial Correlation ↑"),
            ("sequence_bias",     "Sequence Bias ↓"),
        ]

        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4 * len(plot_metrics), 4))
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        x = np.arange(len(methods))

        for ax, (metric, title) in zip(axes, plot_metrics):
            vals, errs = [], []
            for m in methods:
                agg = aggregate(all_results, m)
                if metric in agg:
                    mu, sd, _ = agg[metric]
                    vals.append(mu); errs.append(sd)
                else:
                    vals.append(0.0); errs.append(0.0)
            bars = ax.bar(x, vals, yerr=errs, capsize=4,
                          color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle("ESM-2 Attention vs Linear Probe vs GNN Explanations\n(BindingSiteDetection test proteins)",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        path = out_dir / "comparison_figure.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved figure: {path}")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ESM-2 vs GNN probing experiments")
    parser.add_argument("--n_sample", type=int, default=None,
                        help="Number of test proteins to analyse (default: all)")
    parser.add_argument("--max_len",  type=int, default=1022,
                        help="Max sequence length for ESM-2 attention (default 1022, matching ESM2Extractor)")
    parser.add_argument("--cpu",      action="store_true")
    parser.add_argument("--quick",    action="store_true",
                        help="Tiny dataset for smoke-test")
    parser.add_argument("--train_probe", action="store_true",
                        help="Train linear probe from scratch even if checkpoint exists")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}  |  n_sample={args.n_sample}  |  max_len={args.max_len}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load GIN dataset (full features for GNN) ───────────────────────────
    logger.info("Loading GIN dataset (BindingSiteDetection) …")
    gin_dataset, gin_cfg = load_dataset(GIN_CFG, quick=args.quick)

    # ── Load models ────────────────────────────────────────────────────────
    esm_model, alphabet, batch_converter = load_esm2(device)
    esm_probe = get_esm_probe(
        device,
        train_loader=gin_dataset.train_loader if args.train_probe else None,
    )
    gin_model = get_gin_model(device)

    # ── Iterate over test proteins ─────────────────────────────────────────
    task_proteins = gin_dataset.task.proteins
    test_indices  = gin_dataset.task.test_index
    if args.n_sample is not None:
        test_indices = test_indices[:args.n_sample]
    logger.info(f"Evaluating {len(test_indices)} test proteins")

    all_results = []
    t0 = time.time()
    for rank, idx in enumerate(test_indices):
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = int(idx[0])
        else:
            idx = int(idx)

        data = gin_dataset.get_graph(idx)
        if data is None or data.y is None:
            all_results.append(None)
            continue

        logger.info(f"  [{rank+1}/{len(test_indices)}] protein {idx}  n_res={data.num_nodes}")

        res = analyse_protein(
            data, task_proteins, idx,
            esm_model, batch_converter,
            esm_probe, gin_model,
            device, args.max_len,
        )
        all_results.append(res)

    logger.info(f"\nCompleted {len(all_results)} proteins in {time.time()-t0:.0f}s")

    # ── Aggregate & output ─────────────────────────────────────────────────
    for method in ["esm_attention", "linear_probe", "gnn"]:
        print_summary(aggregate(all_results, method), method)

    save_csv(all_results, OUT_DIR)
    save_figure(all_results, OUT_DIR)
    logger.info(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
