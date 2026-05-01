"""
SoftBlobGIN blob-to-domain mapping analysis.

Fetches Pfam/CATH domain annotations for proteins via PDBe SIFTS API,
computes overlap between learned blob assignments and annotated domains,
and generates PyMOL visualization scripts for case studies.

Deliverable for TODO 2:
  - Quantitative domain-overlap table (Jaccard, ARI)
  - Blob importance vs functional relevance correlation
  - 3D visualization scripts for case study proteins
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EC_NAMES = [
    "Oxidoreductase", "Transferase", "Hydrolase",
    "Lyase", "Isomerase", "Ligase", "Translocase"
]


# ============================================================================
# Domain annotation fetching via PDBe SIFTS API
# ============================================================================

@dataclass
class DomainAnnotation:
    """Domain annotations for a single protein chain."""
    pdb_id: str = ""
    chain_id: str = ""
    n_residues: int = 0

    # Pfam domains: list of (domain_id, name, start, end)
    pfam_domains: list = field(default_factory=list)

    # CATH domains: list of (domain_id, name, start, end)
    cath_domains: list = field(default_factory=list)

    # Active site residues (from PDBe)
    active_site_residues: list = field(default_factory=list)

    # Per-residue domain label: [N] array, -1 = no domain
    residue_domain_labels: np.ndarray = None


def fetch_domain_annotations(pdb_id: str, chain_id: str = "A",
                              cache_dir: str = None) -> DomainAnnotation:
    """Fetch Pfam and CATH domain annotations from PDBe SIFTS API.

    Uses https://www.ebi.ac.uk/pdbe/api/ endpoints.
    Results are cached to disk to avoid repeated API calls.
    """
    import urllib.request
    import urllib.error

    ann = DomainAnnotation(pdb_id=pdb_id.lower(), chain_id=chain_id)

    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"{pdb_id.lower()}_{chain_id}_domains.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            ann.pfam_domains = [tuple(d) for d in cached.get("pfam", [])]
            ann.cath_domains = [tuple(d) for d in cached.get("cath", [])]
            ann.active_site_residues = cached.get("active_sites", [])
            return ann

    pdb_lower = pdb_id.lower()
    base_url = "https://www.ebi.ac.uk/pdbe/api"

    # Fetch Pfam domains
    try:
        url = f"{base_url}/mappings/pfam/{pdb_lower}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if pdb_lower in data:
            for pfam_id, info in data[pdb_lower].get("Pfam", {}).items():
                for mapping in info.get("mappings", []):
                    if mapping.get("chain_id", "").upper() == chain_id.upper():
                        ann.pfam_domains.append((
                            pfam_id,
                            info.get("identifier", ""),
                            mapping.get("start", {}).get("residue_number", 0),
                            mapping.get("end", {}).get("residue_number", 0),
                        ))
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        pass

    # Fetch CATH domains
    try:
        url = f"{base_url}/mappings/cath/{pdb_lower}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if pdb_lower in data:
            for cath_id, info in data[pdb_lower].get("CATH", {}).items():
                for mapping in info.get("mappings", []):
                    if mapping.get("chain_id", "").upper() == chain_id.upper():
                        ann.cath_domains.append((
                            cath_id,
                            info.get("name", ""),
                            mapping.get("start", {}).get("residue_number", 0),
                            mapping.get("end", {}).get("residue_number", 0),
                        ))
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        pass

    # Fetch active/binding site residues via PDB -> UniProt -> features
    try:
        # Step 1: Get UniProt ID from PDB mapping
        url = f"{base_url}/mappings/uniprot/{pdb_lower}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            mapping_data = json.loads(resp.read().decode())

        if pdb_lower in mapping_data:
            uniprot_entries = mapping_data[pdb_lower].get("UniProt", {})
            # Find the UniProt entry matching our chain
            uniprot_id = None
            pdb_offset = 0
            for uid, info in uniprot_entries.items():
                for m in info.get("mappings", []):
                    if m.get("chain_id", "").upper() == chain_id.upper():
                        uniprot_id = uid
                        # PDB residue numbering offset
                        pdb_start = m.get("start", {}).get("residue_number", 1)
                        unp_start = m.get("unp_start", 1)
                        pdb_offset = pdb_start - unp_start
                        break
                if uniprot_id:
                    break

            # Step 2: Fetch UniProt features
            if uniprot_id:
                url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    uni_data = json.loads(resp.read().decode())

                for feat in uni_data.get("features", []):
                    ftype = feat.get("type", "")
                    if ftype in ("Active site", "Binding site", "Site"):
                        loc = feat.get("location", {})
                        start = loc.get("start", {}).get("value")
                        end = loc.get("end", {}).get("value")
                        if start is not None:
                            # Convert UniProt numbering to PDB numbering
                            for r in range(start, (end or start) + 1):
                                pdb_resnum = r + pdb_offset
                                if pdb_resnum > 0:
                                    ann.active_site_residues.append(pdb_resnum)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        pass

    ann.active_site_residues = sorted(set(ann.active_site_residues))

    # Cache results
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_data = {
            "pfam": ann.pfam_domains,
            "cath": ann.cath_domains,
            "active_sites": ann.active_site_residues,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

    # Rate limit
    time.sleep(0.2)
    return ann


def fetch_annotations_batch(pdb_ids: list, cache_dir: str = None,
                             verbose: bool = True) -> dict:
    """Fetch domain annotations for multiple PDB IDs."""
    annotations = {}
    for i, pdb_id in enumerate(pdb_ids):
        # Extract PDB ID and chain (ProteinShake IDs are like "1ABC_A")
        parts = pdb_id.split("_")
        pid = parts[0] if parts else pdb_id
        chain = parts[1] if len(parts) > 1 else "A"

        ann = fetch_domain_annotations(pid, chain, cache_dir=cache_dir)
        annotations[pdb_id] = ann

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"    Fetched annotations {i+1}/{len(pdb_ids)}")

    return annotations


# ============================================================================
# Domain-blob overlap metrics
# ============================================================================

def build_residue_domain_labels(annotation: DomainAnnotation,
                                 n_residues: int) -> np.ndarray:
    """Convert domain annotations to per-residue labels.

    Returns [N] array where each residue gets a domain ID (0, 1, 2, ...)
    or -1 if not in any annotated domain.
    """
    labels = np.full(n_residues, -1, dtype=int)
    domain_id = 0

    # Use Pfam first, fall back to CATH
    domains = annotation.pfam_domains or annotation.cath_domains

    for _, _, start, end in domains:
        # Convert 1-indexed PDB residue numbers to 0-indexed
        s = max(0, start - 1)
        e = min(n_residues, end)
        labels[s:e] = domain_id
        domain_id += 1

    return labels


def compute_domain_blob_overlap(blob_assignments: np.ndarray,
                                 domain_labels: np.ndarray) -> dict:
    """Compute overlap metrics between blob assignments and domain annotations.

    Args:
        blob_assignments: [N] hard blob assignment per residue
        domain_labels: [N] domain label per residue (-1 = unannotated)

    Returns:
        dict with jaccard, ari, and per-blob domain purity
    """
    from sklearn.metrics import adjusted_rand_score

    n = len(blob_assignments)
    assert len(domain_labels) == n

    # Filter to annotated residues only
    annotated = domain_labels >= 0
    if annotated.sum() < 5:
        return {"jaccard": float("nan"), "ari": float("nan"),
                "blob_purity": [], "n_annotated": int(annotated.sum())}

    blob_ann = blob_assignments[annotated]
    domain_ann = domain_labels[annotated]

    # Adjusted Rand Index
    ari = adjusted_rand_score(domain_ann, blob_ann)

    # Per-blob domain purity: for each blob, what fraction of its
    # annotated residues belong to the same domain?
    n_blobs = blob_assignments.max() + 1
    blob_purity = []
    for k in range(n_blobs):
        mask = (blob_assignments == k) & annotated
        if mask.sum() == 0:
            blob_purity.append(float("nan"))
            continue
        domains_in_blob = domain_labels[mask]
        # Purity = fraction of most common domain
        counts = np.bincount(domains_in_blob[domains_in_blob >= 0])
        if len(counts) == 0:
            blob_purity.append(float("nan"))
        else:
            blob_purity.append(counts.max() / counts.sum())

    # Jaccard: treat each domain as a set, each blob as a set,
    # find best-matching pairs
    n_domains = domain_labels.max() + 1
    best_jaccards = []
    for d in range(n_domains):
        domain_set = set(np.where(domain_labels == d)[0])
        if not domain_set:
            continue
        best_j = 0.0
        for k in range(n_blobs):
            blob_set = set(np.where(blob_assignments == k)[0])
            if not blob_set:
                continue
            j = len(domain_set & blob_set) / len(domain_set | blob_set)
            best_j = max(best_j, j)
        best_jaccards.append(best_j)

    mean_jaccard = float(np.mean(best_jaccards)) if best_jaccards else float("nan")

    return {
        "jaccard": mean_jaccard,
        "ari": ari,
        "blob_purity": blob_purity,
        "n_annotated": int(annotated.sum()),
        "n_domains": int(n_domains),
    }


def compute_importance_functional_correlation(blob_importances: np.ndarray,
                                               blob_assignments: np.ndarray,
                                               active_site_residues: list,
                                               n_blobs: int) -> dict:
    """Per-protein test: does the active-site-containing blob have higher π_t?

    Identifies the "active blob" — the blob capturing the most active site
    residues — then checks whether its π_t exceeds the mean π_t of all other
    blobs. This is the correct per-protein test for the requirement
    "active-site-containing blobs should have higher importance".

    Previous approach (Spearman across all 8 blobs within one protein) was
    underpowered: only 8 data points per protein, most blobs containing zero
    active sites, giving effectively a one-vs-seven comparison dressed as a
    correlation.

    Returns dict with per-protein summary + per-blob enrichment for plotting.
    """
    n_residues = len(blob_assignments)
    active_set = set(r - 1 for r in active_site_residues if 0 < r <= n_residues)
    total_active = len(active_set)

    if total_active == 0 or n_residues == 0:
        return None

    overall_rate = total_active / n_residues

    # Per-blob: active site count and enrichment
    blob_active_count = np.zeros(n_blobs, dtype=int)
    blob_enrichment = np.full(n_blobs, float('nan'))
    blob_has_active = np.zeros(n_blobs)

    for k in range(n_blobs):
        blob_residues = set(np.where(blob_assignments == k)[0].tolist())
        if not blob_residues:
            continue
        active_in_blob = len(blob_residues & active_set)
        blob_active_count[k] = active_in_blob
        blob_rate = active_in_blob / len(blob_residues)
        blob_enrichment[k] = blob_rate / (overall_rate + 1e-10)
        blob_has_active[k] = 1.0 if active_in_blob > 0 else 0.0

    # Active blob = blob capturing the most active site residues
    active_blob = int(blob_active_count.argmax())
    n_active_in_blob = int(blob_active_count[active_blob])

    if n_active_in_blob == 0:
        return None

    coverage = n_active_in_blob / total_active
    enrichment = float(blob_enrichment[active_blob]) if not np.isnan(blob_enrichment[active_blob]) else 0.0

    # π_t of active blob vs mean π_t of all other blobs
    other_blobs = [k for k in range(n_blobs) if k != active_blob]
    pi_t_active = float(blob_importances[active_blob])
    pi_t_others_mean = float(np.mean([blob_importances[k] for k in other_blobs]))
    pi_t_delta = pi_t_active - pi_t_others_mean

    # Rank of active blob by π_t (rank 1 = highest importance)
    sorted_blobs = np.argsort(blob_importances)[::-1]
    pi_t_rank = int(np.where(sorted_blobs == active_blob)[0][0]) + 1

    return {
        # Per-protein summary — the primary test
        "active_blob": active_blob,
        "pi_t_active_blob": pi_t_active,
        "pi_t_others_mean": pi_t_others_mean,
        "pi_t_delta": pi_t_delta,
        "pi_t_rank": pi_t_rank,
        "coverage": float(coverage),
        "enrichment": float(enrichment),
        "n_active_sites": total_active,
        "n_active_in_blob": n_active_in_blob,
        # Per-blob data retained for scatter plots
        "blob_enrichment": blob_enrichment.tolist(),
        "blob_has_active": blob_has_active.tolist(),
        "blob_active_count": blob_active_count.tolist(),
        "blob_importance": blob_importances.tolist(),
    }


# ============================================================================
# PyMOL visualization script generation
# ============================================================================

def generate_pymol_script(pdb_id: str, blob_assignments: np.ndarray,
                           blob_importances: np.ndarray,
                           domain_labels: np.ndarray = None,
                           active_site_residues: list = None,
                           out_path: str = None) -> str:
    """Generate a PyMOL .pml script to visualize blob assignments on 3D structure.

    Colors residues by blob assignment, highlights active site residues,
    and annotates domain boundaries.
    """
    parts = pdb_id.split("_")
    pid = parts[0].lower()
    chain = parts[1] if len(parts) > 1 else "A"

    blob_colors = [
        "red", "orange", "yellow", "green",
        "cyan", "blue", "purple", "white"
    ]

    n_blobs = int(blob_assignments.max()) + 1
    lines = [
        f"# SoftBlobGIN blob visualization for {pdb_id}",
        f"# Generated automatically — open in PyMOL",
        f"",
        f"fetch {pid}, async=0",
        f"remove not chain {chain}",
        f"hide everything",
        f"show cartoon, chain {chain}",
        f"color gray80, chain {chain}",
        f"",
        f"# Color residues by blob assignment",
    ]

    for k in range(n_blobs):
        residues = np.where(blob_assignments == k)[0]
        if len(residues) == 0:
            continue
        # Convert to 1-indexed PDB residue numbers
        resi_str = "+".join(str(r + 1) for r in residues)
        color = blob_colors[k % len(blob_colors)]
        imp = blob_importances[k] if k < len(blob_importances) else 0
        lines.append(f"select blob{k+1}, chain {chain} and resi {resi_str}")
        lines.append(f"color {color}, blob{k+1}")
        lines.append(f"# Blob {k+1}: {len(residues)} residues, importance={imp:.4f}")

    # Highlight active site residues
    if active_site_residues:
        resi_str = "+".join(str(r) for r in active_site_residues)
        lines.extend([
            f"",
            f"# Active/binding site residues",
            f"select active_site, chain {chain} and resi {resi_str}",
            f"show sticks, active_site",
            f"color magenta, active_site",
            f"set stick_radius, 0.15, active_site",
        ])

    # Domain boundaries
    if domain_labels is not None:
        lines.append(f"")
        lines.append(f"# Domain boundaries")
        n_domains = int(domain_labels.max()) + 1
        for d in range(n_domains):
            residues = np.where(domain_labels == d)[0]
            if len(residues) == 0:
                continue
            start, end = residues.min() + 1, residues.max() + 1
            lines.append(f"# Domain {d}: residues {start}-{end}")

    lines.extend([
        f"",
        f"# Final styling",
        f"set cartoon_transparency, 0.3",
        f"set ray_shadow, 0",
        f"bg_color white",
        f"orient",
        f"zoom chain {chain}",
    ])

    script = "\n".join(lines)

    if out_path:
        with open(out_path, "w") as f:
            f.write(script)

    return script
