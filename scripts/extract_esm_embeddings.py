#!/usr/bin/env python3
"""
CE7412 Enzyme Classification — ESM-2 Embedding Extraction

Extracts per-residue embeddings from ESM-2 for all proteins in the
ProteinShake EnzymeClassTask dataset. Embeddings are cached to disk
so this only needs to run once.

Usage:
    python scripts/extract_esm_embeddings.py
    python scripts/extract_esm_embeddings.py --model esm2_t12_35M_UR50D --layer 12
    python scripts/extract_esm_embeddings.py --batch-size 4  # reduce for low memory
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import ESM2Extractor


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="ESM-2 model name (default: from config)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Representation layer (default: from config)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    # Load config
    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["features"]["esm2_model"]
    layer = args.layer or cfg["features"]["esm2_layer"]
    cache_dir = cfg["paths"]["esm_cache"]
    data_root = cfg["paths"]["data_root"]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"ESM-2 model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Device: {device}")

    # Load dataset to get protein sequences
    from proteinshake.tasks import EnzymeClassTask
    logger.info("Loading ProteinShake EnzymeClassTask...")
    task = EnzymeClassTask(root=data_root, split="random", split_similarity_threshold=0.7)
    raw_pyg = list(task.dataset.to_graph(eps=8.0).pyg())
    logger.info(f"Loaded {len(raw_pyg)} proteins")

    # Collect (id, sequence) pairs
    proteins = []
    for i, (g, prot) in enumerate(raw_pyg):
        pid = str(prot["protein"].get("ID", f"prot_{i}"))
        seq = prot["protein"].get("sequence", "")
        if len(seq) > 0:
            proteins.append((pid, seq))

    logger.info(f"Extracting embeddings for {len(proteins)} proteins...")

    # Extract
    extractor = ESM2Extractor(
        model_name=model_name, layer=layer,
        cache_dir=cache_dir, device=device
    )

    # Check how many are already cached
    cached = sum(1 for pid, _ in proteins if extractor._cache_path(pid).exists())
    logger.info(f"Already cached: {cached}/{len(proteins)}")

    if cached < len(proteins):
        extractor.extract_batch(proteins, batch_size=args.batch_size)
        logger.info("Extraction complete!")
    else:
        logger.info("All embeddings already cached, nothing to do.")

    # Verify
    cache_dir_path = Path(cache_dir)
    n_files = len(list(cache_dir_path.glob("*.pt")))
    total_size = sum(f.stat().st_size for f in cache_dir_path.glob("*.pt")) / (1024 * 1024)
    logger.info(f"Cache: {n_files} files, {total_size:.1f} MB")


if __name__ == "__main__":
    main()
