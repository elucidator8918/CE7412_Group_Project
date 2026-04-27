#!/usr/bin/env python3
"""
Summarize Benchmark — Evaluates all checkpoints in outputs_benchmark/checkpoints
using ProteinShake's official metrics only.
"""

import argparse
import os
import sys
import torch
import yaml
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import (
    TASK_CLASSES, TASK_METRICS, build_model, evaluate_task, set_seed
)
from src.data.benchmark_dataset import BenchmarkDataset, TASK_TYPE_MAP
from src.data.features import compute_feat_dim, compute_edge_dim

def main():
    parser = argparse.ArgumentParser(description="Summarize Benchmark Models")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs_benchmark/checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} not found.")
        return

    device = torch.device(args.device)
    gin_cfg = cfg["gin"]
    feat_dim = compute_feat_dim(cfg["features"])
    edge_dim = compute_edge_dim(cfg["features"])

    results = []
    
    # Iterate over checkpoints
    for cp_path in sorted(checkpoint_dir.glob("*.pt")):
        # Name format: GIN_{TaskName}_best.pt
        name = cp_path.stem
        if not name.startswith("GIN_") or not name.endswith("_best"):
            continue
            
        task_name = name.replace("GIN_", "").replace("_best", "")
        if task_name not in TASK_CLASSES:
            print(f"Skipping unknown task: {task_name}")
            continue

        print(f"Evaluating {task_name}...")
        set_seed(cfg["seed"])  # Ensure split consistency
        task_class = TASK_CLASSES[task_name]
        task_type = TASK_TYPE_MAP.get(task_name, "graph_multiclass")

        try:
            # 1. Load Dataset (lazy)
            dataset = BenchmarkDataset(task_class, task_name, cfg)
            dataset.prepare()
            
            # 2. Build and Load Model
            model_type = cfg.get("model_type", "gin")
            model = build_model(task_name, task_type, dataset.feat_dim, dataset.edge_dim, 
                                dataset.n_classes, gin_cfg, model_type=model_type)
            
            state_dict = torch.load(cp_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)

            # 3. Evaluate using official metrics only
            metrics = evaluate_task(model, dataset.test_loader, task_type, device, dataset.task)
            
            # Collect standard metrics for a comprehensive table
            res_item = {
                "Task": task_name,
                "Type": task_type,
                "Acc": metrics.get("accuracy", metrics.get("acc", "N/A")),
                "MCC": metrics.get("mcc", "N/A"),
                "F1": metrics.get("f1_macro", metrics.get("f1_micro", "N/A")),
                "AUROC": metrics.get("auroc", "N/A"),
                "R2": metrics.get("r2", "N/A"),
                "Spearman": metrics.get("spearman", "N/A"),
                "Official": metrics.get(TASK_METRICS.get(task_name, "").lower(), "N/A")
            }
            
            # Format numbers
            for k, v in res_item.items():
                if isinstance(v, float):
                    res_item[k] = f"{v:.4f}"
            
            results.append(res_item)

        except Exception as e:
            print(f"Failed to evaluate {task_name}: {e}")

    # Output Table
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BENCHMARK OFFICIAL RESULTS SUMMARY")
    print("="*60)
    print(df.to_markdown(index=False))
    print("="*60 + "\n")

    # Save to a new summary file
    summary_path = Path("outputs_benchmark/official_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Benchmark Official Results Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
