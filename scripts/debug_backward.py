import sys
import os
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGLoader

sys.path.insert(0, os.path.dirname(__file__))

from scripts.deeploc.deeploc_dataset import DeepLocDataset, LOCALIZATION_CLASSES
from scripts.deeploc.train_deeploc import build_model, collate_filter
from src.data.features import ESM2Extractor, compute_feat_dim, compute_edge_dim
from src.training.trainer import Trainer

# Allow memory profiling
import tracemalloc

def debug():
    with open("configs/deeploc.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dummy dataset directly to bypass setup limits
    train_tmp = os.path.join(cfg["paths"]["output_dir"], "_train_split.csv")
    feat_cfg = cfg.get("features", {})
    ds_cfg = cfg.get("dataset", {})

    print("Loading Dataset...")
    train_ds = DeepLocDataset(
        csv_path=train_tmp,
        feat_cfg=feat_cfg,
        esm_extractor=None,
        max_seq_len=ds_cfg.get("max_seq_len", 1000),
        k_neighbors=ds_cfg.get("k_neighbors", 10),
    )

    # Cut dataset to just 16 items (1 batch)
    train_ds.df = train_ds.df.head(16)

    # Use CPU instead of CUDA explicitly to trace CPU RAM if CUDA is what's blowing system RAM
    loader = PyGLoader(
        train_ds, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=False,
        collate_fn=collate_filter
    )

    feat_dim = compute_feat_dim(feat_cfg)
    edge_dim = compute_edge_dim(feat_cfg)
    model = build_model(feat_dim, edge_dim, 10, cfg["model"])
    model = model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()

    print("Fetching batch...")
    batch = next(iter(loader))
    print(f"Batch nodes: {batch.num_nodes}, edges: {batch.num_edges}")
    batch = batch.to(device)

    print("Forward pass...")
    logits = model(batch)
    
    y = batch.y.float()
    if y.dim() == 1 and logits.dim() == 2:
        y = y.unsqueeze(1)
        
    loss = criterion(logits, y)
    
    print("Backward pass...")
    
    # We will use PyTorch's native memory profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True
    ) as prof:
        loss.backward()
        
    print("Backward pass complete!")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

if __name__ == "__main__":
    debug()
