import torch
import yaml
import os
import sys
import pandas as pd
from torch_geometric.loader import DataLoader as PyGLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.deeploc.train_deeploc import build_model
from scripts.deeploc.deeploc_dataset import DeepLocDataset, ESM2Extractor, compute_feat_dim, compute_edge_dim

def check_gate_statistics():
    config_path = "configs/deeploc.yaml"
    ckpt_path = "outputs_deeploc/checkpoints/deeploc_softblobgin_best.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Dataset (Validation only)
    feat_cfg = cfg.get("features", {})
    ds_cfg = cfg.get("dataset", {})
    
    esm_extractor = None
    if feat_cfg.get("use_esm2"):
        esm_extractor = ESM2Extractor(
            model_name=feat_cfg.get("esm2_model"),
            cache_dir=cfg["paths"]["esm_cache"]
        )
    
    val_tmp = "outputs_deeploc/_val_split.csv"
    if not os.path.exists(val_tmp):
        print("Error: Validation split CSV not found. Run training with --quick first.")
        return
        
    val_ds = DeepLocDataset(
        csv_path=val_tmp,
        id_column="ACC" if "ACC" in pd.read_csv(val_tmp).columns else "ACC\n",
        feat_cfg=feat_cfg,
        esm_extractor=esm_extractor,
        max_seq_len=ds_cfg.get("max_seq_len", 1000),
        k_neighbors=ds_cfg.get("k_neighbors", 30)
    )
    val_loader = PyGLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Build Model
    feat_dim = compute_feat_dim(feat_cfg)
    edge_dim = compute_edge_dim(feat_cfg)
    model = build_model(feat_dim, edge_dim, 10, cfg["model"], feat_cfg, device)
    
    # Load weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Instrument the gate
    all_gates = []
    
    def hook_fn(module, input, output):
        all_gates.append(output.detach().cpu())

    # The gate is the Sigmoid layer at the end of model.gate
    handle = model.gate[-1].register_forward_hook(hook_fn)

    print(f"Evaluating gate values for {len(val_ds)} proteins...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            model(batch)

    handle.remove()
    
    if not all_gates:
        print("No gate values captured. Is use_global_skip enabled?")
        return

    gates = torch.cat(all_gates, dim=0)
    
    print("\n" + "="*30)
    print("GATE STATISTICS")
    print("="*30)
    print(f"Mean Gate Value (g): {gates.mean().item():.4f}")
    print(f"Std Dev:            {gates.std().item():.4f}")
    print(f"Min:                {gates.min().item():.4f}")
    print(f"Max:                {gates.max().item():.4f}")
    print("-" * 30)
    
    if gates.mean() > 0.7:
        print("Insight: Model is heavily relying on the Global ESM Pool.")
    elif gates.mean() < 0.3:
        print("Insight: Model is favoring the GNN Graph features.")
    else:
        print("Insight: Model is using a balanced mix of Graph and Global features.")
    print("="*30)

if __name__ == "__main__":
    check_gate_statistics()
