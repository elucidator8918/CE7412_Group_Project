from proteinshake.tasks import GeneOntologyTask, ProteinProteinInterfaceTask
import sys
import numpy as np

try:
    print("--- Gene Ontology ---")
    sys.stdout.flush()
    go = GeneOntologyTask(root='data', use_precomputed=False)
    print("n_classes:", go.num_classes if hasattr(go, 'num_classes') else "N/A")
    if hasattr(go, 'token_map'):
        print("token_map length:", len(go.token_map))
    else:
        print("no token map")
    
    tgt = go.train_targets[0]
    print("Target type:", type(tgt))
    print("Target value:", tgt)
except Exception as e:
    print(f"GO error: {e}")

try:
    print("--- PPI ---")
    sys.stdout.flush()
    ppi = ProteinProteinInterfaceTask(root='data', use_precomputed=False)
    idx = ppi.train_index[0]
    print("Index type:", type(idx))
    print("Index value:", idx)
    
except Exception as e:
    print(f"PPI error: {e}")
