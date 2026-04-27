import os
from Bio.PDB import MMCIFParser
import torch
import numpy as np
from scipy.spatial import cKDTree

parser = MMCIFParser(QUIET=True)
cif_path = "data/alphafold_cif/P82349.cif"

structure = parser.get_structure("AF", cif_path)
coords = []
for model in structure:
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())

coords = np.array(coords)
tree = cKDTree(coords)
# Request k+1 because the query point itself is included
dists, indices = tree.query(coords, k=11)

src = np.repeat(np.arange(len(coords)), 11)
dst = indices.flatten()
# Remove self-loops
mask = src != dst
src = src[mask]
dst = dst[mask]

edge_index = torch.tensor([src, dst], dtype=torch.long)
print(f"Nodes: {len(coords)}, edges: {edge_index.shape[1]}")
