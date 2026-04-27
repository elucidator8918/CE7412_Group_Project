
import os
import torch
from proteinshake.tasks import LigandAffinityTask, ProteinProteinInterfaceTask
from src.data.benchmark_dataset import TASK_TYPE_MAP

def inspect_task(task_name, task_class):
    print(f"--- Inspecting {task_name} ---")
    root = "./data"
    try:
        task = task_class(root=root)
        graphs = iter(task.dataset.to_graph(eps=8.0).pyg())
        data, prot = next(graphs)
        print(f"Data keys: {data.keys()}")
        print(f"Prot keys: {prot.keys()}")
        if "protein" in prot:
            print(f"Protein keys: {prot['protein'].keys()}")
            print(f"Sequence: {prot['protein'].get('sequence', 'MISSING')[:20]}...")
        else:
            print("NO 'protein' key in prot!")
            # Check other keys
            for k in prot.keys():
                if isinstance(prot[k], dict):
                    print(f"  {k} keys: {prot[k].keys()}")
    except Exception as e:
        print(f"Failed to inspect {task_name}: {e}")

if __name__ == "__main__":
    inspect_task("LigandAffinityTask", LigandAffinityTask)
    inspect_task("ProteinProteinInterfaceTask", ProteinProteinInterfaceTask)
