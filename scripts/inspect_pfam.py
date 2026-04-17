
from proteinshake.tasks import ProteinFamilyTask
import torch

try:
    task = ProteinFamilyTask(root='data', use_precomputed=False)
    print(f"Task: {task.name}")
    print(f"Total proteins: {len(task.proteins)}")
    print(f"Train indices: {len(task.train_index)}")
    print(f"Val indices: {len(task.val_index)}")
    print(f"Test indices: {len(task.test_index)}")
    
    # Check labels
    y_train = [task.target(task.proteins[i]) for i in task.train_index]
    unique_labels = set(y_train)
    print(f"Unique train labels: {len(unique_labels)}")
    print(f"Max label: {max(y_train) if y_train else 'N/A'}")
    
    # Check num_classes
    n_classes = task.num_classes
    print(f"Num classes attribute: {n_classes}")

except Exception as e:
    print(f"Error: {e}")
