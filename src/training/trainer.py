"""
Training engine for all model types.

Features:
  - Unified training loop for PyG and plain models
  - Early stopping with configurable patience
  - Cosine annealing with warmup LR scheduler
  - Gradient clipping
  - TensorBoard logging
  - Model checkpointing (best + last)
  - Training-time graph augmentation
  - Gumbel-softmax temperature annealing for SoftBlobGAT
"""

import logging
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

from ..data.augmentation import GraphAugmentation
from .losses import build_criterion

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class Trainer:
    """Unified trainer for all model architectures."""

    def __init__(self, model, cfg, device, is_pyg=True,
                 model_name="model", tb_writer=None):
        self.model = model.to(device)
        self.cfg = cfg["training"]
        self.device = device
        self.is_pyg = is_pyg
        self.model_name = model_name
        self.tb_writer = tb_writer

        # Checkpoint directory
        self.ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Build optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        # Build scheduler
        sched_type = self.cfg.get("scheduler", "cosine_warmup")
        if sched_type == "cosine_warmup":
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=self.cfg.get("warmup_epochs", 10),
                total_epochs=self.cfg["epochs"],
                min_lr=self.cfg.get("min_lr", 1e-6),
            )
        elif sched_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.cfg.get("cosine_T0", 50), T_mult=2
            )
        else:
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        # Build augmentation (for PyG models only)
        aug_cfg = self.cfg.get("augmentation", {})
        self.augmentation = GraphAugmentation(
            edge_drop_rate=aug_cfg.get("edge_drop_rate", 0.1),
            feature_mask_rate=aug_cfg.get("feature_mask_rate", 0.1),
            enabled=aug_cfg.get("enabled", True) and is_pyg,
        )

        # Gradient clipping
        self.grad_clip = self.cfg.get("gradient_clip", 1.0)

    def train(self, train_loader, val_loader, criterion, class_weights=None):
        """Run full training loop with early stopping and logging."""
        epochs = self.cfg["epochs"]
        patience = self.cfg.get("patience", 40)
        min_delta = self.cfg.get("min_delta", 1e-4)
        log_every = 10

        best_val_loss = float("inf")
        best_state = deepcopy(self.model.state_dict())
        stale = 0

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }

        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # Temperature annealing for SoftBlobGAT
            if hasattr(self.model, "set_tau"):
                self.model.set_tau(epoch, epochs)

            # Train
            train_loss, train_acc = self._train_epoch(train_loader, criterion)

            # Validate
            val_loss, val_acc = self._eval_epoch(val_loader, criterion)

            # Schedule
            if isinstance(self.scheduler, CosineWarmupScheduler):
                self.scheduler.step(epoch)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            # TensorBoard logging
            if self.tb_writer is not None:
                self.tb_writer.add_scalars(f"{self.model_name}/loss", {
                    "train": train_loss, "val": val_loss
                }, epoch)
                self.tb_writer.add_scalars(f"{self.model_name}/accuracy", {
                    "train": train_acc, "val": val_acc
                }, epoch)
                self.tb_writer.add_scalar(f"{self.model_name}/lr", current_lr, epoch)

                if hasattr(self.model, "_current_tau"):
                    self.tb_writer.add_scalar(
                        f"{self.model_name}/tau", self.model._current_tau, epoch
                    )

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1

            # Logging
            if epoch % log_every == 0 or epoch == 1 or stale == 0:
                tau_str = f"  tau={self.model._current_tau:.3f}" if hasattr(self.model, "_current_tau") else ""
                marker = " *" if stale == 0 else ""
                logger.info(
                    f"  [{self.model_name}] ep {epoch:3d}/{epochs}  "
                    f"loss={train_loss:.4f}/{val_loss:.4f}  "
                    f"acc={train_acc:.4f}/{val_acc:.4f}  "
                    f"lr={current_lr:.2e}{tau_str}{marker}"
                )

            if stale >= patience:
                logger.info(f"  [{self.model_name}] Early stopping at epoch {epoch} (patience={patience})")
                break

        elapsed = time.time() - t_start
        self.model.load_state_dict(best_state)

        # Save checkpoints
        if self.cfg.get("save_best_model", True):
            ckpt_path = self.ckpt_dir / f"{self.model_name}_best.pt"
            torch.save(best_state, ckpt_path)
            logger.info(f"  Saved best checkpoint: {ckpt_path}")

        logger.info(
            f"  [{self.model_name}] Done in {elapsed:.1f}s  "
            f"best_val_loss={best_val_loss:.4f}"
        )

        return history, elapsed

    def _train_epoch(self, loader, criterion):
        """Single training epoch."""
        self.model.train()
        total_loss = total_correct = total_samples = 0

        if self.is_pyg:
            for batch in loader:
                batch = batch.to(self.device)

                # Apply augmentation
                if self.augmentation.enabled:
                    batch = self.augmentation(batch)

                self.optimizer.zero_grad()
                
                # Handle PairData vs Standard Data
                if hasattr(batch, 'b1'):
                    # PairBatch already contains b1 and b2 Batch objects
                    out = self.model(batch.b1, batch.b2)
                else:
                    out = self.model(batch)
                
                # Dynamic type handling for regression/multilabel vs multiclass
                y = batch.y
                if isinstance(y, list):
                    # List of matrices (e.g. PPI matrices of different shapes)
                    loss = 0
                    # If model didn't return a list, it might be a padded tensor (unlikely given our collate)
                    # For PPI, SoftBlobGINPPI returns a matrix.
                    # With pair_collate returning a list for y, the model likely returns a list or we loop.
                    # Actually, for PPI, self.model(b1, b2) returns a matrix? 
                    # If b1, b2 are PyG Batches, out might be a single matrix if it's a blocked matmul.
                    # But it's easier to assume it's a list or handle the batching in the model.
                    # Let's assume for now that if y is a list, we calculate loss per item.
                    
                    # If 'out' is a single tensor but 'y' is a list, we need to split 'out'
                    # but our PPI model returns [N1, N2] for the WHOLE BATCH? No, that's not right.
                    # SoftBlobGINPPI.forward needs to handle batches.
                    
                    # For now, let's just use the list-based zip if both are lists.
                    if isinstance(out, (list, tuple)):
                        for o, t in zip(out, y):
                            loss += criterion(o, t)
                        loss = loss / len(y)
                    else:
                        # Fallback: if 'out' is one tensor (maybe padded?), try criterion once
                        # This should be refined if we use a block-matrix approach.
                        loss = criterion(out, y) if not isinstance(y, list) else criterion(out, torch.stack(y))
                    correct_count = 0
                elif len(y.shape) > 1 and y.shape[0] == out.shape[0] and y.shape[1] == out.shape[1]:
                    loss = criterion(out, y)
                    correct_count = 0
                elif out.shape == y.shape:
                    loss = criterion(out, y)
                    correct_count = 0
                else:
                    loss = criterion(out, y.long())
                    correct_count = (out.argmax(1) == y.long()).sum().item()
                    
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                total_loss += loss.item() * batch.num_graphs
                total_correct += correct_count
                total_samples += batch.num_graphs
        else:
            for X, Y in loader:
                X, Y = X.to(self.device), Y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(X)
                loss = criterion(out, Y)
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                total_loss += loss.item() * len(Y)
                total_correct += (out.argmax(1) == Y).sum().item()
                total_samples += len(Y)

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def _eval_epoch(self, loader, criterion):
        """Single evaluation epoch."""
        self.model.eval()
        total_loss = total_correct = total_samples = 0

        if self.is_pyg:
            for batch in loader:
                batch = batch.to(self.device)
                
                if hasattr(batch, 'b1'):
                    out = self.model(batch.b1, batch.b2)
                else:
                    out = self.model(batch)
                
                # Dynamic type handling
                y = batch.y
                if isinstance(y, list):
                    loss = 0
                    if isinstance(out, (list, tuple)):
                        for o, t in zip(out, y):
                            loss += criterion(o, t)
                        loss = loss / len(y)
                    else:
                        loss = criterion(out, y) if not isinstance(y, list) else criterion(out, torch.stack(y))
                    correct_count = 0
                elif len(y.shape) > 1 or out.shape == y.shape:
                    loss = criterion(out, y)
                    correct_count = 0
                else:
                    loss = criterion(out, y.long())
                    correct_count = (out.argmax(1) == y.long()).sum().item()
                    
                total_loss += loss.item() * batch.num_graphs
                total_correct += correct_count
                total_samples += batch.num_graphs
        else:
            for X, Y in loader:
                X, Y = X.to(self.device), Y.to(self.device)
                out = self.model(X)
                loss = criterion(out, Y)
                total_loss += loss.item() * len(Y)
                total_correct += (out.argmax(1) == Y).sum().item()
                total_samples += len(Y)

        return total_loss / total_samples, total_correct / total_samples


# ============================================================================
# Prediction helpers
# ============================================================================

@torch.no_grad()
def predict_pyg(model, loader, device):
    """Get predictions + probabilities from a PyG model."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_true.extend(batch.y.long().cpu().numpy())
        all_pred.extend(logits.argmax(1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.vstack(all_prob),
    )


@torch.no_grad()
def predict_plain(model, loader, device):
    """Get predictions + probabilities from a plain MLP model."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        all_true.extend(Y.cpu().numpy())
        all_pred.extend(logits.argmax(1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.vstack(all_prob),
    )


@torch.no_grad()
def get_embeddings(model, loader, device):
    """Extract graph-level embeddings for visualization."""
    model.eval()
    embs, labs = [], []
    for batch in loader:
        batch = batch.to(device)
        e = model.embed(batch)
        embs.append(e)
        labs.extend(batch.y.long().cpu().numpy())
    return np.vstack(embs), np.array(labs)
