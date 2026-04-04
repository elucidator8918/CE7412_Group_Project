"""
Loss functions for enzyme classification.

Includes:
  - FocalLoss: down-weights easy examples, focuses on hard/rare classes
  - LabelSmoothingCE: standard cross-entropy with label smoothing
  - build_criterion: factory function from config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for class-imbalanced classification.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    When γ > 0, the loss down-weights well-classified examples, forcing
    the model to focus on hard, misclassified cases. Particularly useful
    for the rare EC5 (Isomerase) and EC7 (Translocase) classes.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # [n_classes] or None
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw model outputs
            targets: [B] integer class labels

        Returns:
            loss: scalar
        """
        n_classes = logits.shape[1]

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, n_classes).float()

        # Log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Class weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.unsqueeze(0).expand_as(logits)  # [B, C]
            focal_weight = focal_weight * alpha_t

        # Loss
        loss = -focal_weight * smooth_targets * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(cfg, class_weights=None):
    """Factory function to create loss from config.

    Args:
        cfg: training config dict
        class_weights: [n_classes] tensor of inverse-frequency weights
    """
    loss_type = cfg.get("loss", "ce")
    smoothing = cfg.get("label_smoothing", 0.0)

    if loss_type == "focal":
        gamma = cfg.get("focal_gamma", 2.0)
        alpha = class_weights if cfg.get("focal_alpha") is None else None
        return FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=smoothing)

    elif loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=smoothing,
        )

    else:  # "ce"
        return nn.CrossEntropyLoss(weight=class_weights)
