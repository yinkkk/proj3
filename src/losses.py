import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCELoss(nn.Module):
    """
    Focal loss for multi-label classification (BCEWithLogitsLoss + Focal)
    logits: [batch_size, num_classes]
    targets: [batch_size, num_classes], float (0.0 or 1.0)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: raw output from model (before sigmoid)
        target: 0/1 labels
        """
        # BCEWithLogitsLoss per element
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        # Probabilities
        p = torch.sigmoid(logits)
        # Focal factor
        pt = p * target + (1 - p) * (1 - target)  # pt = p if target=1 else 1-p
        focal_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_factor * focal_factor * bce_loss
        else:
            loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
