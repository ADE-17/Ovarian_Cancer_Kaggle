import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=1):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.eps = 1e-6
        self.pos_weight = pos_weight

    def forward(self, probs, target):
        # Predicted probabilities for the negative class
        q = 1 - probs
        p = probs
        # For numerical stability (so we don't inadvertently take the log of 0)
        p = p.clamp(self.eps, 1.0 - self.eps)
        q = q.clamp(self.eps, 1.0 - self.eps)

        # Loss for the positive examples
        pos_loss = -(q**self.gamma) * torch.log(p)
        if self.pos_weight is not None:
            pos_loss *= self.pos_weight

        # Loss for the negative examples
        neg_loss = -(p**self.gamma) * torch.log(q)

        loss = target * pos_loss + (1 - target) * neg_loss

        return loss.sum()