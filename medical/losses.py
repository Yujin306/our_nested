import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, weights=None):
        super().__init__()
        self.loss_fns = loss_fns
        if weights is None:
            self.weights = [1.0] * len(loss_fns)
        else:
            self.weights = weights

        # deep supervision weight
        self.deep_weights = [1.0, 0.5, 0.25, 0.125]

    def forward(self, outputs, targets):
        total_loss = 0.0

        if not isinstance(outputs, list):
            outputs = [outputs]

        for i, out in enumerate(outputs):
            # 해상도 맞춰주기
            if out.shape != targets.shape:
                out = F.interpolate(out, size=targets.shape[2:], mode='trilinear', align_corners=False)

            for loss_fn, weight in zip(self.loss_fns, self.weights):
                loss = weight * loss_fn(out, targets)
                total_loss += self.deep_weights[i] * loss

        return total_loss
