"""
Builds upon: https://github.com/zhangmarvin/memo
Corresponding paper: https://arxiv.org/abs/2110.09506
"""

import torch
import torch.jit
import numpy as np

from tta_methods.base import TTAMethod
from tta_methods.bn import AlphaBatchNorm
from helper.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class MEMO(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    def forward(self, x1, x2):
        if self.episodic:
            self.reset()

        x1_aug = x1
        x2_aug = x2
        for _ in range(self.steps):
            self.forward_and_adapt(x1_aug, x2_aug)

        return self.model(x1, x2)

    def loss_calculation(self, x1, x2):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        outputs = self.model(x1, x2)
        loss, _ = marginal_entropy(outputs[-1])
        return outputs, loss.mean()

    @torch.enable_grad()
    def forward_and_adapt(self, x1, x2):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x1, x2)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x1, x2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

    def configure_model(self):
        self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.cfg.BN.ALPHA).to(self.device)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
