"""
Copyright to ReCAP Authors, ICML 2025 Poster.
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from imageio.v2 import sizes

from helper.registry import ADAPTATION_REGISTRY
from tta_methods.base import TTAMethod

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

@ADAPTATION_REGISTRY.register()
class ReCAP(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.model = model
        self.steps = 1
        assert self.steps > 0, "ReCAP requires >= 1 step(s) to forward and update"
        self.episodic = False
        self.batch_size = self.cfg.TEST.BATCH_SIZE

        self.reset_constant_em = 0.2  # threshold e_m for model recovery scheme, follow SAR
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer()

        self.margin = self.cfg.ReCAP.MARGIN  # margin \tau_RE in Eqn. (9)
        self.margin_L0 = self.cfg.ReCAP.MARGIN_L0 # L_0 in Eqn. (9)
        self.weight_reg = self.cfg.ReCAP.WEIGHT_REG
        self.reweight_threshold = self.cfg.ReCAP.REWEIGHT_THRESHOLD
        self.weight_tau = self.cfg.ReCAP.WEIGHT_TAU
        self.sigma_t = torch.from_numpy(np.load('cov_changeformer.npy')).sum(0)
        try:
            self.W = model.fc.weight # ResNet
        except:
            self.W = model.TDec_x2.linear_c1.proj.weight # ViT

        self.W_cpu = self.W.cpu()
        self._refresh_prob_aug()

    def _refresh_prob_aug(self, scale = 0.01):
        with torch.no_grad():
            sigma_t = torch.clamp(self.sigma_t.view(1, 1, -1), min=1e-8)
            region = sigma_t  * self.weight_tau / (1.0 * scale)
            sqrt_region = torch.sqrt(region).cpu()
            diff = (self.W_cpu.unsqueeze(0) - self.W_cpu.unsqueeze(1)) * sqrt_region
            self.prob_aug = torch.exp(0.5 * torch.einsum('ijb,ijb->ij', diff, diff))
            self.prob_aug = self.prob_aug.cuda()
            self.normW = 0.1 / 2 * (scale ** 2)  * (torch.norm(self.W, dim=1) ** 2)

    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def L_RE(self, x: torch.Tensor) -> torch.Tensor:
        prob_anchor = x.softmax(1)

        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        prob = (x + self.normW).softmax(1)
        return (-prob * torch.log(prob_anchor) + prob * torch.log(prob_aug)).sum(1)

    def L_RI(self, x: torch.Tensor) -> torch.Tensor:
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        return (prob_anchor * torch.log(prob_aug)).sum(1)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_recap(self, x1, x2, ema):

        self.optimizer.zero_grad()
        outputs_list = self.model(x1, x2)
        outputs = outputs_list[-1]
        L_RE = self.L_RE(outputs)
        L_RI = self.L_RI(outputs)

        loss = (L_RE + self.weight_reg * L_RI).mean() # ReCAP replacement for entropy loss
        if not np.isnan(loss.item()):
            ema = update_ema(ema, loss.item() / 2) # record moving average loss values for model recovery
        # print(loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                # print("ema < 0.2, now reset the model")
                reset_flag = True

        return outputs_list, ema, reset_flag


    def forward(self, x1, x2):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt_recap(x1, x2, self.ema)

            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss
 
        return outputs

    def get_features(self, x):
        x = self.model.forward_features(x)
        try:
            x = self.model.global_pool(x) # ResNet
        except:
            x = x[:, 0, :]  # ViT
        return x
        
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.ema = None

    def configure_model(self):
        """Configure model for use with ReCAP."""
        # train mode, because ReCAP optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what ReCAP updates
        self.model.requires_grad_(False)
        # configure norm for ReCAP updates: enable grad + force batch statisics (this only for BN models)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return self.model