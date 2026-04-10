"""
Builds upon: https://github.com/mr-eggplant/SAR/blob/main/sar.py
Corresponding paper: https://openreview.net/pdf?id=g2YraF75Tj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
from tta_methods.base import TTAMethod
from helper.registry import ADAPTATION_REGISTRY
from helper.losses import Entropy
from helper.inject_low_rank_vida import inject_trainable_vida
from copy import deepcopy
logger = logging.getLogger(__name__)

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]

@torch.no_grad()
def update_ema(ema, new_data, alpha=0.9):
    if ema is None:
        return new_data
    else:
        return alpha * ema + (1 - alpha) * new_data


@ADAPTATION_REGISTRY.register()
class TTTA(TTAMethod):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.cfg = cfg
        self.margin_e0 = cfg.TTTA.MARGIN * math.log(100) # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = cfg.TTTA.RESET_EMA  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
        # setup loss function
        self.softmax_entropy = Entropy()
        self.non_router_states = self.get_non_router_states()
        self.input_data = {}
        self.output_data = {}
        self.anchor = deepcopy(self.model.state_dict())
        self.downsampling = 8
        self.pos_coeff = 1
        self.neg_coeff= 3

        self.optimizer = torch.optim.Adam(
            [{"params": self.params[0], "lr": self.cfg.OPTIM.LR, 'weight_decay': self.cfg.OPTIM.WD}, \
             {"params": self.params[1], "lr": self.cfg.TTTA.LR_VIDA, 'weight_decay': self.cfg.TTTA.WD_VIDA}])

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def Original_forward(self, x1, x2):
        """Forward and adapt model input data.
               Measure entropy of the model prediction, take gradients, and update params.
               """
        self.optimizer.zero_grad()
        # self.model.Tenc_x2.patch_embed1.register_forward_hook(self.get_activation('patch_embed1'))
        outputs = self.model(x1, x2)
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = self.softmax_entropy(outputs[-1])
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        self.optimizer.first_step(
            zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = self.softmax_entropy(self.model(x1, x2)[-1])
        entropys2 = entropys2[filter_ids_1]  # second time forward
        filter_ids_2 = torch.where(
            entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = update_ema(self.ema, loss_second.item())  # record moving average loss values for model recovery
        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()

        # for nm, param in self.model.named_parameters():
        #     # if 'compute_vida_route' in nm and 'block4' not in nm:
        #     if 'vida' in nm and 'block4' not in nm:
        #         print(nm)
        #         if param.grad == None:
        #             print(param.grad)
        #         else:
        #             print('yes')
        # print('================================================================')
        self.optimizer.second_step(zero_grad=True)

        # perform model recovery
        print(self.ema)
        if self.ema is not None:
            if self.ema < self.reset_constant_em:
                logger.info(f"ema < {self.reset_constant_em}, now reset the model")
                self.reset()
        # print(outputs[-1].shape)
        return outputs


    def flip_norm_loss_ori(self, feature_maps):
        loss = torch.tensor(0)
        for i in range(len(feature_maps)):
            # print(len(feature_maps[i]))
            x1_norm = feature_maps[i][0]
            x2_norm = feature_maps[i][1]
            B, N, C = x1_norm.shape

            x1_norm_noflip = x1_norm[:B//2]
            x1_norm_flip = x1_norm[B//2:]
            x2_norm_noflip = x2_norm[:B//2]
            x2_norm_flip = x2_norm[B//2:]

            H = int(math.sqrt(N))
            x1_norm_flip = x1_norm_flip.reshape(B//2, H, H, C).permute(0, 3, 1, 2)
            x2_norm_flip = x2_norm_flip.reshape(B//2, H, H, C).permute(0, 3, 1, 2)
            x1_norm_flip = flip(x1_norm_flip, -1).permute(0, 2, 3, 1).reshape(B//2, -1, C)
            x2_norm_flip = flip(x2_norm_flip, -1).permute(0, 2, 3, 1).reshape(B//2, -1, C)


            x1_norm_noflip_mean = x1_norm_noflip.mean(-1, keepdim=True)
            x1_norm_flip_mean = x1_norm_flip.mean(-1, keepdim=True)
            x2_norm_noflip_mean = x2_norm_noflip.mean(-1, keepdim=True)
            x2_norm_flip_mean = x2_norm_flip.mean(-1, keepdim=True)

            x1_norm_noflip_std = x1_norm_noflip.std(-1, keepdim=True)
            x1_norm_flip_std = x1_norm_flip.std(-1, keepdim=True)
            x2_norm_noflip_std = x2_norm_noflip.std(-1, keepdim=True)
            x2_norm_flip_std = x2_norm_flip.std(-1, keepdim=True)

            loss_mean = torch.mean(x1_norm_noflip_mean - x1_norm_flip_mean) + torch.mean(
                x2_norm_noflip_mean - x2_norm_flip_mean)
            loss_std = torch.mean(x1_norm_noflip_std - x1_norm_flip_std) + torch.mean(
                x2_norm_noflip_std - x2_norm_flip_std)

        loss = loss + 3*torch.abs(loss_mean)  + 2*torch.abs(loss_std)
        mean_loss = loss
        return mean_loss

    def flip_norm_loss(self, feature_maps):
        loss = torch.tensor(0)
        for i in range(len(feature_maps)):
            x1_norm = feature_maps[i][0]
            x2_norm = feature_maps[i][1]
            B, N, C = x1_norm.shape

            x1_norm_noflip = x1_norm[:B // 2]
            x1_norm_flip = x1_norm[B // 2:]
            x2_norm_noflip = x2_norm[:B // 2]
            x2_norm_flip = x2_norm[B // 2:]

            H = int(math.sqrt(N))
            x1_norm_flip = x1_norm_flip.reshape(B // 2, H, H, C).permute(0, 3, 1, 2)
            x2_norm_flip = x2_norm_flip.reshape(B // 2, H, H, C).permute(0, 3, 1, 2)
            x1_norm_flip = flip(x1_norm_flip, -1).permute(0, 2, 3, 1).reshape(B // 2, -1, C)
            x2_norm_flip = flip(x2_norm_flip, -1).permute(0, 2, 3, 1).reshape(B // 2, -1, C)

            x1_norm_noflip_mean = x1_norm_noflip.mean(-1, keepdim=True)
            x1_norm_flip_mean = x1_norm_flip.mean(-1, keepdim=True)
            x2_norm_noflip_mean = x2_norm_noflip.mean(-1, keepdim=True)
            x2_norm_flip_mean = x2_norm_flip.mean(-1, keepdim=True)

            x1_norm_noflip_std = x1_norm_noflip.std(-1, keepdim=True)
            x1_norm_flip_std = x1_norm_flip.std(-1, keepdim=True)
            x2_norm_noflip_std = x2_norm_noflip.std(-1, keepdim=True)
            x2_norm_flip_std = x2_norm_flip.std(-1, keepdim=True)

            loss_mean = torch.mean(x1_norm_noflip_mean - x1_norm_flip_mean) + torch.mean(
                x2_norm_noflip_mean - x2_norm_flip_mean)
            loss_std = torch.mean(x1_norm_noflip_std - x1_norm_flip_std) + torch.mean(
                x2_norm_noflip_std - x2_norm_flip_std)

            loss = loss + torch.abs(loss_mean) + torch.abs(loss_std)
        mean_loss = loss
        return mean_loss

    @torch.enable_grad()
    def forward_with_CL(self, x1, x2):

        feature_maps = []
        target_layer = None

        hook_handles = []
        def hook(module, input, output):
            B = output[0].shape[0]
            feature_maps.append(output)
        for name, module in self.model.named_modules():
            if 'norm' in name and 'vida' not in name and 'attn' not in name:
                target_layer = module
                hook_handles.append(target_layer.register_forward_hook(hook))
        if target_layer is None:
            raise ValueError(f"LayerNorm not found in the model")

        B = x1.shape[0]

        x1_flip = flip(x1.clone(), -1)
        x2_flip = flip(x2.clone(), -1)

        input1 = torch.cat([x1, x1_flip], dim=0)
        input2 = torch.cat([x2, x2_flip], dim=0)

        output_all = self.model(input1, input2)
        for handle in hook_handles:
            handle.remove()

        output_s = torch.stack([output_all[-1][0:B], flip(output_all[-1][B:].clone(), -1)], dim=0)
        output_s1 = F.interpolate(output_s[0], size=x1.size()[2:], mode='bilinear', align_corners=True)
        output_s2 = F.interpolate(output_s[1], size=x1.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.stack([output_s1, output_s2], dim=0).mean(0)
        output_all[-1] = pred

        loss = self.flip_norm_loss(feature_maps)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        for nm, m  in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().to(self.device)
                    with torch.no_grad():
                        p.data = self.anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        return output_all

    def forward_and_adapt(self, x1, x2):


        # return self.Original_forward(x1, x2)
        return self.forward_with_CL(x1, x2)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_some_model_and_optimizer()
        self.ema = None

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) and 'vida' not in nm:
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        vida_params = []
        vida_names = []
        for nm, param in self.model.named_parameters():
            if 'vida' in nm :
                param.requires_grad_(True)
                vida_params.append(param)
                vida_names.append(nm)
        return [params, vida_params], [names, vida_names]


    def configure_model(self):
        """Configure model for use with SAR."""
        # self.model.train()
        self.model.eval()
        # disable grad, to (re-)enable only what SAR updates
        self.model.requires_grad_(False)
        self.model = inject_trainable_vida(model=self.model, cfg=self.cfg)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for nm, m in self.model.named_modules():
            if 'vida' in nm:
                m.requires_grad_(True)
            # if ('.kv' in nm) and isinstance(m, nn.Linear):
            #     m.requires_grad_(True)
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, (nn.GroupNorm)):
            # elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def setup_optimizer(self):
        architecture_name = self.cfg.MODEL.ARCH.lower().replace("-", "_")
        if "vit_" in architecture_name or "swin_" in architecture_name:
            logger.info("Overwriting learning rate for transformers, using a learning rate of 0.001.")
            print(f'\nlr:0.001')
            return SAM(self.cfg, self.params[0], self.params[1], torch.optim.Adam, lr=0.001)
        else:
            print(f'\nlr:{self.cfg.OPTIM.LR}')
            return SAM(self.cfg, self.params[0], self.params[1], torch.optim.Adam, lr=self.cfg.OPTIM.LR)

    def load_some_model_and_optimizer(self):
        self.model.load_state_dict(self.non_router_states, strict=False)
        # print(self.optimizer_state)
        # self.optimizer.load_state_dict(self.optimizer_state)

    def get_non_router_states(self):
        non_router_states = {}
        for param_tensor in self.model.state_dict():
            if 'vida' not in param_tensor:
                non_router_states.update({param_tensor: self.model.state_dict()[param_tensor]})
        return non_router_states


class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
    def __init__(self, cfg, params, params_vida,  base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params+params_vida, defaults)

        # self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.base_optimizer = base_optimizer([{"params": params, "lr": cfg.OPTIM.LR, 'weight_decay': cfg.OPTIM.WD, 'adaptive': adaptive, 'rho': rho}, \
                                              {"params": params_vida, "lr": cfg.TTTA.LR_VIDA, 'weight_decay': cfg.TTTA.WD_VIDA, 'adaptive': adaptive,'rho': rho}])
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
