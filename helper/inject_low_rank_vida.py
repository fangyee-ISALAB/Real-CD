import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.linalg
import torch.nn as nn
from matplotlib.pyplot import connect
from sklearn.decomposition import PCA
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def applyPCA(X, numComponents=30):
    B, N, C = X.shape
    X = X.permute(0, 2, 1)
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(np.asarray(X.cpu()).reshape(-1, N))
    X = torch.tensor(np.reshape(newX, (B, C, numComponents))).to('cuda')
    return X


def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(input1, input2):
    feat_mean1, feat_std1 = calc_mean_std(input1)
    feat_mean2, feat_std2 = calc_mean_std(input2)
    feat1 = (input1 - feat_mean1) / feat_std1
    feat2 = (input2 - feat_mean2) / feat_std2
    feat1 = feat1 * feat_std2 + feat_mean2
    feat2 = feat2 * feat_std1 + feat_mean1
    return feat1, feat2

class Vida_Router(nn.Module):
    def __init__(self, in_features, route_hidden_dim, out_feature, dropout):
        super().__init__()
        self.dim = 2
        self.compute_vida_route = nn.Sequential(
            nn.Linear(in_features, route_hidden_dim),
            nn.ReLU(),
            nn.Linear(route_hidden_dim, out_feature),
            nn.Sigmoid(),
        )

    def forward(self, input1):
        return self.compute_vida_route(input1)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class ViDAInjectedLNormLayer(nn.Module):
    def __init__(self, cfg, oldLayerNorm, model_state_dict):
        super().__init__()
        self.in_features = oldLayerNorm.weight.shape[0]
        self.out_features = 2
        self.route_hidden_dim = cfg.TTTA.HIDDEN_DIM
        self.dropout = cfg.TTTA.DROPOUT

        self.weight = oldLayerNorm.weight.to('cuda')
        self.bias = oldLayerNorm.bias.to('cuda')

        self.finetune_ln = True
        if model_state_dict == None:
            self.finetune_ln = False
        else:
            self.vida_scale_w1 = model_state_dict[0]
            self.vida_scale_b1 = model_state_dict[1]
            self.vida_scale_w2 = model_state_dict[2]
            self.vida_scale_b2 = model_state_dict[3]
            self.compute_vida_route1 = Vida_Router(64, self.route_hidden_dim, 2, self.dropout).to(
                'cuda')
            self.compute_vida_route2 = Vida_Router(64, self.route_hidden_dim, self.out_features, self.dropout).to(
                'cuda')
            self.vida_softmax = F.gumbel_softmax
            self.vida_adaptive_avgpool = nn.AdaptiveAvgPool1d(output_size=64)

            self.vida_proj1 = nn.Linear(self.in_features, self.in_features).to("cuda")
            self.vida_proj2 = nn.Linear(self.in_features, self.in_features).to("cuda")
            self.vida_proj3 = nn.Linear(self.in_features, self.in_features).to("cuda")

    def get_logit(self, input, input2):
        B, N, C = input.shape
        input_mean = input.mean(1).unsqueeze(1)
        input_ = input_mean.repeat(1, N, 1)
        dot_product = torch.sum(input2 * input_, dim=-1, keepdim=True)
        norm1 = torch.norm(input_, dim=-1, keepdim=True)
        norm2 = torch.norm(input2, dim=-1, keepdim=True)
        csim = dot_product / (norm1 * norm2)
        for i in range(B):
            if csim[i].max() != csim[i].min():
                csim_clone = csim[i].clone()
                csim[i] = (csim_clone - csim_clone.min()) / (csim_clone.max() - csim_clone.min())
        weighted_logit = torch.mul(input, csim)
        weighted_sum = torch.sum(csim, dim=1).unsqueeze(-1)
        conn = torch.div(weighted_logit, weighted_sum)
        return conn.to("cuda")

    def get_diff(self, input1, input2):
        B, N, C = input1.shape
        f1 = input1.reshape(B, N, 2, C//2).permute(0, 2, 1, 3)
        f2 = input2.reshape(B, N, 2, C//2).permute(0, 2, 1, 3)
        conn = f1 @ f2.permute(0, 1, 3, 2)
        conn = F.softmax(conn, dim=-1)
        diff = conn[:, 0] - conn[:, 1]
        return self.vida_adaptive_avgpool(diff)

    def forward(self, input, input2=None):
        if input2 !=None and self.finetune_ln:

            conn1_logit = self.get_logit(input, input2)
            conn2_logit = self.get_logit(input2, input)
            diff_logit = self.vida_proj1(input-input2)
            conn1_logit = self.vida_proj2(conn1_logit)
            conn2_logit = self.vida_proj2(conn2_logit)
            input1_logit = self.vida_proj3(input)
            input2_logit = self.vida_proj3(input2)

            logit = self.vida_adaptive_avgpool(torch.concat((input1_logit,  input2_logit, conn1_logit, conn2_logit,diff_logit), dim=1).detach().permute(0, 2, 1)).mean(0)

            scale_scores = self.compute_vida_route1(logit)
            shift_router = self.compute_vida_route2(logit)
            shift_router = self.vida_softmax(shift_router, hard=True, dim=-1)[:, 0]

            weight1 = self.weight.data * (1+scale_scores[:,0]) + shift_router * self.vida_scale_w1
            bias1 = self.bias.data * (1+scale_scores[:,1]) + shift_router * self.vida_scale_b1
            mean1 = input.mean(-1, keepdim=True)
            std1 = input.std(-1, keepdim=True)
            output1 = weight1 * ((input - mean1) / (std1 + 1e-10)) + bias1

            weight2 = self.weight.data * (1+scale_scores[:,0]) + shift_router * self.vida_scale_w2
            bias2 = self.bias.data * (1+scale_scores[:,1]) + shift_router * self.vida_scale_b2
            mean2 = input2.mean(-1, keepdim=True)
            std2 = input2.std(-1, keepdim=True)
            output2 = weight2 * ((input2 - mean2) / (std2 + 1e-10)) + bias2

            return output1, output2
        else:
            weight = self.weight.data
            bias = self.bias.data
            mean1 = input.mean(-1, keepdim=True)
            std1 = input.std(-1, keepdim=True)
            output1 = weight * (input - mean1) / (std1 + 1e-9) + bias
            if input2 !=None:
                mean2 = input2.mean(-1, keepdim=True)
                std2 = input2.std(-1, keepdim=True)
                output2 = weight * ((input2 - mean2) / (std2 + 1e-10)) + bias
                return output1, output2
            else:
                return output1


def inject_trainable_vida(model: nn.Module, cfg):
    """
    inject vida into model, and returns vida parameter groups.
    """

    require_grad_params = []
    names = []
    target_replace_norm_module: List[str] = ['OverlapPatchEmbed', 'EncoderTransformer_v3', 'Block']
    all_vida_params = {}

    for mo_name, _module in list(model.named_modules()):
        if _module.__class__.__name__ in target_replace_norm_module:
            for name, _child_module in list(_module.named_modules()):
                if _child_module.__class__.__name__ == "LayerNorm" and '.norm' not in name:
                    if 'block6' in mo_name:
                        print("no adapt")
                        _tmp = ViDAInjectedLNormLayer(
                            cfg,
                            _child_module,
                            None,
                        )
                        _module._modules[name] = _tmp
                    else:
                        params_name = 'vida_' + mo_name + '_' + name
                        params_name = params_name.replace(".", "_")
                        four_params_name_list = [f'{params_name}_{i}' for i in range(4)]
                        if params_name not in all_vida_params.keys():
                            vida_list_params = nn.ParameterList(
                                [nn.Parameter(torch.zeros_like(_child_module.weight)) for i in range(4)])
                            all_vida_params[params_name] = vida_list_params.cuda()

                        _tmp = ViDAInjectedLNormLayer(
                            cfg,
                            _child_module,
                            all_vida_params[params_name],
                        )

                        for i, param in enumerate(all_vida_params[params_name]):
                            param.requires_grad = True
                            model.register_parameter(four_params_name_list[i], param)
                            require_grad_params.extend(
                                list(four_params_name_list[i])
                            )

                        # switch the module
                        _module._modules[name] = _tmp

                        require_grad_params.extend(
                            list(_module._modules[name].compute_vida_route1.parameters())
                        )
                        require_grad_params.extend(
                            list(_module._modules[name].compute_vida_route2.parameters())
                        )

                        require_grad_params.extend(
                            list(_module._modules[name].vida_adaptive_avgpool.parameters())
                        )
                        require_grad_params.extend(
                            list(_module._modules[name].vida_proj1.parameters())
                        )
                        require_grad_params.extend(
                            list(_module._modules[name].vida_proj2.parameters())
                        )
                        require_grad_params.extend(
                            list(_module._modules[name].vida_proj3.parameters())
                        )


                        _module._modules[name].compute_vida_route1.requires_grad = True
                        _module._modules[name].vida_adaptive_avgpool.requires_grad = True
                        _module._modules[name].compute_vida_route2.requires_grad = True
                        _module._modules[name].vida_softmax.requires_grad = True
                        _module._modules[name].vida_proj1.requires_grad = True
                        _module._modules[name].vida_proj2.requires_grad = True
                        _module._modules[name].vida_proj3.requires_grad = True
                        names.append(name)
    return model