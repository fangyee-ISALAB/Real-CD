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


class Cattn(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        hidden_features = hidden_features
        self.vida_fc1 = nn.Linear(in_features, hidden_features)
        self.vida_dwconv = DWConv(hidden_features)
        self.vida_act = nn.GELU()
        self.vida_fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)
        # self.vida_last_act = nn.Tanh()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.vida_fc1(x)
        x = self.vida_dwconv(x, H, W)
        x = self.vida_act(x)
        x = self.vida_fc2(x)
        # x = self.vida_last_act(x)
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

            # self.vida_proj1 = nn.Linear(self.in_features, int(self.in_features/4)).to("cuda")
            # self.vida_proj2 = nn.Linear(self.in_features, int(self.in_features/4)).to("cuda")
            # self.vida_proj3 = nn.Linear(self.in_features, int(self.in_features/4)).to("cuda")


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

            # diff_logit = self.get_diff(input, input2)
            # conn_logit = self.vida_adaptive_avgpool(torch.concat((input.detach(), input2.detach()), dim=1))


            conn1_logit = self.get_logit(input, input2)
            conn2_logit = self.get_logit(input2, input)
            diff_logit = self.vida_proj1(input-input2)
            conn1_logit = self.vida_proj2(conn1_logit)
            conn2_logit = self.vida_proj2(conn2_logit)
            input1_logit = self.vida_proj3(input)
            input2_logit = self.vida_proj3(input2)

            logit = self.vida_adaptive_avgpool(torch.concat((input1_logit,  input2_logit, conn1_logit, conn2_logit,diff_logit), dim=1).detach().permute(0, 2, 1)).mean(0)

            # diff_logit, conn_logit = self.get_logit(input, input2)
            # diff_logit = self.vida_proj1(diff_logit

            # conn_logit = self.vida_proj2(conn_logit)
            # input1_logit BUTONG= self.vida_proj3(input)
            # input2_logit = self.vida_proj3(input2)
            # logit = self.vida_adaptive_avgpool(torch.concat((input1_logit, input2_logit, conn_logit, diff_logit), dim=-1).detach().permute(0, 2, 1)).mean(0)
            # print(logit.shape)

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



class ViDAInjectGatedLinear(nn.Module):
    def __init__(self, cfg, in_features, out_feature, weight, bias):
        super().__init__()
        self.route_hidden_dim = cfg.TTTA.HIDDEN_DIM
        self.dropout = cfg.TTTA.DROPOUT
        self.out_dim2 = 64
        self.in_features = in_features
        self.vida_refine = Cattn(in_features, self.route_hidden_dim).to('cuda')
        self.compute_vida_route = Vida_Router(1, self.route_hidden_dim, 2, self.dropout).to('cuda')
        self.vida_softmax = F.gumbel_softmax
        self.qkv = nn.Linear(in_features, out_feature).to('cuda')
        self.qkv.weight.data = weight
        self.qkv.bias.data = bias

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
        return csim

    def forward(self, input, H, W, input2=None):
        if input2 != None:
            q1 = self.qkv(input)
            refine_x2 = input2 + self.vida_refine(input2, H, W)
            q2 = self.qkv(refine_x2)
            csim = self.get_logit(input, refine_x2)
            router = self.compute_vida_route(csim)
            router = self.vida_softmax(router, hard=False, dim=-1)[:, :, 0]
            return q1, q2, router
        else:
            return self.qkv(input)



def inject_trainable_vida(model: nn.Module, cfg):
    """
    inject vida into model, and returns vida parameter groups.
    """

    require_grad_params = []
    names = []
    # target_replace_linear_module: List[str] = ["Attention"]
    target_replace_linear_module: List[str] = []
    target_replace_norm_module: List[str] = ['OverlapPatchEmbed', 'EncoderTransformer_v3', 'Block']
    # target_replace_norm_module: List[str] = ['OverlapPatchEmbed', 'EncoderTransformer_v3']
    # target_replace_norm_module: List[str] = []
    all_vida_params = {}

    for mo_name, _module in list(model.named_modules()):
        if _module.__class__.__name__ in target_replace_linear_module:
            for name, _child_module in list(_module.named_modules()):
                if 'q' in name and _child_module.__class__.__name__ == "Linear":
                    if 'block3' in mo_name or 'block4' in mo_name or 'block2' in mo_name:
                        continue
                    print(f"{mo_name}_{name}")
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = ViDAInjectGatedLinear(
                         cfg,
                        _child_module.in_features,
                        _child_module.out_features,
                        weight,
                        bias,
                    )

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].vida_refine.parameters())
                    )
                    _module._modules[name].vida_refine.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].compute_vida_route.parameters())
                    )
                    _module._modules[name].compute_vida_route.requires_grad = True

                    _module._modules[name].vida_softmax.requires_grad = True
                    names.append(mo_name)
        if _module.__class__.__name__ in target_replace_norm_module:
            for name, _child_module in list(_module.named_modules()):
                if _child_module.__class__.__name__ == "LayerNorm" and '.norm' not in name:
                    print(f"{mo_name}_{name}")
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
                        # print('yes')
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