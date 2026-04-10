import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial

# from detectron2.model_zoo.configs.common.models.mask_rcnn_vitdet import num_heads
from fontTools.ttLib.tables.S__i_l_f import attrs_attributes
from mmcv.cnn import Linear

from models.ChangeFormerBaseNetworks import *
from models.help_funcs import TwoLayerConv2d, save_to_mat
import torch.nn.functional as F

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import normal_init
# from mmcv.cnn import ConvModule
import pdb

from scipy.io import savemat

from models.pixel_shuffel_up import PS_UP


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

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

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

    def forward(self, x1, x2):
        # pdb.set_trace()
        x1 = self.proj(x1)
        _, _, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)

        x2 = self.proj(x2)
        _, _, H, W = x2.shape
        x2 = x2.flatten(2).transpose(1, 2)

        x1, x2 = self.norm(x1, x2)
        # x1 = self.norm(x1)
        # x2 = self.norm(x2)

        return x1, x2, H, W


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

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
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

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

    def forward(self, x1, x2, H, W , Cross_flag):

        # print(f"tx1:{torch.all(x1).item()}")
        if Cross_flag:
            B, N, C = x1.shape
            q1, q2, router = self.q(x1, H, W, x2)
            q1 = q1.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q2 = q2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if self.sr_ratio > 1:
                x1_ = x1.permute(0, 2, 1).reshape(B, C, H, W)
                x1_ = self.sr(x1_).reshape(B, C, -1).permute(0, 2, 1)

                x2_ = x2.permute(0, 2, 1).reshape(B, C, H, W)
                x2_ = self.sr(x2_).reshape(B, C, -1).permute(0, 2, 1)
                # x1_ = self.norm(x1_)
                x1_, _ = self.norm(x1_, x2_)

                kv1 = self.kv(x1_)
                kv1 = kv1.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv1 = self.kv(x1)
                kv1 = kv1.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]
            k11 = k1
            v11 = v1

            router = router.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, k1.shape[-2])
            attn1 = ((q1 @ k11.transpose(-2, -1)) * self.scale).to(torch.float64)
            attn2 = ((q2 @ k11.transpose(-2, -1)) * self.scale).to(torch.float64)
            attn = attn1  + attn2 * (1-router)
            attn = attn.to(torch.float32).softmax(dim=-1)

            attn = self.attn_drop(attn)
            x1 = (attn @ v11).transpose(1, 2).reshape(B, N, C)
            x1 = self.proj(x1)
            x1 = self.proj_drop(x1)
        else:
            B, N, C = x1.shape
            q1 = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if self.sr_ratio > 1:
                x1_ = x1.permute(0, 2, 1).reshape(B, C, H, W)
                x1_ = self.sr(x1_).reshape(B, C, -1).permute(0, 2, 1)

                x1_ = self.norm(x1_)

                kv1 = self.kv(x1_)
                kv1 = kv1.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv1 = self.kv(x1)
                kv1 = kv1.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]
            attn = (q1 @ k1.transpose(-2, -1) * self.scale).to(torch.float64)
            attn = attn.to(torch.float32).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x1 = (attn @ v1).transpose(1, 2).reshape(B, N, C)
            x1 = self.proj(x1)
            x1 = self.proj_drop(x1)
        return x1


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

        # self.modify_weight = nn.Sequential(
        #     nn.Linear(dim, mlp_hidden_dim),
        #     nn.Dropout(drop),
        #     nn.ReLU(),
        #     nn.Linear(mlp_hidden_dim, dim),
        # )

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

    def forward(self, x1, x2, H, W, Cross_flag):

        x1_norm, x2_norm = self.norm1(x1, x2)
        # x1_norm = self.norm1(x1)
        # x2_norm = self.norm1(x2)


        x1_ = self.attn(x1_norm, x2_norm, H, W, Cross_flag)
        x2_ = self.attn(x2_norm, x1_norm, H, W, Cross_flag)
        x1 = x1 + self.drop_path(x1_)
        x2 = x2 + self.drop_path(x2_)

        x1_, x2_ = self.norm2(x1, x2)
        # x1_ = self.norm2(x1)
        # x2_ = self.norm2(x2)

        x1 = x1 + self.drop_path(self.mlp(x1_, H, W))
        x2 = x2 + self.drop_path(self.mlp(x2_, H, W))

        return x1, x2


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


# Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


# Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


# Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer_v3(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)



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

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward(self, x1, x2):

        B = x1.shape[0]
        out1s = []
        out2s = []

        # stage 1
        x1, x2, H1, W1 = self.patch_embed1(x1, x2)
        for i, blk in enumerate(self.block1):
            x1, x2 = blk(x1, x2, H1, W1, False)

        x1, x2 = self.norm1(x1, x2)
        # x1 = self.norm1(x1)
        # x2 = self.norm1(x2)

        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        out1s.append(x1)
        out2s.append(x2)

        # stage 2
        x1, x2, H1, W1 = self.patch_embed2(x1, x2)
        # print(x1.shape, H1, W1)
        for i, blk in enumerate(self.block2):
            x1, x2 = blk(x1, x2, H1, W1, False)

        x1, x2 = self.norm2(x1, x2)
        # x1 = self.norm2(x1)
        # x2 = self.norm2(x2)

        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        out1s.append(x1)
        out2s.append(x2)

        # stage 3
        x1, x2, H1, W1 = self.patch_embed3(x1, x2)
        for i, blk in enumerate(self.block3):
            x1, x2 = blk(x1, x2, H1, W1, False)

        x1, x2 = self.norm3(x1, x2)
        # x1 = self.norm3(x1)
        # x2 = self.norm3(x2)

        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        out1s.append(x1)
        out2s.append(x2)

        # stage 4+
        x1, x2, H1, W1 = self.patch_embed4(x1, x2)
        for i, blk in enumerate(self.block4):
            x1, x2 = blk(x1, x2, H1, W1, False)

        x1, x2 = self.norm4(x1, x2)
        # x1 = self.norm4(x1)
        # x2 = self.norm4(x2)

        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        out1s.append(x1)
        out2s.append(x2)
        return out1s, out2s


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer_v3, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


# ChangeFormerV6:
class ChangeFormer_TTTA(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(ChangeFormer_TTTA, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        # Transformer Encoder
        self.Tenc_x2 = EncoderTransformer_v3(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                             embed_dims=self.embed_dims,
                                             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                             qk_scale=None, drop_rate=self.drop_rate,
                                             attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             depths=self.depths, sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                             output_nc=output_nc,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16])

    def forward(self, x1, x2):
        # print(f"inx1:{torch.all(torch.isnan(x1)).item()}")

        fx1, fx2 = self.Tenc_x2(x1, x2)
        cp = self.TDec_x2(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp