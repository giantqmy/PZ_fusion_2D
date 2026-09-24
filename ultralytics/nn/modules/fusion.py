import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from timm.layers import DropPath, to_2tuple, trunc_normal_
DEPTH_CACHE = None
from .conv import Conv
class FusionDEA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.dea = DEA(channel=channel)

    def forward(self, x1, x2):
        # DEA 期望的是 [x_vi, x_ir] 这种结构
        out = self.dea([x1, x2])
        # DEYOLO 原版是 result_vi + result_ir → sigmoid
        # 这里我们把结果作为“融合引导”
        return x1 * out, x2 * out

class DEA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x = [x, x]
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)


class DECA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        """convolution pyramid"""
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, int(self.kernel_size/kernel1/kernel2), int(self.kernel_size/kernel1/kernel2), 0,
                      groups=channel),
            nn.SiLU()
        )

    def forward(self, x):
        b, c, h, w = x[0].size()
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        # glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t))) if min(h, w) >= self.kernel_size else torch.mean(
        #                                                                             glob_t, dim=[2, 3], keepdim=True)
        glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return result_vi, result_ir


class DEPA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        """convolution merge"""
        if m_kernel is None:
            m_kernel = [3, 7]
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], 1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], 1))
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))
        #todo glob = F.adaptive_avg_pool2d(self.compress1(x[0]) + self.compress2(x[1]), 1)
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ConvMlp(nn.Module):
    """Channel MLP implemented with pointwise convolutions."""

    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class HaarDWT(nn.Module):
    """Fixed Haar wavelet decomposition into low- and high-frequency features."""

    def __init__(self, channels):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2
        weight = torch.stack([ll, lh, hl, hh], dim=0)
        self.register_buffer("weight", weight[:, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"HaarDWT expected {self.channels} channels, got {c}")
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2))

        y = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=c)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        low = y[:, :, 0]
        high = torch.cat([y[:, :, 1], y[:, :, 2], y[:, :, 3]], dim=1)
        return low, high


class WaveletS0DoLPFusion(nn.Module):
    """Adaptively fuse S0 and DoLP in Haar low- and high-frequency bands."""

    def __init__(self, dim):
        super().__init__()
        self.dwt = HaarDWT(dim)
        self.low_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, 2, 1),
        )
        self.high_gate = nn.Sequential(
            nn.Conv2d(dim * 6, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, 2, 1),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x_s0, x_dolp):
        h, w = x_s0.shape[-2:]
        ll_s0, hf_s0 = self.dwt(x_s0)
        ll_dolp, hf_dolp = self.dwt(x_dolp)

        low_w = self.low_gate(torch.cat([ll_s0, ll_dolp], dim=1)).softmax(dim=1)
        low = ll_s0 * low_w[:, 0:1] + ll_dolp * low_w[:, 1:2]

        high_w = self.high_gate(torch.cat([hf_s0, hf_dolp], dim=1)).softmax(dim=1)
        high = hf_s0 * high_w[:, 0:1] + hf_dolp * high_w[:, 1:2]
        high = self.high_proj(high)

        low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        high = F.interpolate(high, size=(h, w), mode="bilinear", align_corners=False)
        return self.out_proj(torch.cat([low, high], dim=1))


class PIM(nn.Module):
    """PoolFormer-style branch refinement with wavelet S0-DoLP fusion."""

    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.norm1_s0 = nn.BatchNorm2d(dim)
        self.norm1_dolp = nn.BatchNorm2d(dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2_s0 = nn.BatchNorm2d(dim)
        self.norm2_dolp = nn.BatchNorm2d(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp_s0 = ConvMlp(dim, hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_dolp = ConvMlp(dim, hidden_dim, act_layer=act_layer, drop=drop)
        self.wavelet_fusion = WaveletS0DoLPFusion(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        scale = layer_scale_init_value * torch.ones(dim)
        self.layer_scale_1_s0 = nn.Parameter(scale.clone())
        self.layer_scale_1_dolp = nn.Parameter(scale.clone())
        self.layer_scale_2_s0 = nn.Parameter(scale.clone())
        self.layer_scale_2_dolp = nn.Parameter(scale.clone())
        self.fusion_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _scale(gamma, x):
        return gamma.view(1, -1, 1, 1) * x

    def forward(self, x_s0, x_dolp):
        out_s0 = x_s0 + self.drop_path(
            self._scale(self.layer_scale_1_s0, self.token_mixer(self.norm1_s0(x_s0)))
        )
        out_dolp = x_dolp + self.drop_path(
            self._scale(self.layer_scale_1_dolp, self.token_mixer(self.norm1_dolp(x_dolp)))
        )

        n_s0 = self.norm2_s0(out_s0)
        n_dolp = self.norm2_dolp(out_dolp)
        wave_fusion = self.wavelet_fusion(n_s0, n_dolp)

        out_s0 = out_s0 + self.drop_path(self._scale(self.layer_scale_2_s0, self.mlp_s0(n_s0)))
        out_dolp = out_dolp + self.drop_path(self._scale(self.layer_scale_2_dolp, self.mlp_dolp(n_dolp)))
        return out_s0 + self.fusion_scale * wave_fusion, out_dolp + self.fusion_scale * wave_fusion

# class Mlp2(nn.Module):
#     """
#     Implementation of MLP with 1*1 convolutions.
#     Input: tensor with shape [B, C, H, W]
#     """
#     def __init__(self, in_features, hidden_features=None, 
#                  out_features=None, act_layer=nn.GELU, drop=0.05):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features,in_features*4, 1)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(in_features*4, in_features*2, 1)
#         self.fc3 = nn.Conv2d(in_features*2, out_features, 1)
#         self.drop = nn.Dropout(drop)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         x = self.act(x)
#         x = self.drop(x)
#         return x
#MLP3
class IdentityMlp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
class Mlp(nn.Module):
    """
    MLP 里 不建议每一层都用 GELU            Linear → Act → Linear
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.05):###我在mpf改了0.05
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features #不管这个
        self.dw = nn.Conv2d(in_features*4, in_features*4, 3, padding=1, groups=in_features*4)
        self.fc1 = nn.Conv2d(in_features,in_features*4, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_features*4, in_features*2, 1)
        self.fc3 = nn.Conv2d(in_features*2, out_features, 1)
        self.drop = nn.Dropout(drop)#训练时：10% 的特征通道 / 像素会被随机丢弃
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x) #dim 4dim
        x=  self.dw(x) #4dim 4dim
        x = self.act(x) #
        x = self.drop(x)
        x = self.fc2(x) #4dim 2dim
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        # x = self.act(x)
        # x = self.drop(x)
        return x
# class Mlp(nn.Module):
    # """
    # Implementation of MLP with 1*1 convolutions.
    # Input: tensor with shape [B, C, H, W]
    # """
    # def __init__(self, in_features, hidden_features=None, 
    #              out_features=None, act_layer=nn.GELU, drop=0.05):
    #     super().__init__()
    #     out_features = out_features or in_features
    #     hidden_features = hidden_features or in_features
    #     self.dw = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
    #     self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
    #     self.act = act_layer()
    #     self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
    #     self.drop = nn.Dropout(drop)
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.dw(x)
    #     x = self.act(x)
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


    
# class FeaturePoolingModule(nn.Module):
#     def __init__(self, dim, pool_size=3, mlp_ratio=4.,#hys你说的消融实验是改这个的大小么
#                  act_layer=nn.GELU, norm_layer=GroupNorm,
#                  drop=0., drop_path=0.,
#                  use_layer_scale=True, layer_scale_init_value=1e-5):

#         super().__init__()
#         dim=dim//2#什么意思

#         self.norm1 = norm_layer(dim)
#         self.token_mixer = Pooling(pool_size)
#         self.norm2 = norm_layer(dim)

#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
#         self.use_layer_scale = use_layer_scale
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.mlp_max = nn.Sequential(
#                     nn.Linear(dim * 2, dim * 2),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(dim * 2, 2))

#         # 为不同模态使用不同 LayerScale（更合理）
#         if use_layer_scale:
#             self.layer_scale_1_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
#             self.layer_scale_1_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
#             self.layer_scale_2_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
#             self.layer_scale_2_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

#         # 融合控制参数
#         self.alpha = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape

#         # Token mixing
#         out_1 = x1 + self.drop_path(
#             self.layer_scale_1_1.unsqueeze(-1).unsqueeze(-1) *
#             self.token_mixer(self.norm1(x1))
#         )
#         out_2 = x2 + self.drop_path(
#             self.layer_scale_1_2.unsqueeze(-1).unsqueeze(-1) *
#             self.token_mixer(self.norm1(x2))
#         )

#         n_1 = self.norm2(out_1)
#         n_2 = self.norm2(out_2)

#         n_f = torch.cat((n_1,n_2),dim=1)

#         max = self.max_pool(n_f).view(B, 2 * C)
#         max_attn = self.mlp_max(max).softmax(dim=-1)

#         w1 = max_attn[:, 0].view(B, 1, 1, 1)
#         w2 = max_attn[:, 1].view(B, 1, 1, 1)

#         out_1_weighted = n_1 * w1
#         out_2_weighted = n_2 * w2
#         # MLP
#         mlp1 = self.drop_path(
#             self.layer_scale_2_1.unsqueeze(-1).unsqueeze(-1) *
#             self.mlp(self.norm2(out_1))
#         )
#         mlp2 = self.drop_path(
#             self.layer_scale_2_2.unsqueeze(-1).unsqueeze(-1) *
#             self.mlp(self.norm2(out_2))
#         )

#         mlp_f1=out_1_weighted+mlp1
#         mlp_f2=out_2_weighted+mlp2

#         # 融合输出
#         out_x1 = out_1+mlp_f1
#         out_x2 = out_2+mlp_f2

#         return out_x1, out_x2
class FeaturePoolingModule(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,#hys你说的消融实验是改这个的大小么
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        dim=dim//2#什么意思

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # if(mlp_ratio>0)
        # self.mlp1 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp2 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp3 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if mlp_ratio > 0:
            self.mlp1 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp2 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp3 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp1 = IdentityMlp()
            self.mlp2 = IdentityMlp()
            self.mlp3 = IdentityMlp()


        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_max = nn.Sequential(
                    nn.Linear(dim * 2, dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim * 2, 2))

        # 为不同模态使用不同 LayerScale（更合理）
        if use_layer_scale:
            self.layer_scale_1_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_1_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        # 融合控制参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        B, C, H, W = x1.shape#单分支的数据

        # Token mixing
        out_1 = x1 + self.drop_path(
            self.layer_scale_1_1.unsqueeze(-1).unsqueeze(-1) *
            self.token_mixer(self.norm1(x1))
        )
        out_2 = x2 + self.drop_path(
            self.layer_scale_1_2.unsqueeze(-1).unsqueeze(-1) *
            self.token_mixer(self.norm1(x2))
        )

        n_1 = self.norm2(out_1)
        n_2 = self.norm2(out_2)

        n_f = torch.cat((n_1,n_2),dim=1)

        max = self.max_pool(n_f).view(B, 2 * C)
        max_attn = self.mlp_max(max).softmax(dim=-1)

        w1 = max_attn[:, 0].view(B, 1, 1, 1)
        w2 = max_attn[:, 1].view(B, 1, 1, 1)

        out_1_weighted = n_1 * w1
        out_2_weighted = n_2 * w2
        fusion=out_1_weighted+out_2_weighted
        fusion=self.mlp3(fusion)
        # MLP
        mlp1 = self.drop_path(
            self.layer_scale_2_1.unsqueeze(-1).unsqueeze(-1) *
            self.mlp1(self.norm2(out_1))
        )
        mlp2 = self.drop_path(
            self.layer_scale_2_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp2(self.norm2(out_2))
        )

        mlp_f1=out_1_weighted+mlp1
        mlp_f2=out_2_weighted+mlp2

        # 融合输出
        out_x1 = out_1+mlp_f1+0.5*fusion
        out_x2 = out_2+mlp_f2+0.5*fusion

        return out_x1, out_x2

class FeatureAlignmentModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureAlignmentModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

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
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2
    




class ChannelWeights(nn.Module):#通道注意力，也就是什么 
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，(B, 96, 256, 256) → (B, 96, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_avg = nn.Sequential(
                    nn.Linear(self.dim, self.dim),#如果我的输入向量是96，但是全连接层在
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp_max = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, self.dim),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # print("!!!!!!!!!!!!")
        # print(B, C, H, W)#(1,12,256,256)
        x = torch.cat((x1, x2), dim=1)
        # print("a")
        # print(x.shape)

        # Avg. Adaptive normalization
        avg = self.avg_pool(x).view(B, 2 * C)
        # print("b")
        # print("avg shape:", avg.shape)
        avg_attn = self.mlp_avg(avg).softmax(dim=-1)
        avg_x1, avg_x2 = (avg_attn.view(B, 2, 1) * avg.view(B, 2, C)).chunk(2, dim=1)
        avg_x = (avg_x1 + avg_x2).view(B, C)

        # Max. Adaptive normalization
        max = self.max_pool(x).view(B, 2 * C)
        max_attn = self.mlp_max(max).softmax(dim=-1)
        max_x1, max_x2 = (max_attn.view(B, 2, 1) * max.view(B, 2, C)).chunk(2, dim=1)
        max_x = (max_x1 + max_x2).view(B, C)

        y = torch.cat((avg_x, max_x), dim=1)
        y = self.mlp(y).view(B, self.dim, 1)
        channel_weights = y.reshape(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights
    



def _grid2win(x, nwindows, H, W):
    B, N, C = x.size()
    x = x.view(B, H, W, C)
    window_h, window_w = H//nwindows[0],  W//nwindows[1]
    x = x.view(B, window_h, nwindows[0], window_w, nwindows[1], C)
    x = torch.einsum('bhpwqc->bpqhwc', x).flatten(1, 2).flatten(-3, -2)
    return x

# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, sr_ratio=8, drop_path=0.):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.f1, self.f2, self.f3 = 0, 0, 0
        self.lepe1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim)
        self.lepe2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim)
        self.proj1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.proj2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if self.sr_ratio>1:
            if self.sr_ratio == 8:
                self.f1, self.f2, self.f3 = 4800, 9600, 4800
            elif self.sr_ratio == 4:
                self.f1, self.f2, self.f3 = 1200, 2400, 1200
            elif self.sr_ratio == 2:
                self.f1, self.f2, self.f3 = 300, 600, 300
            # elif self.sr_ratio == 1:
            #     self.f1, self.f2, self.f3 = 120, 240, 120
            self.fuse1_1 = nn.Linear(2 * dim, dim)
            self.fuse1_2 = nn.Linear(2 * dim, dim)
            self.fuse1_3 = nn.Linear(2 * dim, dim)
            self.f1_1 = nn.Linear(self.f1, 150)
            self.f1_2 = nn.Linear(self.f2, 120)
            self.f1_3 = nn.Linear(self.f3, 30)

            self.fuse2_1 = nn.Linear(2 * dim, dim)
            self.fuse2_2 = nn.Linear(2 * dim, dim)
            self.fuse2_3 = nn.Linear(2 * dim, dim)
            self.f2_1 = nn.Linear(self.f1, 150)
            self.f2_2 = nn.Linear(self.f2, 120)
            self.f2_3 = nn.Linear(self.f3, 30)
        else:
            self.fuse = nn.Linear(2 * head_dim, head_dim)

        self.end_fuse = nn.Linear(2 * dim, dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        nwindows = [16, 32]
        window_size = [H//nwindows[0], W//nwindows[1]]
        x1 = nn.Linear(12, 24)(x1)
        x2 = nn.Linear(12, 24)(x2)


        q1, k1, v1 = self.qkv1(x1).chunk(3, dim=-1)
        q2, k2, v2 = self.qkv2(x2).chunk(3, dim=-1)
        lepe1 = self.lepe1(v1.transpose(-2, -1).reshape(B, -1, H, W)).flatten(2).transpose(1, 2).contiguous()
        lepe2 = self.lepe2(v1.transpose(-2, -1).reshape(B, -1, H, W)).flatten(2).transpose(1, 2).contiguous()

        if self.sr_ratio>1:
            v12 = torch.cat([v1, v2], dim=-1)
            v_r = _grid2win(v12, nwindows, H, W)
            k1 = k1.transpose(-2, -1).view(B, 2 * C, H, W)

            k2 = k2.transpose(-2, -1).view(B, 2 * C, H, W)

            # Region Mapping
            k_r1 = F.avg_pool2d(k1.detach(), kernel_size=window_size, ceil_mode=True, count_include_pad=False)
            k_r2 = F.avg_pool2d(k2.detach(), kernel_size=window_size, ceil_mode=True, count_include_pad=False)
            k_r1 = k_r1.flatten(-2, -1).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1).contiguous()
            k_r2 = k_r2.flatten(-2, -1).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1).contiguous()

            q1 = q1.contiguous().view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q2 = q2.contiguous().view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # Construct the significance map
            a_r1 = (q1 @ k_r1.transpose(-2, -1)) * self.scale
            graph_x1 = a_r1.softmax(dim=-1)
            avg_a_r1 = torch.mean(a_r1.mean(1), dim=-2)         # B, head, 19200, 300 -> B, 19200, 300 -> B, 300
            a_r2 = (q2 @ k_r2.transpose(-2, -1)) * self.scale
            graph_x2 = a_r2.softmax(dim=-1)
            avg_a_r2 = torch.mean(a_r2.mean(1), dim=-2)

            # RGB: Rank regions by significance
            mask_sort1, mask_sort_index1 = torch.sort(avg_a_r1, dim=1)

            # top_k 

            # RGB: Aggregation Feature
            p1_scale1 = torch.gather(v_r, 1, mask_sort_index1[:, :75].contiguous().
                                     view(B, 75, 1, 1).repeat(1, 1, window_size[0]*window_size[1], 2 * C))
            p1_scale1 = self.fuse1_1(p1_scale1).flatten(1, 2).transpose(-2, -1)
            p1_scale2 = torch.gather(v_r, 1, mask_sort_index1[:, 75:225].contiguous().
                                     view(B, 150, 1, 1).repeat(1, 1, window_size[0]*window_size[1], 2 * C))
            p1_scale2 = self.fuse1_2(p1_scale2).flatten(1, 2).transpose(-2, -1)
            p1_scale3 = torch.gather(v_r, 1, mask_sort_index1[:, 225:].contiguous().
                                     view(B, 75, 1, 1).repeat(1, 1, window_size[0]*window_size[1], 2 * C))
            p1_scale3 = self.fuse1_3(p1_scale3).flatten(1, 2).transpose(-2, -1)

            seq1 = (torch.cat([self.f1_1(p1_scale1), self.f1_2(p1_scale2), self.f1_3(p1_scale3)], dim=-1).
                    reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1))


            # X: Rank regions by significance
            mask_sort2, mask_sort_index2 = torch.sort(avg_a_r2, dim=1)

            # X: Aggregation Feature
            p2_scale1 = torch.gather(v_r, 1, mask_sort_index2[:, :75].contiguous().
                                     view(B, 75, 1, 1).repeat(1, 1, window_size[0] * window_size[1], 2 * C))
            p2_scale1 = self.fuse2_1(p2_scale1).flatten(1, 2).transpose(-2, -1)
            p2_scale2 = torch.gather(v_r, 1, mask_sort_index2[:, 75:225].contiguous().
                                     view(B, 150, 1, 1).repeat(1, 1, window_size[0] * window_size[1], 2 * C))
            p2_scale2 = self.fuse2_2(p2_scale2).flatten(1, 2).transpose(-2, -1)
            p3_scale3 = torch.gather(v_r, 1, mask_sort_index2[:, 225:].contiguous().
                                     view(B, 75, 1, 1).repeat(1, 1, window_size[0] * window_size[1], 2 * C))
            p3_scale3 = self.fuse2_3(p3_scale3).flatten(1, 2).transpose(-2, -1)

            seq2 = (torch.cat([self.f2_1(p2_scale1), self.f2_2(p2_scale2), self.f2_3(p3_scale3)], dim=-1).
                    reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1))

            x1 = (graph_x1 @ seq1).transpose(1, 2).flatten(-2, -1).contiguous()
            x1 = self.drop_path(self.norm1(self.proj1(x1 + lepe1)))
            x2 = (graph_x2 @ seq2).transpose(1, 2).flatten(-2, -1).contiguous()
            x2 = self.drop_path(self.norm2(self.proj2(x2 + lepe2)))

            out_fuse = self.end_fuse(torch.cat([x1, x2], dim=-1))
            out_fuse = out_fuse.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        else:
            q1 = q1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            k1 = k1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v1 = v1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            q2 = q2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            k2 = k2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v2 = v2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

            v12 = self.fuse(torch.cat([v1, v2], dim=-1))

            ctx1 = (q1 @ k1.transpose(-2, -1)) * self.scale
            ctx1 = ctx1.softmax(dim=-1)
            ctx2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            ctx2 = ctx2.softmax(dim=-1)

            x1 = (ctx1 @ v12).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
            x1 = self.drop_path(self.norm1(self.proj1(x1 + lepe1)))
            x2 = (ctx2 @ v12).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
            x2 = self.drop_path(self.norm2(self.proj2(x2 + lepe2)))
            out_fuse = self.end_fuse(torch.cat([x1, x2], dim=-1))
            out_fuse = out_fuse.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        return out_fuse



class SelfGuidedFeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=8, drop_path=0.):
        super().__init__()
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads, sr_ratio=sr_ratio,
                                                     drop_path=drop_path)
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
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        out_fuse = self.cross_attn(x1, x2, H, W)

        return out_fuse
    


    """
Core of BiFormer, Bi-Level Routing Attention.

To be refactored.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,#沿着哪个窗口去选
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v


class MBFormer(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim, num_heads=4, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.feature_integrator = FeatureIntegrator()
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing

        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling   不做下采样操作返回原输出
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, x2, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2   如果图像的长或宽不能被 n_win整除，在第二维度左右补（0，2），在第三维上下补（0，3）
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size

            N2, H_in2, W_in2, C2 = x2.size()

            x2 = F.pad(x2, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2   如果图像的长或宽不能被 n_win整除，在第二维度左右补（0，2），在第三维上下补（0，3）
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x2.size()  # padded size
        else:
            N, H, W, C = x.size()
            print("!!!!!!!!!!!!!!!!")
            print(f"[MBF] H={H}, W={W}, n_win={self.n_win}")

            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################
        #rearrange(tensor, "输入模式 -> 输出模式", 已知维度)
        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)   #n=batch,   H = j * h,W = i * w  [B,token,H,W,c]
        x2 = rearrange(x2, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)   #n=batch,   H = j * h,W = i * w  [B,token,H,W,c]
        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)
        q2, kv2 = self.qkv(x2)
        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        ##一个窗口
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')     #[B,token,H,W,c]----->[B,token,H*W,c]
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))   #[B,token,H,W,c]----->[B*token,C,H,W]
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win) ##[B,token,H,W,c]----->[B,token,H*W,c]

        q_pix2 = rearrange(q2, 'n p2 h w c -> n p2 (h w) c')     #[B,token,H,W,c]----->[B,token,H*W,c]
        kv_pix2 = self.kv_down(rearrange(kv2, 'n p2 h w c -> (n p2) c h w'))   #[B,token,H,W,c]----->[B*token,C,H,W]
        kv_pix2 = rearrange(kv_pix2, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win) ##[B,token,H,W,c]----->[B,token,H*W,c]

        #图里面的D2是token 
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)      #[B,token,H,W,c]--->[B,token,c]
        q_win2, k_win2 = q2.mean([2, 3]), kv2[..., 0:self.qk_dim].mean([2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)      #[B,token,H,W,c]--->[B,token,c]
        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)        #[B,token,H*W,c]--->[B,H,W,C]

        lepe2 = self.lepe(rearrange(kv2[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe2 = rearrange(lepe2, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)        #[B,token,H*W,c]--->[B,H,W,C]
        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors            ##[B,token,topk]
        r_weight2, r_idx2 = self.router(q_win2, k_win2)  # both are (n, p^2, topk) tensors            ##[B,token,topk]

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)                  ##[B,token,topk,H*W,C]
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        kv_pix_sel2 = self.kv_gather(r_idx=r_idx2, r_weight=r_weight2, kv=kv_pix2)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel2, v_pix_sel2 = kv_pix_sel2.split([self.qk_dim, self.dim], dim=-1)                  ##[B,token,topk,H*W,C]
        ######### do attention as normal ####################

        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        k_pix_sel2 = rearrange(k_pix_sel2, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        # k_pix_sel = self.feature_integrator(torch.cat([k_pix_sel,k_pix_sel2],dim=-1))

        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        v_pix_sel2 = rearrange(v_pix_sel2, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        # v_pix_sel = self.feature_integrator(torch.cat([v_pix_sel.permute(0, 1, 3, 2),v_pix_sel2.permute(0, 1, 3, 2)],dim=-1))
        # v_pix_sel=v_pix_sel.permute(0, 1, 3, 2)


        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',                ###[B*token,head,token_size,c_]
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
        q_pix2 = rearrange(q_pix2, 'n p2 w2 (m c) -> (n p2) m w2 c',                ###[B*token,head,token_size,c_]
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)


        # param-free multihead attention 四维做后面两维的乘法操作 广播机制
        attn_weight = (q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)       #[B*token,head,token_size,c_]->[B*token,head,token_size,topk*h_kv*w_kv]
        attn_weight2 = (q_pix2 * self.scale) @ k_pix_sel2  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight2 = self.attn_act(attn_weight2)       #[B*token,head,token_size,c_]->[B*token,head,token_size,topk*h_kv*w_kv]

        out = attn_weight @ v_pix_sel2 # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)[B*token,head,token_size,c]
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)            #[B,H,W,C]
        out2 = attn_weight2 @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)[B*token,head,token_size,c]
        out2 = rearrange(out2, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)            #[B,H,W,C]
        out = out + lepe
        # output linear
        out = self.wo(out)

        out2 = out2 + lepe2
        # output linear
        out2 = self.wo(out2)
        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()
            out2 = out2[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out,out2


class SAFF(nn.Module):
    """Sparse attention fusion with aligned cross-branch value routing."""

    def __init__(self, dim, num_heads=4, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None,
                 kv_downsample_mode="identity", topk=4, param_attention="qkvo",
                 param_routing=False, diff_routing=False, soft_routing=False,
                 residual_kernel=3, auto_pad=False):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.auto_pad = auto_pad

        assert self.qk_dim % num_heads == 0 and dim % num_heads == 0
        assert 0 < topk <= n_win * n_win, f"topk={topk} exceeds {n_win * n_win} routing windows"
        assert not (param_routing and not diff_routing)

        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=topk,
            diff_routing=diff_routing,
            param_routing=param_routing,
        )
        if soft_routing:
            mul_weight = "soft"
        elif diff_routing:
            mul_weight = "hard"
        else:
            mul_weight = "none"
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # Keep modality-specific projections: branch 1 is RGB and branch 2 is DoLP.
        self.qkv1 = QKVLinear(dim, self.qk_dim)
        self.qkv2 = QKVLinear(dim, self.qk_dim)
        if param_attention == "qkvo":
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
        elif param_attention == "qkv":
            self.proj1 = nn.Identity()
            self.proj2 = nn.Identity()
        else:
            raise ValueError(f"Unsupported param_attention mode: {param_attention}")

        self.kv_down = self._make_kv_downsample(
            kv_downsample_mode, kv_per_win, kv_downsample_ratio, kv_downsample_kernel
        )
        padding = residual_kernel // 2
        self.residual1 = nn.Conv2d(dim, dim, residual_kernel, padding=padding, bias=False)
        self.residual2 = nn.Conv2d(dim, dim, residual_kernel, padding=padding, bias=False)
        self.attn_act = nn.Softmax(dim=-1)
        self._shape_debug_printed = False

    @staticmethod
    def _make_kv_downsample(mode, kv_per_win, ratio, kernel):
        if mode == "identity":
            return nn.Identity()
        if mode == "ada_avgpool":
            return nn.AdaptiveAvgPool2d(kv_per_win)
        if mode == "ada_maxpool":
            return nn.AdaptiveMaxPool2d(kv_per_win)
        if mode == "avgpool":
            return nn.AvgPool2d(ratio) if ratio > 1 else nn.Identity()
        if mode == "maxpool":
            return nn.MaxPool2d(ratio) if ratio > 1 else nn.Identity()
        if mode in {"fracpool", "conv"}:
            raise NotImplementedError(f"kv_downsample_mode={mode} is not implemented")
        raise ValueError(f"Unsupported kv_downsample_mode: {mode}")

    def _partition(self, x):
        return rearrange(
            x, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win
        )

    def _flatten_kv(self, kv):
        kv = self.kv_down(rearrange(kv, "n p h w c -> (n p) c h w"))
        return rearrange(kv, "(n j i) c h w -> n (j i) (h w) c", j=self.n_win, i=self.n_win)

    def _format_key(self, key):
        return rearrange(
            key, "n p k w (m c) -> (n p) m c (k w)", m=self.num_heads
        )

    def _format_value(self, value):
        return rearrange(
            value, "n p k w (m c) -> (n p) m (k w) c", m=self.num_heads
        )

    def _format_query(self, query):
        return rearrange(
            query, "n p w (m c) -> (n p) m w c", m=self.num_heads
        )

    def _restore(self, x, batch, height, width):
        return rearrange(
            x,
            "(n j i) m (h w) c -> n (m c) (j h) (i w)",
            n=batch,
            j=self.n_win,
            i=self.n_win,
            h=height // self.n_win,
            w=width // self.n_win,
        )

    def forward(self, x1, x2, ret_attn_mask=False):
        """Fuse two aligned NCHW feature maps and preserve their branch identities."""
        if x1.shape != x2.shape:
            raise ValueError(f"SAFF inputs must have identical shapes, got {x1.shape} and {x2.shape}")
        if x1.ndim != 4 or x1.shape[1] != self.dim:
            raise ValueError(f"SAFF expects NCHW inputs with {self.dim} channels, got {x1.shape}")

        batch, _, input_h, input_w = x1.shape
        residual1 = self.residual1(x1)
        residual2 = self.residual2(x2)

        pad_h = (self.n_win - input_h % self.n_win) % self.n_win
        pad_w = (self.n_win - input_w % self.n_win) % self.n_win
        if (pad_h or pad_w) and not self.auto_pad:
            raise ValueError(f"Feature size {(input_h, input_w)} must be divisible by n_win={self.n_win}")
        if pad_h or pad_w:
            x1 = F.pad(x1, (0, pad_w, 0, pad_h))
            x2 = F.pad(x2, (0, pad_w, 0, pad_h))
        height, width = x1.shape[-2:]

        win1, win2 = self._partition(x1), self._partition(x2)
        q1, kv1 = self.qkv1(win1)
        q2, kv2 = self.qkv2(win2)
        q_pix1 = rearrange(q1, "n p h w c -> n p (h w) c")
        q_pix2 = rearrange(q2, "n p h w c -> n p (h w) c")
        kv_pix1, kv_pix2 = self._flatten_kv(kv1), self._flatten_kv(kv2)

        q_win1 = q1.mean((2, 3))
        q_win2 = q2.mean((2, 3))
        k_win1 = kv1[..., :self.qk_dim].mean((2, 3))
        k_win2 = kv2[..., :self.qk_dim].mean((2, 3))
        route_weight1, route_idx1 = self.router(q_win1, k_win1)
        route_weight2, route_idx2 = self.router(q_win2, k_win2)

        query1, query2 = self._format_query(q_pix1), self._format_query(q_pix2)

        # Process one routed cross-modal path at a time.  The route semantics are
        # unchanged (route 1: K1+V2, route 2: K2+V1), but the two large gather
        # results are never kept simultaneously, reducing the training peak.
        kv_for_route1 = torch.cat((kv_pix1[..., :self.qk_dim], kv_pix2[..., self.qk_dim:]), dim=-1)
        selected_route1 = self.kv_gather(route_idx1, route_weight1, kv_for_route1)
        del kv_for_route1
        key1 = self._format_key(selected_route1[..., :self.qk_dim])
        value2_by_1 = self._format_value(selected_route1[..., self.qk_dim:])
        del selected_route1
        if not self._shape_debug_printed:
            attention_shape = (*query1.shape[:-1], key1.shape[-1])
            attention_numel = math.prod(attention_shape)
            attention_mib = attention_numel * query1.element_size() / (1024 ** 2)
            rank = os.environ.get("LOCAL_RANK", "0")
            print(
                f"[SAFF DEBUG rank={rank}] dim={self.dim} n_win={self.n_win} topk={self.router.topk} "
                f"input={tuple(x1.shape)} dtype={x1.dtype} "
                f"query={tuple(query1.shape)} key={tuple(key1.shape)} "
                f"attention={attention_shape} estimated={attention_mib:.2f} MiB",
                flush=True,
            )
            self._shape_debug_printed = True
        attention1 = self.attn_act((query1 * self.scale) @ key1)
        cross_to_2 = self._restore(attention1 @ value2_by_1, batch, height, width)
        del key1, value2_by_1

        kv_for_route2 = torch.cat((kv_pix2[..., :self.qk_dim], kv_pix1[..., self.qk_dim:]), dim=-1)
        selected_route2 = self.kv_gather(route_idx2, route_weight2, kv_for_route2)
        del kv_for_route2
        key2 = self._format_key(selected_route2[..., :self.qk_dim])
        value1_by_2 = self._format_value(selected_route2[..., self.qk_dim:])
        del selected_route2
        attention2 = self.attn_act((query2 * self.scale) @ key2)
        cross_to_1 = self._restore(attention2 @ value1_by_2, batch, height, width)
        del key2, value1_by_2

        cross_to_1 = self.proj1(cross_to_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        cross_to_2 = self.proj2(cross_to_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        cross_to_1 = cross_to_1[..., :input_h, :input_w]
        cross_to_2 = cross_to_2[..., :input_h, :input_w]

        output1 = residual1 + cross_to_1
        output2 = residual2 + cross_to_2
        if ret_attn_mask:
            routing = {
                "branch1": (route_weight1, route_idx1, attention1),
                "branch2": (route_weight2, route_idx2, attention2),
            }
            return output1, output2, routing
        return output1, output2


class FeatureIntegrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_cache = {}

    def forward(self, v_pix_sel):
        """
        v_pix_sel: (B, M, L, C)
        输出: (B, M, L_out, C)
        """
        B, M, C, L = v_pix_sel.shape
        # print(v_pix_sel.shape)

        # 动态创建 MLP: L -> 2L -> L/4
        if L not in self.fc_cache:
            out_dim = max(L // 4, 1)  # 避免 L 太小导致 0
            self.fc_cache[L] = nn.Sequential(
                nn.Linear(L, 2 * L),
                nn.ReLU(inplace=True),
                nn.Linear(2 * L, out_dim)
            ).to(v_pix_sel.device, dtype=v_pix_sel.dtype)

        mlp = self.fc_cache[L].to(v_pix_sel.device)

        # 全连接作用于序列维度 L
        v_fc = mlp(v_pix_sel)  # (B, M, C, L)
        v_fc = v_fc         #.permute(0, 1, 3, 2)            # (B, M, L_out, C)
        # print(v_fc.shape)
        return v_fc
