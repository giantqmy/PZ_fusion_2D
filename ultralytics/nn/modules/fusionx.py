import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath, to_2tuple, trunc_normal_

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

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class FeaturePoolingModuleown(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale

        #两条支路的layer_scale也要区别
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x1, x2):
        out_1 = x1 + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x1)))
        out_2 = x2 + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x2)))
        
        out_3 = out_1 + out_2

        mlp1 = self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(out_1)))
        
        mlp2 = self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(out_2)))
        
        mlp3 = self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(out_3)))
        
          
        out_x1 = mlp1 + mlp3
        out_x2 = mlp2 + mlp3

        return out_x1,out_x2
    
class FeaturePoolingModule(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_max = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2, 2))

        # 为不同模态使用不同 LayerScale（更合理）
        if use_layer_scale:
            self.layer_scale_1_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_1_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        # 融合控制参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

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
        # MLP
        mlp1 = self.drop_path(
            self.layer_scale_2_1.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(out_1))
        )
        mlp2 = self.drop_path(
            self.layer_scale_2_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(out_2))
        )

        mlp_f1=out_1_weighted+mlp1
        mlp_f2=out_2_weighted+mlp2

        # 融合输出
        out_x1 = out_1+mlp_f1
        out_x2 = out_2+mlp_f2

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