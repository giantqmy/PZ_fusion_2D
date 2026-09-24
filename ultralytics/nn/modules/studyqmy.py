import torch
import torch.nn as nn
from .fusion import FeatureAlignmentModule as FAM
from .fusion import SelfGuidedFeatureFusionModule as SGFFM
from .fusion import FeaturePoolingModule as FPM
from .fusion import MBFormer as MBF
import torch.nn.functional as F

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                  # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class channel1(nn.Module):
    def __init__(self, c_in, dim=96, **kwargs):
        super().__init__()
        assert dim % 2 == 0, f"dim should be even to split equally, got {dim}"

        half_in_1 = 6
        half_in_2=4
        half_out = dim // 2
        self.conv = nn.Conv2d(2, 3, kernel_size=1) 

        # 两支的输入通道用 half_in，而不是写死 4
        self.branch1 = nn.Sequential(
            nn.Conv2d(half_in_1, half_in_1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(half_in_1, half_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(half_out),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(half_in_2, half_in_2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(half_in_2, half_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(half_out),
            nn.SiLU()
        )
        # 下面保持你已有模块初始化（参数按需调整）
        self.fam = FAM(dim=dim, reduction=1)
        self.sgfm = SGFFM(dim=dim, reduction=1, num_heads=4, sr_ratio=8, drop_path=0.1)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.fpm = FPM(dim=dim,pool_size=3,mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5)
        
        self.mbf = MBF(dim=dim//2, num_heads=1, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False)

        self.conv = nn.Conv2d(2, 3, kernel_size=1) 
        self.depth_gate = nn.Sequential(
                    nn.Conv2d(1, dim // 4, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(dim // 4),
                    nn.SiLU(),
                    nn.Conv2d(dim // 4, dim, kernel_size=1),
                    nn.Sigmoid()
            )
        self.c2f = C2f(c1=dim, c2=dim, n=2)
        self.rgb_att = ECA(k_size=3)
        self.pol_att = ECA(k_size=3)
        self.nir_att = ECA(k_size=3)
        self.polnir_att = ECA(k_size=3)


    def forward(self, x):
        AoLP_rgb = x[:,6:7,:,:].float() / 255.0 * math.pi - math.pi/2
        AoLP_nir = x[:,7:8,:,:].float() / 255.0 * math.pi - math.pi/2

        AoLP_rgb_sin = torch.sin(2 * AoLP_rgb)
        AoLP_rgb_cos = torch.cos(2 * AoLP_rgb)

        AoLP_nir_sin = torch.sin(2 * AoLP_nir)
        AoLP_nir_cos = torch.cos(2 * AoLP_nir)
        x1_1=x[:,0:3,:,:] #BCHW 切片
        # x1_2=x[:,6:7,:,:]
        x1_3=x[:,4:5,:,:]
        # ---- 反归一化 AoLP ----
        rgb = self.rgb_att(x1_1)
        pol_rgb= torch.concat((AoLP_rgb_sin, AoLP_rgb_cos, x1_3), dim=1)
        pol_rgb = self.pol_att(pol_rgb)


        x1=torch.concat((rgb,pol_rgb),dim=1)

        x2_1=x[:,3:4,:,:]
        # x2_2=x[:,7:8,:,:]
        x2_3=x[:,5:6,:,:]
        nir=self.nir_att(x2_1)
        pol_nir=torch.concat((AoLP_nir_sin,AoLP_nir_cos,x2_3),dim=1)
        pol_nir=self.polnir_att(pol_nir)
        x2=torch.concat((nir,pol_nir),dim=1)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3=x[:,8:9,:,:]

        global DEPTH_CACHE
        DEPTH_CACHE = x3
        x_feat = torch.cat([x1, x2], dim=1) 
        x_feat = self.c2f(x_feat)
        depth_norm = torch.exp(-(4 - torch.floor_divide(x3.float(), 64)))  # (B,1,H,W)
        depth_norm = F.interpolate(
        depth_norm, 
        size=x_feat.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    )
        depth_norm = depth_norm.to(x_feat.dtype) 

    # ================= 深度门控 + 残差 =================
        gate = self.depth_gate(depth_norm)  # (B, dim, H, W)
        x_next = x_feat * gate + x_feat      # 残差连接，保留原特征

         # (B, dim, H, W)
        return x_next
    
class ChannelSplitBlockFusion(nn.Module):
    """
    支持任意偶数输入通道 c_in：
    - 将输入分成两半 (c_in//2, c_in//2)
    - 两支分别经过 conv -> BN -> SiLU 变为 (dim//2, dim//2)
    - 经过 FAM / SGFFM 产生 fused (dim)
    返回: x_next (B, dim, H, W), x_fused (B, dim, H, W)
    注意：该模块返回 tuple，通常用于被上层模块调用（例如 StackedChannelSplitFusion）。
    """
    def __init__(self, c_in, dim=96, **kwargs):
        super().__init__()
        assert c_in % 2 == 0, f"in channels must be even, got {c_in}"
        assert dim % 2 == 0, f"dim should be even to split equally, got {dim}"

        half_in = c_in // 2
        half_out = dim // 2

        # 两支的输入通道用 half_in，而不是写死 4
        self.branch1 = nn.Sequential(
            nn.Conv2d(half_in, half_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(half_in, half_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(half_out),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(half_in, half_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(half_in, half_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(half_out),
            nn.SiLU()
        )

        self.fam = FAM(dim=dim, reduction=1)
        self.sgfm = SGFFM(dim=dim, reduction=1, num_heads=4, sr_ratio=8, drop_path=0.1)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.fpm = FPM(dim=dim,pool_size=3,mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5)
        self.mbf = MBF(dim=dim//2, num_heads=1, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkv", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=True)

        self.depth_gate = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.c2f = C2f(c1=dim, c2=dim, n=2)

    def forward(self, x):
        c=x.shape[1]
        half = c // 2
        x1, x2 = torch.split(x, half, dim=1)  # 

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x1, x2 = self.fam(x1, x2)
        x1, x2 = self.fpm(x1, x2)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1, x2 = self.mbf(x1, x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x_next = torch.cat([x1, x2], dim=1)  # (B, dim, H, W)
        x_next = self.c2f(x_next)

        depth=DEPTH_CACHE 
        depth_norm = F.interpolate(
            depth,
            size=x_next.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        gate = self.depth_gate(depth_norm)

        x_next = x_next * gate + x_next 
        
        return x_next
