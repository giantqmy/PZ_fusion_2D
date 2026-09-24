# Copyright (c) 2024, MaskWPSFM Module
"""
MaskWPSFM - 集成掩码注意力机制的WPSFM模块
从block.py中提取的相关模块集合

## 核心特性
- **MaskWPSFM**: 集成MaskAttention机制的WPSFM，完全兼容原WPSFM的输入输出
- **MaskWGEFM**: 在WGEFM基础上集成掩码机制，掩码只作用在注意力计算部分
- **DenseLayer**: 密集连接层，用于特征提取
- **BBasicConv2d**: 基础卷积块，包含卷积、批归一化和ReLU激活

## 掩码机制
- 掩码只作用在注意力计算的scores上
- 总共作用2次（RGB分支和INF分支各1次）
- 使用伯努利分布生成随机掩码
- 支持可配置的掩码比例

## 使用示例
```python
# 创建MaskWPSFM模块
mask_wpsfm = MaskWPSFM(Channel=256, area=4, mask_ratio=0.5)

# 输入数据格式: data = (rgb, depth)
# rgb和depth都是 (B, C, H, W)
output = mask_wpsfm((rgb_features, depth_features))
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入einops用于张量重排
try:
    import einops
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class BBasicConv2d(nn.Module):
    """基础卷积块，包含卷积、批归一化和ReLU激活"""
    
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class DenseLayer(nn.Module):
    """密集连接层，用于特征提取"""
    
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, padding=1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, 3, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class MaskWGEFM(nn.Module):
    """
    参考源代码MaskAttention，在WGEFM基础上集成掩码机制
    掩码只作用在注意力计算的scores上，作用2次（RGB分支和INF分支各1次）
    """
    
    def __init__(self, in_C, out_C, area=1, mask_ratio=0.5):
        super(MaskWGEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.Q = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, padding=1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, padding=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.area = area
        self.mask_ratio = mask_ratio
        
        # 用于缓存掩码，避免重复计算
        self.register_buffer('rgb_mask', None)
        self.register_buffer('inf_mask', None)

    def generate_attention_mask(self, batch_size, seq_len, device, dtype=None):
        """
        参考源代码生成注意力掩码
        """
        # 如果没有指定dtype，使用float32作为默认
        if dtype is None:
            dtype = torch.float32
            
        # 生成二值掩码
        binary_mask = torch.bernoulli(torch.full((batch_size, seq_len), self.mask_ratio, 
                                               dtype=dtype, device=device))
        
        # 转换为注意力掩码格式：>0.5的设为0（不遮蔽），<=0.5的设为-inf（遮蔽）
        processed_mask = torch.where(binary_mask > 0.5, 
                                   torch.tensor(0.0, dtype=dtype, device=device), 
                                   torch.tensor(-float('inf'), dtype=dtype, device=device))
        
        # 扩展为注意力矩阵形状 (batch_size, seq_len, seq_len)
        return processed_mask.unsqueeze(1).expand(-1, seq_len, -1)

    def forward(self, x, y):
        # x: RGB特征, y: 深度或INF特征
        B, C, H, W = x.size()
        N = H * W
        Q = self.Q(torch.cat([x, y], dim=1))

        #### RGB 分支 ####
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        
        # 原始区域注意力计算
        if self.area == 1:
            RGB_V_ = RGB_V.reshape(B, -1, N)
            RGB_K_ = RGB_K.reshape(B, -1, N).permute(0, 2, 1)
            RGB_Q_ = Q.reshape(B, -1, N)
            seq_len = N
        else:
            assert N % self.area == 0, "H*W must be divisible by area."
            RGB_V_ = RGB_V.reshape(B * self.area, -1, N // self.area)
            RGB_K_ = RGB_K.reshape(B * self.area, -1, N // self.area).permute(0, 2, 1)
            RGB_Q_ = Q.reshape(B * self.area, -1, N // self.area)
            seq_len = N // self.area

        # 计算RGB注意力分数
        RGB_scores = torch.bmm(RGB_K_, RGB_Q_)
        
        # 第1次掩码应用：RGB分支注意力scores
        effective_batch = B if self.area == 1 else B * self.area
        rgb_attention_mask = self.generate_attention_mask(effective_batch, seq_len, x.device, x.dtype)
        RGB_scores = RGB_scores + rgb_attention_mask
        
        RGB_mask = self.softmax(RGB_scores)
        RGB_refine = torch.bmm(RGB_V_, RGB_mask.permute(0, 2, 1))

        if self.area > 1:
            RGB_refine = RGB_refine.reshape(B, self.area, -1, N // self.area)
            RGB_refine = RGB_refine.reshape(B, -1, N)
        
        RGB_refine = RGB_refine.reshape(B, -1, H, W)
        RGB_refine = self.gamma1 * RGB_refine + y

        #### INF 分支 ####
        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        
        if self.area == 1:
            INF_V_ = INF_V.reshape(B, -1, N)
            INF_K_ = INF_K.reshape(B, -1, N).permute(0, 2, 1)
            INF_Q_ = Q.reshape(B, -1, N)
        else:
            INF_V_ = INF_V.reshape(B * self.area, -1, N // self.area)
            INF_K_ = INF_K.reshape(B * self.area, -1, N // self.area).permute(0, 2, 1)
            INF_Q_ = Q.reshape(B * self.area, -1, N // self.area)

        # 计算INF注意力分数
        INF_scores = torch.bmm(INF_K_, INF_Q_)
        
        # 第2次掩码应用：INF分支注意力scores  
        inf_attention_mask = self.generate_attention_mask(effective_batch, seq_len, x.device, x.dtype)
        INF_scores = INF_scores + inf_attention_mask
        
        INF_mask = self.softmax(INF_scores)
        INF_refine = torch.bmm(INF_V_, INF_mask.permute(0, 2, 1))
        
        if self.area > 1:
            INF_refine = INF_refine.reshape(B, self.area, -1, N // self.area)
            INF_refine = INF_refine.reshape(B, -1, N)
        
        INF_refine = INF_refine.reshape(B, -1, H, W)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class MaskWPSFM(nn.Module):
    """
    集成MaskAttention机制的WPSFM，完全兼容原WPSFM的输入输出
    掩码只作用在注意力计算部分，总共作用2次
    """
    
    def __init__(self, Channel, area=4, mask_ratio=0.5):
        """
        Args:
            Channel: 输入通道数，与原WPSFM一致
            area: 区域划分数量，与原WPSFM一致  
            mask_ratio: 掩码比例，控制掩码密度
        """
        super(MaskWPSFM, self).__init__()
        # 保持与原WPSFM完全一致的结构
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        # 使用带掩码的融合层
        self.obj_fuse = MaskWGEFM(Channel * 2, Channel, area=area, mask_ratio=mask_ratio)

    def forward(self, data):
        """
        与原WPSFM保持完全一致的输入输出接口
        Input: data = (rgb, depth)，其中rgb和depth都是 (B, C, H, W)
        Output: fused_features (B, C, H, W)
        """
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out


class LayerNormProxy(nn.Module):
    """Layer normalization proxy for 2D feature maps"""
    
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if not EINOPS_AVAILABLE:
            # 如果einops不可用，使用原生pytorch操作
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H*W, C)
            x = self.norm(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return x
        else:
            x = einops.rearrange(x, 'b c h w -> b h w c')
            x = self.norm(x)
            return einops.rearrange(x, 'b h w c -> b c h w')


# 工厂函数
def create_mask_wpsfm(Channel, area=4, mask_ratio=0.5):
    """
    创建MaskWPSFM模块的工厂函数
    
    Args:
        Channel: 输入通道数
        area: 区域划分数量，默认4
        mask_ratio: 掩码比例，默认0.5
    
    Returns:
        MaskWPSFM: 掩码WPSFM模块
    """
    return MaskWPSFM(Channel=Channel, area=area, mask_ratio=mask_ratio)


def create_mask_wgefm(in_C, out_C, area=1, mask_ratio=0.5):
    """
    创建MaskWGEFM模块的工厂函数
    
    Args:
        in_C: 输入通道数
        out_C: 输出通道数
        area: 区域划分数量，默认1
        mask_ratio: 掩码比例，默认0.5
    
    Returns:
        MaskWGEFM: 掩码WGEFM模块
    """
    return MaskWGEFM(in_C=in_C, out_C=out_C, area=area, mask_ratio=mask_ratio)


# 测试函数
def test_mask_wpsfm():
    """测试MaskWPSFM模块"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 测试MaskWPSFM模块")
    print(f"📱 设备: {device}")
    print(f"🔧 EINOPS_AVAILABLE: {EINOPS_AVAILABLE}")
    print("=" * 60)
    
    # 测试参数
    test_cases = [
        # (Channel, area, mask_ratio, input_shape)
        (128, 1, 0.3, (2, 128, 32, 32)),
        (256, 4, 0.5, (2, 256, 16, 16)),
        (512, 8, 0.7, (2, 512, 8, 8)),
    ]
    
    for Channel, area, mask_ratio, input_shape in test_cases:
        print(f"\n🧪 测试配置:")
        print(f"   通道数: {Channel}, 区域数: {area}, 掩码比例: {mask_ratio}")
        print(f"   输入形状: {input_shape}")
        
        try:
            # 创建模型
            model = MaskWPSFM(Channel=Channel, area=area, mask_ratio=mask_ratio).to(device)
            
            # 测试输入
            rgb = torch.randn(*input_shape).to(device)
            depth = torch.randn(*input_shape).to(device)
            data = (rgb, depth)
            
            # 前向传播
            with torch.no_grad():
                output = model(data)
            
            print(f"   ✅ 测试通过，输出形状: {output.shape}")
            
            # 简单性能测试
            import time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10 * 1000
            print(f"   ⏱️ 平均推理时间: {avg_time:.2f}ms")
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   📊 参数量: {total_params:,}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 MaskWPSFM特性总结:")
    print("   • 集成掩码注意力机制的WPSFM")
    print("   • 掩码只作用在注意力计算部分，总共作用2次")
    print("   • 完全兼容原WPSFM的输入输出接口")
    print("   • 支持可配置的掩码比例和区域划分")
    print("   • 使用伯努利分布生成随机掩码")
    
    print("\n💡 使用建议:")
    print("   • area=1: 全局注意力，适合小特征图")
    print("   • area=4: 区域注意力，平衡性能和效率")
    print("   • area=8: 局部注意力，适合大特征图")
    print("   • mask_ratio=0.3-0.7: 根据任务调整掩码密度")


if __name__ == "__main__":
    test_mask_wpsfm()
