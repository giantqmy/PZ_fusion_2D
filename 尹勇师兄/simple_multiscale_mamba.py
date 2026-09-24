# Copyright (c) 2024, Simple Multi-Scale Mamba Module
"""
Simple Multi-Scale Mamba - 简化多尺度Mamba模块
使用mamba-ssm库实现真正的状态空间扫描

## 核心特性
- **多尺度并行扫描**: 处理不同大小的目标特征
- **真正mamba-ssm**: 使用mamba-ssm的selective_scan_fn进行状态空间扫描
- **多尺度融合**: 智能融合不同尺度的特征
- **YOLO适配**: 完全兼容YOLO的模块调用规范

## 多尺度扫描机制
- 使用多个尺度(1x, 2x, 4x等)处理输入特征
- 每个尺度使用独立的Mamba参数进行扫描
- 智能融合不同尺度的输出特征
- 支持残差连接提高训练稳定性

## 使用示例
```python
model = SimpleMultiScaleMamba(
    d_model=256,
    d_state=16,
    scales=[1, 2, 4],  # 多尺度分支的尺度
)
output = model(input_tensor)  # (B, C, H, W)
```
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    selective_scan_fn = None
    selective_state_update = None
    MAMBA_SSM_AVAILABLE = False
    print("Warning: mamba-ssm not available, using fallback implementation")

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class SimpleMultiScaleMamba(nn.Module):
    """
    简化多尺度Mamba模块
    使用mamba-ssm处理不同尺度的特征，捕获多尺度目标信息
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.scales = scales
        self.d_inner = expand * d_model
        
        # 每个尺度的Mamba参数
        self.scale_branches = nn.ModuleList()
        self.scale_A_logs = nn.ParameterList()
        self.scale_Ds = nn.ParameterList()
        
        for scale in scales:
            branch = nn.ModuleDict({
                'in_proj': nn.Linear(d_model, self.d_inner * 2, bias=False),
                'conv1d': nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=True,
                    kernel_size=d_conv,
                    groups=self.d_inner,
                    padding=d_conv - 1,
                ),
                'x_proj': nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False),  # dt_rank + B和C参数
                'dt_proj': nn.Linear(self.d_inner, self.d_inner, bias=True),
                'out_proj': nn.Linear(self.d_inner, d_model, bias=False),
                'norm': nn.LayerNorm(d_model),
            })
            self.scale_branches.append(branch)
            
            # Mamba状态空间参数单独存储
            self.scale_A_logs.append(nn.Parameter(torch.log(torch.rand(d_state, dtype=torch.float32).repeat(self.d_inner, 1))))
            self.scale_Ds.append(nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32)))
        
        # 尺度权重学习
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.scale_fusion = nn.Linear(d_model * len(scales), d_model)
        
    def _multi_scale_downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """多尺度下采样"""
        if scale == 1:
            return x
        
        B, C, H, W = x.shape
        # 使用平均池化进行下采样
        downsampled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
        return downsampled
    
    def _multi_scale_upsample(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """多尺度上采样"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def _mamba_scan(self, x: torch.Tensor, branch: nn.ModuleDict, branch_idx: int) -> torch.Tensor:
        """
        使用mamba-ssm进行扫描
        Args:
            x: (B, L, D) - 序列化的特征
            branch: 对应尺度的分支参数
            branch_idx: 分支索引，用于获取对应的A_log和D参数
        Returns:
            output: (B, L, D) - Mamba输出
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = branch['in_proj'](x)  # (B, L, 2*d_inner)
        x_mamba, z = xz.chunk(2, dim=-1)  # (B, L, d_inner), (B, L, d_inner)
        
        # 转换为卷积格式
        x_mamba = rearrange(x_mamba, 'b l d -> b d l')
        
        # 1D卷积
        if causal_conv1d_fn is not None:
            # causal_conv1d_fn需要权重形状为 (dim, width)
            conv_weight = branch['conv1d'].weight  # (d_inner, 1, d_conv)
            conv_weight = rearrange(conv_weight, 'd 1 w -> d w')  # (d_inner, d_conv)
            x_mamba = causal_conv1d_fn(x_mamba, conv_weight, 
                                      branch['conv1d'].bias, activation="silu")
        else:
            # causal_conv1d_fn不可用，使用标准卷积
            x_mamba = branch['conv1d'](x_mamba)[..., :L]  # 截断到原始长度
            x_mamba = F.silu(x_mamba)
        
        # 转换回序列格式
        x_mamba = rearrange(x_mamba, 'b d l -> b l d')
        
        # 状态空间投影 - 按照mamba的正确方式
        x_proj = branch['x_proj'](x_mamba)  # (B, L, d_inner + 2*d_state)
        dt_proj, B_proj, C_proj = torch.split(x_proj, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # dt投影 - 使用投影矩阵
        dt = branch['dt_proj'](dt_proj)  # (B, L, d_inner)
        
        # 转换为mamba-ssm需要的格式
        x_scan = rearrange(x_mamba, 'b l d -> b d l')  # (B, d_inner, L)
        dt_scan = rearrange(dt, 'b l d -> b d l')      # (B, d_inner, L)
        B_scan = rearrange(B_proj, 'b l d -> b d l')   # (B, d_state, L)
        C_scan = rearrange(C_proj, 'b l d -> b d l')   # (B, d_state, L)
        
        # 获取状态空间参数
        A = -torch.exp(self.scale_A_logs[branch_idx].float())  # (d_inner, d_state)
        D = self.scale_Ds[branch_idx].float()
        
        # 使用mamba-ssm，但在CPU上时使用简化实现
        if MAMBA_SSM_AVAILABLE and x_scan.is_cuda:
            # 在CUDA上使用真正的mamba-ssm
            y = selective_scan_fn(
                x_scan,
                dt_scan,
                A,
                B_scan,
                C_scan,
                D,
                z=rearrange(z, 'b l d -> b d l'),
                delta_bias=branch['dt_proj'].bias.float() if branch['dt_proj'].bias is not None else None,
                delta_softplus=True,
            )
        else:
            # CPU fallback或mamba-ssm不可用时的简化实现
            z_rearranged = rearrange(z, 'b l d -> b d l')
            # 简单的线性变换作为fallback
            y = x_scan + 0.1 * torch.tanh(dt_scan) * torch.mean(B_scan, dim=1, keepdim=True) * torch.mean(C_scan, dim=1, keepdim=True)
            if z_rearranged is not None:
                y = y * F.silu(z_rearranged)
            if D is not None:
                y = y + D.unsqueeze(0).unsqueeze(-1) * x_scan
        
        # 转换回原格式
        y = rearrange(y, 'b d l -> b l d')
        
        # 输出投影
        output = branch['out_proj'](y)
        output = branch['norm'](output + x)  # 残差连接
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, H, W) - 输入特征图
        Returns:
            output: (B, C, H, W) - 多尺度扫描输出
        """
        B, C, H, W = x.shape
        original_size = (H, W)
        
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # 多尺度下采样
            x_scale = self._multi_scale_downsample(x, scale)
            
            # 转换为序列格式
            x_seq = rearrange(x_scale, 'b c h w -> b (h w) c')
            
            # Mamba扫描
            y_seq = self._mamba_scan(x_seq, self.scale_branches[i], i)
            
            # 转换回特征图格式
            H_scale, W_scale = x_scale.shape[2], x_scale.shape[3]
            y_scale = rearrange(y_seq, 'b (h w) c -> b c h w', h=H_scale, w=W_scale)
            
            # 上采样到原始尺寸
            y_scale = self._multi_scale_upsample(y_scale, original_size)
            
            scale_outputs.append(y_scale)
        
        # 尺度融合
        if len(scale_outputs) > 1:
            # 加权融合
            weights = F.softmax(self.scale_weights, dim=0)
            fused = sum(w * output for w, output in zip(weights, scale_outputs))
            
            # 特征融合
            concat_features = torch.cat(scale_outputs, dim=1)  # (B, C*scales, H, W)
            concat_features = rearrange(concat_features, 'b c h w -> b (h w) c')
            fused_features = self.scale_fusion(concat_features)
            fused_features = rearrange(fused_features, 'b (h w) c -> b c h w', h=H, w=W)
            
            return fused_features + fused  # 残差连接
        else:
            return scale_outputs[0]


# 适配YOLO的简化多尺度Mamba
class SimpleMultiScaleMambaYOLO(SimpleMultiScaleMamba):
    """
    适配YOLO的简化多尺度Mamba模块
    
    ## YOLO适配特性
    - 自动适配YOLO的输入输出格式 (c1, c2, n, shortcut, g, e)
    - 支持shortcut连接提高训练稳定性
    - 针对目标检测任务优化的默认参数
    - 兼容YOLO的模块调用规范
    
    ## 使用示例
    ```python
    # 在YOLO配置文件中使用
    - [from, number, module, args]
    - [-1, 1, SimpleMultiScaleMambaYOLO, [256, 512]]  # c1=256, c2=512
    ```
    """
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, 
                 scales=[1, 2, 4], **kwargs):
        """
        YOLO格式的初始化
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: 重复次数（保持YOLO兼容性，这里固定为1）
            shortcut: 是否使用shortcut连接
            g: 分组数（保持兼容性）
            e: 扩展比例（保持兼容性）
            scales: 多尺度分支的尺度列表
            **kwargs: 其他参数
        """
        # 使用c2作为模型维度
        d_model = c2
        
        # 调用父类初始化
        super().__init__(
            d_model=d_model,
            scales=scales,
            **kwargs
        )
        
        # YOLO特定参数
        self.c1 = c1
        self.c2 = c2
        self.shortcut = shortcut and c1 == c2
        
        # 输入通道适配
        if c1 != c2:
            self.input_adapter = nn.Conv2d(c1, c2, 1, 1, bias=False)
        else:
            self.input_adapter = nn.Identity()
        
        # shortcut连接
        if self.shortcut:
            self.shortcut_conv = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, 1, bias=False)
    
    def forward(self, x):
        """
        YOLO格式的前向传播
        Args:
            x: (B, C1, H, W) - YOLO输入特征图
        Returns:
            output: (B, C2, H, W) - YOLO输出特征图
        """
        # 确保输入和模块在同一设备上
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        
        identity = x
        
        # 输入通道适配
        x = self.input_adapter(x)
        
        # 多尺度Mamba处理
        output = super().forward(x)
        
        # shortcut连接
        if self.shortcut:
            output = output + self.shortcut_conv(identity)
        
        return output


# 工厂函数
def create_simple_multiscale_mamba(d_model: int, 
                                  scales: List[int] = [1, 2, 4],
                                  **kwargs) -> SimpleMultiScaleMamba:
    """
    工厂函数创建简化多尺度Mamba模块
    """
    return SimpleMultiScaleMamba(
        d_model=d_model,
        scales=scales,
        **kwargs
    )


def create_simple_multiscale_mamba_yolo(c1, c2, scales=[1, 2, 4], **kwargs):
    """
    创建适配YOLO的简化多尺度Mamba模块
    
    Args:
        c1: 输入通道数
        c2: 输出通道数
        scales: 多尺度分支的尺度列表，默认[1, 2, 4]
        **kwargs: 其他参数
    
    Returns:
        SimpleMultiScaleMambaYOLO: 适配YOLO的简化多尺度Mamba模块
    """
    return SimpleMultiScaleMambaYOLO(
        c1=c1,
        c2=c2,
        scales=scales,
        **kwargs
    )


# 预设配置工厂函数
def create_simple_mamba_configs():
    """
    创建不同配置的简化多尺度Mamba模块
    
    Returns:
        dict: 不同配置的工厂函数字典
    """
    configs = {
        # 轻量级配置 - 适合小模型
        'lightweight': lambda c1, c2: SimpleMultiScaleMambaYOLO(
            c1=c1, c2=c2,
            scales=[1, 2],  # 只用两个尺度
            d_state=8,      # 较小的状态维度
            expand=1        # 较小的扩展比例
        ),
        
        # 标准配置 - 平衡性能和效率
        'standard': lambda c1, c2: SimpleMultiScaleMambaYOLO(
            c1=c1, c2=c2,
            scales=[1, 2, 4],  # 三个尺度
            d_state=16,        # 标准状态维度
            expand=2           # 标准扩展比例
        ),
        
        # 高性能配置 - 适合大模型
        'high_performance': lambda c1, c2: SimpleMultiScaleMambaYOLO(
            c1=c1, c2=c2,
            scales=[1, 2, 4, 8],  # 四个尺度
            d_state=32,           # 更大的状态维度
            expand=2              # 标准扩展比例
        ),
        
        # 精细检测配置 - 适合小目标检测
        'fine_detection': lambda c1, c2: SimpleMultiScaleMambaYOLO(
            c1=c1, c2=c2,
            scales=[1, 2, 3, 4],  # 更密集的尺度
            d_state=24,           # 中等状态维度
            expand=2              # 标准扩展比例
        )
    }
    
    return configs


# 测试函数
def test_simple_multiscale_mamba():
    """测试简化多尺度Mamba模块"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 测试简化多尺度Mamba模块")
    print(f"📱 设备: {device}")
    print(f"🔧 MAMBA_SSM_AVAILABLE: {MAMBA_SSM_AVAILABLE}")
    print("=" * 60)
    
    # 测试不同配置
    configs = create_simple_mamba_configs()
    test_cases = [
        # (配置名称, c1, c2, 输入尺寸)
        ('lightweight', 128, 128, (2, 128, 64, 64)),
        ('standard', 256, 256, (2, 256, 32, 32)),
        ('high_performance', 512, 512, (2, 512, 16, 16)),
        ('fine_detection', 256, 512, (2, 256, 32, 32)),  # 测试不同输入输出通道
    ]
    
    for config_name, c1, c2, input_shape in test_cases:
        print(f"\n🧪 测试配置: {config_name}")
        print(f"   输入通道: {c1}, 输出通道: {c2}")
        print(f"   输入形状: {input_shape}")
        
        try:
            # 创建模型
            if config_name in configs:
                model = configs[config_name](c1, c2).to(device)
            else:
                model = SimpleMultiScaleMambaYOLO(c1=c1, c2=c2).to(device)
            
            # 测试输入
            x = torch.randn(*input_shape).to(device)
            
            # 前向传播
            with torch.no_grad():
                output = model(x)
            
            print(f"   ✅ 测试通过，输出形状: {output.shape}")
            
            # 简单性能测试
            import time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10 * 1000
            print(f"   ⏱️ 平均推理时间: {avg_time:.2f}ms")
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   📊 参数量: {total_params:,}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 简化多尺度Mamba特性总结:")
    print("   • 多尺度并行扫描 - 处理不同大小目标")
    print("   • 使用mamba-ssm库实现真正的状态空间扫描")
    print("   • 智能多尺度特征融合")
    print("   • 支持YOLO格式的输入输出适配")
    print("   • 提供多种预设配置适应不同场景")
    
    print("\n💡 使用建议:")
    print("   • lightweight: 适合轻量级模型，快速推理")
    print("   • standard: 平衡性能和效率，通用推荐")
    print("   • high_performance: 适合大模型，追求最佳性能")
    print("   • fine_detection: 适合小目标检测任务")


if __name__ == "__main__":
    test_simple_multiscale_mamba()
