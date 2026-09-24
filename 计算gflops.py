import torch
from thop import profile
from models.yolo import Model  # 根据你的工程路径改

# ===============================
# 1. 构建模型
# ===============================
cfg = r"G:\paper_sen\code\fusion\PZ_8\ultralytics\cfg\models\11\yolo11n.yaml"   # 或你修改后的 yaml
model = Model(cfg).cuda()
model.eval()

# ===============================
# 2. 构造假输入
# ===============================
# 如果是 RGB
x = torch.randn(1, 3, 640, 640).cuda()

# 如果你是多模态（举例）
# x_rgb = torch.randn(1, 3, 640, 640).cuda()
# x_depth = torch.randn(1, 1, 640, 640).cuda()
# macs, params = profile(model, inputs=(x_rgb, x_depth))

macs, params = profile(model, inputs=(x,), verbose=False)

print(f"GFLOPs: {macs / 1e9:.2f}")
print(f"Params: {params / 1e6:.2f} M")
