from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 1. Hook 容器 =====================
FEATURES = {}

def hook_fn(name):
    def hook(m, i, o):
        if isinstance(o, (list, tuple)):
            o = o[0]
        FEATURES[name] = o.detach().cpu()
    return hook


# ===================== 2. 加载模型 =====================
model = YOLO(
    "/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/116_2改_2003/weights/best.pt"
)

# ===================== 3. 注册 backbone hook（按你的 YAML） =====================
# backbone:
# 0: channel1
# 1: CSBF stage1
# 3: CSBF stage2
# 5: CSBF stage3
# 7: CSBF stage4

hook_layers = {
    0: "stage0_channel1",
    1: "stage1_csbf",
    3: "stage2_csbf",
    5: "stage3_csbf",
    7: "stage4_csbf",
}

for idx, name in hook_layers.items():
    model.model.model[idx].register_forward_hook(hook_fn(name))

print(">>> Feature hooks registered.")


# ===================== 4. 运行 9 通道推理 =====================
path = "/media/ddc/新加卷/hys/qmy/pz/val.txt"

model.predict(
    source=path,
    ch=9,
    device="1",
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.5,
    project="/media/ddc/新加卷/hys/qmy/PZ_8/runs/predict116_2_200没跑完",
    name="valold_best"
)

print(">>> Inference finished.")


# ===================== 5. 保存特征图 =====================
save_root = "/media/ddc/新加卷/hys/qmy/PZ_8/feature_vis"
os.makedirs(save_root, exist_ok=True)

for name, feat in FEATURES.items():
    feat = feat[0]            # batch=0
    fmap = feat.mean(0)       # channel mean

    plt.figure(figsize=(5, 5))
    plt.imshow(fmap, cmap="jet")
    plt.colorbar()
    plt.axis("off")

    out_path = os.path.join(save_root, f"{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved feature map: {out_path}")

print(">>> All feature maps saved.")
