import os
import shutil

val_txt = r"/media/ddc/新加卷/hys/qmy/pz/val.txt"          # 你的 val 列表
labels_dir = r"/media/ddc/新加卷/hys/qmy/pz/labels"        # 原标签文件夹
save_dir = r"/media/ddc/新加卷/hys/qmy/PZ_8/AP/valgt"      # 要保存的新文件夹

os.makedirs(save_dir, exist_ok=True)

with open(val_txt, "r") as f:
    lines = f.readlines()

count = 0

for line in lines:
    line = line.strip()

    if not line:
        continue

    # 取文件名（不带后缀）
    base = os.path.basename(line)
    name = os.path.splitext(base)[0]

    label_path = os.path.join(labels_dir, name + ".txt")

    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(save_dir, name + ".txt"))
        count += 1
    else:
        print(f"⚠️ 找不到标签: {label_path}")

print(f"✔️ 复制完成，共复制 {count} 个标签文件")
