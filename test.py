import os
from ultralytics import YOLO
from pathlib import Path

# 设置基础路径
base_dir = r"F:\data_use\data_all"
image_folder = os.path.join(base_dir, '')  # 原始图片路径
output_folder = os.path.join(base_dir, '')  # 检测结果保存路径
val_txt_path = os.path.join(base_dir, 'val.txt')  # 验证集列表文件

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 加载YOLO模型
model = YOLO(model=r"G:\paper_sen\code\PZ_8\runs/train/18+rgb+10s_200/weights/last.pt")

# 类别名称映射
class_names = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian',
    3: 'Bus',
    4: 'Truck'
}

# 读取验证集文件列表
with open(val_txt_path, 'r') as f:
    val_images = [line.strip() for line in f.readlines() if line.strip()]

print(f"Found {len(val_images)} validation images")

# 处理验证集中的每张图片
for image_rel_path in val_images:
    # 构建完整图片路径
    image_path = os.path.join(image_folder, image_rel_path)
    
    # 确保图片存在
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    print(f"Processing {image_path}...")
    
    # 预测图片
    results = model(image_path)
    
    # 处理检测结果
    for result in results:
        # 创建对应的输出目录结构
        output_rel_dir = os.path.dirname(image_rel_path)
        output_dir = os.path.join(output_folder, output_rel_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建输出文件路径
        output_file = os.path.join(output_dir, os.path.basename(image_path))
        
        # 保存检测结果（带边界框的可视化图片）
        result.save(filename=output_file)
        
        # 打印检测信息
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                class_name = class_names.get(class_id, f'Unknown_{class_id}')
                print(f"  Detected: {class_name} ({conf:.2f}) at {box.xyxy[0].tolist()}")
        else:
            print("  No detections")
        
        print(f"Saved result to {output_file}")

print("Validation complete!")