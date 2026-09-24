import os

# 设置图片文件夹路径
image_folder = r"F:\YOLO_data\data_day\images"  # 请替换为你的实际路径

# 目标后缀列表
targets = ['_I0', '_I45', '_I90', '_I135']

# 遍历文件夹中的文件
for filename in os.listdir(image_folder):
    if any(suffix in filename for suffix in targets):
        name, ext = os.path.splitext(filename)
        # 查找最后一个 "_Ixx" 的位置
        for suffix in targets:
            if name.endswith(suffix):
                index = name.rfind('_')
                if index != -1:
                    new_name = name[:index] + '-' + name[index+1:] + ext
                    old_path = os.path.join(image_folder, filename)
                    new_path = os.path.join(image_folder, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} → {new_name}")
                break
