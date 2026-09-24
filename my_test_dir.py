from ultralytics import YOLO
import os
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getPolarizationImages(base_path):
    """
    读取一组 8 通道的偏振-双光数据并堆叠成 (H,W,8)
    通道顺序:
    S0_rgb(3) + S0_nir(1) + DoLP_rgb(1) + DoLP_nir(1) + AoLP_rgb(1) + AoLP_nir(1)
    """
    ret_list = None
    
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    clean_name = os.path.splitext(base_name)[0]

    # 文件后缀 + 是否灰度
    channel_files = [
        ("_S0_rgb.png", False),    # RGB 0 1 2
        ("_S0_nir.png", True),     # 灰度 3 
        ("_dolp_rgb.png", True),   # 灰度 4
        ("_dolp_nir.png", True),   # 灰度 5
        ("_aolp_rgb.png", True),   # 灰度 6 
        ("_aolp_nir.png", True) ,  # 灰度 7
        ("_depth.png", True) , # 灰度 8
    ]
    
    for suffix, is_gray in channel_files:
        img_path = os.path.join(dir_path, clean_name + suffix)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing required polarization file: {img_path}")

        # 灰度图强制单通道
        if is_gray:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=2)  # (H,W,1)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # 转成 RGB

        if ret_list is None:
            ret_list = img
        else:
            ret_list = np.concatenate((ret_list, img), axis=2)
    # print("aaaaa")
    # print(ret_list.shape)
    # 最终检查
    if ret_list.shape[2] != 9:
        raise ValueError(f"Expected 9 channels, got {ret_list.shape[2]}")

    return ret_list
    

if __name__ == "__main__":
    model = YOLO(r"/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/1228+RGB+AOLP+depth2/weights/best.pt")

    img_dir = r"/media/ddc/新加卷/hys/qmy/PZ_8/test"

    # 只保留“索引图片”（没有其他后缀）
    files = []
    for f in os.listdir(img_dir):
        if not f.endswith(".png"):
            continue
        
        # 过滤掉所有附加通道文件
        if any(x in f.lower() for x in ["_s0", "dolp", "aolp", "depth", "nir"]):
            continue
        
        files.append(f)

    print("将要推理的索引图数量:", len(files))

    for f in sorted(files):
        base_path = os.path.join(img_dir, f)

        img = getPolarizationImages(base_path)   # (H,W,9)

        results = model.predict(
            source=img,
            ch=9,
            save=True,
            save_txt=True,
            save_conf=True,
            conf=0.5
        )