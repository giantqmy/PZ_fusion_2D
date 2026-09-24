from ultralytics import YOLO
import datetime
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    # model = YOLO(r"/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/124-最高不加depthbuquchong2/weights/best.pt")
    model = YOLO(r"/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/329/weights/epoch40.pt")
    path = r"/media/ddc/新加卷/hys/qmy/pz/valfeature.txt"
    model.predict(source=path,ch=9, device='1',save=True,save_txt=True, save_conf=True,conf=0.5,
    project="/media/ddc/新加卷/hys/qmy/PZ_8/runs",
    name="329")



    # # ##尝试推理
    # results = model.val(
    #     data=r"/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml",   # 训练用的 data.yaml
    #     split='val',
        
    # )

    # print(results) 这可以不print
