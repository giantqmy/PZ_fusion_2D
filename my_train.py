from ultralytics import YOLO
import datetime
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.use_deterministic_algorithms(False)

if __name__ == '__main__':
    # # model = YOLO(r'F:/PZ/ultralytics/cfg/models/11/yolo11n.yaml')
    model = YOLO(r'/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/models/11/pz_911.yaml')

    model.train(data=r"/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6qu.yaml",#数据集划分
                    # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                    cache=False,
                    imgsz=512,
                    epochs=200,
                    single_cls=False,  # 是否是单类别检测
                    batch=4,
                    save_period=5,
                    # patience=50, 
                    workers=8,
                    device='0,1',
                    optimizer='SGD',  # using SGD 优化器
                    resume= False, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                    augment=False,
                    project='runs/train',
                    name='329',
                    pretrained=False,)
                    
    
