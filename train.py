import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
        model = YOLO(r'F:/PZ/ultralytics/cfg/models/11/yolo11n.yaml')  # 模型pt文件路径
        # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
        # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
        # model.load('yolov11n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
        model.train(data=r"F:\PZ\ultralytics\cfg\datasets\pz5.yaml", # yz32.yaml文件路径
                    # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                    cache=False,
                    imgsz=640,
                    epochs=200,
                    single_cls=False,  # 是否是单类别检测
                    batch=4,
                    close_mosaic=0,
                    workers=4,
                    device='0',
                    optimizer='SGD',  # using SGD 优化器
                    resume=False, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                    amp=True,
                    project='runs/train',
                    name='exp',
                    )
