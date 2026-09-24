from ultralytics import YOLO
model = YOLO('/home/qmy/PZ/runs/train/exp14/weights/best.pt')
results = model.export(format='onnx')