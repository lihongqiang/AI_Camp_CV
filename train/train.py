import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
IMGSZ = 1080

import warnings
warnings.filterwarnings('ignore')


from ultralytics import YOLO
model = YOLO("yolov10x.pt")
results = model.train(data="yolo-dataset/yolo.yaml", epochs=20, imgsz=IMGSZ,\
                       batch=64, verbose=True, freeze=10,
                       optimizer='SGD', lr0=0.0001, warmup_epochs=0, lrf=0.01,
                       plots=True,
                       patience=3, 
                       mixup=0.4, copy_paste=0.5, mosaic=0,
                       device=[1,2])
