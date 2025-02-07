### 编程思路：简要说明算法编程实现方法和路径，提供核心代码段。
base模型采用yolov10x，通过对训练集的视频构建训练数据集，调整参数优化模型效果。
构建数据集代码：code/generate_data.py
```python
import pandas as pd
import cv2
for anno_path, video_path in zip(train_annos[:-3], train_videos[:-3]):
    print(video_path)
    anno_df = pd.read_json(anno_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]
        
        frame_anno = anno_df[anno_df['frame_id'] == frame_idx]
        cv2.imwrite('./temp_data/train/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.jpg', frame)

        if len(frame_anno) != 0:
            with open('./temp_data/train/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.txt', 'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)
                    
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    if x_center > 1:
                        print(bbox)
                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')
        
        frame_idx += 1
```
训练代码：train/train.py
```python
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
                       device=[0,1,2,3,4,5,6,7])
```

### 特色亮点：简要说明算法和程序在运算速度、效率等方面的特点和优势。
模型小，运行速度快，支持实时识别。

### 编程语言：采用的主要编程语言（必须注明版本号，如python3.7）。
python3.12.4

### 软件运行环境：说明算法程序运行需配置的操作系统、语言环境、依赖包等（注明版本号）。
依赖包见requirement.txt
环境|参数
----|---
内核版本|Linux 6b9d144d4892 5.4.0-144-generic #161-Ubuntu SMP Fri Feb 3 14:49:04 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
操作系统版本|Ubuntu 22.04.3 LTS
GPU Driver Version|535.183.01
CUDA Version|12.2
cuDNN | v8.9.7


### 硬件配置环境：说明算法程序运行建议硬件环境配置。
CPU|Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz 128核
GPU|8*A100

### 运行说明：算法程序运行安装、配置、运行操作（必须）、算法结果等简要说明。
1. 数据生成
```bash
python3 code/generate_data.py
```
2. 训练
```bash
python3 train/train.py
```
3. 预测
```
python3 code/main.py
```