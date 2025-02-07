import os
import glob

if not os.path.exists('temp_data/'):
    os.mkdir('temp_data/')
if not os.path.exists('temp_data/train'):
    os.mkdir('temp_data/train')
if not os.path.exists('temp_data/val'):
    os.mkdir('temp_data/val')

dir_path = os.path.abspath('./') + '/'

# 需要按照你的修改path
with open('temp_data/yolo.yaml', 'w', encoding='utf-8') as up:
    up.write(f'''
path: {dir_path}/temp_data/
train: train/
val: val/

names:
    0: 非机动车违停
    1: 机动车违停
    2: 垃圾桶满溢
    3: 违法经营
''')
    

train_annos = glob.glob('raw_data/训练集(有标注第一批)/标注/*.json')
train_videos = glob.glob('raw_data/训练集(有标注第一批)/视频/*.mp4')
train_annos.sort(); train_videos.sort();
category_labels = ["非机动车违停", "机动车违停", "垃圾桶满溢", "违法经营"]

# 构建train
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

# 构建test
for anno_path, video_path in zip(train_annos[-3:], train_videos[-3:]):
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
        cv2.imwrite('./temp_data/val/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.jpg', frame)

        if len(frame_anno) != 0:
            with open('./temp_data/val/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.txt', 'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)
                    
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')
        
        frame_idx += 1