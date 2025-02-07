import os
from ultralytics import YOLO
import glob
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
num = 127
model = YOLO(f"model/train{num}/weights/best.pt")
category_labels = {0:"非机动车违停", 1:"机动车违停", 2:"垃圾桶满溢", 3:"违法经营"}

if not os.path.exists(f'result_{num}/'):
    os.mkdir(f'result_{num}')

IMGSZ=1080

for cnt, path in enumerate(glob.glob('测试集/*.mp4')):
    submit_json = []
    results = model(path, conf=0.05,  visualize=False,\
                    imgsz=IMGSZ,  verbose=False, batch=128, stream=True, device=1)
    for idx, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        if idx == 0:
            print(f'cnt:{cnt}, path:{path}')
            # result.show()
            
        if len(boxes.cls) == 0:
            continue
        
        xywh = boxes.xyxy.data.cpu().numpy().round()
        cls = boxes.cls.data.cpu().numpy().round()
        conf = boxes.conf.data.cpu().numpy()
        for i, (ci, xy, confi) in enumerate(zip(cls, xywh, conf)):
            if int(ci) in category_labels:
                submit_json.append(
                    {
                        'frame_id': idx,
                        'event_id': i+1,
                        'category': category_labels[int(ci)],
                        'bbox': list([int(x) for x in xy]),
                        "confidence": float(confi)
                    }
                )

    with open(f'./result_{num}/' + path.split('/')[-1][:-4] + '.json', 'w', encoding='utf-8') as up:
        json.dump(submit_json, up, indent=4, ensure_ascii=False)

