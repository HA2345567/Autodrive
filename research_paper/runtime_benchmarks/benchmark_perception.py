from pathlib import Path
import json
import time
import cv2
import torch
from ultralytics import YOLO

root = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
report = {'device': 'cpu', 'threads': torch.get_num_threads(), 'cuda_available': torch.cuda.is_available(), 'frames': []}
models = {
    'lane': YOLO(str(root/'saved_models/lane_detection_model/lane_detection_best.pt')),
    'object': YOLO(str(root/'saved_models/object_detection_model/object_detection_best.pt')),
}
for name, model in models.items():
    print(name, 'task=', model.task, 'classes=', model.names, flush=True)
for frame_id in [2400, 3760]:
    frame = cv2.imread(str(root/f'data/driving_dataset/{frame_id}.jpg'))
    entry = {'frame': frame_id, 'shape': list(frame.shape)}
    for name, model in models.items():
        start = time.perf_counter()
        result = model.predict(frame, imgsz=640 if name == 'lane' else 480, conf=.20 if name == 'lane' else .25, device='cpu', verbose=False)[0]
        entry[name] = {'wall_ms': (time.perf_counter()-start)*1000, 'speed': result.speed, 'boxes': len(result.boxes), 'masks': 0 if result.masks is None else len(result.masks.xy), 'confidences': result.boxes.conf.tolist()}
        cv2.imwrite(str(Path(__file__).parent/f'{name}_{frame_id}.jpg'), result.plot())
        print(frame_id, name, json.dumps(entry[name]), flush=True)
    report['frames'].append(entry)
(Path(__file__).parent/'perception_baseline.json').write_text(json.dumps(report, indent=2))
