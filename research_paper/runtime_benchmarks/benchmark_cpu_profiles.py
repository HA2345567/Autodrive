import json
from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO

root=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
frame=cv2.imread(str(root/'data/driving_dataset/2400.jpg'))
lane=YOLO(str(root/'saved_models/lane_detection_model/lane_detection_best.pt'))
obj=YOLO(str(root/'saved_models/object_detection_model/object_detection_best.pt'))
records=[]
for threads in [1,2]:
 torch.set_num_threads(threads)
 for size_l,size_o in [(480,320),(640,480)]:
  record={'threads':threads,'lane_size':size_l,'object_size':size_o}
  for name,model,size in [('lane',lane,size_l),('object',obj,size_o)]:
   results=[]
   for repeat in range(2):
    t=time.perf_counter()
    result=model.predict(frame,imgsz=size,conf=.2 if name=='lane' else .25,device='cpu',verbose=False)[0]
    results.append((time.perf_counter()-t)*1000)
   record[name]={'wall_ms':results[-1],'boxes':len(result.boxes),'masks':0 if result.masks is None else len(result.masks.xy)}
   if size_l==480 and threads==1:
    cv2.imwrite(str(Path(__file__).parent/f'cpu_profile_{name}.jpg'),result.plot())
  print(json.dumps(record),flush=True)
  records.append(record)
(Path(__file__).parent/'cpu_profiles.json').write_text(json.dumps(records,indent=2))
