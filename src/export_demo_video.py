"""Render a smooth MP4 from real, frame-aligned offline model predictions.

Run: python -m src.export_demo_video --start 2400 --seconds 20 --fps 20
Inference is resumable. Playback FPS is an export setting, not inference speed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=2400)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--lane-imgsz', type=int, default=640)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lane-conf', type=float, default=.20)
    parser.add_argument('--obj-conf', type=float, default=.25)
    parser.add_argument('--output', type=Path, default=ROOT/'output/video/AutoDrive_20s_Demo.mp4')
    return parser.parse_args()


def infer(args, records, config, cache, status_path):
    saved = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            record = json.loads(line)
            saved[record['frame_id']] = record
    required = list(range(args.start, args.start+config['frame_count']))
    todo = [i for i in required if i not in saved]
    if not todo:
        return [saved[i] for i in required]
    import torch
    from ultralytics import YOLO
    from src.inference.run_realtime_demo import load_steering
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    print(f'Loading models; device={device}; pending={len(todo)}', flush=True)
    session, steering, head = load_steering('huber')
    lane_model = YOLO(str(ROOT/config['lane_weights']),task='segment')
    object_model = YOLO(str(ROOT/config['object_weights']),task='detect')
    warm = cv2.imread(str(records[todo[0]][0]))
    lane_model.predict(warm,imgsz=args.lane_imgsz,conf=args.lane_conf,device=device,verbose=False)
    object_model.predict(warm,imgsz=args.imgsz,conf=args.obj_conf,device=device,verbose=False)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    initial_count = len(saved)
    try:
        with cache.open('a',encoding='utf-8') as stream:
            for offset in range(0,len(todo),args.batch):
                ids = todo[offset:offset+args.batch]
                frames = [cv2.imread(str(records[i][0])) for i in ids]
                if any(frame is None for frame in frames):
                    raise ValueError(f'Unreadable image in batch {ids}')
                tick = time.perf_counter()
                lane_results = lane_model.predict(frames,imgsz=args.lane_imgsz,conf=args.lane_conf,
                                                  device=device,verbose=False)
                object_results = object_model.predict(frames,imgsz=args.imgsz,conf=args.obj_conf,
                                                      device=device,verbose=False)
                for i, frame, lane_result, object_result in zip(ids,frames,lane_results,object_results):
                    lanes = []
                    if lane_result.masks is not None:
                        for mask in lane_result.masks.xy:
                            if len(mask)>=3 and float(np.max(mask[:,1]))>=frame.shape[0]*.42:
                                lanes.append(np.round(mask,2).tolist())
                    objects = []
                    for box in object_result.boxes:
                        cls = int(box.cls[0])
                        objects.append({'class_id':cls,'class_name':object_model.names[cls],
                                        'confidence':float(box.conf[0]),'box':box.xyxy[0].tolist()})
                    record = {'frame_id':i,'image':records[i][0].relative_to(ROOT).as_posix(),
                              'image_sha256':digest(records[i][0]),'actual_deg':records[i][1],
                              'ai_deg':float(steering(frame)),'steering_head':head,
                              'lanes':lanes,'objects':objects}
                    if not np.isfinite(record['ai_deg']):
                        raise ValueError(f'Non-finite steering on frame {i}')
                    stream.write(json.dumps(record,separators=(',',':'))+'\n')
                    saved[i] = record
                stream.flush()
                elapsed = time.perf_counter()-started
                completed = len(saved)-initial_count
                remaining = (len(required)-len(saved))*elapsed/max(1,completed)
                status = {'completed':len(saved),'total':len(required),'last_frame':ids[-1],
                          'elapsed_seconds':round(elapsed,1),'estimated_remaining_seconds':round(remaining,1),
                          'last_batch_seconds':round(time.perf_counter()-tick,2),'device':device,
                          'state':'inference'}
                status_path.write_text(json.dumps(status,indent=2))
                print(json.dumps(status),flush=True)
    finally:
        session.close()
    return [saved[i] for i in required]


def label(image,value,xy,size=.55,color=(230,235,240),thickness=1):
    cv2.putText(image,value,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def compose(record, history, wheel, smoothed, index, count, fps, plot_limit):
    canvas = np.full((720,1280,3),(20,16,13),dtype=np.uint8)
    canvas[:5] = (30,185,245)
    label(canvas,'AUTODRIVE',(24,45),1.05,(245,245,245),2)
    label(canvas,'Perception + AI steering research demo',(270,44),.65)
    label(canvas,'OFFLINE RENDER  |  Every frame inferred  |  Playback: '+str(fps)+' FPS',
          (24,78),.46,(180,190,200))
    label(canvas,f"LANE MASKS {len(record['lanes'])}   OBJECTS {len(record['objects'])}   SOURCE FRAME {record['frame_id']}",
          (24,105),.48,(90,225,245))
    frame = cv2.imread(str(ROOT/record['image']))
    if frame is None or digest(ROOT/record['image']) != record['image_sha256']:
        raise ValueError('Source frame changed after inference: '+record['image'])
    road = cv2.resize(frame,(884,498),interpolation=cv2.INTER_CUBIC)
    scale = np.array([884/frame.shape[1],498/frame.shape[0]])
    overlay = road.copy()
    polygons = []
    for lane in record['lanes']:
        pts = (np.asarray(lane)*scale).astype(np.int32).reshape(-1,1,2)
        polygons.append(pts)
        cv2.fillPoly(overlay,[pts],(235,195,25))
    road = cv2.addWeighted(overlay,.22,road,.78,0)
    for pts in polygons:
        cv2.polylines(road,[pts],True,(245,220,80),1,cv2.LINE_AA)
    for obj in record['objects']:
        box=(np.asarray(obj['box'])*np.tile(scale,2)).astype(int)
        x1,y1,x2,y2=box
        color=(80,230,145) if obj['class_name']!='traffic light' else (70,170,255)
        cv2.rectangle(road,(x1,y1),(x2,y2),color,2)
        value=f"{obj['class_name']} {obj['confidence']:.2f}"
        (tw,th),_=cv2.getTextSize(value,cv2.FONT_HERSHEY_SIMPLEX,.42,1)
        tx=min(max(0,x1),max(0,884-tw-8)); ty=max(th+8,y1)
        cv2.rectangle(road,(tx,ty-th-8),(tx+tw+7,ty+2),(25,30,25),-1)
        label(road,value,(tx+3,ty-3),.42,color)
    canvas[116:614,24:908]=road
    cv2.line(canvas,(927,100),(927,649),(65,65,65),1)
    label(canvas,'AI STEERING',(952,110),.72,(90,225,245),2)
    rotation=cv2.getRotationMatrix2D((140,140),-smoothed,1.)
    rotated=cv2.warpAffine(wheel,rotation,(280,280),borderValue=(250,250,250))
    canvas[128:408,952:1232]=rotated
    label(canvas,f"{record['ai_deg']:+.2f} deg",(950,455),1.07,(90,225,245),2)
    label(canvas,'Model: CNN + Huber (RGB)',(952,485),.46)
    label(canvas,f"Recorded: {record['actual_deg']:+.2f} deg",(952,520),.56)
    label(canvas,f"Error: {record['ai_deg']-record['actual_deg']:+.2f} deg",(952,552),.56,(100,180,250))
    label(canvas,'Wheel animation smoothed',(952,583),.39,(160,170,180))
    label(canvas,'Numeric AI output is unsmoothed',(952,604),.36,(160,170,180))
    label(canvas,f'{(index+1)/fps:05.2f} / {count/fps:.0f} s',(952,641),.57)
    # A compact trace keeps the AI/reference difference visible throughout.
    cv2.rectangle(canvas,(24,632),(908,678),(30,29,26),-1)
    middle=655
    cv2.line(canvas,(24,middle),(908,middle),(65,65,65),1)
    segment=history[max(0,index-149):index+1]
    for key,color in [('actual_deg',(200,205,210)),('ai_deg',(90,225,245))]:
        points=[]
        for j,item in enumerate(segment):
            points.append([int(24+j*884/149),int(middle-item[key]/plot_limit*21)])
        if len(points)>1:
            cv2.polylines(canvas,[np.asarray(points,np.int32)],False,color,1,cv2.LINE_AA)
    label(canvas,'AI / recorded steering trace',(24,627),.35,(160,170,180))
    label(canvas,'Harsh Bhardwaj  |  Shrishti Singh  |  Rajat Kandpal',(24,700),.44)
    label(canvas,'github.com/HA2345567/Autodrive',(952,686),.36,(160,170,180))
    label(canvas,'Research prototype: model outputs may contain misses or errors. Playback FPS is not inference speed.',
          (24,717),.34,(150,160,170))
    return canvas


def render(args, predictions, status_path):
    import imageio_ffmpeg
    wheel=cv2.imread(str(ROOT/'data/sterring_wheel_image.png'))
    if wheel is None:
        raise FileNotFoundError('Missing steering-wheel image')
    wheel=cv2.resize(wheel,(280,280),interpolation=cv2.INTER_CUBIC)
    temporary=args.output.with_name(args.output.stem+'.rendering.mp4')
    log=args.output.with_suffix('.ffmpeg.log')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','warning',
             '-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(args.fps),'-i','-',
             '-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(temporary)]
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    plot_limit=max(30,max(abs(row[key]) for row in predictions for key in ['ai_deg','actual_deg'])*1.05)
    smoothed=predictions[0]['ai_deg']
    # Match the original cockpit's 0.15 EMA at 20 FPS. Scale with playback
    # time so the same gentle movement is retained at other export rates.
    wheel_alpha=1-(1-.15)**(20/args.fps)
    with log.open('w') as errors:
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors,**kwargs)
        try:
            for j,record in enumerate(predictions):
                smoothed+=wheel_alpha*(record['ai_deg']-smoothed)
                frame=compose(record,predictions,wheel,smoothed,j,len(predictions),args.fps,plot_limit)
                encoder.stdin.write(frame.tobytes())
                if j in (0,len(predictions)//2,len(predictions)-1):
                    cv2.imwrite(str(args.output.with_name(args.output.stem+f'_frame_{j:04d}.png')),frame)
            encoder.stdin.close()
            code=encoder.wait()
            if code:
                raise RuntimeError(f'FFmpeg failed ({code}); see {log}')
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
    capture=cv2.VideoCapture(str(temporary))
    observed={'frame_count':int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
              'fps':capture.get(cv2.CAP_PROP_FPS),'width':int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height':int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    ok,_=capture.read(); capture.release()
    if not ok or observed['frame_count']!=len(predictions) or abs(observed['fps']-args.fps)>.01:
        raise RuntimeError('Encoded video metadata or decoding does not match the requested clip')
    temporary.replace(args.output)
    observed.update(state='complete',output=str(args.output),duration_seconds=len(predictions)/args.fps,
                    bytes=args.output.stat().st_size,per_frame_inference=True,real_time_benchmark=False,
                    wheel_animation='original cockpit EMA',wheel_alpha_at_20_fps=.15,
                    numeric_ai_output='unsmoothed')
    status_path.write_text(json.dumps(observed,indent=2))
    print(json.dumps(observed),flush=True)


def main():
    args=arguments()
    if min(args.seconds,args.fps,args.batch,args.threads)<=0 or args.start<0:
        raise ValueError('Start must be nonnegative; seconds, FPS, batch and threads must be positive')
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    index=ROOT/'data/driving_dataset/data.txt'
    records=[(index.parent/row.split()[0],float(row.split()[1])) for row in index.read_text().splitlines() if len(row.split())>=2]
    count=round(args.seconds*args.fps)
    if args.start+count>len(records):
        raise ValueError('Requested clip extends beyond the dataset')
    config={'start':args.start,'frame_count':count,'fps':args.fps,'lane_imgsz':args.lane_imgsz,
            'imgsz':args.imgsz,'lane_conf':args.lane_conf,'obj_conf':args.obj_conf,
            'lane_weights':'saved_models/lane_detection_model/lane_detection_best.pt',
            'object_weights':'saved_models/object_detection_model/object_detection_best.pt',
            'steering_head':'output/steering_audit/improved_readout.npz',
            'index_sha256':digest(index)}
    for key in ['lane_weights','object_weights','steering_head']:
        config[key+'_sha256']=digest(ROOT/config[key])
    manifest=args.output.with_suffix('.manifest.json')
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Existing cache uses a different configuration; select a different --output path')
    manifest.write_text(json.dumps(config,indent=2))
    cache=args.output.with_suffix('.predictions.jsonl')
    status=args.output.with_suffix('.status.json')
    predictions=infer(args,records,config,cache,status)
    with args.output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['source_frame','image','recorded_deg','ai_deg','signed_error_deg','lane_masks','objects'])
        for row in predictions:
            writer.writerow([row['frame_id'],row['image'],row['actual_deg'],row['ai_deg'],row['ai_deg']-row['actual_deg'],len(row['lanes']),len(row['objects'])])
    status.write_text(json.dumps({'state':'encoding','completed':len(predictions),'total':count}))
    render(args,predictions,status)


if __name__=='__main__':
    main()
