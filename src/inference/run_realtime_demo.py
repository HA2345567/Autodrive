"""Interactive, frame-aligned AutoDrive inference with honest runtime timing."""
import argparse
import hashlib
import logging
import os
from pathlib import Path
import time

# Set bounds before importing the numerical runtimes. Three independent models
# must not each create a CPU-sized thread pool on a small laptop.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np

from src.inference.lane_detector import LaneDetector
from src.inference.object_detector import ObjectDetector
from src.inference.realtime_pipeline import RealtimePipeline, align_prediction
from src.inference.runtime_backend import prepare_weights

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def arguments():
    parser = argparse.ArgumentParser(description='AutoDrive lane, object and AI steering demo')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--fps', type=int, default=20, help='UI/target playback rate, not detection throughput')
    parser.add_argument('--lane-model', default=str(ROOT/'saved_models/lane_detection_model/lane_detection_best.pt'))
    parser.add_argument('--object-model', default=str(ROOT/'saved_models/object_detection_model/object_detection_best.pt'))
    parser.add_argument('--lane-conf', type=float, default=.20)
    parser.add_argument('--obj-conf', type=float, default=.25)
    parser.add_argument('--lane-imgsz', type=int, default=None)
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--profile', choices=('balanced', 'quality'), default='balanced',
                        help='balanced CPU: 480 lane / 480 object; quality or CUDA: 640 / 640')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, etc.')
    parser.add_argument('--backend', choices=('auto', 'torch', 'openvino'), default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--no-lanes', action='store_true')
    parser.add_argument('--no-objects', action='store_true')
    parser.add_argument('--show-actual', action='store_true', help='Display recorded steering instead of AI; M toggles')
    parser.add_argument('--steering-head', choices=('auto', 'original', 'huber'), default='auto')
    parser.add_argument('--playback-mode', choices=('aligned', 'tracked'), default='aligned',
                        help='aligned: exact source frames; tracked: continuous video with bounded optical-flow overlays')
    parser.add_argument('--max-overlay-age', type=float, default=.75)
    parser.add_argument('--limit', type=int, default=None, help='Stop after this many distinct displayed frames')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--snapshot-dir', type=Path, help='Save the first three rendered prediction frames')
    return parser.parse_args()


def load_steering(head):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from src.models import model
    from src.run import resolve_steering_checkpoint
    folder = resolve_steering_checkpoint()
    checkpoint = tf.train.latest_checkpoint(folder) or str(Path(folder)/'model.ckpt')
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    # Reserve CUDA for perception; the small steering graph uses one CPU thread.
    config.device_count['GPU'] = 0
    session = tf.Session(config=config)
    tf.train.Saver().restore(session, checkpoint)
    readout = None
    path = ROOT/'output/steering_audit/improved_readout.npz'
    if head != 'original':
        try:
            with np.load(path, allow_pickle=False) as data:
                fingerprint = hashlib.sha256(Path(checkpoint+'.data-00000-of-00001').read_bytes()).hexdigest()
                if str(data['checkpoint_sha256']) != fingerprint:
                    raise ValueError('Readout belongs to a different backbone checkpoint')
                if str(data['channel_order']) != 'RGB' or str(data['loss']) != 'huber':
                    raise ValueError('Unexpected readout preprocessing or loss')
                readout = {k: data[k].copy() for k in ('feature_mean', 'feature_scale', 'coefficients', 'target_mean_deg')}
                if readout['coefficients'].shape != (1152,) or np.any(readout['feature_scale'] <= 0):
                    raise ValueError('Invalid readout feature dimensions or scale')
        except Exception:
            if head == 'huber':
                session.close()
                raise
            LOGGER.warning('Verified Huber readout unavailable; using original CNN', exc_info=True)
    label = 'HUBER RGB' if readout else 'CNN BGR'

    def predict(frame):
        image = cv2.resize(frame[-150:], (200, 66)).astype(np.float32) / 255.
        if readout is not None:
            image = image[:, :, ::-1].copy()
            features = session.run(model.h_conv5_flat, {model.x: [image], model.keep_prob: 1.})[0]
            return float(((features-readout['feature_mean'])/readout['feature_scale']) @ readout['coefficients'] + readout['target_mean_deg'])
        return float(session.run(model.y_pred, {model.x: [image], model.keep_prob: 1.})[0, 0] * 180 / np.pi)
    return session, predict, label


def text(image, value, point, color=(240, 240, 240), scale=.43):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
    import torch
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    quality = args.profile == 'quality' or device != 'cpu'
    args.lane_imgsz = args.lane_imgsz or (640 if quality else 480)
    args.imgsz = args.imgsz or (640 if quality else 480)
    LOGGER.info('device=%s | PyTorch threads=%s | playback=%s | AI steering is default', device, torch.get_num_threads(), args.playback_mode)
    index = ROOT/'data/driving_dataset/data.txt'
    records = [(index.parent/line.split()[0], float(line.split()[1]))
               for line in index.read_text().splitlines() if len(line.split()) >= 2]
    if not records:
        raise ValueError('Driving dataset index is empty')
    i = max(0, min(len(records)-1, args.start))
    wheel = cv2.imread(str(ROOT/'data/sterring_wheel_image.png'))
    if wheel is None:
        raise FileNotFoundError('Missing data/sterring_wheel_image.png')
    wheel = cv2.resize(wheel, (300, 300))
    session, steering, head_label = load_steering(args.steering_head)
    lane_path, lane_backend = prepare_weights(args.lane_model, args.lane_imgsz, args.backend, device)
    object_path, object_backend = prepare_weights(args.object_model, args.imgsz, args.backend, device)
    lane = LaneDetector(lane_path, args.lane_conf, args.lane_imgsz, device)
    objects = ObjectDetector(object_path, args.obj_conf, args.imgsz, device)
    if lane.model.task != 'segment' or len(lane.model.names) != 1:
        session.close()
        raise ValueError('Expected the trained single-class lane segmentation model')
    LOGGER.info('Loaded %s lane + %s object | steering=%s', lane_backend, object_backend, head_label)
    # Ultralytics select_device() can reset PyTorch's CPU thread count during
    # the first predictor setup. Initialize each predictor once, THEN enforce
    # the requested cap before concurrent work begins.
    warm_frame = cv2.imread(str(records[i][0]))
    if warm_frame is None:
        session.close()
        raise ValueError(f'Unreadable starting frame: {records[i][0]}')
    LOGGER.info('Warming up the two predictors once...')
    lane.detect(warm_frame)
    objects.detect(warm_frame)
    torch.set_num_threads(max(1, args.threads))
    cv2.setNumThreads(1)
    LOGGER.info('Predictors ready | enforced PyTorch threads=%s', torch.get_num_threads())
    if args.snapshot_dir:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    use_lanes, use_objects = not args.no_lanes, not args.no_objects
    show_actual, paused, corridor = args.show_actual, False, False
    # Steering gets the ORIGINAL source image, preserving its 150-row crop.
    # Perception also sees this clean image; rendering happens after inference.
    pipeline = RealtimePipeline(lane, objects, steering,
                    parallel_detectors=(device == 'cpu' and args.threads == 1
                                        and lane_backend == object_backend == 'torch'))
    submitted = None
    loaded_id, source_frame = None, None
    displayed = set()
    last_log, last_loop, last_seen_result = 0., time.perf_counter(), None
    fps, smoothed = float(args.fps), 0.
    if not args.headless:
        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Steering & Actuation', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('frame', 60, 60)
        cv2.moveWindow('Steering & Actuation', 730, 60)
    LOGGER.info('M: AI/reference | P/Space: pause | L/O: lanes/objects | C: corridor | [/]: seek | Q/Esc: quit')
    try:
        while i < len(records):
            started = time.perf_counter()
            dt = started-last_loop
            last_loop = started
            if dt > .001:
                fps = .9*fps+.1/dt
            if loaded_id != i:
                source_frame = cv2.imread(str(records[i][0]))
                loaded_id = i
                if source_frame is None:
                    LOGGER.error('Unreadable frame %s: %s', i, records[i][0])
                    i += 1
                    continue
            key_id = (i, use_lanes, use_objects)
            if submitted != key_id:
                pipeline.submit(i, source_frame, records[i][1], use_lanes, use_objects)
                submitted = key_id
            result = pipeline.latest()
            valid = result is not None and result.frame_id in (i, i-1)
            if args.playback_mode == 'aligned' and valid:
                base = result.frame
                view_id = result.frame_id
                angle = result.steering_deg
                actual = result.actual_deg
                lane_info = {'lanes': result.lanes if use_lanes else []}
                obj_info = {'objects': result.objects if use_objects else []}
                state = 'INFERENCE ERROR' if result.errors else 'FRAME ALIGNED'
                age = (started-result.captured_at)*1000
            else:
                base, view_id, actual = source_frame, i, records[i][1]
                if args.playback_mode == 'tracked':
                    aligned = align_prediction(result, source_frame, i, args.max_overlay_age,
                                               max_frame_gap=max(1, int(args.fps*args.max_overlay_age)))
                    lane_info = {'lanes': aligned['lanes'] if use_lanes else []}
                    obj_info = {'objects': aligned['objects'] if use_objects else []}
                    state, age = aligned['state'], aligned['age_ms']
                    angle = result.steering_deg if result and aligned['state'] in ('DETECTED', 'TRACKED') else None
                else:
                    lane_info, obj_info = {'lanes': []}, {'objects': []}
                    state, age, angle = 'WARMING UP / SEEKING', None, None
            road = base.copy()
            if lane_info['lanes']:
                road = lane.draw_lanes(road, lane_info['lanes'], fill_corridor=corridor)
            if obj_info['objects']:
                road = objects.draw_objects(road, obj_info['objects'])
            road = cv2.resize(road, (640, 360))
            chosen = actual if show_actual else angle
            if chosen is not None:
                smoothed += min(.35, 1-np.exp(-10*dt))*(chosen-smoothed)
            rotation = cv2.getRotationMatrix2D((150, 150), -smoothed, 1.)
            cockpit = np.full((420, 320, 3), 20, np.uint8)
            cockpit[10:310, 10:310] = cv2.warpAffine(wheel, rotation, (300, 300), borderValue=(255,255,255))
            mode = 'REFERENCE' if show_actual else 'AI'
            color = (245, 245, 245) if show_actual else (0, 225, 255)
            angle_text = 'WAITING' if chosen is None else f'{chosen:+.1f} deg'
            text(cockpit, f'STEERING [{mode}]: {angle_text}', (12, 337), color)
            text(cockpit, f'Model: {head_label}', (12, 360), scale=.4)
            text(cockpit, f'Recorded reference: {actual:+.1f} deg', (12, 382), scale=.4)
            error = 'WAITING' if angle is None else f'{angle-actual:+.1f} deg'
            text(cockpit, f'AI minus reference: {error}', (12, 404), scale=.4)
            cv2.rectangle(road, (0,0), (640,56), (24,24,24), -1)
            text(road, f"LANES {len(lane_info['lanes'])} | OBJECTS {len(obj_info['objects'])} | {state}", (10,20))
            detection_hz = 1000/result.timings['total'] if result and result.timings['total'] else 0.
            text(road, f'Display {min(fps,999):.1f} FPS | Inference {detection_hz:.2f} Hz | {device}', (10,43), scale=.40)
            cv2.rectangle(road, (0,310), (640,360), (24,24,24), -1)
            text(road, f'Frame {view_id} | Steering {mode}: {angle_text}', (10,330), color)
            if result:
                text(road, f"Lane {result.timings.get('lane',0):.0f} ms | Object {result.timings.get('object',0):.0f} ms | Source age {age or 0:.0f} ms", (10,350), scale=.38)
            if result and result.frame_id == view_id and view_id != last_seen_result:
                last_seen_result = view_id
                displayed.add(view_id)
                if args.snapshot_dir and len(displayed) <= 3:
                    cv2.imwrite(str(args.snapshot_dir/f'road_{view_id}.png'), road)
                    cv2.imwrite(str(args.snapshot_dir/f'steering_{view_id}.png'), cockpit)
            if started-last_log >= 1:
                LOGGER.info('frame=%s mode=%s head=%s AI=%s reference=%.2f lanes=%s objects=%s | display=%.1f FPS detection=%.2f Hz timings=%s state=%s',
                            view_id, mode, head_label, None if angle is None else round(angle,2), actual,
                            len(lane_info['lanes']), len(obj_info['objects']), fps, detection_hz,
                            {} if result is None else {k:round(v,1) for k,v in result.timings.items()}, state)
                last_log = started
            if not args.headless:
                cv2.imshow('frame', road)
                cv2.imshow('Steering & Actuation', cockpit)
            if args.headless:
                key = -1
            else:
                # Long waitKey delays are rounded up by Windows GUI timers.
                # Pump events briefly, then pace using Python's precise sleep.
                key = cv2.waitKey(1) & 0xff
                if (cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1
                        or cv2.getWindowProperty('Steering & Actuation', cv2.WND_PROP_VISIBLE) < 1):
                    break
            remaining = 1/max(1,args.fps) - (time.perf_counter()-started)
            if remaining > 0:
                time.sleep(remaining)
            if key in (27, ord('q')):
                break
            elif key == ord('m'):
                show_actual = not show_actual
            elif key in (32,ord('p')):
                paused = not paused
            elif key == ord('l'):
                use_lanes = not use_lanes
            elif key == ord('o'):
                use_objects = not use_objects
            elif key == ord('c'):
                corridor = not corridor
            elif key in (ord('['),ord(']')):
                i = max(0,min(len(records)-1,i+(100 if key == ord(']') else -100)))
                submitted = None
                continue
            elif key in (ord('+'),ord('=')):
                args.fps = min(60,args.fps+2)
            elif key == ord('-'):
                args.fps = max(5,args.fps-2)
            if args.limit and len(displayed) >= args.limit:
                break
            if not paused and (args.playback_mode == 'tracked' or (result and result.frame_id == i)):
                i += 1
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline.stop():
            session.close()
        cv2.destroyAllWindows()
        LOGGER.info('Demo stopped cleanly')


if __name__ == '__main__':
    main()
