"""AutoDrive Unified Perception & Control HUD.

Integrates:
1. Steering Angle Prediction (NVIDIA CNN Model & smooth steering wheel rotation)
2. Lane Detection & Segmentation (YOLOv11 segmentation model & drivable corridor)
3. Dynamic Cockpit Pedal Telemetry (Brake & Accelerator pedals)

Dual-Window Layout:
- Window 'frame': Live road camera view with lane segmentation and HUD telemetry.
- Window 'Steering & Actuation': Crisp rotating steering wheel + status bar + cockpit pedal bay.
"""

import argparse
import os
import sys
import time
import threading
from typing import Optional, List
import cv2
import numpy as np

import warnings
import logging

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['YOLO_VERBOSE'] = 'False'
warnings.filterwarnings("ignore")

# pyrefly: ignore [missing-import]
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Project Root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore [missing-import]
from src.models import model
from src.inference.lane_detector import AsyncLaneDetector, resolve_lane_model_path
from src.inference.object_detector import AsyncObjectDetector, resolve_object_model_path

# ==============================================================================
# Configuration Toggles
# ==============================================================================
SHOW_ACTUAL = True             # True: Actual dataset steering angle; False: Predicted AI model angle
SHOW_LANE_DETECTION = True     # True: Real-time YOLOv11 Lane Segmentation Overlay
SHOW_OBJECT_DETECTION = True   # True: Real-time YOLOv11 Object Detection Overlay
SHOW_CORRIDOR = False          # False: Crisp lane markings; True: Drivable ego-path corridor


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates image by angle (in degrees) around its center, with clean background fill."""
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1.0)
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def resolve_lane_model() -> Optional[str]:
    """Finds the trained YOLOv11 lane segmentation model checkpoint."""
    candidates = [
        os.path.join(PROJECT_ROOT, "saved_models", "lane_detection_model", "lane_detection_best.pt"),
        os.path.join(PROJECT_ROOT, "saved_models", "lane_detection_best.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return resolve_lane_model_path()


def resolve_object_model() -> Optional[str]:
    """Finds the trained YOLOv11 object detection model checkpoint."""
    candidates = [
        os.path.join(PROJECT_ROOT, "saved_models", "object_detection_model", "object_detection_best.pt"),
        os.path.join(PROJECT_ROOT, "saved_models", "object_detection_best.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return resolve_object_model_path()


def resolve_steering_checkpoint() -> str:
    """Finds the trained NVIDIA CNN steering model checkpoint."""
    candidates = [
        os.path.join(PROJECT_ROOT, "saved_models", "regression_model"),
        os.path.join(PROJECT_ROOT, "saved_models", "steering_angle"),
        os.path.join(PROJECT_ROOT, "model_training", "train_steering_angle", "save"),
        os.path.join(PROJECT_ROOT, "src", "training", "train_steering_angle", "save"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and (
            tf.compat.v1.train.latest_checkpoint(candidate)
            or os.path.exists(os.path.join(candidate, 'model.ckpt.meta'))
        ):
            return candidate
    return os.path.join(PROJECT_ROOT, "saved_models", "regression_model")


def main():
    global SHOW_ACTUAL, SHOW_LANE_DETECTION, SHOW_OBJECT_DETECTION, SHOW_CORRIDOR

    parser = argparse.ArgumentParser(description="AutoDrive Unified Perception & Control HUD")
    parser.add_argument("--fps", type=int, default=20, help="Target playback frame rate (default: 20 FPS)")
    parser.add_argument("--start", type=int, default=0, help="Starting frame index in dataset (e.g. --start 2400)")
    parser.add_argument("--lane-model", type=str, default=None, help="Path to lane detection model checkpoint")
    parser.add_argument("--object-model", type=str, default=None, help="Path to object detection model checkpoint (default: saved_models/object_detection_model/object_detection_best.pt)")
    parser.add_argument("--obj-conf", type=float, default=0.35, help="Confidence threshold for object detection (default: 0.35)")
    parser.add_argument("--lane-conf", type=float, default=0.25, help="Confidence threshold for lane segmentation (default: 0.25)")
    parser.add_argument("--no-lanes", action="store_true", help="Start with lane detection disabled")
    parser.add_argument("--no-objects", action="store_true", help="Start with object detection disabled")
    cli_args, _ = parser.parse_known_args()

    if cli_args.no_lanes:
        SHOW_LANE_DETECTION = False
    if cli_args.no_objects:
        SHOW_OBJECT_DETECTION = False

    dataset_txt = os.path.join(PROJECT_ROOT, "data", "driving_dataset", "data.txt")
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "driving_dataset")
    wheel_image_path = os.path.join(PROJECT_ROOT, "data", "sterring_wheel_image.png")

    # Verify input files
    if not os.path.exists(dataset_txt):
        print(f"Error: Dataset index not found at '{dataset_txt}'.")
        return
    if not os.path.exists(wheel_image_path):
        print(f"Error: Steering wheel image not found at '{wheel_image_path}'.")
        return

    checkpoint_dir = resolve_steering_checkpoint()
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Steering model checkpoint directory not found at '{checkpoint_dir}'.")
        return

    # Load dataset index
    xs: List[str] = []
    ys: List[float] = []
    with open(dataset_txt, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                xs.append(os.path.join(dataset_dir, parts[0]))
                ys.append(float(parts[1]))

    print(f"Loaded {len(xs)} images from driving dataset.")

    # Load and prepare steering wheel image
    wheel_img = cv2.imread(wheel_image_path)
    if wheel_img is None:
        print("Error: Could not load steering wheel image.")
        return
    wheel_img = cv2.resize(wheel_img, (300, 300))

    # Initialize TensorFlow session with memory growth to prevent CUDA conflicts
    print("[1/5] Initializing Steering Angle Model...", flush=True)
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=tf_config)
    saver = tf.compat.v1.train.Saver()

    checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        checkpoint = os.path.abspath(os.path.join(checkpoint_dir, 'model.ckpt'))
    else:
        checkpoint = os.path.abspath(checkpoint)

    print(f"      Restoring steering weights from: {checkpoint}", flush=True)
    try:
        saver.restore(sess, checkpoint)
    except Exception as e:
        print(f"Error restoring model: {e}", flush=True)
        return

    # Asynchronous steering inference worker (non-blocking for buttery smooth 30+ FPS)
    _steering_lock = threading.Lock()
    _steering_event = threading.Event()
    _steering_input: list = [None]
    _predicted_deg_cache: list = [0.0]

    def _steering_worker():
        while True:
            _steering_event.wait()
            _steering_event.clear()
            with _steering_lock:
                inp = _steering_input[0]
            if inp is None:
                break
            try:
                rad = model.y_pred.eval(
                    session=sess,
                    feed_dict={model.x: [inp], model.keep_prob: 1.0}
                )[0][0]
                with _steering_lock:
                    _predicted_deg_cache[0] = float(rad * 180.0 / np.pi)
            except Exception:
                pass

    _steering_thread = threading.Thread(target=_steering_worker, daemon=True)
    _steering_thread.start()

    # Initialize YOLOv11 Lane Detector
    print("[2/5] Initializing YOLOv11 Lane Detector...", flush=True)
    lane_model_path = cli_args.lane_model or resolve_lane_model()
    try:
        lane_detector = AsyncLaneDetector(
            model_name=lane_model_path,
            conf_threshold=cli_args.lane_conf,
            imgsz=640
        )
    except Exception as e:
        print(f"[LaneDetector] Initialization warning: {e}", flush=True)
        lane_detector = None

    # Initialize YOLOv11 Object Detector (object_detection_best.pt)
    print("[3/5] Initializing YOLOv11 Object Detector...", flush=True)
    obj_model_path = cli_args.object_model or resolve_object_model()
    try:
        object_detector = AsyncObjectDetector(
            model_name=obj_model_path,
            conf_threshold=cli_args.obj_conf,
            imgsz=480
        )
        if hasattr(object_detector.detector, "class_names") and object_detector.detector.class_names:
            print(f"      Loaded model: {obj_model_path}")
            print(f"      Classes: {', '.join(object_detector.detector.class_names.values())}", flush=True)
    except Exception as e:
        print(f"[ObjectDetector] Initialization warning: {e}", flush=True)
        object_detector = None


    # Create OpenCV display windows
    print("[4/4] Launching Perception HUD & Steering Actuation Windows...", flush=True)
    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Steering & Actuation", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("frame", 60, 60)
    cv2.moveWindow("Steering & Actuation", 730, 60)

    print("=" * 75)
    print("AutoDrive Autonomous Perception & Control HUD")
    print("Integrated: Steering Prediction + YOLOv11 Lane Detection + YOLOv11 Object Detection")
    print("Controls:")
    print("  'q' or ESC : Exit")
    print("  'l'        : Toggle Lane Detection & Segmentation ON / OFF")
    print("  'o'        : Toggle Object Detection ON / OFF")
    print("  'c'        : Toggle Drivable Ego Corridor Fill ON / OFF")
    print("  'm'        : Toggle Steering Mode (ACTUAL vs PREDICTED AI Model)")
    print("  'p' / SPACE: Pause / Resume playback")
    print("  '[' / ']'  : Skip 100 frames backward / forward")
    print("  '+' / '-'  : Increase / Decrease playback speed")
    print("=" * 75)

    # Visualization loop variables
    smoothed_angle = 0.0
    is_paused = False
    prev_time = time.time()
    fps = 0.0
    target_fps = max(5, min(60, cli_args.fps))
    _frame_deadline = time.time()
    start_idx = max(0, min(len(xs) - 1, cli_args.start)) if xs else 0
    i = start_idx
    if start_idx > 0:
        print(f"[Playback] Starting at Frame {start_idx} of {len(xs)}...")

    try:
        while i < len(xs):
            # Compute FPS
            cur_time = time.time()
            dt = cur_time - prev_time
            prev_time = cur_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            img_path = xs[i]
            actual_deg = ys[i]

            full_image = cv2.imread(img_path)
            if full_image is None:
                i += 1
                continue

            # 1. STEERING PREDICTION (NVIDIA CNN)
            # Crop bottom 150px, resize to 200x66, normalize to [0,1]
            model_input = cv2.resize(full_image[-150:], (200, 66)) / 255.0
            with _steering_lock:
                _steering_input[0] = model_input
                predicted_deg = _predicted_deg_cache[0]
            _steering_event.set()

            # Angle selection and smoothing (time-delta invariant exponential smoothing)
            angle_to_show = actual_deg if SHOW_ACTUAL else predicted_deg
            decay = 10.0
            alpha = 1.0 - np.exp(-decay * dt) if dt > 0 else 0.15
            alpha = max(0.04, min(0.35, alpha))
            smoothed_angle += alpha * (angle_to_show - smoothed_angle)

            # 2. ROTATE STEERING WHEEL & BUILD COCKPIT ACTUATION PANEL (Window: 'Steering & Actuation')
            # Clockwise turn matches positive angle -> negate for warpAffine
            rotated_wheel = rotate_image(wheel_img, -smoothed_angle)

            # Clean High-Tech Cockpit Actuation Panel (320w x 380h)
            cockpit_canvas = np.full((380, 320, 3), 18, dtype=np.uint8)
            cockpit_canvas[10:310, 10:310] = cv2.resize(rotated_wheel, (300, 300))
            cv2.rectangle(cockpit_canvas, (9, 9), (311, 311), (50, 50, 58), 1)

            mode_tag = "ACTUAL" if SHOW_ACTUAL else "AI"
            mode_color = (240, 240, 240) if SHOW_ACTUAL else (0, 230, 255)

            cv2.rectangle(cockpit_canvas, (10, 318), (310, 370), (26, 26, 32), -1)
            cv2.rectangle(cockpit_canvas, (10, 318), (310, 370), (48, 48, 56), 1)
            cv2.putText(cockpit_canvas, f"STEER [{mode_tag}]: {angle_to_show:+.1f} deg", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
            cv2.putText(cockpit_canvas, f"AI PRED : {predicted_deg:+.1f} deg", (20, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)

            # 3. ROAD VIEW OVERLAYS (Window: 'frame')
            road_display: np.ndarray = cv2.resize(full_image, (640, 360))

            # Lane Detection & Segmentation
            lane_info: dict = {"num_lanes": 0, "lanes": []}
            if SHOW_LANE_DETECTION and lane_detector is not None:
                det_frame, det_info = lane_detector.detect(road_display, fill_corridor=SHOW_CORRIDOR)
                if det_frame is not None:
                    road_display = det_frame
                lane_info = det_info

            # Object Detection (YOLOv11 object_detection_best.pt)
            obj_info: dict = {"num_objects": 0, "objects": []}
            if SHOW_OBJECT_DETECTION and object_detector is not None:
                det_frame, det_info = object_detector.detect(road_display)
                if det_frame is not None:
                    road_display = det_frame
                obj_info = det_info


            # Sleek Top-Bar Status Badges (Left-aligned from x=16, y=14)
            badge_x = 16
            badge_y = 14
            badge_h = 24
            badge_gap = 8

            # 1. Lane Detection Badge
            corr_label = "+C" if SHOW_CORRIDOR else ""
            lane_text = f"LANE: {'ON' if SHOW_LANE_DETECTION else 'OFF'}{corr_label}"
            (lw, _), _ = cv2.getTextSize(lane_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            lane_w = lw + 18
            lane_bg = (18, 105, 42) if SHOW_LANE_DETECTION else (26, 26, 32)
            lane_border = (40, 195, 80) if SHOW_LANE_DETECTION else (55, 55, 65)
            lane_fg = (255, 255, 255) if SHOW_LANE_DETECTION else (155, 155, 165)

            cv2.rectangle(road_display, (badge_x, badge_y), (badge_x + lane_w, badge_y + badge_h), lane_bg, -1)
            cv2.rectangle(road_display, (badge_x, badge_y), (badge_x + lane_w, badge_y + badge_h), lane_border, 1)
            cv2.putText(road_display, lane_text, (badge_x + 9, badge_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lane_fg, 1, cv2.LINE_AA)
            badge_x += lane_w + badge_gap

            # 2. Object Detection Badge
            obj_text = f"OBJ: {'ON' if SHOW_OBJECT_DETECTION else 'OFF'}"
            (ow, _), _ = cv2.getTextSize(obj_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            obj_w = ow + 18
            obj_bg = (180, 95, 0) if SHOW_OBJECT_DETECTION else (26, 26, 32)
            obj_border = (235, 140, 20) if SHOW_OBJECT_DETECTION else (55, 55, 65)
            obj_fg = (255, 255, 255) if SHOW_OBJECT_DETECTION else (155, 155, 165)

            cv2.rectangle(road_display, (badge_x, badge_y), (badge_x + obj_w, badge_y + badge_h), obj_bg, -1)
            cv2.rectangle(road_display, (badge_x, badge_y), (badge_x + obj_w, badge_y + badge_h), obj_border, 1)
            cv2.putText(road_display, obj_text, (badge_x + 9, badge_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, obj_fg, 1, cv2.LINE_AA)
            badge_x += obj_w + badge_gap

            # 3. FPS Performance Badge
            fps_text = f"{fps:3.0f} FPS"
            (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            fps_w = fw + 16
            cv2.rectangle(road_display, (badge_x, badge_y), (badge_x + fps_w, badge_y + badge_h), (24, 24, 30), -1)
            cv2.rectangle(road_display, (badge_x, badge_y), (badge_x + fps_w, badge_y + badge_h), (55, 55, 65), 1)
            cv2.putText(road_display, fps_text, (badge_x + 8, badge_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 245, 195), 1, cv2.LINE_AA)

            # HUD Badge 4: Steering Mode & Angle (Bottom-Left)
            mode_label = f"STEERING: {'ACTUAL' if SHOW_ACTUAL else 'AI'} ({angle_to_show:+.1f} deg)"
            mode_color = (0, 255, 255) if not SHOW_ACTUAL else (240, 240, 240)
            (mw, _), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(road_display, (16, 320), (16 + mw + 18, 346), (24, 24, 30), -1)
            cv2.rectangle(road_display, (16, 320), (16 + mw + 18, 346), (55, 55, 65), 1)
            cv2.putText(road_display, mode_label, (25, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_color, 1, cv2.LINE_AA)

            # HUD Badge 5: Perception Tracking Status (Bottom-Right)
            num_lanes = int(lane_info.get("num_lanes", 0) or 0) if SHOW_LANE_DETECTION else 0
            num_objs = int(obj_info.get("num_objects", 0) or 0) if SHOW_OBJECT_DETECTION else 0
            status_items = []
            if SHOW_LANE_DETECTION:
                status_items.append(f"LANES: {num_lanes}")
            if SHOW_OBJECT_DETECTION:
                status_items.append(f"OBJS: {num_objs}")
            if not status_items:
                status_items.append("PERCEPTION OFF")
            status_text = " | ".join(status_items)
            status_color = (0, 255, 120) if (num_lanes > 0 or num_objs > 0) else (140, 140, 150)
            (stw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            bx2 = 624
            bx1 = bx2 - stw - 18
            cv2.rectangle(road_display, (bx1, 320), (bx2, 346), (24, 24, 30), -1)
            cv2.rectangle(road_display, (bx1, 320), (bx2, 346), (55, 55, 65), 1)
            cv2.putText(road_display, status_text, (bx1 + 9, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

            # Render display windows
            cv2.imshow("frame", road_display)
            cv2.imshow("Steering & Actuation", cockpit_canvas)

            # Real-time console telemetry
            lanes_cnt = int(lane_info.get("num_lanes", 0) or 0) if SHOW_LANE_DETECTION else 0
            objs_cnt = int(obj_info.get("num_objects", 0) or 0) if SHOW_OBJECT_DETECTION else 0
            sys.stdout.write(
                f"\rFrame {i:5d} | Act: {actual_deg:+5.1f} | AI: {predicted_deg:+5.1f} | Mode: {mode_tag} | Lanes: {lanes_cnt} | Objs: {objs_cnt} | FPS: {fps:4.1f}"
            )
            sys.stdout.flush()

            # Keyboard interaction & smooth frame pacing
            target_period = 1.0 / target_fps
            elapsed = time.time() - cur_time
            sleep_budget = target_period - elapsed
            budget_ms = max(1, int(sleep_budget * 1000)) if sleep_budget > 0.001 else 1
            key = cv2.waitKey(budget_ms if not is_paused else 200) & 0xFF
            _frame_deadline = time.time() + target_period

            if key == ord('q') or key == 27:
                break
            elif key == ord('l') or key == ord('L'):
                SHOW_LANE_DETECTION = not SHOW_LANE_DETECTION
                print(f"\n[HUD] Lane Detection: {'ENABLED' if SHOW_LANE_DETECTION else 'DISABLED'}")
            elif key == ord('o') or key == ord('O'):
                SHOW_OBJECT_DETECTION = not SHOW_OBJECT_DETECTION
                print(f"\n[HUD] Object Detection: {'ENABLED' if SHOW_OBJECT_DETECTION else 'DISABLED'}")
            elif key == ord('c') or key == ord('C'):
                SHOW_CORRIDOR = not SHOW_CORRIDOR
                print(f"\n[HUD] Drivable Corridor Fill: {'ENABLED' if SHOW_CORRIDOR else 'DISABLED'}")
            elif key == ord('m') or key == ord('M'):
                SHOW_ACTUAL = not SHOW_ACTUAL
                print(f"\n[HUD] Steering Mode: {'ACTUAL (Dataset)' if SHOW_ACTUAL else 'PREDICTED (AI Model)'}")
            elif key == ord('p') or key == ord('P') or key == 32:
                is_paused = not is_paused
                print(f"\n[HUD] {'PAUSED' if is_paused else 'RESUMED'}")
            elif key in (ord('+'), ord('=')):
                target_fps = min(60, target_fps + 2)
                print(f"\n[HUD] Target Speed: {target_fps} FPS")
            elif key == ord('-'):
                target_fps = max(5, target_fps - 2)
                print(f"\n[HUD] Target Speed: {target_fps} FPS")
            elif key == ord(']'):
                i = min(len(xs) - 1, i + 100)
                print(f"\n[HUD] Jumped forward +100 to Frame {i}")
            elif key == ord('['):
                i = max(0, i - 100)
                print(f"\n[HUD] Jumped backward -100 to Frame {i}")

            if not is_paused:
                i += 1

        print()

    except KeyboardInterrupt:
        print("\n[HUD] Visualization stopped by user.")
    finally:
        try:
            with _steering_lock:
                _steering_input[0] = None
            _steering_event.set()
            _steering_thread.join(timeout=1.0)
            if lane_detector is not None and hasattr(lane_detector, 'stop'):
                lane_detector.stop()
            if object_detector is not None and hasattr(object_detector, 'stop'):
                object_detector.stop()
            sess.close()
            cv2.destroyAllWindows()
            print("AutoDrive HUD shutdown cleanly.")
        except (Exception, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
