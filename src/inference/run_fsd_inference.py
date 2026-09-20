"""AutoDrive Full Self-Driving (FSD) End-to-End Perception & Control Pipeline.

Integrates:
1. NVIDIA PilotNet CNN Steering Angle Prediction (TensorFlow 1.x / tf.compat.v1)
2. YOLOv11 Lane Segmentation & Drivable Corridor
3. YOLOv11 / Pretrained Object Detection (Vehicles, Pedestrians, Obstacles)
4. Dynamic Cockpit Pedal Telemetry (PedalController - Brake & Gas)
5. Multi-View Real-Time OpenCV HUD Display

PERFORMANCE NOTES (fixes applied to address playback lag):
  - Both YOLO models now request GPU ('cuda:0') when available, falling back
    to CPU only if no GPU is present. Running two YOLO models on CPU per
    frame was the dominant cause of lag.
  - Frame reads (cv2.imread) are moved off the main thread into a small
    prefetch queue so disk I/O overlaps with inference instead of blocking it.
  - Per-stage timings (read / steer / lane+obj / render) are tracked with a
    rolling average and printed + overlaid on the HUD, so bottlenecks are
    visible instead of guessed at.
  - The loop no longer pretends to hit a fixed 30fps: it reports the actual
    measured FPS instead of just padding sleep time.
"""

import os
import sys
import time
import threading
import queue
import colorsys
import argparse
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np

import warnings
import logging

# Suppress TensorFlow logging and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
# Suppress the Ultralytics 'half is deprecated, use quantize' warning that
# floods the terminal on every background inference cycle.
warnings.filterwarnings("ignore", message=".*half.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*quantize.*", category=UserWarning)
try:
    import absl.logging
    handler = getattr(absl.logging, "_absl_handler", None)
    if handler is not None:
        logging.root.removeHandler(handler)
    absl.logging.set_verbosity("error")
    absl.logging.set_stderrthreshold("error")
except Exception:
    pass

# pyrefly: ignore [missing-import]
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models import model
from src.inference.lane_detector import resolve_lane_model_path
from src.inference.pedal_dashboard import PedalController, PedalRenderer
from ultralytics import YOLO

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False


def resolve_device() -> str:
    """Picks 'cuda:0' if a GPU is available, otherwise falls back to CPU.

    Running both YOLO models on CPU per-frame is the single biggest cause
    of simulator lag, so this is checked explicitly instead of relying on
    Ultralytics' implicit default.
    """
    return "cuda:0" if _CUDA_AVAILABLE else "cpu"


def print_device_diagnostics():
    """Loudly reports GPU availability so a silent CPU fallback is never
    mistaken for 'the code is slow' when it's actually 'no GPU is being used'.
    """
    print("=" * 70)
    print("[Diagnostics] Checking compute device...")
    try:
        import torch
        print(f"[Diagnostics] torch version       : {torch.__version__}")
        print(f"[Diagnostics] torch.cuda available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Diagnostics] GPU name             : {torch.cuda.get_device_name(0)}")
            print(f"[Diagnostics] CUDA build           : {torch.version.cuda}")
        else:
            print("[Diagnostics] *** NO GPU DETECTED — YOLO will run on CPU. ***")
            print("[Diagnostics] Lane/object inference in the multi-second range per")
            print("[Diagnostics] frame (as opposed to tens of ms) is expected on CPU")
            print("[Diagnostics] for these model sizes. This is very likely your lag.")
            print("[Diagnostics] Fix options:")
            print("[Diagnostics]   1) If you have an NVIDIA GPU: reinstall torch with")
            print("[Diagnostics]      CUDA support, e.g.:")
            print("[Diagnostics]      pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("[Diagnostics]      (pick the cuXXX tag matching your installed CUDA/driver)")
            print("[Diagnostics]   2) If you don't have an NVIDIA GPU: switch to the")
            print("[Diagnostics]      smallest YOLO checkpoints (yolo11n.pt / nano) and")
            print("[Diagnostics]      run with --imgsz 192 --detect-every 10 or higher.")
    except Exception as e:
        print(f"[Diagnostics] Could not query torch/cuda: {e}")
    print("=" * 70)


def resolve_steering_checkpoint(model_dir: str = "saved_models/regression_model") -> str:
    """Finds the trained PilotNet CNN steering model checkpoint prefix."""
    candidates = [
        model_dir,
        "saved_models/steering_angle",
        "src/training/train_steering_angle/save",
        os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "regression_model"),
        os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "steering_angle"),
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c):
            ckpt = tf.train.latest_checkpoint(abs_c)
            if ckpt:
                return ckpt
            direct_meta = os.path.join(abs_c, "model.ckpt.meta")
            if os.path.exists(direct_meta):
                return os.path.join(abs_c, "model.ckpt")
    # Fallback to model.ckpt in specified directory
    return os.path.abspath(os.path.join(model_dir, "model.ckpt"))


def resolve_object_model_path(path: str = "saved_models/object_detection_model/object_detection_best.pt") -> str:
    """Resolves object detection model weights with graceful fallback to standard YOLO."""
    if os.path.exists(path):
        return os.path.abspath(path)
    # Check if directory exists and has .pt files
    obj_dir = os.path.dirname(os.path.abspath(path))
    if os.path.exists(obj_dir):
        for f in os.listdir(obj_dir):
            if f.endswith(".pt"):
                return os.path.join(obj_dir, f)
    # Fallback to pretrained lightweight COCO model
    return "yolo11n.pt"


def maybe_export_to_openvino(pt_path: str, imgsz: int) -> str:
    """Exports a .pt YOLO model to OpenVINO IR format for faster CPU inference,
    caching the result on disk so export only happens once. Falls back to the
    original .pt path if OpenVINO isn't installed or export fails.

    OpenVINO is Intel's inference runtime, purpose-built for CPU speed —
    on a no-GPU machine this is typically a 2-4x speedup over raw PyTorch
    for the same model, with no accuracy loss.
    """
    try:
        from ultralytics import YOLO as _YOLO
    except Exception:
        return pt_path

    base, _ = os.path.splitext(pt_path)
    ov_dir = f"{base}_openvino_model"
    ov_xml = os.path.join(ov_dir, os.path.basename(base) + ".xml")

    if os.path.isdir(ov_dir) and os.path.exists(ov_xml):
        return ov_dir  # already exported

    try:
        # pyrefly: ignore [missing-import]
        import openvino  # noqa: F401
    except ImportError:
        print(f"[Perception] OpenVINO not installed — skipping CPU acceleration for {pt_path}.")
        print("[Perception] Install with: pip install openvino")
        return pt_path

    try:
        print(f"[Perception] Exporting {pt_path} -> OpenVINO IR (one-time, cached at {ov_dir}) ...")
        m = _YOLO(pt_path)
        m.export(format="openvino", imgsz=imgsz, half=False)
        if os.path.isdir(ov_dir):
            print(f"[Perception] OpenVINO export complete: {ov_dir}")
            return ov_dir
        return pt_path
    except Exception as e:
        print(f"[Perception] OpenVINO export failed ({e}); falling back to .pt on CPU.")
        return pt_path


def resolve_wheel_image_path(path: str = "data/sterring_wheel_image.png") -> str:
    """Resolves steering wheel image asset path."""
    candidates = [
        path,
        "data/sterring_wheel_image.png",
        "assets/steering_wheel.jpg",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "sterring_wheel_image.png"),
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c):
            return abs_c
    return path


def resolve_pedal_paths(
    brake_path: str = "data/brake_pedal.png",
    gas_path: str = "data/gas_pedal.png"
) -> Tuple[str, str]:
    """Resolves brake and gas pedal image asset paths."""
    resolved_brake = brake_path
    brake_candidates = [
        brake_path,
        "data/brake_pedal.png",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "brake_pedal.png"),
    ]
    for c in brake_candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c):
            resolved_brake = abs_c
            break

    resolved_gas = gas_path
    gas_candidates = [
        gas_path,
        "data/gas_pedal.png",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "gas_pedal.png"),
    ]
    for c in gas_candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c):
            resolved_gas = abs_c
            break

    return resolved_brake, resolved_gas


class FramePrefetcher:
    """Reads frames from disk on a background thread into a bounded queue,
    along with ground-truth actual steering angles parsed from data.txt.
    """                                 

    def __init__(self, data_path: str, start_idx: int = 0, limit: Optional[int] = None, queue_size: int = 8):
        self.data_path = data_path
        self.limit = limit
        self.queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()

        # Parse ground-truth steering angles from data.txt
        self.angles: dict = {}
        data_txt = os.path.join(data_path, "data.txt")
        if not os.path.exists(data_txt):
            data_txt = os.path.join(os.path.dirname(__file__), "..", "..", "data", "driving_dataset", "data.txt")
        if os.path.exists(data_txt):
            with open(data_txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            f_idx = int(parts[0].replace(".jpg", ""))
                            self.angles[f_idx] = float(parts[1])
                        except Exception:
                            pass

        self._thread = threading.Thread(target=self._worker, args=(start_idx,), daemon=True)
        self._thread.start()

    def _worker(self, start_idx: int):
        i = start_idx
        while not self._stop_event.is_set():
            if self.limit is not None and i >= self.limit:
                self.queue.put((i, None, 0.0))  # sentinel: no more frames expected
                return
            frame_file = os.path.join(self.data_path, f"{i}.jpg")
            img = cv2.imread(frame_file)
            actual_angle = self.angles.get(i, 0.0)
            self.queue.put((i, img, actual_angle))
            if img is None:
                return
            i += 1

    def get(self, timeout: float = 5.0) -> Tuple[int, Optional[np.ndarray], float]:
        return self.queue.get(timeout=timeout)

    def stop(self):
        self._stop_event.set()


class RollingTimer:
    """Tracks a rolling average duration (ms) for a named pipeline stage."""

    def __init__(self, window: int = 30):
        self.window = window
        self.samples: List[float] = []

    def add(self, seconds: float):
        self.samples.append(seconds * 1000.0)
        if len(self.samples) > self.window:
            self.samples.pop(0)

    @property
    def avg_ms(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


class SteeringAnglePredictor:
    """Predicts steering angle in degrees using NVIDIA PilotNet CNN under TF1 compat."""

    def __init__(self, checkpoint_path: str):
        self.sess = tf.compat.v1.InteractiveSession()
        self.saver = tf.compat.v1.train.Saver()
        resolved_ckpt = resolve_steering_checkpoint(checkpoint_path)
        print(f"[SteeringPredictor] Restoring checkpoint: {resolved_ckpt}")
        self.saver.restore(self.sess, resolved_ckpt)

    def predict_angle(self, image: np.ndarray) -> float:
        """Predicts steering angle in degrees from preprocessed camera frame."""
        with self.sess.as_default():
            # Model outputs angle in radians; convert to degrees
            rad = model.y_pred.eval(
                session=self.sess,
                feed_dict={model.x: [image], model.keep_prob: 1.0},
            )[0][0]
            return float(rad * 180.0 / np.pi)

    def close(self):
        if hasattr(self, "sess") and self.sess is not None:
            self.sess.close()


class ImageSegmentation:
    """Handles lane detection, corridor polygon filling, and object bounding boxes."""

    def __init__(
        self,
        lane_model_path: str,
        object_model_path: str,
        device: Optional[str] = None,
        imgsz: int = 480,
        lane_imgsz: Optional[int] = None,
        detect_every_n: int = 3,
        use_openvino: bool = False,
        obj_conf: float = 0.35,
        lane_conf: float = 0.25,
    ):
        resolved_lane = resolve_lane_model_path(lane_model_path)
        resolved_obj = resolve_object_model_path(object_model_path)
        self.device = device or resolve_device()
        self.obj_conf = float(obj_conf)
        self.lane_conf = float(lane_conf)

        if use_openvino and self.device == "cpu":
            resolved_lane = maybe_export_to_openvino(resolved_lane, lane_imgsz or max(imgsz, 640))
            resolved_obj = maybe_export_to_openvino(resolved_obj, imgsz)
            # OpenVINO IR models are CPU-only and don't take a torch device string
            self._ov_active = resolved_lane.endswith("_openvino_model") or resolved_obj.endswith("_openvino_model")
        else:
            self._ov_active = False
        # NOTE: with async detection (AsyncDetectionWorker), the main video
        # loop no longer blocks on inference — so there is no longer a
        # reason to shrink imgsz for FPS's sake. A small imgsz (e.g. 192)
        # was previously forced here to keep the blocking call fast, but
        # that resolution is too low for the lane segmentation model to
        # reliably detect thin lane lines, which is why the corridor was
        # showing up missing/broken/incorrect. We now keep imgsz reasonably
        # high by default for detection QUALITY; the only cost of a larger
        # imgsz now is that the background thread updates boxes/lanes a
        # little less often — the video itself stays smooth regardless.
        self.imgsz = imgsz
        # Lane segmentation needs more resolution than object detection to
        # reliably pick up thin lane lines — default it higher unless the
        # caller overrides. This directly fixes "lane not showing correctly".
        self.lane_imgsz = lane_imgsz or max(imgsz, 640)
        self.half = self.device.startswith("cuda")
        self._warned_no_lanes = False
        # Only run the (expensive) YOLO passes every Nth frame; the cached
        # overlay is reused on the frames in between. This is the single
        # biggest lever for reducing lag, since objects/lanes barely move
        # between consecutive frames at typical playback rates.
        self.detect_every_n = max(1, detect_every_n)
        self._frame_count = 0
        self._cached_overlay: Optional[np.ndarray] = None
        self._cached_lane_ms = 0.0
        self._cached_object_ms = 0.0

        print(f"[Perception] Loading Lane Model    : {resolved_lane} (device={self.device}, imgsz={self.lane_imgsz}, half={self.half})")
        self.lane_model = YOLO(resolved_lane)
        print(f"[Perception] Loading Object Model  : {resolved_obj} (device={self.device}, imgsz={self.imgsz}, half={self.half})")
        self.object_model = YOLO(resolved_obj)
        self.colors = self._generate_colors(len(self.object_model.names))
        print(f"[Perception] Object model classes  : {', '.join(self.object_model.names.values())}")
        print(f"[Perception] Lane conf threshold   : {self.lane_conf:.2f}")
        print(f"[Perception] Object conf threshold : {self.obj_conf:.2f}")

        # Warm up both models once so the first real frame isn't hit with
        # CUDA-context / cuDNN autotune overhead (that overhead can look
        # exactly like "lag" on frame 0-1 otherwise).
        try:
            lane_dummy = np.zeros((self.lane_imgsz, self.lane_imgsz, 3), dtype=np.uint8)
            object_dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.lane_model.predict(lane_dummy, imgsz=self.lane_imgsz, device=self.device, half=self.half, verbose=False)
            self.object_model.predict(object_dummy, imgsz=self.imgsz, device=self.device, half=self.half, verbose=False)
        except Exception as e:
            print(f"[Perception] Warmup skipped: {e}")

    @staticmethod
    def _generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
        colors: List[Tuple[int, int, int]] = []
        for i in range(max(1, num_colors)):
            hue = i / max(1, num_colors)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def process(self, img: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, float, float]:
        """Synchronous, blocking detect + draw. Kept for reference / simple
        use cases; the live simulator now uses `detect_raw` + `draw_cached`
        via AsyncDetectionWorker so the main loop never blocks on this.
        """
        lane_results, object_results, lane_ms, object_ms = self.detect_raw(img)
        overlay = img.copy()
        self._draw_lane_overlay(overlay, lane_results)
        self._draw_object_overlay(overlay, object_results)
        blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        return blended, lane_ms, object_ms

    def detect_raw(self, img: np.ndarray):
        """Runs both YOLO models and returns raw results (no drawing).
        This is the expensive, blocking part — meant to be called from a
        background thread (see AsyncDetectionWorker), never from the main
        render loop directly.
        """
        common_kwargs: dict = {"verbose": False}
        if not self._ov_active:
            common_kwargs["device"] = self.device
            common_kwargs["half"] = self.half

        t0 = time.time()
        lane_results: Any = self.lane_model.predict(img, conf=self.lane_conf, imgsz=self.lane_imgsz, **common_kwargs)
        lane_ms = (time.time() - t0) * 1000.0

        # Debug visibility: if the lane model consistently finds 0 masks,
        # that's a model/confidence issue, not a drawing bug — surface it
        # instead of silently rendering an empty corridor.
        num_lane_masks = 0
        for r in lane_results:
            if getattr(r, "masks", None) is not None:
                num_lane_masks += len(getattr(r.masks, "xy", []))

        if num_lane_masks == 0 and not self._warned_no_lanes:
            print(
                f"\n[Perception][WARN] Lane model returned 0 masks on this frame. "
                f"If this persists, try lowering conf (currently {self.lane_conf:.2f}) or check "
                f"that --lane-model points at a segmentation checkpoint "
                f"(mask-producing), not a plain detection .pt."
            )
            self._warned_no_lanes = True
        elif num_lane_masks > 0:
            self._warned_no_lanes = False  # reset so future silence re-warns

        t0 = time.time()
        object_results = self.object_model.predict(img, conf=self.obj_conf, imgsz=self.imgsz, **common_kwargs)
        object_ms = (time.time() - t0) * 1000.0

        return lane_results, object_results, lane_ms, object_ms

    def draw_cached(self, img: np.ndarray, lane_results, object_results, alpha: float = 0.5) -> np.ndarray:
        """Cheap: draws (possibly stale/cached) detection results onto the
        CURRENT live frame. This is what makes async detection look correct
        instead of double-exposed — we never reuse an old frame's pixels,
        only its detected shapes, redrawn fresh on top of the live image.
        """
        overlay = img.copy()
        if lane_results is not None:
            self._draw_lane_overlay(overlay, lane_results)
        if object_results is not None:
            self._draw_object_overlay(overlay, object_results)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    def _draw_lane_overlay(self, overlay: np.ndarray, lane_results):
        for result in lane_results:
            if result.masks is None or len(result.masks.xy) == 0:
                continue
            corridor_pts = []
            for mask in result.masks.xy:
                if len(mask) < 3:
                    continue
                # cv2.polylines expects (N, 1, 2), not (1, N, 2)
                points = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                corridor_pts.append(points)
                cv2.polylines(overlay, [points], isClosed=False, color=(255, 220, 0), thickness=3, lineType=cv2.LINE_AA)

            # Fill drivable corridor if 2+ lane markings found
            if len(corridor_pts) >= 2:
                try:
                    sorted_lanes = sorted(corridor_pts, key=lambda p: float(np.mean(p[:, 0, 0])))
                    left = sorted_lanes[0][:, 0, :]
                    right = sorted_lanes[-1][:, 0, :]
                    left_s = left[np.argsort(left[:, 1])]
                    right_s = right[np.argsort(-right[:, 1])]
                    corridor = np.vstack([left_s, right_s])
                    cv2.fillPoly(overlay, [corridor.astype(np.int32)], (0, 220, 100))
                except Exception:
                    pass

    def _draw_object_overlay(self, overlay: np.ndarray, object_results):
        for result in object_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                color = self.colors[class_id % len(self.colors)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                label = f"{self.object_model.names[class_id]}: {float(box.conf[0]):.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, max(0, y1 - lh - 8)), (x1 + lw, y1), color, -1)
                cv2.putText(overlay, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class AsyncDetectionWorker:
    """Runs YOLO detection on a background thread so the main render loop
    NEVER blocks waiting for it. This is the real fix for CPU-bound lag:
    instead of freezing the whole app for 2-3 seconds per detection call,
    the video keeps playing at full speed using the most recently available
    detection result, and that result updates in the background whenever
    the model finishes — a few times a second instead of every frame.

    Only the single most recent submitted frame is ever processed (older
    ones are dropped), so the worker never falls behind on a growing queue.
    """

    def __init__(self, segmentation: "ImageSegmentation"):
        self.segmentation = segmentation
        self._input: "queue.Queue" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_lane_results = None
        self._latest_object_results = None
        self._latest_lane_ms = 0.0
        self._latest_object_ms = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray):
        """Non-blocking: replaces any not-yet-processed frame with this one."""
        try:
            self._input.get_nowait()  # drop stale pending frame, if any
        except queue.Empty:
            pass
        try:
            self._input.put_nowait(frame)
        except queue.Full:
            pass  # worker is mid-inference; this frame will be skipped, that's fine

    def get_latest(self):
        """Returns (lane_results, object_results, lane_ms, object_ms) — may be
        None/0.0 until the first detection completes."""
        with self._lock:
            return (
                self._latest_lane_results,
                self._latest_object_results,
                self._latest_lane_ms,
                self._latest_object_ms,
            )

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._input.get(timeout=0.5)
            except queue.Empty:
                continue
            lane_results, object_results, lane_ms, object_ms = self.segmentation.detect_raw(frame)
            with self._lock:
                self._latest_lane_results = lane_results
                self._latest_object_results = object_results
                self._latest_lane_ms = lane_ms
                self._latest_object_ms = object_ms

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)


class SelfDrivingCarSimulator:
    """Simulates FSD-style perception + cockpit HUD playback (visualization only;
    this does not actuate a real or simulated vehicle physics model)."""

    def __init__(
        self,
        steering_model: SteeringAnglePredictor,
        segmentation_model: ImageSegmentation,
        data_path: str = "data/driving_dataset",
        wheel_img_path: str = "data/sterring_wheel_image.png",
        brake_pedal_path: str = "data/brake_pedal.png",
        gas_pedal_path: str = "data/gas_pedal.png",
        verbose_profiling: bool = True,
        show_actual: bool = True,
    ):
        self.steering_model = steering_model
        self.segmentation_model = segmentation_model
        self.data_path = data_path
        self.verbose_profiling = verbose_profiling
        self.show_actual = show_actual

        resolved_wheel = resolve_wheel_image_path(wheel_img_path)
        self.wheel_img = cv2.imread(resolved_wheel)
        if self.wheel_img is None:
            # Procedural steering wheel fallback if image missing
            self.wheel_img = np.full((300, 300, 3), 255, dtype=np.uint8)
            cv2.circle(self.wheel_img, (150, 150), 120, (50, 50, 50), 16)
        else:
            self.wheel_img = cv2.resize(self.wheel_img, (300, 300))

        self.smoothed_angle = 0.0
        self.pedal_controller = PedalController(smoothing_alpha=0.25)
        resolved_brake, resolved_gas = resolve_pedal_paths(brake_pedal_path, gas_pedal_path)
        self.pedal_renderer = PedalRenderer(
            brake_asset_path=resolved_brake,
            gas_asset_path=resolved_gas,
        )
        self.rows, self.cols = self.wheel_img.shape[:2]

        # Rolling per-stage timers for lag diagnosis
        self.t_read = RollingTimer()
        self.t_steer = RollingTimer()
        self.t_lane = RollingTimer()
        self.t_object = RollingTimer()
        self.t_render = RollingTimer()
        self.t_frame = RollingTimer()

    def start_simulation(self, frame_interval: float = 1.0 / 30.0, limit: Optional[int] = None):
        cv2.namedWindow("AutoDrive FSD Perception", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Steering & Actuation", cv2.WINDOW_AUTOSIZE)

        print("=" * 70)
        print("AutoDrive Full Self-Driving (FSD) Simulation Running")
        print(f"  Perception device : {self.segmentation_model.device}")
        print("  Controls:")
        print("    'q' or ESC : Exit simulation")
        print("    'm'        : Toggle Steering Mode (ACTUAL Dataset vs AI Model)")
        print("    'p' / SPACE: Pause / Resume simulation")
        print("=" * 70)

        prefetcher = FramePrefetcher(self.data_path, start_idx=0, limit=limit)
        detector = AsyncDetectionWorker(self.segmentation_model)

        i = 0
        is_paused = False
        while True:
            frame_start = time.time()

            t0 = time.time()
            try:
                idx, full_image, actual_deg = prefetcher.get(timeout=5.0)
            except queue.Empty:
                print("\nFrame prefetch timed out; stopping.")
                break
            read_ms = (time.time() - t0) * 1000.0
            self.t_read.add(read_ms / 1000.0)

            if full_image is None:
                if i == 0:
                    print(f"Error: no frames found in '{self.data_path}'.")
                else:
                    print(f"\nCompleted playback of {i} frames from dataset.")
                break

            # 1. Steering Prediction (NVIDIA CNN Input Preprocessing)
            t0 = time.time()
            cropped = cv2.resize(full_image[-150:], (200, 66)) / 255.0
            pred_deg = self.steering_model.predict_angle(cropped)
            self.t_steer.add(time.time() - t0)

            # Determine active steering angle (Dataset Ground Truth vs AI Model Prediction)
            active_deg = actual_deg if self.show_actual else pred_deg

            # Exponential smoothing for fluid steering rotation and pedal stability
            self.smoothed_angle += 0.15 * (active_deg - self.smoothed_angle)

            # 2. Perception Overlay (Lane lines + Object boxes) — NON-BLOCKING.
            detector.submit(full_image)
            lane_results, object_results, lane_ms, object_ms = detector.get_latest()
            segmented_image = self.segmentation_model.draw_cached(full_image, lane_results, object_results)
            self.t_lane.add(lane_ms / 1000.0)
            self.t_object.add(object_ms / 1000.0)

            # 3. Dynamic Pedal Actuation Physics (driven by smoothed active angle)
            brake_pct, gas_pct = self.pedal_controller.update(self.smoothed_angle)

            # 4. Render Cockpit HUD & Road View
            t0 = time.time()
            self._update_display(
                degrees=active_deg,
                segmented_image=segmented_image,
                brake_pct=brake_pct,
                gas_pct=gas_pct,
                frame_idx=i,
                pred_deg=pred_deg,
                actual_deg=actual_deg,
            )
            self.t_render.add(time.time() - t0)

            i += 1
            self.t_frame.add(time.time() - frame_start)

            elapsed = time.time() - frame_start
            sleep_time = max(0.001, frame_interval - elapsed)
            key = cv2.waitKey(max(1, int(sleep_time * 1000))) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('m'), ord('M')):
                self.show_actual = not self.show_actual
                mode_str = "ACTUAL (Dataset Ground Truth)" if self.show_actual else "PREDICTED (PilotNet CNN AI)"
                print(f"\n[Steering Mode] Switched to: {mode_str}")
            elif key in (ord('p'), ord('P'), 32):
                is_paused = not is_paused
                print(f"\n[Playback] {'PAUSED' if is_paused else 'RESUMED'}")
                while is_paused:
                    p_key = cv2.waitKey(100) & 0xFF
                    if p_key in (ord('p'), ord('P'), 32):
                        is_paused = False
                        print("[Playback] RESUMED")
                        break
                    elif p_key in (ord('q'), 27):
                        is_paused = False
                        key = p_key
                        break
                if key in (ord('q'), 27):
                    break

            if self.verbose_profiling and i % 30 == 0:
                fps = 1000.0 / self.t_frame.avg_ms if self.t_frame.avg_ms > 0 else 0.0
                print(
                    f"\n[Profile] read={self.t_read.avg_ms:5.1f}ms  "
                    f"steer={self.t_steer.avg_ms:5.1f}ms  "
                    f"lane={self.t_lane.avg_ms:5.1f}ms  "
                    f"object={self.t_object.avg_ms:5.1f}ms  "
                    f"render={self.t_render.avg_ms:5.1f}ms  "
                    f"total={self.t_frame.avg_ms:5.1f}ms  "
                    f"fps={fps:4.1f}"
                )

        prefetcher.stop()
        detector.stop()
        cv2.destroyAllWindows()
        self.steering_model.close()
        print("\nAutoDrive FSD simulation completed.")

    def _update_display(
        self,
        degrees: float,
        segmented_image: np.ndarray,
        brake_pct: float,
        gas_pct: float,
        frame_idx: int,
        pred_deg: Optional[float] = None,
        actual_deg: Optional[float] = None,
    ):
        if pred_deg is None:
            pred_deg = degrees
        if actual_deg is None:
            actual_deg = degrees

        # Rotate steering wheel smoothly using smoothed angle
        # In OpenCV getRotationMatrix2D, negative angle rotates clockwise (right turn)
        M = cv2.getRotationMatrix2D((self.cols / 2.0, self.rows / 2.0), -self.smoothed_angle, 1.0)
        wheel = self.wheel_img if self.wheel_img is not None else np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        dst_wheel = cv2.warpAffine(wheel, M, (self.cols, self.rows), borderValue=(255, 255, 255))

        # Cockpit Actuation Panel below wheel (320w x 580h)
        cockpit_canvas = np.full((580, 320, 3), 18, dtype=np.uint8)
        cockpit_canvas[10:310, 10:310] = cv2.resize(dst_wheel, (300, 300))
        cv2.rectangle(cockpit_canvas, (9, 9), (311, 311), (50, 50, 58), 1)

        # Vehicle State & Status Bar
        # Vehicle State & Status Bar
        if brake_pct > 0.05:
            status_str = "BRAKING"
            status_color = (0, 60, 255)
        elif gas_pct > 0.05:
            status_str = "CRUISING"
            status_color = (0, 230, 120)
        else:
            status_str = "COASTING"
            status_color = (180, 180, 180)

        mode_tag = "ACTUAL" if self.show_actual else "AI"
        mode_color = (240, 240, 240) if self.show_actual else (0, 230, 255)

        cv2.rectangle(cockpit_canvas, (10, 318), (310, 368), (26, 26, 32), -1)
        cv2.rectangle(cockpit_canvas, (10, 318), (310, 368), (48, 48, 56), 1)
        cv2.putText(cockpit_canvas, f"STEER [{mode_tag}]: {degrees:+.1f} deg", (20, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
        cv2.putText(cockpit_canvas, f"STATUS   : {status_str} | AI: {pred_deg:+.1f} deg", (20, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

        # Render High-Tech Footwell Cockpit Pedal Bay (Brake & Accelerator)
        self.pedal_renderer.render(
            cockpit_canvas,
            box_rect=(10, 376, 300, 194),
            brake_pct=brake_pct,
            gas_pct=gas_pct,
        )

        # Road View Telemetry HUD
        display_road = cv2.resize(segmented_image, (640, 360))
        cv2.rectangle(display_road, (20, 20), (230, 52), (30, 30, 30), -1)
        cv2.putText(display_road, f"STEER [{mode_tag}]: {degrees:+.1f} deg", (26, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, mode_color, 1, cv2.LINE_AA)

        if brake_pct > 0.05:
            cv2.rectangle(display_road, (238, 20), (365, 52), (0, 0, 220), -1)
            cv2.putText(display_road, f"BRAKE {int(brake_pct*100)}%", (246, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        elif gas_pct > 0.05:
            cv2.rectangle(display_road, (238, 20), (365, 52), (0, 160, 60), -1)
            cv2.putText(display_road, f"GAS {int(gas_pct*100)}%", (256, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(display_road, (238, 20), (365, 52), (55, 55, 60), -1)
            cv2.putText(display_road, "COASTING", (248, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # Live measured FPS overlay
        fps = 1000.0 / self.t_frame.avg_ms if self.t_frame.avg_ms > 0 else 0.0
        cv2.putText(display_road, f"FPS: {fps:4.1f}", (500, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("AutoDrive FSD Perception", display_road)
        cv2.imshow("Steering & Actuation", cockpit_canvas)

        sys.stdout.write(f"\r[FSD Frame {frame_idx:4d}] Steer: {degrees:+5.1f} deg ({mode_tag}) | AI: {pred_deg:+5.1f} deg | Brake: {int(brake_pct*100):3d}% | Gas: {int(gas_pct*100):3d}% | FPS: {fps:4.1f}")
        sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run AutoDrive End-to-End FSD Perception and Control.")
    parser.add_argument("--data", type=str, default="data/driving_dataset", help="Path to driving dataset images.")
    parser.add_argument("--steering-model", type=str, default="saved_models/regression_model", help="Path to steering model checkpoint dir.")
    parser.add_argument("--lane-model", type=str, default="saved_models/lane_detection_model/lane_detection_best.pt", help="Path to lane model.")
    parser.add_argument("--object-model", type=str, default="saved_models/object_detection_model/object_detection_best.pt", help="Path to object model.")
    parser.add_argument("--wheel", type=str, default="data/sterring_wheel_image.png", help="Path to steering wheel image.")
    parser.add_argument("--brake-pedal", type=str, default="data/brake_pedal.png", help="Path to brake pedal image.")
    parser.add_argument("--gas-pedal", type=str, default="data/gas_pedal.png", help="Path to accelerator pedal image.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of frames to run (default: entire dataset).")
    parser.add_argument("--device", type=str, default=None, help="Force perception device, e.g. 'cuda:0' or 'cpu' (default: auto-detect).")
    parser.add_argument("--no-profile", action="store_true", help="Disable per-stage profiling printout.")
    parser.add_argument("--imgsz", type=int, default=480, help="Object-detection YOLO inference resolution (lower = faster, less accurate).")
    parser.add_argument("--lane-imgsz", type=int, default=None, help="Lane-segmentation YOLO inference resolution. Defaults to max(imgsz, 640) — lane lines need more resolution than object boxes to detect reliably. Raise this if lane fill is still missing/broken.")
    parser.add_argument("--detect-every", type=int, default=3, help="(Legacy, mostly unused now with async detection) Kept for compatibility.")
    parser.add_argument("--obj-conf", type=float, default=0.35, help="Confidence threshold for object detection (default: 0.35).")
    parser.add_argument("--lane-conf", type=float, default=0.25, help="Confidence threshold for lane segmentation (default: 0.25).")
    parser.add_argument("--openvino", action="store_true", help="Export/use OpenVINO IR models for faster CPU inference (requires: pip install openvino). Recommended if you have no GPU.")
    parser.add_argument("--use-ai-steering", action="store_true", help="Start simulation in AI steering mode instead of dataset ground truth.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print_device_diagnostics()

    steering_predictor = SteeringAnglePredictor(args.steering_model)
    image_segmentation = ImageSegmentation(
        lane_model_path=args.lane_model,
        object_model_path=args.object_model,
        device=args.device,
        imgsz=args.imgsz,
        lane_imgsz=args.lane_imgsz,
        detect_every_n=args.detect_every,
        use_openvino=args.openvino,
        obj_conf=args.obj_conf,
        lane_conf=args.lane_conf,
    )
    simulator = SelfDrivingCarSimulator(
        steering_predictor,
        image_segmentation,
        data_path=args.data,
        wheel_img_path=args.wheel,
        brake_pedal_path=args.brake_pedal,
        gas_pedal_path=args.gas_pedal,
        verbose_profiling=not args.no_profile,
        show_actual=not args.use_ai_steering,
    )
    simulator.start_simulation(limit=args.limit)