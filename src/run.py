"""AutoDrive entry point and compatibility helpers.

Run `python -m src.run` for clean-frame perception and AI steering playback.
Heavy runtimes are imported only when the demo starts.
"""
import os
for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ.setdefault(_thread_setting, '1')
from typing import Optional
import cv2
import numpy as np
from src.inference.lane_detector import resolve_lane_model_path
from src.inference.object_detector import resolve_object_model_path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
            os.path.exists(os.path.join(candidate, 'model.ckpt.index'))
            or os.path.exists(os.path.join(candidate, 'model.ckpt.meta'))
        ):
            return candidate
    return os.path.join(PROJECT_ROOT, "saved_models", "regression_model")


def main():
    from src.inference.run_realtime_demo import main as run_demo
    run_demo()


if __name__ == "__main__":
    main()
