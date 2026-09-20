"""AutoDrive YOLOv11 Lane Detection & Drivable Corridor Module.

Provides both synchronous (LaneDetector) and threaded asynchronous (AsyncLaneDetector)
inference engines for real-time lane line segmentation and ego-path corridor filling.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def resolve_lane_model_path(model_path: Optional[str] = None) -> str:
    """Resolves lane segmentation model path checking standard locations."""
    if model_path and os.path.exists(model_path):
        return os.path.abspath(model_path)

    candidates = [
        "saved_models/lane_detection_model/lane_detection_best.pt",
        "saved_models/lane_detection_best.pt",
        os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "lane_detection_model", "lane_detection_best.pt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "lane_detection_best.pt"),
    ]
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if os.path.exists(abs_path):
            return abs_path

    # Fallback to yolo11s-seg.pt if custom weights not found. This is a
    # generic COCO-pretrained segmentation model, NOT a lane detector --
    # callers should warn the user rather than let this pass silently.
    return "yolo11s-seg.pt"


class LaneDetector:
    """Synchronous YOLOv11 lane segmentation and drivable corridor engine."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.20,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        resolved_path = resolve_lane_model_path(model_name)

        # BUG FIX: previously this fallback happened silently, so a missing
        # weights file quietly degraded to a generic object-segmentation model
        # with no lane-specific training, producing meaningless "lane" masks
        # with zero indication anything was wrong.
        if os.path.basename(resolved_path) == "yolo11s-seg.pt" and not os.path.isabs(resolved_path):
            print(
                "[LaneDetector] Warning: no trained lane-detection weights found "
                "under 'saved_models/'. Falling back to generic 'yolo11s-seg.pt' "
                "(COCO-pretrained) -- lane segmentation output will likely be "
                "incorrect. Pass a valid model_name/--model to use your trained weights."
            )

        from ultralytics import YOLO
        self.model = YOLO(resolved_path)

    def detect(
        self,
        frame: Optional[np.ndarray],
        imgsz: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        fill_corridor: bool = False,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Detects lane lines on an input BGR frame and optionally fills drivable corridor."""
        if frame is None or frame.size == 0:
            return None, {"num_lanes": 0, "lanes": []}

        h, w = frame.shape[:2]
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        sz = imgsz if imgsz is not None else self.imgsz

        predict_kwargs: Dict[str, Any] = {
            "conf": conf,
            "imgsz": sz,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results: Any = self.model.predict(frame, **predict_kwargs)
        result: Any = results[0] if isinstance(results, (list, tuple)) else next(iter(results))

        lanes: List[np.ndarray] = []

        if result is not None and getattr(result, "masks", None) is not None:
            masks_xy = getattr(result.masks, "xy", [])
            for mask_xy in masks_xy:
                if len(mask_xy) < 3:
                    continue
                # Filter out sky, ceiling, or upper-tree false positives:
                # Genuine road lane markings always extend into the lower half of the road view
                max_y = float(np.max(mask_xy[:, 1]))
                if max_y < h * 0.42:
                    continue
                pts = np.asarray(mask_xy, dtype=np.int32).reshape((-1, 1, 2))
                lanes.append(pts)

        lane_info = {
            "num_lanes": len(lanes),
            "lanes": lanes,
        }
        rendered = self.draw_lanes(frame, lanes, fill_corridor=fill_corridor) if lanes else frame.copy()
        return rendered, lane_info

    def draw_lanes(
        self,
        frame: np.ndarray,
        lanes: List[np.ndarray],
        fill_corridor: bool = False,
        alpha: float = 0.50,
    ) -> np.ndarray:
        """Renders high-fidelity lane markings and optional ego drivable corridor."""
        if frame is None or frame.size == 0 or not lanes:
            return frame.copy() if frame is not None else frame

        h, w = frame.shape[:2]
        rendered = frame.copy()
        overlay = frame.copy()

        # 1. Accurately highlight each detected road lane line
        for pts in lanes:
            contour = pts if pts.ndim == 3 else pts.reshape((-1, 1, 2))
            # Fill the lane stripe with luminous electric cyan/blue
            cv2.fillPoly(overlay, [contour], color=(255, 190, 0))
            # Crisp, high-contrast anti-aliased edge glow
            cv2.polylines(overlay, [contour], isClosed=True, color=(255, 245, 80), thickness=2, lineType=cv2.LINE_AA)

        # 2. Ego Drivable Corridor: only formed if two distinct lanes flank the vehicle
        if fill_corridor and len(lanes) >= 2:
            try:
                ego_center = w / 2.0
                left_candidates = []
                right_candidates = []

                for pts in lanes:
                    contour = pts.reshape((-1, 2))
                    mean_x = float(np.mean(contour[:, 0]))
                    max_y = float(np.max(contour[:, 1]))
                    # Only consider lane lines that reach into the near-vehicle road zone
                    if max_y > h * 0.52:
                        if mean_x < ego_center + (w * 0.08):
                            left_candidates.append((mean_x, contour))
                        if mean_x > ego_center - (w * 0.08):
                            right_candidates.append((mean_x, contour))

                if left_candidates and right_candidates:
                    # Select the innermost left lane and innermost right lane
                    left_lane = max(left_candidates, key=lambda c: c[0])[1]
                    right_lane = min(right_candidates, key=lambda c: c[0])[1]

                    # Verify distinct lanes with realistic lateral vehicle width
                    if np.mean(right_lane[:, 0]) - np.mean(left_lane[:, 0]) > w * 0.18:
                        left_sorted = left_lane[np.argsort(left_lane[:, 1])]
                        right_sorted = right_lane[np.argsort(-right_lane[:, 1])]

                        corridor_pts = np.vstack([left_sorted, right_sorted])
                        corridor_overlay = overlay.copy()
                        cv2.fillPoly(corridor_overlay, [corridor_pts.astype(np.int32)], color=(0, 210, 80))
                        overlay = cv2.addWeighted(corridor_overlay, 0.40, overlay, 0.60, 0)
            except Exception:
                pass

        return cv2.addWeighted(overlay, alpha, rendered, 1.0 - alpha, 0)

    def predict_single(self, frame: np.ndarray, conf: Optional[float] = None) -> Any:
        """Executes YOLO segmentation for single frame and returns raw result object."""
        c = conf if conf is not None else self.conf_threshold
        kwargs: Dict[str, Any] = {"conf": c, "imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results: Any = self.model.predict(frame, **kwargs)
        return results[0] if isinstance(results, (list, tuple)) else next(iter(results))

    def draw_overlay(self, frame: np.ndarray, result: Any, alpha: float = 0.5) -> np.ndarray:
        """Overlays segmentation masks from a result object onto the frame."""
        if result is None or getattr(result, "masks", None) is None:
            return frame.copy()

        masks_xy = getattr(result.masks, "xy", [])
        if len(masks_xy) == 0:
            return frame.copy()

        overlay = frame.copy()
        for mask_xy in masks_xy:
            if len(mask_xy) < 3:
                continue
            pts = np.asarray(mask_xy, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=(0, 220, 255))
            cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


class AsyncLaneDetector:
    """Asynchronous threaded lane detector ensuring non-blocking 30+ FPS HUD display."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.20,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.detector = LaneDetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
        )
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._input_frame: Optional[np.ndarray] = None
        self._fill_corridor: bool = True

        self._latest_lanes: List[np.ndarray] = []
        self._latest_info: Dict[str, Any] = {"num_lanes": 0, "lanes": []}

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()

            with self._lock:
                frame = self._input_frame
                fill = self._fill_corridor

            if frame is None:
                continue

            try:
                _, info = self.detector.detect(frame, fill_corridor=fill)
                with self._lock:
                    self._latest_lanes = info.get("lanes", [])
                    self._latest_info = info
            except Exception:
                pass

    def detect(
        self, frame: Optional[np.ndarray], fill_corridor: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Submits frame for processing and returns latest rendered overlay instantly."""
        if frame is None or frame.size == 0:
            return None, {"num_lanes": 0, "lanes": []}

        with self._lock:
            self._input_frame = frame.copy()
            self._fill_corridor = fill_corridor
            lanes = list(self._latest_lanes)
            info = dict(self._latest_info)

        self._event.set()

        if lanes:
            rendered = self.detector.draw_lanes(frame, lanes, fill_corridor=fill_corridor)
            return rendered, info
        return frame.copy(), info

    def stop(self) -> None:
        """Stops background inference worker thread cleanly."""
        self._stop_event.set()
        self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)