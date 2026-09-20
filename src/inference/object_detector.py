"""AutoDrive YOLOv11 Object Detection Module.

Provides both synchronous (ObjectDetector) and threaded asynchronous (AsyncObjectDetector)
inference engines for real-time 2D bounding-box detection (cars, pedestrians, traffic signs, etc.).
"""

import colorsys
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def resolve_object_model_path(model_path: Optional[str] = None) -> str:
    """Resolves object detection model path checking standard locations."""
    if model_path and os.path.exists(model_path):
        return os.path.abspath(model_path)

    candidates = [
        "saved_models/object_detection_model/object_detection_best.pt",
        "saved_models/object_detection_best.pt",
        os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "object_detection_model", "object_detection_best.pt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "object_detection_best.pt"),
    ]
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if os.path.exists(abs_path):
            return abs_path

    # Fallback to yolo11n.pt if custom weights not found
    return "yolo11n.pt"


class ObjectDetector:
    """Synchronous YOLOv11 object detection engine."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 480,
        device: Optional[str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        resolved_path = resolve_object_model_path(model_name)
        if os.path.basename(resolved_path) == "yolo11n.pt" and not os.path.isabs(resolved_path):
            print(
                "[ObjectDetector] Warning: custom object detection weights not found "
                "under 'saved_models/object_detection_model/'. Falling back to 'yolo11n.pt'."
            )

        from ultralytics import YOLO
        self.model = YOLO(resolved_path)
        self.class_names = self.model.names or {}
        self.colors = self._generate_colors(len(self.class_names))

    @staticmethod
    def _generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
        colors: List[Tuple[int, int, int]] = []
        for i in range(max(1, num_colors)):
            hue = i / max(1, num_colors)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def detect(
        self,
        frame: Optional[np.ndarray],
        imgsz: Optional[int] = None,
        conf_threshold: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Detects objects on an input BGR frame and returns annotated image and metadata."""
        if frame is None or frame.size == 0:
            return None, {"num_objects": 0, "objects": []}

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

        detected_objects: List[Dict[str, Any]] = []

        if result is not None and getattr(result, "boxes", None) is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])
                coords = [int(v) for v in box.xyxy[0].tolist()]
                detected_objects.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": confidence,
                    "box": coords,  # [x1, y1, x2, y2]
                })

        annotated = self.draw_objects(frame, detected_objects)
        info = {
            "num_objects": len(detected_objects),
            "objects": detected_objects,
        }
        return annotated, info

    def draw_objects(
        self,
        frame: np.ndarray,
        objects: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draws bounding boxes and labels onto the given frame."""
        if not objects:
            return frame

        canvas = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            cls_id = obj["class_id"]
            cls_name = obj["class_name"]
            conf = obj["confidence"]
            color = self.colors[cls_id % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Draw label tag
            label = f"{cls_name}: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, y1 - lh - 6)
            cv2.rectangle(canvas, (x1, tag_y1), (x1 + lw + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas


class AsyncObjectDetector:
    """Asynchronous threaded object detector ensuring non-blocking 30+ FPS HUD display."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 480,
        device: Optional[str] = None,
    ) -> None:
        self.detector = ObjectDetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
        )
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._input_frame: Optional[np.ndarray] = None

        self._latest_objects: List[Dict[str, Any]] = []
        self._latest_info: Dict[str, Any] = {"num_objects": 0, "objects": []}

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()

            with self._lock:
                frame = self._input_frame

            if frame is None:
                continue

            try:
                _, info = self.detector.detect(frame)
                with self._lock:
                    self._latest_objects = info.get("objects", [])
                    self._latest_info = info
            except Exception:
                pass

    def detect(
        self, frame: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Submits frame for processing and returns latest rendered overlay instantly."""
        if frame is None or frame.size == 0:
            return None, {"num_objects": 0, "objects": []}

        with self._lock:
            self._input_frame = frame.copy()
            objects = list(self._latest_objects)
            info = dict(self._latest_info)

        self._event.set()

        if objects:
            rendered = self.detector.draw_objects(frame, objects)
            return rendered, info
        return frame.copy(), info

    def stop(self) -> None:
        """Stops background inference worker thread cleanly."""
        self._stop_event.set()
        self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
