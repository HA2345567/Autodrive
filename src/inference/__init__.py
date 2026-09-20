"""AutoDrive Inference Package.

Exposes inference detectors, predictors, and runners with PEP 562 lazy loading
to prevent heavy PyTorch / TensorFlow initialization penalties on module import.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lane_detector import LaneDetector, AsyncLaneDetector
    from .object_detector import ObjectDetector, AsyncObjectDetector
    from .run_steering_angle_prediction import SteeringAnglePredictor

__all__ = [
    "LaneDetector",
    "AsyncLaneDetector",
    "ObjectDetector",
    "AsyncObjectDetector",
    "SteeringAnglePredictor",
]


def __getattr__(name: str):
    if name in ("LaneDetector", "AsyncLaneDetector"):
        from . import lane_detector
        return getattr(lane_detector, name)
    elif name in ("ObjectDetector", "AsyncObjectDetector"):
        from . import object_detector
        return getattr(object_detector, name)
    elif name == "SteeringAnglePredictor":
        from . import run_steering_angle_prediction
        return getattr(run_steering_angle_prediction, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
