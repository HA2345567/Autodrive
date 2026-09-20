"""Integration verification script testing Steering Prediction and Lane Detection."""

import os
import sys
import unittest
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.lane_detector import LaneDetector, AsyncLaneDetector
from src.run import rotate_image, resolve_lane_model


class TestIntegratedPipeline(unittest.TestCase):
    """Verifies that Steering Prediction and Lane Detection integrate seamlessly."""

    @classmethod
    def setUpClass(cls):
        cls.lane_model_path = resolve_lane_model()
        cls.wheel_img_path = "data/sterring_wheel_image.png"

        if not os.path.exists(cls.wheel_img_path):
            raise unittest.SkipTest(f"Wheel image '{cls.wheel_img_path}' not found.")

        raw_wheel = cv2.imread(cls.wheel_img_path)
        if raw_wheel is None:
            raise unittest.SkipTest(f"Failed to read wheel image '{cls.wheel_img_path}'.")
        cls.wheel_img: np.ndarray = cv2.resize(raw_wheel, (300, 300))

        cls.lane_detector = AsyncLaneDetector(model_name=cls.lane_model_path, conf_threshold=0.20, imgsz=320, device="cpu")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "lane_detector") and cls.lane_detector is not None:
            cls.lane_detector.stop()

    def test_rotate_steering_wheel(self):
        """Wheel rotation should return valid 300x300 image."""
        assert self.wheel_img is not None
        rotated = rotate_image(self.wheel_img, 25.0)
        self.assertEqual(rotated.shape, (300, 300, 3))

    def test_pipeline_on_dataset_images(self):
        """Processes dataset images with lane detector and wheel rotation."""
        dataset_dir = "data/driving_dataset"
        if not os.path.exists(dataset_dir):
            raise unittest.SkipTest(f"Dataset directory '{dataset_dir}' not found.")

        test_images = ["0.jpg", "100.jpg", "500.jpg"]
        for img_name in test_images:
            img_path = os.path.join(dataset_dir, img_name)
            if not os.path.exists(img_path):
                continue
            frame = cv2.imread(img_path)
            self.assertIsNotNone(frame, f"Failed to load {img_path}")
            assert frame is not None

            road_display = cv2.resize(frame, (640, 360))

            # 1. Lane detection
            detected_display, lane_info = self.lane_detector.detect(road_display, fill_corridor=True)
            self.assertIsNotNone(detected_display)
            assert detected_display is not None
            self.assertEqual(detected_display.shape, (360, 640, 3))
            self.assertIn("num_lanes", lane_info)

            # 2. Steering wheel rotation
            assert self.wheel_img is not None
            rotated = rotate_image(self.wheel_img, -10.5)
            self.assertEqual(rotated.shape, (300, 300, 3))


if __name__ == "__main__":
    unittest.main()
