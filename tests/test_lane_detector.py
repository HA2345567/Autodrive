"""Unit tests for YOLOv11 Lane Detector and AsyncLaneDetector."""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.lane_detector import LaneDetector, AsyncLaneDetector


class TestLaneDetector(unittest.TestCase):
    """Tests for LaneDetector class."""

    @classmethod
    def setUpClass(cls):
        cls.detector = LaneDetector(conf_threshold=0.20, imgsz=320, device="cpu")

    def test_detector_initialization(self):
        """Detector should load model without errors."""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.conf_threshold, 0.20)

    def test_detect_dummy_frame(self):
        """Detector should handle arbitrary BGR frames gracefully."""
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        rendered, lane_info = self.detector.detect(dummy_frame)
        self.assertIsInstance(rendered, np.ndarray)
        self.assertEqual(rendered.shape, (360, 640, 3))
        self.assertIn("num_lanes", lane_info)
        self.assertIn("lanes", lane_info)

    def test_detect_empty_frame(self):
        """Detector should handle None or empty frames safely."""
        rendered, lane_info = self.detector.detect(None)
        self.assertIsNone(rendered)
        self.assertEqual(lane_info["num_lanes"], 0)


class TestAsyncLaneDetector(unittest.TestCase):
    """Tests for AsyncLaneDetector class."""

    @classmethod
    def setUpClass(cls):
        cls.async_detector = AsyncLaneDetector(conf_threshold=0.20, imgsz=320, device="cpu")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "async_detector") and cls.async_detector is not None:
            cls.async_detector.stop()

    def test_async_detect(self):
        """AsyncLaneDetector should return rendered frame and lane info non-blocking."""
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        rendered, lane_info = self.async_detector.detect(dummy_frame)
        self.assertIsInstance(rendered, np.ndarray)
        self.assertEqual(rendered.shape, (360, 640, 3))
        self.assertIn("num_lanes", lane_info)


if __name__ == "__main__":
    unittest.main()
