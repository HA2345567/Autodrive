"""Tests for Pedal Dashboard integration into run_fsd_inference."""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.run_fsd_inference import (
    resolve_pedal_paths,
    SelfDrivingCarSimulator,
)


class TestFsdPedalIntegration(unittest.TestCase):
    """Verifies that Pedal Dashboard components are correctly integrated into FSD inference."""

    def test_resolve_pedal_paths(self):
        """Brake and gas pedal images should be found successfully."""
        brake_path, gas_path = resolve_pedal_paths()
        self.assertTrue(os.path.exists(brake_path), f"Brake pedal not found at {brake_path}")
        self.assertTrue(os.path.exists(gas_path), f"Gas pedal not found at {gas_path}")

    def test_simulator_pedal_renderer_initialization(self):
        """Simulator should initialize PedalRenderer with loaded RGBA pedal sprites."""
        mock_steering = MagicMock()
        mock_seg = MagicMock()

        sim = SelfDrivingCarSimulator(
            steering_model=mock_steering,
            segmentation_model=mock_seg,
            wheel_img_path="data/sterring_wheel_image.png",
            brake_pedal_path="data/brake_pedal.png",
            gas_pedal_path="data/gas_pedal.png",
        )

        self.assertIsNotNone(sim.pedal_renderer)
        self.assertIsNotNone(sim.pedal_renderer.brake_img)
        self.assertIsNotNone(sim.pedal_renderer.gas_img)
        self.assertEqual(sim.pedal_renderer.brake_img.shape[2], 4)
        self.assertEqual(sim.pedal_renderer.gas_img.shape[2], 4)

    def test_update_display_renders_pedals_and_badges(self):
        """_update_display should render without error and update displays."""
        import cv2

        mock_steering = MagicMock()
        mock_seg = MagicMock()

        sim = SelfDrivingCarSimulator(
            steering_model=mock_steering,
            segmentation_model=mock_seg,
        )

        test_road = np.zeros((360, 640, 3), dtype=np.uint8)

        # Mock cv2.imshow to prevent popup windows during test
        original_imshow = cv2.imshow
        shown_windows = {}

        def mock_imshow(winname, mat):
            shown_windows[winname] = mat.copy()

        # pyrefly: ignore [bad-assignment]
        cv2.imshow = mock_imshow
        try:
            # Test 1: Cruising state (high gas, low brake)
            sim._update_display(
                degrees=2.0,
                segmented_image=test_road,
                brake_pct=0.0,
                gas_pct=0.88,
                frame_idx=0,
            )
            self.assertIn("Steering & Actuation", shown_windows)
            self.assertIn("AutoDrive FSD Perception", shown_windows)
            cockpit = shown_windows["Steering & Actuation"]
            self.assertEqual(cockpit.shape, (580, 320, 3))

            # Test 2: Braking state (high brake, zero gas)
            sim._update_display(
                degrees=25.0,
                segmented_image=test_road,
                brake_pct=0.85,
                gas_pct=0.0,
                frame_idx=1,
            )
            cockpit_braking = shown_windows["Steering & Actuation"]
            self.assertEqual(cockpit_braking.shape, (580, 320, 3))
            # Pixels in pedal bay should be non-zero
            self.assertGreater(np.sum(cockpit_braking[376:570, 10:310]), 0)

        finally:
            cv2.imshow = original_imshow


if __name__ == "__main__":
    unittest.main()
