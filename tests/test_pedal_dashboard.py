"""Unit and Integration Tests for Dynamic Pedal Telemetry & HUD Renderer."""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.pedal_dashboard import PedalController, PedalRenderer


class TestPedalController(unittest.TestCase):
    """Tests the dynamic vehicle pedal physics controller."""

    def setUp(self):
        self.controller = PedalController(smoothing_alpha=1.0)  # Immediate response for unit testing

    def test_straight_line_cruising(self):
        """Straight path should apply high throttle and zero braking."""
        brake, gas = self.controller.update(steering_angle_deg=0.0)
        self.assertEqual(brake, 0.0)
        self.assertGreaterEqual(gas, 0.80)

    def test_gentle_curve(self):
        """Gentle curve should feather throttle and apply no braking."""
        brake, gas = self.controller.update(steering_angle_deg=8.0)
        self.assertEqual(brake, 0.0)
        self.assertLess(gas, 0.85)
        self.assertGreater(gas, 0.35)

    def test_moderate_curve(self):
        """Moderate curve should initiate progressive braking and cut throttle."""
        brake, gas = self.controller.update(steering_angle_deg=17.0)
        self.assertGreater(brake, 0.20)
        self.assertLess(gas, 0.30)

    def test_sharp_turn(self):
        """Sharp cornering should cut throttle completely and apply heavy braking."""
        brake, gas = self.controller.update(steering_angle_deg=30.0)
        self.assertGreaterEqual(brake, 0.70)
        self.assertEqual(gas, 0.0)

    def test_bounds_clamping(self):
        """Pedal percentages must always remain strictly within [0.0, 1.0]."""
        for test_angle in [-45.0, -25.0, -5.0, 0.0, 5.0, 25.0, 45.0]:
            brake, gas = self.controller.update(test_angle)
            self.assertGreaterEqual(brake, 0.0)
            self.assertLessEqual(brake, 1.0)
            self.assertGreaterEqual(gas, 0.0)
            self.assertLessEqual(gas, 1.0)


class TestPedalRenderer(unittest.TestCase):
    """Tests the footwell HUD pedal bay renderer."""

    def setUp(self):
        self.renderer = PedalRenderer(
            brake_asset_path="data/brake_pedal.png",
            gas_asset_path="data/gas_pedal.png"
        )

    def test_assets_loaded(self):
        """Brake and gas pedal RGBA sprites should be loaded successfully."""
        self.assertIsNotNone(self.renderer.brake_img)
        self.assertIsNotNone(self.renderer.gas_img)
        self.assertEqual(self.renderer.brake_img.shape[2], 4)
        self.assertEqual(self.renderer.gas_img.shape[2], 4)

    def test_render_on_canvas(self):
        """Rendering into a HUD canvas must complete cleanly and modify pixels."""
        canvas = np.zeros((600, 340, 3), dtype=np.uint8)
        box = (15, 415, 310, 172)
        
        # Initial canvas in box is empty
        self.assertEqual(np.sum(canvas[415:587, 15:325]), 0)

        # Render active turn state
        self.renderer.render(canvas, box, brake_pct=0.75, gas_pct=0.0)

        # Verify pixels were drawn
        self.assertGreater(np.sum(canvas[415:587, 15:325]), 0)

    def test_fallback_rendering(self):
        """Renderer should handle missing asset files gracefully with procedural fallback."""
        fallback_renderer = PedalRenderer(
            brake_asset_path="non_existent_brake.png",
            gas_asset_path="non_existent_gas.png"
        )
        canvas = np.zeros((600, 340, 3), dtype=np.uint8)
        # Should not raise exception
        fallback_renderer.render(canvas, (15, 415, 310, 172), brake_pct=0.5, gas_pct=0.5)
        self.assertGreater(np.sum(canvas[415:587, 15:325]), 0)


if __name__ == "__main__":
    unittest.main()
