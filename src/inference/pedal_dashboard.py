"""AutoDrive Dynamic Cockpit Pedal Telemetry & Renderer.

Provides real-time pedal physics simulation (Brake & Accelerator) driven by
steering angle severity and vehicle dynamics, along with high-tech HUD rendering
of realistic vehicle pedals with depression stroke, dynamic neon glow, and pressure meters.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np


class PedalController:
    """Simulates real-time brake and throttle pedal inputs from vehicle dynamics.

    Ensures physically grounded longitudinal vehicle dynamics:
    - Single-foot mutual exclusion: Brake and accelerator NEVER depress simultaneously.
    - Smooth progressive curve deceleration: Higher steering angles progressively apply brakes.
    - Natural lift-off throttle before brake engagement (coasting deadzone).
    - Responsive exponential smoothing for jitter-free actuator display.
    """

    # Highway speed range: min speed during sharp cornering, max speed on straights
    _MIN_SPEED_MPH: float = 18.0
    _MAX_SPEED_MPH: float = 62.0

    def __init__(self, smoothing_alpha: float = 0.25) -> None:
        """Initializes pedal controller with smoothing filter.

        Args:
            smoothing_alpha: Exponential moving average smoothing factor (0.0 to 1.0).
        """
        self.alpha = smoothing_alpha
        self.brake_pct = 0.0  # 0.0 (released) to 1.0 (fully depressed)
        self.gas_pct = 0.85   # Starts at cruising throttle
        self.speed_mph = 55.0

    def update(self, steering_angle_deg: float, speed_mph: Optional[float] = None) -> Tuple[float, float]:
        """Updates and returns smoothed (brake_pct, gas_pct) based on steering angle and speed.

        Uses angle-derived speed estimation for dataset replay (car is always moving on a
        highway — never stopped). If real GPS speed is available, pass it as speed_mph to
        override the estimate.

        Args:
            steering_angle_deg: Current steering angle in degrees.
            speed_mph: Real vehicle speed in mph (optional). When provided it overrides the
                       internal angle-derived estimate.

        Returns:
            Tuple of (brake_pct, gas_pct) each in range [0.0, 1.0], strictly mutually exclusive.
        """
        abs_deg = abs(steering_angle_deg)

        # --- Speed estimate ---
        # Derive speed from steering severity: straight road = fast, sharp turn = slow.
        # Uses a smooth sigmoid-style curve so there are no discontinuities.
        # This NEVER reaches 0, which prevents the integrator-deadlock seen with a
        # physics-based model on a highway dataset (car is always moving).
        if speed_mph is not None:
            self.speed_mph = float(np.clip(speed_mph, self._MIN_SPEED_MPH, self._MAX_SPEED_MPH))
        else:
            # Higher angle → slower; clamp between highway min/max
            speed_range = self._MAX_SPEED_MPH - self._MIN_SPEED_MPH
            angle_factor = np.clip(abs_deg / 30.0, 0.0, 1.0)  # 0 at straight, 1 at 30°+
            target_speed = self._MAX_SPEED_MPH - speed_range * angle_factor
            # Smooth speed transitions with a light EMA
            self.speed_mph += 0.12 * (target_speed - self.speed_mph)
            self.speed_mph = float(np.clip(self.speed_mph, self._MIN_SPEED_MPH, self._MAX_SPEED_MPH))

        # --- Pedal mapping based on road curvature severity ---
        # - Straight (|angle| < 4 deg):     High cruising throttle (85%), zero brake
        # - Gentle curve (4 - 12 deg):      Feathered throttle (85%→40%), zero brake
        # - Transition zone (12 - 14 deg):  Lift-off throttle (40%→0%), coasting
        # - Moderate curve (14 - 22 deg):   Progressive braking (22%→67%), zero throttle
        # - Sharp turn (> 22 deg):          Heavy cornering braking (70%→100%), zero throttle
        if True:  # always in normal cruise range (car always moving on highway)
            # - Straight (|angle| < 4 deg): High cruising throttle, zero brake
            # - Gentle curve (4 - 12 deg): Feathered throttle (85% -> 40%), zero brake
            # - Transition zone (12 - 14 deg): Lift-off throttle (40% -> 0%), coasting
            # - Moderate curve (14 - 22 deg): Progressive braking (22% -> 67%), zero throttle
            # - Sharp turn (> 22 deg): Heavy cornering braking (70% -> 100%), zero throttle
            if abs_deg < 4.0:
                target_gas = 0.85
                target_brake = 0.0
            elif abs_deg < 12.0:
                ratio = (abs_deg - 4.0) / 8.0
                target_gas = 0.85 - ratio * 0.45  # 0.85 -> 0.40
                target_brake = 0.0
            elif abs_deg < 14.0:
                ratio = (abs_deg - 12.0) / 2.0
                target_gas = max(0.0, 0.40 * (1.0 - ratio))
                target_brake = 0.0
            elif abs_deg < 22.0:
                target_gas = 0.0
                ratio = (abs_deg - 14.0) / 8.0
                target_brake = 0.22 + ratio * 0.45  # 0.22 -> 0.67
            else:
                target_gas = 0.0
                sharp_ratio = min(1.0, (abs_deg - 22.0) / 15.0)
                target_brake = min(1.0, 0.70 + sharp_ratio * 0.30)  # 0.70 -> 1.0

        # --- 4. Strict single-foot mutual exclusion ---
        # A driver's foot cannot press both pedals simultaneously.
        # Priority: if braking is commanded, immediately snap throttle to 0.
        if target_brake > 0.01:
            target_gas = 0.0
            # Instantly clear any lingering gas EMA
            self.gas_pct = max(0.0, self.gas_pct - self.alpha * 4 * self.gas_pct)

        # If throttle is commanded (and brake just released), instantly clear brake EMA
        if target_gas > 0.01 and target_brake <= 0.01:
            self.brake_pct = max(0.0, self.brake_pct - self.alpha * 4 * self.brake_pct)

        # --- 5. Exponential smoothing for realistic physical pedal stroke travel ---
        self.brake_pct += self.alpha * (target_brake - self.brake_pct)
        self.gas_pct += self.alpha * (target_gas - self.gas_pct)

        # --- 6. Dead-zone snap to clean zero ---
        if self.brake_pct < 0.01:
            self.brake_pct = 0.0
        if self.gas_pct < 0.01:
            self.gas_pct = 0.0

        # Safe bounds clamping
        self.brake_pct = float(np.clip(self.brake_pct, 0.0, 1.0))
        self.gas_pct = float(np.clip(self.gas_pct, 0.0, 1.0))

        return self.brake_pct, self.gas_pct


class PedalRenderer:
    """Renders authentic footwell pedal bay inside the steering dashboard HUD.

    Uses the authentic footwell_pedals.png asset with a true perspective-warp
    mechanical pedal depression:
    - Hinge is fixed at the TOP of each pedal (like a real suspended brake pedal
      or organ-style accelerator top pivot).
    - Bottom of the pedal swings DOWN and INWARD (foreshortens) under pressure,
      using cv2.warpPerspective for a proper 3D rotation illusion.
    - A cast shadow deepens in the recess behind the released pedal.
    - Metallic sheen brightens when the pedal snaps back to rest (foot off).
    """

    # Maximum downward swing in pixels for each pedal (at 100% depression)
    _BRAKE_MAX_TRAVEL: int = 22   # brake is heavier – bigger stroke
    _GAS_MAX_TRAVEL:   int = 18   # accelerator lighter stroke
    # Maximum inward keystone squeeze at bottom corners (perspective convergence)
    _KEYSTONE_MAX: float = 0.18   # fraction of sprite width

    def __init__(
        self,
        brake_asset_path: str = "data/brake_pedal.png",
        gas_asset_path:   str = "data/gas_pedal.png",
        footwell_asset_path: str = "data/footwell_pedals.png",
    ) -> None:
        """Loads footwell base plate and extracts articulated pedal sprites."""
        self.bw, self.bh = 300, 194
        self.use_footwell = False

        if os.path.exists(footwell_asset_path):
            f_img = cv2.imread(footwell_asset_path)
            if f_img is not None:
                if f_img.shape[0] > 700:
                    f_img = f_img[:690, :]
                base = cv2.resize(f_img, (self.bw, self.bh), interpolation=cv2.INTER_AREA)

                # Calibrated pedal-pad bounding boxes inside the footwell image.
                # Brake is the large left pad; Accelerator is the narrow right pad.
                self.bx1 = int(self.bw * 0.13)
                self.by1 = int(self.bh * 0.22)
                self.bx2 = int(self.bw * 0.56)
                self.by2 = int(self.bh * 0.72)

                self.gx1 = int(self.bw * 0.66)
                self.gy1 = int(self.bh * 0.18)
                self.gx2 = int(self.bw * 0.88)
                self.gy2 = int(self.bh * 0.70)

                # Extract resting pedal-face sprites
                self.brake_sprite = base[self.by1:self.by2, self.bx1:self.bx2].copy()
                self.gas_sprite   = base[self.gy1:self.gy2, self.gx1:self.gx2].copy()

                # Background plate: inpaint pedal slots so they can be re-drawn each frame
                inpaint_mask = np.zeros((self.bh, self.bw), dtype=np.uint8)
                cv2.rectangle(inpaint_mask, (self.bx1, self.by1), (self.bx2, self.by2), 255, -1)
                cv2.rectangle(inpaint_mask, (self.gx1, self.gy1), (self.gx2, self.gy2), 255, -1)
                self.bg_plate = cv2.inpaint(base, inpaint_mask, 7, cv2.INPAINT_TELEA)

                # Darken the recess slots — they sit in shadow when pedal is raised
                slot_dark = 0.52
                for (x1, y1, x2, y2) in [
                    (self.bx1, self.by1, self.bx2, self.by2),
                    (self.gx1, self.gy1, self.gx2, self.gy2),
                ]:
                    xe1 = max(0, x1 - 4);  ye1 = max(0, y1 - 4)
                    xe2 = min(self.bw, x2 + 4);  ye2 = min(self.bh, y2 + 4)
                    self.bg_plate[ye1:ye2, xe1:xe2] = (
                        self.bg_plate[ye1:ye2, xe1:xe2].astype(np.float32) * slot_dark
                    ).astype(np.uint8)

                self.use_footwell = True

        # Fallback standalone sprites
        self.brake_img = self._load_rgba(brake_asset_path, target_size=(108, 68))
        self.gas_img   = self._load_rgba(gas_asset_path,   target_size=(54, 82))

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rgba(path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _perspective_depress(
        sprite: np.ndarray,
        pct: float,
        max_travel: int,
        keystone_max: float,
    ) -> np.ndarray:
        """Apply a perspective warp simulating a hinge-pivot pedal depression.

        The pedal is hinged at the TOP edge:
          - Top two corners stay FIXED.
          - Bottom two corners move DOWN by ``travel`` px and converge INWARD
            by ``keystone`` px (the face foreshortens as it tilts away from camera).

        This produces a proper trapezoid that looks like a 3D rotating surface.
        """
        h, w = sprite.shape[:2]
        travel   = int(pct * max_travel)
        keystone = int(pct * keystone_max * w)

        src = np.float32([
            [0,              0    ],
            [w - 1,          0    ],
            [w - 1,          h - 1],
            [0,              h - 1],
        ])

        new_h = max(8, h + travel)
        dst = np.float32([
            [0,                  0        ],
            [w - 1,              0        ],
            [w - 1 - keystone,   new_h - 1],
            [keystone,           new_h - 1],
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            sprite, M, (w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _apply_sheen(sprite: np.ndarray, pct: float) -> np.ndarray:
        """Adjust metallic brightness based on pedal angle.

        Released pedal (pct=0) catches overhead cabin light -> brighter.
        Fully depressed (pct=1) tilts away from light -> slightly darker.
        """
        gain = 1.0 + (1.0 - pct) * 0.32 - pct * 0.10
        return np.clip(sprite.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    @staticmethod
    def _cast_shadow(
        panel: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        pct: float,
    ) -> None:
        """Draw a graduated shadow in the recess slot ABOVE the pedal face.

        Shadow deepest when pct=0 (pedal fully raised, recess exposed).
        Shrinks when pct=1 (pedal depressed, face fills the slot).
        """
        shadow_h = max(2, int((1.0 - pct) * 12))
        sy2 = min(panel.shape[0], y1 + shadow_h)
        if sy2 > y1 and x2 > x1:
            roi = panel[y1:sy2, x1:x2].astype(np.float32)
            rows = sy2 - y1
            grad = np.linspace(0.20, 0.78, rows, dtype=np.float32).reshape(rows, 1, 1)
            panel[y1:sy2, x1:x2] = np.clip(roi * grad, 0, 255).astype(np.uint8)

    def _blend_sprite(
        self,
        panel: np.ndarray,
        warped: np.ndarray,
        px: int, py: int,
    ) -> None:
        """Blend the perspective-warped sprite into the panel with soft edges."""
        h, w = warped.shape[:2]
        y2 = min(panel.shape[0], py + h)
        x2 = min(panel.shape[1], px + w)
        if y2 <= py or x2 <= px or px < 0 or py < 0:
            return
        clip_h = y2 - py
        clip_w = x2 - px

        mask = np.ones((clip_h, clip_w), dtype=np.float32)
        fade = max(2, min(8, clip_h // 6, clip_w // 6))
        for k in range(fade):
            a = (k + 1) / (fade + 1)
            if k < clip_h:
                mask[k, :]          = np.minimum(mask[k, :],          a)
                mask[clip_h-1-k, :] = np.minimum(mask[clip_h-1-k, :], a)
            if k < clip_w:
                mask[:, k]          = np.minimum(mask[:, k],          a)
                mask[:, clip_w-1-k] = np.minimum(mask[:, clip_w-1-k], a)

        mask = mask[:, :, np.newaxis]
        roi = panel[py:y2, px:x2].astype(np.float32)
        src = warped[:clip_h, :clip_w].astype(np.float32)
        panel[py:y2, px:x2] = np.clip(
            src * mask + roi * (1.0 - mask), 0, 255
        ).astype(np.uint8)

    def _draw_telemetry_bar(
        self,
        panel: np.ndarray,
        brake_pct: float,
        gas_pct: float,
    ) -> None:
        """Render OEM-style telemetry strip at the bottom of the footwell bay."""
        BAR_H  = 30
        bar_y1 = self.bh - BAR_H - 2
        gauge_w = 108
        gy_fill = bar_y1 + 18

        cv2.rectangle(panel, (6, bar_y1), (self.bw - 6, self.bh - 2), (16, 16, 22), -1)
        cv2.rectangle(panel, (6, bar_y1), (self.bw - 6, self.bh - 2), (40, 40, 50), 1)

        # Brake label + gradient bar
        b_col = (100, 120, 255) if brake_pct > 0.04 else (90, 90, 100)
        cv2.putText(panel, f"BRK {int(brake_pct * 100):3d}%",
                    (12, bar_y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, b_col, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (12, gy_fill), (12 + gauge_w, gy_fill + 5), (28, 28, 36), -1)
        for col_x in range(int(gauge_w * brake_pct)):
            t = col_x / max(1, gauge_w - 1)
            cv2.line(panel, (12 + col_x, gy_fill), (12 + col_x, gy_fill + 5),
                     (int(40 + t * 160), int(40 + t * 80), 255))
        cv2.rectangle(panel, (12, gy_fill), (12 + gauge_w, gy_fill + 5), (46, 46, 58), 1)

        # Accel label + gradient bar
        mid_x = self.bw // 2 + 10
        g_col = (40, 220, 80) if gas_pct > 0.04 else (90, 90, 100)
        cv2.putText(panel, f"ACC {int(gas_pct * 100):3d}%",
                    (mid_x, bar_y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, g_col, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (mid_x, gy_fill), (mid_x + gauge_w, gy_fill + 5), (28, 28, 36), -1)
        for col_x in range(int(gauge_w * gas_pct)):
            t = col_x / max(1, gauge_w - 1)
            cv2.line(panel, (mid_x + col_x, gy_fill), (mid_x + col_x, gy_fill + 5),
                     (0, int(140 + t * 115), int(30 + t * 60)))
        cv2.rectangle(panel, (mid_x, gy_fill), (mid_x + gauge_w, gy_fill + 5), (46, 46, 58), 1)

    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------

    def render(
        self,
        canvas: np.ndarray,
        box_rect: Tuple[int, int, int, int],
        brake_pct: float,
        gas_pct: float,
    ) -> None:
        """Renders the physically moving pedals onto canvas.

        Args:
            canvas:    Cockpit dashboard HUD canvas image.
            box_rect:  (x, y, w, h) bounding rectangle for the pedal footwell bay.
            brake_pct: Brake depression percentage  (0.0 released -> 1.0 full press).
            gas_pct:   Throttle depression percentage (0.0 released -> 1.0 full press).
        """
        bx, by, bw, bh = box_rect
        ch, cw = canvas.shape[:2]
        if bx < 0 or by < 0 or bx + bw > cw or by + bh > ch:
            return

        # ------------------------------------------------------------------
        # PRIMARY PATH: footwell_pedals.png + perspective warp
        # ------------------------------------------------------------------
        if self.use_footwell:
            panel = self.bg_plate.copy()

            # Brake pedal
            b_warped = self._perspective_depress(
                self._apply_sheen(self.brake_sprite, brake_pct),
                brake_pct, self._BRAKE_MAX_TRAVEL, self._KEYSTONE_MAX
            )
            self._cast_shadow(panel, self.bx1, self.by1, self.bx2, self.by2, brake_pct)
            self._blend_sprite(panel, b_warped, self.bx1, self.by1)

            # Accelerator pedal
            g_warped = self._perspective_depress(
                self._apply_sheen(self.gas_sprite, gas_pct),
                gas_pct, self._GAS_MAX_TRAVEL, self._KEYSTONE_MAX
            )
            self._cast_shadow(panel, self.gx1, self.gy1, self.gx2, self.gy2, gas_pct)
            self._blend_sprite(panel, g_warped, self.gx1, self.gy1)

            self._draw_telemetry_bar(panel, brake_pct, gas_pct)

            if (bw, bh) != (self.bw, self.bh):
                panel = cv2.resize(panel, (bw, bh), interpolation=cv2.INTER_AREA)

            canvas[by:by + bh, bx:bx + bw] = panel
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (40, 40, 50), 1)
            return

        # ------------------------------------------------------------------
        # FALLBACK: dark slot pockets + standalone sprite files
        # ------------------------------------------------------------------
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (16, 16, 20), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (40, 40, 50), 1)

        pocket_y1 = by + 10
        pocket_y2 = by + bh - 30
        for px1, px2 in [(bx + 8, bx + 148), (bx + 154, bx + bw - 8)]:
            cv2.rectangle(canvas, (px1, pocket_y1), (px2, pocket_y2), (22, 22, 28), -1)
            cv2.rectangle(canvas, (px1, pocket_y1), (px2, pocket_y2), (36, 36, 44), 1)

        for sprite_img, pct, max_t, slot_x, slot_w in [
            (self.brake_img, brake_pct, self._BRAKE_MAX_TRAVEL, bx + 8,   140),
            (self.gas_img,   gas_pct,   self._GAS_MAX_TRAVEL,   bx + 154, bw - 162),
        ]:
            if sprite_img is None:
                continue
            warped = self._perspective_depress(
                sprite_img[:, :, :3], pct, max_t, self._KEYSTONE_MAX
            )
            sw, sh = warped.shape[1], warped.shape[0]
            px_pos = slot_x + (slot_w - sw) // 2
            py_pos = pocket_y1 + 6
            alpha_ch = sprite_img[:sh, :sw, 3:4].astype(np.float32) / 255.0
            if (py_pos + sh <= canvas.shape[0] and
                    px_pos + sw <= canvas.shape[1] and px_pos >= 0):
                roi = canvas[py_pos:py_pos + sh, px_pos:px_pos + sw].astype(np.float32)
                canvas[py_pos:py_pos + sh, px_pos:px_pos + sw] = np.clip(
                    warped.astype(np.float32) * alpha_ch + roi * (1.0 - alpha_ch),
                    0, 255
                ).astype(np.uint8)

        bar_y1 = by + bh - 24
        cv2.putText(canvas, f"BRK {int(brake_pct*100):3d}%",
                    (bx + 12, bar_y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (100, 120, 255) if brake_pct > 0.04 else (90, 90, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ACC {int(gas_pct*100):3d}%",
                    (bx + 162, bar_y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (40, 220, 80) if gas_pct > 0.04 else (90, 90, 100), 1, cv2.LINE_AA)

