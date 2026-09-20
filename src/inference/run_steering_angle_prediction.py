"""AutoDrive Steering Angle Prediction CLI Runner.

Executes steering angle inference using the NVIDIA PilotNet / DAVE-2 9-layer CNN
trained on the Sully Chen driving dataset under TensorFlow 1.x (tf.compat.v1).
Supports both single-frame prediction and batch dataset evaluation with real-time HUD telemetry.
"""

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

# Suppress TensorFlow logging warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# pyrefly: ignore [missing-import]
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from src.models import model


class SteeringAnglePredictor:
    """Predicts the steering angle of an autonomous vehicle from road images."""

    def __init__(self, checkpoint_dir: str = "saved_models/regression_model") -> None:
        """Initializes the TensorFlow 1.x session and restores model weights.

        Args:
            checkpoint_dir: Directory containing the PilotNet model checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint cannot be located.
        """
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            for fallback in [
                "saved_models/regression_model",
                "saved_models/steering_angle",
                os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "regression_model"),
                os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", "steering_angle"),
            ]:
                if os.path.exists(fallback):
                    self.checkpoint_dir = fallback
                    break

        self.session = tf.InteractiveSession()
        self.saver = tf.compat.v1.train.Saver()

        checkpoint = tf.compat.v1.train.latest_checkpoint(self.checkpoint_dir)
        if checkpoint is None:
            checkpoint = os.path.abspath(os.path.join(self.checkpoint_dir, "model.ckpt"))
        else:
            checkpoint = os.path.abspath(checkpoint)

        if not os.path.exists(f"{checkpoint}.meta") and not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Steering angle checkpoint not found at '{checkpoint}'. "
                "Ensure training has completed or checkpoint files are placed in saved_models/regression_model/."
            )

        print(f"[SteeringAnglePredictor] Restoring checkpoint: {checkpoint}")
        self.saver.restore(self.session, checkpoint)

    def predict(self, frame: np.ndarray) -> float:
        """Predicts the steering angle in degrees for a given camera frame.

        Args:
            frame: BGR image frame from the vehicle camera.

        Returns:
            The predicted steering angle in degrees.
        """
        if frame is None or frame.size == 0:
            return 0.0

        # Preprocessing: Crop bottom 150 pixels, resize to 200x66, normalize to [0, 1]
        cropped = frame[-150:]
        resized = cv2.resize(cropped, (200, 66))
        normalized = resized / 255.0

        # Run prediction through TensorFlow graph
        predicted_rad = model.y_pred.eval(
            session=self.session,
            feed_dict={model.x: [normalized], model.keep_prob: 1.0},
        )[0][0]

        # Convert radians to degrees
        return float(predicted_rad * 180.0 / np.pi)

    def close(self) -> None:
        """Closes the TensorFlow session and releases resources."""
        if self.session:
            self.session.close()


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the steering prediction runner."""
    parser = argparse.ArgumentParser(
        description="Run AutoDrive Steering Angle Prediction (NVIDIA PilotNet / DAVE-2)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single road image to predict steering angle.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/driving_dataset",
        help="Path to dataset directory containing data.txt and image frames (default: 'data/driving_dataset').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/regression_model",
        help="Path to checkpoint directory (default: 'saved_models/regression_model').",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display visualization window with predicted angle overlay (default: on).",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive display and output predictions to console only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of frames to process in batch/dataset mode (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the steering prediction CLI runner."""
    args = parse_args()

    predictor = SteeringAnglePredictor(checkpoint_dir=args.checkpoint)

    try:
        # Single image mode
        if args.image is not None:
            if not os.path.exists(args.image):
                print(f"[Error] Image path '{args.image}' does not exist.")
                sys.exit(1)
            img = cv2.imread(args.image)
            if img is None:
                print(f"[Error] Failed to read image from '{args.image}'.")
                sys.exit(1)

            angle = predictor.predict(img)
            print("=" * 60)
            print(f"  Input Image              : {args.image}")
            print(f"  Predicted Steering Angle : {angle:+.2f} deg")
            print("=" * 60)

            if args.show:
                display = img.copy()
                cv2.putText(
                    display,
                    f"Pred: {angle:+.1f} deg",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Steering Prediction", display)
                print("Press any key to close the window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return

        # Dataset mode
        data_txt = os.path.join(args.source, "data.txt")
        if not os.path.exists(data_txt):
            print(f"[Error] Dataset file not found at '{data_txt}'. Specify a valid --source or --image.")
            sys.exit(1)

        frames: List[str] = []
        actuals: List[float] = []
        with open(data_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    frames.append(os.path.join(args.source, parts[0]))
                    actuals.append(float(parts[1]))

        total = min(len(frames), args.limit)
        print(f"[Steering] Running inference on {total} frames from '{args.source}'...")

        for idx in range(total):
            frame = cv2.imread(frames[idx])
            if frame is None:
                continue
            pred_deg = predictor.predict(frame)
            act_deg = actuals[idx]
            diff = abs(pred_deg - act_deg)

            print(
                f"Frame {idx:4d} | Actual: {act_deg:+6.1f} deg | Predicted: {pred_deg:+6.1f} deg | Error: {diff:5.1f} deg"
            )

            if args.show:
                display = cv2.resize(frame, (640, 360))
                cv2.putText(
                    display,
                    f"AI Pred: {pred_deg:+.1f} deg  |  Actual: {act_deg:+.1f} deg",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("AutoDrive Steering Prediction", display)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break

        if args.show:
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n[Steering] Prediction stopped by user.")
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
