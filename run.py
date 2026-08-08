import os
import sys
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

# ==============================================================================
# Configuration Toggle
# Set to True to rotate the wheel by the ACTUAL steering angle from the dataset.
# Set to False to rotate by the model's PREDICTED steering angle (AI).
# ==============================================================================
SHOW_ACTUAL = True

# Add path to import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model_traning', 'train_steering_angle')))
import model

# Disable TensorFlow v2 behavior
tf.disable_v2_behavior()

def rotate_image(image, angle):
    """Rotate image by angle (in degrees) around its center, filling borders with white."""
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    # Fill background borders with white (255, 255, 255) so the steering wheel image displays exactly as is
    return cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))

def main():
    dataset_txt = "data/driving_dataset/data.txt"
    dataset_dir = "data/driving_dataset/"
    wheel_image_path = "data/sterring_wheel_image.png"
    checkpoint_dir = "model_traning/train_steering_angle/save"

    # Verify files exist
    if not os.path.exists(dataset_txt):
        print(f"Error: Dataset text file not found at {dataset_txt}")
        return
    if not os.path.exists(wheel_image_path):
        print(f"Error: Steering wheel image not found at {wheel_image_path}")
        return
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return

    # Load dataset index
    xs = []
    ys = []
    with open(dataset_txt, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                xs.append(os.path.join(dataset_dir, parts[0]))
                ys.append(float(parts[1])) # Keep actual angle in degrees

    print(f"Loaded {len(xs)} images from dataset.")

    # Load steering wheel image exactly as it is
    wheel_img = cv2.imread(wheel_image_path)
    if wheel_img is None:
        print("Error: Could not load steering wheel image.")
        return
    # Resize steering wheel image to 300x300
    wheel_img = cv2.resize(wheel_img, (300, 300))

    # Initialize TensorFlow session and restore model
    sess = tf.InteractiveSession()
    saver = tf.compat.v1.train.Saver()
    
    # Restore latest checkpoint
    checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        checkpoint = os.path.join(checkpoint_dir, 'model.ckpt')
    
    print(f"Restoring model from: {checkpoint}")
    try:
        saver.restore(sess, checkpoint)
    except Exception as e:
        print(f"Error restoring model: {e}")
        print("Please make sure training has run at least once to save a checkpoint.")
        return

    # Create two separate windows matching reference names
    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("steering wheel", cv2.WINDOW_AUTOSIZE)

    # Position windows side-by-side on screen
    cv2.moveWindow("frame", 100, 100)
    cv2.moveWindow("steering wheel", 760, 100)

    mode_str = "ACTUAL" if SHOW_ACTUAL else "PREDICTED (AI)"
    print(f"Starting visualization (Mode: {mode_str}) from index 0. Press 'q' or 'ESC' to exit.")

    # Loop through the dataset images starting from index 0
    start_index = 0
    smoothed_angle = 0.0
    for i in range(start_index, len(xs)):
        img_path = xs[i]
        actual_deg = ys[i]

        # Load road image
        full_image = cv2.imread(img_path)
        if full_image is None:
            continue

        # Preprocess image for the model input (crop bottom 150, resize to 200x66, normalize to [0,1])
        model_input = cv2.resize(full_image[-150:], (200, 66)) / 255.0

        # Predict steering angle (in radians)
        predicted_rad = model.y_pred.eval(feed_dict={model.x: [model_input], model.keep_prob: 1.0})[0][0]
        # Convert prediction to degrees
        predicted_deg = predicted_rad * 180.0 / np.pi

        # Select which angle to use for rotation and HUD calculation
        angle_to_show = actual_deg if SHOW_ACTUAL else predicted_deg

        # Smooth out the steering wheel rotation to prevent jitters (Exponential Moving Average)
        smoothed_angle += 0.05 * (angle_to_show - smoothed_angle)

        # Rotate the steering wheel image (negate angle so clockwise matches positive steering angle)
        rotated_wheel = rotate_image(wheel_img, -smoothed_angle)

        # Resize road image to match reference layout size (640x360)
        road_display = cv2.resize(full_image, (640, 360))

        # Calculate simulated telemetry (Brake / Gas / Speed)
        abs_angle = abs(angle_to_show)
        is_braking = abs_angle > 15.0 # Brake if steering turns sharply (more than 15 degrees)
        simulated_speed = max(15.0, 55.0 - (abs_angle * 1.5)) # Speed drops as steering increases

        # Draw HUD overlays on the road view window
        # 1. Draw Brake/Gas Indicator box (Top-Left)
        if is_braking:
            cv2.rectangle(road_display, (20, 20), (120, 55), (0, 0, 255), -1) # Red background
            cv2.putText(road_display, "BRAKE", (32, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(road_display, (20, 20), (120, 55), (0, 150, 0), -1) # Green background
            cv2.putText(road_display, "GAS", (48, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Draw Speedometer box (Top-Right)
        cv2.rectangle(road_display, (490, 20), (620, 55), (50, 50, 50), -1) # Dark gray background
        cv2.putText(road_display, f"{simulated_speed:.0f} MPH", (505, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the separate frames in their respective windows
        cv2.imshow("frame", road_display)
        cv2.imshow("steering wheel", rotated_wheel)

        # Print real-time progress update to console
        sys.stdout.write(f"\rFrame {i:5d} | Actual: {actual_deg:6.2f} deg | Predicted: {predicted_deg:6.2f} deg | Status: {'BRAKING' if is_braking else 'GAS'}")
        sys.stdout.flush()

        # Wait for 15ms (approx 66 fps) for buttery smooth playback
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    print() # New line after loop ends
    sess.close()
    cv2.destroyAllWindows()
    print("Visualization finished.")

if __name__ == "__main__":
    main()
