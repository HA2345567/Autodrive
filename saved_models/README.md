# Saved Models Directory

This directory stores trained model checkpoints and serialized weights for the AutoDrive project.

## Structure:
- `steering_angle/`: Model checkpoints for the end-to-end steering angle prediction network (NVIDIA CNN architecture).
- `lane_detection/`: Model weights / calibration matrices for advanced lane detection.

## Notes:
- Model checkpoint binaries (`.ckpt`, `.pb`, `.h5`, `.keras`, etc.) are ignored by git to keep the repository lightweight.
