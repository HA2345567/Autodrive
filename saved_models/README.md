# Saved Models Directory

This directory stores trained model checkpoints and serialized weights for the AutoDrive project.

## Structure

- `lane_detection_model/lane_detection_best.pt`: trained YOLO11s lane segmentation model.
- `object_detection_model/object_detection_best.pt`: trained ten-class YOLO11s object detector.
- `regression_model/model.ckpt.*`: TensorFlow steering CNN checkpoint. Keep its `.index`, `.meta` and `.data-00000-of-00001` files together with the `checkpoint` pointer file.

The selected steering readout is stored separately at `output/steering_audit/improved_readout.npz`; it must match the CNN backbone. See [the asset transfer and GPU setup guide](../docs/NVIDIA_GPU_SETUP.md).

## Notes:
- Model checkpoint binaries (`.ckpt`, `.pb`, `.h5`, `.keras`, etc.) are ignored by git to keep the repository lightweight.
