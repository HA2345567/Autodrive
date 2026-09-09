# Codebase Restructuring and Deduplication Design

**Date**: 2026-09-09  
**Status**: Approved  

---

## 1. Objective

Restructure the AutoDrive project to establish a clean, standard Python repository layout under `src/`, eliminate duplicate model architectures and wrapper scripts, consolidate overlapping CLI tools, and fix path references across batch/PowerShell launchers and documentation.

---

## 2. Directory Layout Architecture

All project source code is consolidated under `src/`:

```
AutoDrive/
├── data/                                # Driving dataset and visual assets
├── saved_models/                        # Model weights and checkpoints
│   ├── lane_detection_best.pt
│   └── steering_angle/                  # TensorFlow 1.x model checkpoints
├── src/                                 # Unified Python packages
│   ├── __init__.py
│   ├── models/                          # Neural network architectures
│   │   ├── __init__.py
│   │   └── model.py                     # NVIDIA PilotNet / DAVE-2 CNN graph
│   ├── inference/                       # Real-time inference engines & CLI tools
│   │   ├── __init__.py
│   │   ├── lane_detector.py             # Lane detection & segmentation engine
│   │   ├── run_lane_detection.py        # Standalone lane detection CLI
│   │   ├── yolo_detector.py             # YOLO object detector & ByteTrack tracker
│   │   ├── run_yolo_object_detection.py # Standalone YOLO detection CLI
│   │   ├── run_steering_angle_prediction.py # Consolidated steering CLI (single & batch)
│   │   └── run_hog_yolo_detection.py    # Classical HOG + YOLO detector
│   └── training/                        # Relocated from model_traning/
│       ├── __init__.py
│       ├── train_steering_angle/
│       │   ├── __init__.py
│       │   ├── driving_data.py          # Data generator for steering angle
│       │   ├── train.py                 # Steering angle training pipeline
│       │   └── logs/                    # TensorBoard event logs
│       └── train_lane_detection/
│           └── lane_detection.ipynb     # YOLOv11 segmentation notebook
├── run.py                               # Unified HUD dashboard (Steering + Lanes + YOLO)
├── run_train.bat                        # Launcher for steering training
├── run_train.ps1                        # PowerShell launcher for steering training
├── run_visualization.bat                # Launcher for unified HUD
├── run_visualization.ps1                # PowerShell launcher for unified HUD
├── requirements.txt                     # Dependencies
├── setup.py                             # Package setup configuration
└── README.md                            # Documentation and quickstart
```

---

## 3. Detailed Deduplications & Changes

### 3.1 Model Architecture Deduplication
- **Problem**: `model_traning/train_steering_angle/model.py` and `src/models/model.py` are identical duplicates.
- **Solution**: Remove `model_traning/train_steering_angle/model.py`. The single source of truth for the PilotNet CNN architecture is `src/models/model.py`.
- **Imports**: `src/training/train_steering_angle/train.py` imports `model` directly via `from src.models import model`.

### 3.2 Steering Inference Consolidation
- **Problem**: `src/inference/run_steering_angle_prediction.py` and `src/inference/run_steering_prediction.py` duplicate `SteeringAnglePredictor`. The former is documented in `README.md`, while the latter has advanced features (single frame mode, batch dataset evaluation, error computation, and interactive visualization).
- **Solution**: Consolidate the full-featured implementation into `src/inference/run_steering_angle_prediction.py`. Delete `src/inference/run_steering_prediction.py`.

### 3.3 Redundant Wrapper Removal
- **Problem**: `src/inference/yolo_run_object_detection.py` is an unneeded 17-line pass-through script that only calls `src/inference/run_yolo_object_detection.py`.
- **Solution**: Delete `src/inference/yolo_run_object_detection.py`.

### 3.4 File Renaming
- **Problem**: `src/inference/run_segmentation_obi_det.py` contains typos and misrepresents its functionality (it implements HOG + YOLO detection, not segmentation).
- **Solution**: Rename to `src/inference/run_hog_yolo_detection.py`.

### 3.5 Training Directory Migration
- **Problem**: `model_traning/` is spelled incorrectly and sits outside `src/`.
- **Solution**: Move to `src/training/` with packages `train_steering_angle/` and `train_lane_detection/`. Remove the old `model_traning/` directory.

### 3.6 Clean Temporary Checkpoints
- **Problem**: Leftover temporary files like `model.ckpt.data-00000-of-00001.tempstate*` exist in `saved_models/steering_angle/`.
- **Solution**: Delete temporary `.tempstate*` files.

### 3.7 Runner Scripts and Documentation Updates
- Update `run.py` to remove any `model_traning` path additions.
- Update `run_train.bat` and `run_train.ps1` to target `src\training\train_steering_angle\train.py`.
- Update `README.md` directory tree diagram and command examples.

---

## 4. Verification Plan

1. **Syntax / Import Validation**:
   - Run Python bytecode compilation check across all `.py` files in `src/` and root: `python -m py_compile ...`
2. **CLI Runner Verification**:
   - Verify `python src/inference/run_steering_angle_prediction.py --help`
   - Verify `python src/inference/run_yolo_object_detection.py --help`
   - Verify `python src/inference/run_lane_detection.py --help`
   - Verify `python src/inference/run_hog_yolo_detection.py --help`
3. **Training Script Validation**:
   - Verify `python -c "from src.models import model; print('Model loaded successfully')"`
   - Verify `python -c "from src.training.train_steering_angle import driving_data; print('Data module imported successfully')"`
4. **Launcher Scripts**:
   - Inspect `run_train.bat` and `run_train.ps1` paths.
