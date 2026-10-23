# AutoDrive Project Architecture

This document is the handoff map for the AutoDrive project. It explains where the code lives, how the runtime works, and which large or private files must be transferred separately.

## 1. Repository Map

```text
AutoDrive/
├── README.md                         Short project entry point
├── ARCHITECTURE.md                   This handoff and file-structure guide
├── pyproject.toml                    Build-system and Pyrefly configuration
├── setup.py                          Python package metadata and dependencies
├── requirements.txt                  Runtime and training dependencies
├── data/
│   └── driving_dataset/
│       └── data.txt                  Image paths and steering-angle labels
├── model_training/
│   ├── train_lane_detection/
│   │   └── tusimple_lane_yolo11.ipynb       Lane segmentation training notebook
│   ├── train_object_detection/
│   │   └── coco_object_detection_yolo11.ipynb Object detection training notebook
│   └── train_steering_angle/
│       ├── driving_data.py            Driving dataset loader/preprocessor
│       ├── train.py                   NVIDIA CNN steering model training
│       ├── logs/                      TensorBoard output; generated
│       └── save/                      Training checkpoint; generated
├── saved_models/
│   ├── lane_detection_model/
│   │   └── lane_detection_best.pt     YOLO lane segmentation weights
│   ├── object_detection_model/
│   │   └── object_detection_best.pt   YOLO object detection weights
│   └── regression_model/
│       └── model.ckpt.*               TensorFlow steering weights
├── src/
│   ├── run.py                         Main unified runtime and HUD
│   ├── models/
│   │   └── model.py                   TensorFlow steering model graph
│   └── inference/
│       ├── lane_detector.py           Synchronous/asynchronous lane inference
│       ├── object_detector.py         Synchronous/asynchronous object inference
│       ├── pedal_dashboard.py         Brake/accelerator dashboard rendering
│       ├── run_fsd_inference.py       FSD-style combined inference runner
│       ├── run_lane_detection.py      Lane-only runner
│       └── run_steering_angle_prediction.py Steering-only runner
└── tests/
    ├── test_integration_pipeline.py  Lane and steering integration tests
    ├── test_lane_detector.py          Lane detector tests
    ├── test_pedal_dashboard.py        Pedal dashboard tests
    └── test_fsd_pedal_integration.py  FSD/pedal integration tests
```

The repository also contains standard Python package marker files (`__init__.py`) and local tooling folders such as `.vscode/`, `.git/`, and `venv/`. The virtual environment should not be transferred; recreate it on the recipient's machine.

Research documentation and figures live in `docs/`. Manuscript and presentation builders live in `research_paper/`, including `research_paper/presentation/` for the slide sources and original assets. `tools/build_gpu_handoff.py` packages the runtime for another machine. The generated `tmp/` directory is disposable and ignored by Git. Transfer archives, cached steering features and video inference caches remain local; final deliverables and recorded result reports remain available under `output/`.

## 2. Runtime Flow

The main entry point is `src/run.py`.

1. It reads `data/driving_dataset/data.txt` and resolves image paths below `data/driving_dataset/`.
2. It loads the steering-wheel image from `data/sterring_wheel_image.png`.
3. It restores the TensorFlow steering checkpoint from `saved_models/regression_model/`.
4. It loads the YOLO lane model from `saved_models/lane_detection_model/lane_detection_best.pt`.
5. It loads the YOLO object model from `saved_models/object_detection_model/object_detection_best.pt`.
6. It runs frame-by-frame inference and displays the road view plus the `Steering & Actuation` HUD window.
7. `src/inference/run_realtime_demo.py` coordinates bounded work in `realtime_pipeline.py`. Both detectors receive the same clean source frame; a two-worker CPU pool uses one PyTorch thread per model. Default aligned playback displays that frame with its own predictions; optional tracked playback applies bounded optical flow and hides expired results.
8. AI steering is the default. A verified RGB Huber readout is used when its backbone hash matches; otherwise the original BGR CNN is reported as the fallback. Recorded labels are displayed separately for comparison.
9. `runtime_backend.py` supports native CUDA/CPU and opt-in OpenVINO exports (`--backend openvino`). Auto uses PyTorch: OpenVINO measured slower on the local Pentium. Balanced CPU inference uses one PyTorch thread with 480-pixel lane/object inputs. Display FPS and inference Hz are separate measurements.

The model responsibilities are:

| Component | Input | Output | Implementation |
|---|---|---|---|
| Steering angle | Road image | Predicted steering angle in degrees | TensorFlow v1 graph in `src/models/model.py` |
| Lane detection | Road frame | Lane masks/overlay and lane metadata | YOLO segmentation in `src/inference/lane_detector.py` |
| Object detection | Road frame | Detected objects and confidence scores | YOLO detection in `src/inference/object_detector.py` |
| Pedal dashboard | Telemetry/state | Brake and accelerator UI | `src/inference/pedal_dashboard.py` |

## 3. What Can Be Shared in Git

Share the source and configuration files:

- `src/`
- `model_training/` notebooks and Python scripts
- `tests/`
- `requirements.txt`, `setup.py`, and `pyproject.toml`
- `README.md` and `ARCHITECTURE.md`
- The dataset index format in `data/driving_dataset/data.txt`, if it does not expose sensitive paths

## 4. What Must Be Transferred Separately

The current `.gitignore` excludes these files because they are large generated assets or local data:

- All driving images under `data/driving_dataset/`
- `data/sterring_wheel_image.png` (required by `src/run.py` and some tests)
- `saved_models/lane_detection_model/lane_detection_best.pt`
- `saved_models/object_detection_model/object_detection_best.pt`
- `saved_models/regression_model/model.ckpt.*`
- Training checkpoints in `model_training/**/save/`
- TensorBoard logs in `model_training/**/logs/`

Use a private file share, artifact store, or encrypted archive for these assets. Preserve the directory names shown above when extracting them. Do not commit credentials, `.env` files, virtual environments, or private datasets.

If the recipient does not receive the trained models, they can still inspect the code and retrain the models using the notebooks/scripts, but the unified runtime will not produce the intended predictions until the checkpoints are restored.

## 5. Setup on Another Machine

From the project root on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Then copy the private data and model assets into the exact paths listed in section 4.

Run the unified application with:

```powershell
python -m src.run
```

Useful options include:

```powershell
python -m src.run --start 2400 --fps 20
python -m src.run --no-lanes
python -m src.run --no-objects
python -m src.run --lane-model C:\path\to\lane_detection_best.pt
python -m src.run --object-model C:\path\to\object_detection_best.pt
```

## 6. Validation

Run the tests from the project root:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Tests that require images or trained weights may skip or fail when the private assets have not been copied. That indicates an incomplete local asset transfer, not necessarily a source-code failure.

## 7. Important Path Notes

- Run commands from the repository root, not from inside `src/`.
- The steering-wheel filename is currently `sterring_wheel_image.png` with the spelling shown above. Keep that spelling unless both the code and asset are renamed together.
- `src/run.py` has fallback model paths, but the standard layout in `saved_models/` is the expected handoff layout.
- Model files are binary artifacts and should be versioned separately from the source code.
