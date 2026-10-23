# AutoDrive: doosre Windows laptop par NVIDIA GPU setup

Yeh guide current local runtime ke liye hai. **Latest fixes uncommitted local files mein hain**, isliye pehle transfer ZIP ya current source copy karein. Sirf remote GitHub clone se latest fixes milna guaranteed nahi hai.

The setup commands below are for Windows PowerShell and 64-bit Python 3.11. No GPU performance measurement has been made on the destination laptop.

## 1. Kya copy karna hai

Recommended: `AutoDrive_GPU_Handoff.zip` ko doosre laptop par extract karein. Isme current runtime source, setup guide, required model weights, steering readout, dataset index, aur wheel image hain. **Driving JPG images is ZIP mein nahi hain.**

Current checkout se handoff archive dobara banane ke liye `python tools/build_gpu_handoff.py` run karein. Generated ZIP `output/transfer/` mein rehta hai aur Git se excluded hai.

Current laptop se `data/driving_dataset/` ka poora image collection bhi copy karein (current folder approximately 2.33 GB). `data.txt` ke paths image files se match rehne chahiye. Destination structure:

```text
AutoDrive/
├── src/                                  # poora latest src folder
├── requirements.txt
├── setup.py
├── pyproject.toml
├── data/
│   ├── sterring_wheel_image.png           # exact spelling required
│   └── driving_dataset/
│       ├── data.txt
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ... remaining recorded images
├── saved_models/
│   ├── lane_detection_model/
│   │   └── lane_detection_best.pt
│   ├── object_detection_model/
│   │   └── object_detection_best.pt
│   └── regression_model/
│       ├── checkpoint
│       ├── model.ckpt.index
│       ├── model.ckpt.meta
│       └── model.ckpt.data-00000-of-00001
└── output/
    └── steering_audit/
        └── improved_readout.npz           # verified RGB Huber steering head
```

| File / folder | Required for |
|---|---|
| `lane_detection_best.pt` | Trained lane instance segmentation |
| `object_detection_best.pt` | Trained 10-class object detection |
| Entire `saved_models/regression_model/` | Original TensorFlow CNN backbone/checkpoint |
| `output/steering_audit/improved_readout.npz` | Improved Huber AI steering head; must match this backbone |
| Entire `data/driving_dataset/` | Actual recorded frames and steering labels |
| `data/sterring_wheel_image.png` | Steering-wheel display |
| Entire current `src/` | Latest input, timing, alignment and AI-mode fixes |

`src/inference/realtime_pipeline.py`, `run_realtime_demo.py`, and `runtime_backend.py` must be present. Copying only the old `src/run.py` is insufficient.

Do not copy `venv/` or `.venv/` to the new machine; create a fresh environment. Generated `*_openvino_model/` folders are not needed for NVIDIA inference. The feature cache, training logs and research PDFs are not required to run the demo.

The checkpoint pointer file already uses relative paths:

```text
model_checkpoint_path: "model.ckpt"
all_model_checkpoint_paths: "model.ckpt"
```

Keep the three `model.ckpt.*` files together. The `.npz` readout alone cannot replace the CNN checkpoint.

## 2. Driver, Python, aur project location

1. Install a current NVIDIA driver compatible with the laptop GPU from [NVIDIA](https://www.nvidia.com/Download/index.aspx), then restart if the installer requires it.
2. Install **Python 3.11, 64-bit**, with the Python launcher from [python.org](https://www.python.org/downloads/windows/).
3. Extract the handoff to a folder such as `D:\Projects\AutoDrive`.
4. Copy the driving images into its `data\driving_dataset\` folder.

PowerShell mein:

```powershell
nvidia-smi
py -0p
Set-Location D:\Projects\AutoDrive
```

Change the project path if you extracted elsewhere. `nvidia-smi` should list the NVIDIA GPU. Its CUDA version is a driver capability, not proof that PyTorch can use CUDA.

## 3. Fresh virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation, use the environment's interpreter directly for every subsequent Python command:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

This avoids changing the machine's execution policy. Replace `python` below with `.\.venv\Scripts\python.exe` if activation was blocked.

## 4. CUDA-enabled PyTorch install

Use **one** of the following commands, according to the GPU. These pinned PyTorch/torchvision pairs are listed in the official [PyTorch release installation instructions](https://pytorch.org/get-started/previous-versions/).

### RTX 20 / 30 / 40 / 50 series, or Turing/newer GPU

Use an up-to-date compatible NVIDIA driver, then:

```powershell
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
```

### Older Maxwell / Pascal / Volta GPU

For older cards such as GTX 10-series, use the CUDA 12.6 wheel family that retains those architectures:

```powershell
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
```

Very old cards may fall outside supported architectures. Check the [PyTorch architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) rather than installing arbitrary CUDA builds. RTX 50-series needs a Blackwell-capable build; do not use the legacy cu126 branch for it.

A separate system CUDA Toolkit is not the first setup step for these prebuilt inference wheels. Start with the NVIDIA driver and the matching PyTorch wheel; a source build or custom CUDA extension has additional requirements.

## 5. Repo dependencies

Install the repository dependencies while retaining the TensorFlow and Ultralytics versions used by the current local runtime:

```powershell
python -m pip install -r requirements.txt "tensorflow==2.20.0" "ultralytics==8.4.117"
python -m pip install -e . --no-deps
python -m pip check
```

These are setup pins, not a claim that the NVIDIA destination environment has been fully validated. The dependency list is otherwise unpinned. No OpenVINO installation is needed for this NVIDIA path.

**TensorFlow GPU list empty aana is setup mein normal hai.** Current code explicitly runs the steering graph on CPU and reserves NVIDIA CUDA for PyTorch perception. Modern TensorFlow native-Windows packages do not provide the old native CUDA support; there is no need to downgrade to TensorFlow 2.10 for this application. [TensorFlow installation guidance](https://www.tensorflow.org/install/pip#windows-native).

## 6. GPU actually work kar raha hai ya nahi

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable: check driver, wheel and Python environment'; print('GPU:', torch.cuda.get_device_name(0)); x=torch.rand((256,256),device='cuda:0'); print('CUDA operation:', (x@x).mean().item())"
```

Expected:

```text
CUDA build: <a CUDA version, not None>
CUDA available: True
GPU: <your NVIDIA GPU name>
CUDA operation: <a numeric value>
```

The matrix operation checks that a real GPU kernel runs, not just that a device is enumerated.

Also confirm the local interpreter and runtime dependencies:

```powershell
python -c "import sys,tensorflow,ultralytics; print(sys.executable); print('TensorFlow:',tensorflow.__version__); print('Ultralytics:',ultralytics.__version__)"
```

## 7. Run commands

Run all commands from the project root.

### Smooth recorded-video playback on NVIDIA GPU

```powershell
python -m src.run --device cuda:0 --backend torch --playback-mode tracked --fps 20 --start 2400 --steering-head huber
```

- `tracked` advances the video without waiting for every detector call. Short-term optical flow moves cached detections; old or unreliable overlays are hidden.
- AI steering is the default; `huber` explicitly requires the verified readout instead of silently falling back.
- Start frame 2400 is a road scene. Starting at 0 shows the parking-garage beginning of the sequence, where highway-style lanes may not exist.
- The current tracked mode still receives steering from completed inference packets. If perception is too slow, an expired packet can show `WAITING`; it does not substitute recorded steering.
- Display FPS is not fresh detection FPS. The HUD reports inference Hz and source age separately.

### Exact frame-matched predictions

```powershell
python -m src.run --device cuda:0 --backend torch --playback-mode aligned --fps 20 --start 2400 --steering-head huber
```

`aligned` advances source playback only when a matching prediction is ready. It can look choppy if combined inference cannot keep up. This is the current default, so use the explicit `tracked` command when testing smooth playback.

### Lower perception resolution if necessary

```powershell
python -m src.run --device cuda:0 --backend torch --playback-mode tracked --fps 20 --start 2400 --lane-imgsz 480 --imgsz 480 --steering-head huber
```

CUDA otherwise defaults to 640 pixels for both detectors. Smaller sizes can improve latency but may miss thin lanes or distant objects. Exact FPS depends on GPU model, VRAM, CPU, storage, thermals and the scene; no fixed FPS is guaranteed.

### Original CNN head, for comparison

```powershell
python -m src.run --device cuda:0 --backend torch --playback-mode tracked --fps 20 --start 2400 --steering-head original
```

This still displays genuine AI output, but uses the original BGR CNN instead of the Huber readout.

## 8. Running output mein kya dikhna chahiye

Look for:

```text
device=cuda:0
playback=tracked
Loaded torch lane + torch object | steering=HUBER RGB
mode=AI
```

The main window shows lane/object overlays and source age. The wheel window shows AI steering, the recorded reference, and their difference. `REFERENCE` mode is recorded steering, not an AI prediction.

Second PowerShell window mein GPU activity dekhein:

```powershell
nvidia-smi -l 1
```

Python GPU memory/activity should appear when CUDA inference is running. Stop this monitoring command with Ctrl+C.

Controls: **M** = AI/reference, **P/Space** = pause, **L/O** = lane/object overlays, **C** = corridor, **[ / ]** = seek, **Q/Esc** = quit.

## 9. Common problems

| Symptom | Action |
|---|---|
| `CUDA available: False` | Check `sys.executable`, install the matching CUDA wheel in that environment, and check/update the NVIDIA driver. |
| PyTorch version ends in `+cpu`, or CUDA build is `None` | In this project environment, uninstall `torch torchvision`, then reinstall the selected CUDA wheel command. |
| `no kernel image is available` | The wheel does not support that GPU architecture. Use the architecture-appropriate branch and rerun the CUDA operation check. |
| `CUDA out of memory` | Close other GPU-heavy applications; retry 480-pixel inputs and one overlay at a time with `--no-lanes` or `--no-objects`. |
| Huber file missing or hash mismatch | Copy `improved_readout.npz` with its matching regression checkpoint. Do not rename a different readout or bypass the hash check. |
| Wheel image missing | Copy `data/sterring_wheel_image.png` with its exact filename. |
| Runtime arguments not recognized | The destination has old source. Copy the entire latest `src/` folder from this handoff. |
| Video pauses despite GPU | Confirm `device=cuda:0` and `playback=tracked`; aligned mode intentionally waits for model completion. |
| Boxes/masks expire in tracked mode | Read the reported latency. GPU presence alone does not guarantee timely inference or accurate tracking. |
| `cv2.imshow` error | Use desktop Windows and the normal `opencv-python` package, not a headless-only OpenCV installation. |

For an accidental CPU-only PyTorch install, use these commands only inside the new project environment, then run the selected CUDA install command again:

```powershell
python -m pip uninstall -y torch torchvision
```

## Official references

- [PyTorch install commands and version pairs](https://pytorch.org/get-started/previous-versions/)
- [PyTorch GPU architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [TensorFlow native Windows installation](https://www.tensorflow.org/install/pip#windows-native)

Guide prepared from the current repository and official documentation on 23 October 2026. Model accuracy and smoothness on the destination laptop remain to be measured.
