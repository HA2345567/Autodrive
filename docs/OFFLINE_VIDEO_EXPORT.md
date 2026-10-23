# Export a smooth AutoDrive demonstration video

The exporter runs the trained lane model, object model, and verified Huber steering head on every selected source frame before encoding. The MP4 therefore plays smoothly even when inference on the rendering machine is slow. It does not claim real-time inference or perfect model accuracy.

## Setup

Use the existing project environment and restore the assets listed in [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md). Install the video encoder dependency:

```powershell
python -m pip install -r requirements-video.txt
```

## Create the requested 20-second clip

From the repository root, on the NVIDIA laptop:

```powershell
python -m src.export_demo_video --device cuda:0 --start 2400 --seconds 20 --fps 20 --batch 4
```

On a CPU machine:

```powershell
python -m src.export_demo_video --device cpu --start 2400 --seconds 20 --fps 20 --batch 1 --threads 1
```

The default input sizes are 640 pixels for both perception models. The clip contains 400 consecutive frames, source rows 2400–2799. Playback is set to 20 FPS; no source capture rate is inferred from that setting.

This segment belongs to the original steering training partition. It illustrates model outputs and is not a held-out evaluation or evidence of generalization.

## Outputs

All files appear under `output/video/`:

| File | Contents |
|---|---|
| `AutoDrive_20s_Demo.mp4` | 1280 × 720 H.264 MP4, with the road and AI steering panel |
| `AutoDrive_20s_Demo.csv` | Per-frame AI/reference angles, error, lane-mask and object counts |
| `AutoDrive_20s_Demo.predictions.jsonl` | Source-image hashes, polygons, boxes, confidence scores, and steering outputs |
| `AutoDrive_20s_Demo.manifest.json` | Selected frames, settings and model/readout hashes |
| `AutoDrive_20s_Demo.status.json` | Progress and final video metadata |
| `AutoDrive_20s_Demo_frame_*.png` | First, middle and final rendered frames |

The wheel animation uses the original cockpit's gentle smoothing (a 0.15 exponential moving average at 20 FPS, adjusted by playback time for other rates). The displayed AI number and exported CSV retain the unsmoothed prediction. The recorded steering reference and signed prediction error remain visible. All road overlays belong to the same source image as their prediction; there is no stale-frame tracking in this export.

## Resume an interrupted export

Run the same command again. Completed frame predictions are reused when the manifest matches, and missing frames are inferred. Keep the output filename/settings unchanged to resume. Use a new `--output` path when changing model settings:

```powershell
python -m src.export_demo_video --device cuda:0 --start 2400 --seconds 20 --fps 20 --batch 4 --output output/video/AutoDrive_Alternate.mp4
```

Rendering time depends on the hardware and is independent of the final 20-second playback duration. CPU rendering can take many minutes. The exporter does not add detections where the models returned none.
