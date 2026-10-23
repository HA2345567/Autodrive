# Detailed research results

[Back to AutoDrive](../README.md) · [Research manuscript](../output/pdf/AutoDrive_Research_Paper_Reference_Style.pdf)

This appendix retains the detailed tables used in the research manuscript and adds the saved full-validation steering experiment. Values are historical records, not a new model run. Reference table numbering is retained for comparison with the paper.

## Contents

- [COCO-10 class distribution](#coco-10-class-distribution)
- [TuSimple dataset](#tusimple-dataset)
- [Model architecture](#model-architecture)
- [Training hyperparameters](#training-hyperparameters)
- [Recorded comparisons](#recorded-comparisons)
- [Object detection metrics](#object-detection-metrics)
- [Lane metrics](#lane-metrics)
- [Legacy BGR steering audit](#legacy-bgr-steering-audit)
- [Per-class object detection](#per-class-object-detection)
- [Perception timing](#perception-timing)
- [Full-validation steering experiment](#full-validation-steering-experiment)

## COCO-10 class distribution

Instance counts are different from image counts. Person and car dominate this subset.

Table II(a). Dataset distribution: COCO-10 instances per class and split (images: 8,080 / 975 / 525).

| Class | Train instances | Train share | Val instances | Test instances |
| --- | --- | --- | --- | --- |
| person | 24,661 | 43.8% | 2,728 | 1,567 |
| bicycle | 2,887 | 5.1% | 210 | 104 |
| car | 12,443 | 22.1% | 1,276 | 642 |
| motorcycle | 3,054 | 5.4% | 220 | 147 |
| bus | 2,188 | 3.9% | 169 | 114 |
| truck | 2,914 | 5.2% | 267 | 147 |
| traffic light | 4,768 | 8.5% | 403 | 231 |
| stop sign | 1,118 | 2.0% | 53 | 22 |
| dog | 1,221 | 2.2% | 146 | 72 |
| cat | 1,013 | 1.8% | 132 | 70 |
| Total | 56,267 | 100% | 5,604 | 3,116 |

## TuSimple dataset

Lane annotations are derived from sparse polylines; no independent lane test split is reported.

Table II(b). Dataset distribution: TuSimple lane dataset as used in this work.

| Item | Value |
| --- | --- |
| Annotated frames (valid records) | 3,626 (0 missing images) |
| Resolution | 1280×720 px (native); 640 px network input |
| Class | 1 (lane) |
| Train / validation frames | 2,900 / 726 (random split, seed 42) |
| Validation lane instances | 2,732 (≈3.8 per frame) |
| Frames skipped during conversion | 0 train, 0 val |

## Model architecture

The DAVE-2 rows describe the canonical reference, including YUV input, not the saved local checkpoint. The actual local CNN has 1,595,511 parameters and an additional 1,164-unit dense layer. The readout experiment uses RGB and the 1,152-dimensional frozen convolutional features.

Table III. Model architecture: DAVE-2 reference network (parameters computed from layer shapes) and the two YOLO11 models (fused, from training logs).

| Layer | Parameters | Output |
| --- | --- | --- |
| Input (YUV) | – | 66×200×3 |
| Normalisation | – | 66×200×3 |
| Conv1: 24 filters, 5×5, stride 2 | 1,824 | 31×98×24 |
| Conv2: 36 filters, 5×5, stride 2 | 21,636 | 14×47×36 |
| Conv3: 48 filters, 5×5, stride 2 | 43,248 | 5×22×48 |
| Conv4: 64 filters, 3×3 | 27,712 | 3×20×64 |
| Conv5: 64 filters, 3×3 | 36,928 | 1×18×64 |
| Flatten | – | 1,152 |
| FC 100 | 115,300 | 100 |
| FC 50 | 5,050 | 50 |
| FC 10 | 510 | 10 |
| Output (steering) | 11 | 1 |
| Total DAVE-2 | 252,219 |  |
| YOLO11s detector (10 classes) | 9,416,670 | 100 layers, 21.4 GFLOPs |
| YOLO11s-seg lane network (1 class) | 10,067,203 | 113 layers, 32.9 GFLOPs |

## Training hyperparameters

Steering entries describe the original trainer configuration, not the Huber readout. The historical training exposure of the saved checkpoint is unknown. Perception columns come from the recorded training configurations.

Table IV. Hyperparameters of the three modules (values taken from the logged training configurations).

| Parameter | Detection (YOLO11s) | Lane (YOLO11s-seg) | Steering (DAVE-2) |
| --- | --- | --- | --- |
| Initial weights | yolo11s.pt (COCO) | yolo11s-seg.pt | Random initialization |
| Epochs / patience | 50 / 15 | 60 / 20 | 30 / none |
| Batch size | 16 | 16 | 100 |
| Input size | 640×640 | 640×640 | 66×200 |
| Optimiser | auto | auto | Adam |
| Loss | CIoU + BCE + DFL | CIoU + BCE + DFL + mask | MSE |
| LR schedule | cosine, 3 warm-up ep. | linear, 3 warm-up ep. | Fixed 1e-4 |
| Weight decay | 5×10−4 | 5×10−4 | L2 0.001 |
| Mosaic | 1.0 (off last 10 ep.) | 0.0 | - |
| Flip / scale / translate | 0.5 / 0.5 / 0.1 | 0.5 / 0.4 / 0.1 | None |
| HSV (h, s, v) | 0.015, 0.7, 0.4 | 0.015, 0.7, 0.4 | - |
| Rotation / shear / perspective | default | 0 / 0 / 0 | - |
| mask_ratio | – | 2 | - |
| Dropout | 0 | 0 | Keep probability 0.5 |
| Seed / deterministic | 0 / yes | 0 / yes | Not set |
| Mixed precision | yes | yes | No |
| Training time (T4) | 2.14 h | 2.31 h | Not recorded |

## Recorded comparisons

These are training progress, checkpoint re-evaluation, and split comparisons, not controlled ablations. Deltas use the precision printed in notebook outputs. Source cells and hashes are preserved in [recorded_comparisons.json](../research_paper/recorded_comparisons.json).

Table V. Recorded training and evaluation comparisons from the project notebooks [36]. Change = observed - reference, at printed log precision.

| Comparison | Metric | Reference | Observed | Change |
| --- | --- | --- | --- | --- |
| Detection: epoch 1 to 50 | Validation mAP50-95 | 0.331 | 0.482 | +0.151 |
| Lane: epoch 1 to 60 | Box mAP50-95 | 0.672 | 0.831 | +0.159 |
| Lane: epoch 1 to 60 | Mask mAP50-95 | 0.173 | 0.363 | +0.190 |
| Lane: checkpoint re-evaluation | Mask mAP50-95 | 0.363 | 0.389 | +0.026 |
| Detection: validation to test | mAP50-95 | 0.4855 | 0.4711 | -0.0144 |
| Detection: validation to test | mAP50 | 0.6883 | 0.6723 | -0.0160 |

## Object detection metrics

All-class detection metrics. The first epoch, final epoch, and best-checkpoint evaluation are distinct records. mAP is not classification accuracy.

Table VI(a). Main results: detection, COCO-10 (all classes, best.pt).

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | mAP75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Epoch-1 reference (val) | 975 | 5,604 | 0.575 | 0.498 | 0.510 | 0.331 | – |
| Validation | 975 | 5,604 | 0.724 | 0.621 | 0.688 | 0.486 | 0.518 |
| Test (held out) | 525 | 3,116 | 0.712 | 0.614 | 0.672 | 0.471 | 0.511 |

## Lane metrics

YOLO box/mask metrics on derived TuSimple annotations, not official TuSimple scores. The end-of-training best-checkpoint values can differ from the last epoch line.

Table VI(b). Main results: lane segmentation (TuSimple, 726 validation frames, 2,732 lanes). No held-out test split exists.

| Metric | Epoch-1 reference | End of training (best.pt) | Standalone re-evaluation |
| --- | --- | --- | --- |
| Box P | 0.955 | 0.976 | 0.976 |
| Box R | 0.953 | 0.974 | 0.974 |
| Box mAP50 | 0.974 | 0.992 | 0.992 |
| Box mAP50-95 | 0.672 | 0.831 | 0.831 |
| Mask P | 0.744 | 0.918 | 0.925 |
| Mask R | 0.735 | 0.916 | 0.923 |
| Mask mAP50 | 0.595 | 0.877 | 0.890 |
| Mask mAP50-95 | 0.173 | 0.363 | 0.389 |

## Legacy BGR steering audit

A separate 505-frame audit sampled every 18th validation row. Do not pool these values with the 9,082-frame RGB evaluation below.

Table VI(c). Legacy BGR steering checkpoint audit (n = 505); closed-loop autonomy was not evaluated.

| Metric | Result |
| --- | --- |
| Model | DAVE-2 |
| MAE | 22.9616° |
| RMSE | 33.2350° |
| Mean bias | +20.9977° |
| R² | −0.8769 |
| Pearson correlation | −0.0404 |
| Within ±5° | 9.11% |

## Per-class object detection

Held-out test results; validation AP50–95 is supplied for comparison. P = precision; R = recall; inst. = annotated instances.

Table VII. Class-wise detection results on the held-out test split (AP50-95 on validation shown for comparison).

| Class | Test P | Test R | Test AP50 | Test AP50-95 | Val AP50-95 | Test inst. |
| --- | --- | --- | --- | --- | --- | --- |
| person | 0.812 | 0.613 | 0.717 | 0.457 | 0.469 | 1,567 |
| bicycle | 0.686 | 0.399 | 0.498 | 0.270 | 0.311 | 104 |
| car | 0.763 | 0.607 | 0.680 | 0.456 | 0.406 | 642 |
| motorcycle | 0.802 | 0.633 | 0.771 | 0.472 | 0.422 | 147 |
| bus | 0.842 | 0.711 | 0.791 | 0.662 | 0.669 | 114 |
| truck | 0.561 | 0.463 | 0.521 | 0.331 | 0.347 | 147 |
| traffic light | 0.636 | 0.476 | 0.494 | 0.265 | 0.271 | 231 |
| stop sign | 0.488 | 0.607 | 0.619 | 0.556 | 0.736 | 22 |
| dog | 0.651 | 0.806 | 0.771 | 0.596 | 0.558 | 72 |
| cat | 0.882 | 0.829 | 0.861 | 0.645 | 0.667 | 70 |

## Perception timing

All timings are milliseconds per image on Tesla T4. The sums project 32–44 FPS for the two YOLO modules only and exclude capture, steering, display, and control I/O.

Table VIII. Per-image latency on a Tesla T4 (milliseconds, from framework timers) and weight size.

| Module | Source of timing | Pre | Inference | Post | Total | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Detection | Test evaluation (batch 16) | 1.9 | 7.3 | 1.9 | 11.1 | 19.2 MB |
| Lane segmentation | Re-evaluation (batch 16) | 1.2 | 8.1 | 2.5 | 11.8 | 20.5 MB |
| Detection | Single-image predict | 3.7 | 11.0 | 2.3 | 17.0 | – |
| Lane segmentation | Single-image predict | 1.1 | 10.2 | 3.0 | 14.3 | – |
| Both, sequential | Batch-16 sum / single-image sum |  |  |  | 22.9 / 31.3 | 39.7 MB |

## Full-validation steering experiment

All methods use the same 9,082 internal validation rows. The selected CNN + Huber readout improves on the original RGB CNN but trails both constant baselines; R² remains negative. No independent-drive or closed-loop result is claimed.

| Method | MAE ° | RMSE ° | Bias ° | R² | Pearson r | Within ±5° |
| --- | --- | --- | --- | --- | --- | --- |
| Original CNN (RGB) | 23.2570 | 33.6313 | +21.4106 | -0.8067 | 0.0508 | 9.82% |
| Frozen CNN + Huber | 14.6822 | 26.6395 | +3.1230 | -0.1336 | 0.1682 | 31.35% |
| Training-mean constant | 12.4793 | 25.0502 | +1.2196 | -0.0024 | Undefined | 42.57% |
| Training-median constant | 12.6900 | 25.0212 | -0.1859 | -0.0001 | Undefined | 41.62% |

### Selection protocol

| Item | Recorded value |
| --- | --- |
| Original split | 36,324 training / 9,082 validation rows |
| Training sample | 9,081 rows, every fourth original training row |
| Inner chronological split | 7,265 fit / 1,816 development |
| Preprocessing | RGB; crop bottom 150 rows; resize 200 × 66; divide by 255 |
| Feature representation | 1,152 frozen CNN features; training-derived standardization |
| Candidate search | 5 squared-error penalties and 4 Huber configurations |
| Selected readout | Huber threshold 10°; penalty α = 100 |
| Final optimizer | L-BFGS-B; 96 iterations; convergence reported |
| Uncertainty | Circular block bootstrap; 100-frame blocks; 1,000 replicates; seed 20261021 |
| Training-label audit | -152.87° to 501.78°; 219 labels outside ±180° |
| Validation labels outside ±180° | 0 |

The original `2 × atan` output is bounded to ±180°; the new linear degree readout removes that representational limit. Validation labels are excluded from readout fitting and selection. Historical backbone exposure and driving-data provenance are not established. The main demo now supports this readout after verifying the backbone hash; the historical metrics above retain their original protocol.

### Bootstrap intervals

Circular block bootstrap, block length 100 consecutive validation frames; 1000 replicates; seed 20261021. Intervals depend on the chosen block length.

| Method | MAE 95% interval (°) | RMSE 95% interval (°) |
| --- | --- | --- |
| Original CNN (RGB) | 19.677–27.703 | 24.927–42.875 |
| Frozen CNN + Huber | 11.631–18.629 | 18.241–35.675 |
| Training-mean constant | 9.254–16.665 | 16.089–34.413 |
| Training-median constant | 9.462–16.824 | 16.212–34.242 |

Paired MAE reduction versus the original RGB CNN: **6.849°–10.250°** (95% interval).

### Evidence

- [Selected-readout audit](../output/steering_audit/improved_readout_audit.json)
- [Comparison summary](../output/steering_audit/comparison_summary.json)
- [Frame-level predictions](../output/steering_audit/improved_validation_predictions.csv)
- [Experiment report](../output/steering_audit/STEERING_EXPERIMENT.md)
- [Research manuscript source](../research_paper/build_reference_style.py)

Numeric precision follows the source tables and JSON records. Nothing here represents an independent rerun of the perception evaluators.
