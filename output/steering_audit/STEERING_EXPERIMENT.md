# Steering model improvement experiment

This report uses newly computed frame-aligned predictions. No performance numbers were replaced with assumed targets.

## Protocol

- Original ordered split: 36,324 training rows and 9,082 validation rows.
- New readout fitted on 9,081 training frames (every fourth row).
- Inner training split: 7,265 fit / 1,816 development frames.
- Readout penalty selected on inner-development MAE: alpha = 100; loss = huber.
- Frozen five-convolution backbone; RGB, bottom 150 rows, resize to 200x66, divide by 255.
- Original RGB head and new readout are evaluated on identical validation frames.
- Validation labels are excluded from fitting and hyperparameter selection. Historical backbone checkpoint exposure is not independently established.
- This is internal validation on an existing ordered split. No independent driving test or closed-loop simulation is claimed.

## Full validation results

| Model | MAE (deg) | RMSE (deg) | Bias (deg) | R2 | Pearson r | Within +/-5 deg |
|---|---:|---:|---:|---:|---:|---:|
| Original CNN (RGB) | 23.2570 | 33.6313 | +21.4106 | -0.8067 | 0.0508 | 9.82% |
| Frozen CNN + Huber | 14.6822 | 26.6395 | +3.1230 | -0.1336 | 0.1682 | 31.35% |
| Training-mean constant | 12.4793 | 25.0502 | +1.2196 | -0.0024 | undefined (constant) | 42.57% |
| Training-median constant | 12.6900 | 25.0212 | -0.1859 | -0.0001 | undefined (constant) | 41.62% |

## Diagnosis

- The saved checkpoint and the training-folder checkpoint have identical weights.
- The earlier BGR audit is reproduced on 505 frames: MAE 22.9616 deg, RMSE 33.2350 deg, R2 -0.8769.
- RGB preprocessing alone does not improve the 505-frame validation sample.
- The old 2*atan output is bounded by +/-180 degrees. There are 219 original training labels outside that range, with a maximum of 501.78 degrees.
- The new linear degree head removes that representational limit. Its actual usefulness is determined by the validation results above.

## Uncertainty

Circular block bootstrap, block length 100 consecutive validation frames; 1000 replicates; seed 20261021. Intervals depend on the chosen block length.

- Original CNN (RGB): MAE 95% interval [19.677, 27.703] deg; RMSE [24.927, 42.875] deg.
- Frozen CNN + Huber: MAE 95% interval [11.631, 18.629] deg; RMSE [18.241, 35.675] deg.
- Training-mean constant: MAE 95% interval [9.254, 16.665] deg; RMSE [16.089, 34.413] deg.
- Training-median constant: MAE 95% interval [9.462, 16.824] deg; RMSE [16.212, 34.242] deg.

## Evidence files

- `pilot_audit.json`: preprocessing comparison and training-only calibration.
- `improved_readout_audit.json`: selected loss, penalty search, optimizer status and full validation.
- `readout_audit.json`: the initial squared-error readout comparison.
- `improved_validation_predictions.csv`: every evaluated row, image, target and selected prediction.
- `improved_readout.npz`: selected readout, feature normalization and backbone hash.
- `comparison_summary.json`: metrics and block-bootstrap intervals.
