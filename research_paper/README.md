# AutoDrive research sources

Start with the [project README](../README.md), [detailed result tables](../docs/RESEARCH_RESULTS.md), and [latest research manuscript](../output/pdf/AutoDrive_Research_Paper_Reference_Style.pdf).

## Manuscript and figures

Presentation builders and original slide assets are in [presentation/](presentation/). The [README asset builder](build_readme_assets.py) uses those preserved inputs. Temporary drafts and preview renders belong in the ignored `tmp/` folder.

| Source | Purpose |
|---|---|
| [build_reference_style.py](build_reference_style.py) | Latest reference-style manuscript builder |
| [build_manuscript.py](build_manuscript.py) | Earlier completed manuscript source |
| [assets/](assets/) | Original full-resolution paper figures |
| [diagrams/](diagrams/) | Mermaid sources and vector exports |
| [manuscript_data.json](manuscript_data.json) | Original figure provenance, references, and legacy 505-frame audit |
| [recorded_comparisons.json](recorded_comparisons.json) | Saved perception notebook records, source cells, and hashes |

The manuscript builder uses local Windows fonts and the bundled Computer Modern fonts. Its imports include ReportLab, PyMuPDF, pdfplumber, and Pillow. Inspect the rendered pages after changing the manuscript. Earlier final/completed PDF filenames are retained as historical artifacts; the linked reference-style PDF is the latest manuscript used for the presentation and project README.

## Steering experiment

| Source | Purpose |
|---|---|
| [steering_readout_plan.json](steering_readout_plan.json) | Candidate search and selection protocol |
| [train_steering_readout.py](train_steering_readout.py) | RGB feature extraction and squared-error readouts |
| [train_robust_steering_readout.py](train_robust_steering_readout.py) | Huber candidate search and refit |
| [summarize_steering_experiment.py](summarize_steering_experiment.py) | Frame-aligned metrics, baselines, bootstrap intervals, and plots |
| [Experiment report](../output/steering_audit/STEERING_EXPERIMENT.md) | Recorded protocol and results |
| [Selected-readout audit](../output/steering_audit/improved_readout_audit.json) | Selection record, optimizer, and full-validation metrics |

The 505-frame BGR audit and the 9,082-frame RGB evaluation are separate records. The selected Huber readout reports 14.6822° MAE, improving on the original RGB CNN while trailing constant baselines. Its R² remains negative. The main demo can now load this readout after verifying its backbone hash; that runtime integration does not change the recorded evaluation metrics.

## Reporting scope

- Perception metrics and curves come from saved project notebook outputs and figures; this documentation does not claim an independent evaluator rerun.
- Lane scores use derived masks and YOLO metrics, not official TuSimple leaderboard evaluation.
- Steering uses an existing ordered internal split. Historical backbone exposure and dataset provenance are not established.
- The original dataset and checkpoint are required to reproduce feature extraction; the Huber experiment also requires SciPy. Preserve saved experiment outputs before rerunning scripts.
- Independent-drive and closed-loop vehicle evaluations remain outstanding.
- All three authors and the ECE affiliation at Maharaja Surajmal Institute of Technology, New Delhi, are listed in the project README and manuscript.
