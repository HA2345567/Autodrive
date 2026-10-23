"""Assemble README figures and detailed tables from existing research artifacts."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
IMAGES = DOCS / 'images'
IMAGES.mkdir(parents=True, exist_ok=True)

sources = {
    'prototype_display.jpeg': ('research_paper/presentation/assets/prototype_display.jpeg', 'Original prototype screenshot extracted from the user-supplied minor synopsis PDF; display demonstration only.'),
    'object_training.png': ('research_paper/assets/object_training.png', 'Original object-detection training figure preserved from the supplied research paper.'),
    'lane_training.png': ('research_paper/assets/lane_training.png', 'Original lane-segmentation training figure preserved from the supplied research paper.'),
    'lane_predictions.jpg': ('research_paper/assets/lane_predictions.jpg', 'Original lane validation examples preserved from the supplied research paper.'),
    'object_predictions.jpg': ('research_paper/assets/object_predictions.jpg', 'Original object test examples preserved from the supplied research paper.'),
    'class_ap.png': ('research_paper/assets/class_ap.png', 'Original class-wise AP figure preserved from the supplied research paper.'),
    'object_confusion.jpg': ('research_paper/assets/object_confusion.jpg', 'Original object confusion matrix preserved from the supplied research paper.'),
    'lane_confusion.jpg': ('research_paper/assets/lane_confusion.jpg', 'Original lane confusion matrix preserved from the supplied research paper.'),
    'steering_full_validation_scatter.png': ('output/steering_audit/steering_full_validation_scatter.png', 'Generated from saved full-validation predictions by research_paper/summarize_steering_experiment.py.'),
    'steering_validation_sequence.png': ('output/steering_audit/steering_validation_sequence.png', 'Generated from saved full-validation predictions by research_paper/summarize_steering_experiment.py.'),
}
manifest = []
for name, (source, description) in sources.items():
    shutil.copy2(ROOT / source, IMAGES / name)
    manifest.append({'file': name, 'source': source, 'description': description,
                     'sha256': hashlib.sha256((IMAGES / name).read_bytes()).hexdigest()})
(IMAGES / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
lines = ['# Figure provenance', '', '[Back to the project README](../../README.md)', '',
         'These are copies of existing project figures. No result was changed or simulated for the README.', '',
         '| Figure | Origin and scope |', '|---|---|']
for item in manifest:
    lines.append(f"| [{item['file']}]({item['file']}) | {item['description']} |")
lines += ['', 'The perception extraction records, original PDF pages, and pixel dimensions are documented in [manuscript_data.json](../../research_paper/manuscript_data.json).', '',
          'The prototype source is the user-supplied `Minor_synopsis PPT (2).pdf`. Steering plots correspond to [improved_validation_predictions.csv](../../output/steering_audit/improved_validation_predictions.csv) and the [comparison summary](../../output/steering_audit/comparison_summary.json).', '',
          'File hashes and original repository paths are retained in [provenance.json](provenance.json). Copying these assets does not assign new rights to third-party imagery; original terms still apply.', '']
(IMAGES / 'README.md').write_text('\n'.join(lines), encoding='utf-8')

tables = json.loads((ROOT / 'research_paper/presentation/assets/tables.json').read_text(encoding='utf-8'))
table_by_label = {table['label']: table for table in tables}
def markdown_table(rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    result = ['| ' + ' | '.join(map(cell, rows[0])) + ' |',
              '| ' + ' | '.join('---' for _ in rows[0]) + ' |']
    result += ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows[1:]]
    return '\n'.join(result)

sections = [
    ('COCO-10 class distribution', 'II(a)', 'Instance counts are different from image counts. Person and car dominate this subset.'),
    ('TuSimple dataset', 'II(b)', 'Lane annotations are derived from sparse polylines; no independent lane test split is reported.'),
    ('Model architecture', 'III', 'The DAVE-2 rows describe the canonical reference, including YUV input, not the saved local checkpoint. The actual local CNN has 1,595,511 parameters and an additional 1,164-unit dense layer. The readout experiment uses RGB and the 1,152-dimensional frozen convolutional features.'),
    ('Training hyperparameters', 'IV', 'Steering entries describe the original trainer configuration, not the Huber readout. The historical training exposure of the saved checkpoint is unknown. Perception columns come from the recorded training configurations.'),
    ('Recorded comparisons', 'V', 'These are training progress, checkpoint re-evaluation, and split comparisons, not controlled ablations. Deltas use the precision printed in notebook outputs. Source cells and hashes are preserved in [recorded_comparisons.json](../research_paper/recorded_comparisons.json).'),
    ('Object detection metrics', 'VI(a)', 'All-class detection metrics. The first epoch, final epoch, and best-checkpoint evaluation are distinct records. mAP is not classification accuracy.'),
    ('Lane metrics', 'VI(b)', 'YOLO box/mask metrics on derived TuSimple annotations, not official TuSimple scores. The end-of-training best-checkpoint values can differ from the last epoch line.'),
    ('Legacy BGR steering audit', 'VI(c)', 'A separate 505-frame audit sampled every 18th validation row. Do not pool these values with the 9,082-frame RGB evaluation below.'),
    ('Per-class object detection', 'VII', 'Held-out test results; validation AP50–95 is supplied for comparison. P = precision; R = recall; inst. = annotated instances.'),
    ('Perception timing', 'VIII', 'All timings are milliseconds per image on Tesla T4. The sums project 32–44 FPS for the two YOLO modules only and exclude capture, steering, display, and control I/O.'),
]
lines = ['# Detailed research results', '', '[Back to AutoDrive](../README.md) · [Research manuscript](../output/pdf/AutoDrive_Research_Paper_Reference_Style.pdf)', '',
         'This appendix retains the detailed tables used in the research manuscript and adds the saved full-validation steering experiment. Values are historical records, not a new model run. Reference table numbering is retained for comparison with the paper.', '',
         '## Contents', '']
for title, _, _ in sections:
    anchor = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
    lines.append(f'- [{title}](#{anchor})')
lines += ['- [Full-validation steering experiment](#full-validation-steering-experiment)', '']
for title, label, note in sections:
    table = table_by_label[label]
    lines += ['## ' + title, '', note, '', table['caption'], '', markdown_table(table['rows']), '']

summary = json.loads((ROOT / 'output/steering_audit/comparison_summary.json').read_text(encoding='utf-8'))
audit = json.loads((ROOT / 'output/steering_audit/improved_readout_audit.json').read_text(encoding='utf-8'))
rows = [['Method', 'MAE °', 'RMSE °', 'Bias °', 'R²', 'Pearson r', 'Within ±5°']]
for name, metrics in summary['metrics'].items():
    rows.append([name, f"{metrics['mae_deg']:.4f}", f"{metrics['rmse_deg']:.4f}", f"{metrics['bias_deg']:+.4f}",
                 f"{metrics['r_squared']:.4f}", 'Undefined' if metrics['pearson_r'] is None else f"{metrics['pearson_r']:.4f}",
                 f"{metrics['within_5_deg_pct']:.2f}%"])
protocol = audit['protocol']
lines += ['## Full-validation steering experiment', '',
          'All methods use the same 9,082 internal validation rows. The selected CNN + Huber readout improves on the original RGB CNN but trails both constant baselines; R² remains negative. No independent-drive or closed-loop result is claimed.', '', markdown_table(rows), '',
          '### Selection protocol', '',
          markdown_table([['Item', 'Recorded value'],
            ['Original split', '36,324 training / 9,082 validation rows'],
            ['Training sample', '9,081 rows, every fourth original training row'],
            ['Inner chronological split', '7,265 fit / 1,816 development'],
            ['Preprocessing', 'RGB; crop bottom 150 rows; resize 200 × 66; divide by 255'],
            ['Feature representation', '1,152 frozen CNN features; training-derived standardization'],
            ['Candidate search', '5 squared-error penalties and 4 Huber configurations'],
            ['Selected readout', 'Huber threshold 10°; penalty α = 100'],
            ['Final optimizer', 'L-BFGS-B; 96 iterations; convergence reported'],
            ['Uncertainty', 'Circular block bootstrap; 100-frame blocks; 1,000 replicates; seed 20261021'],
            ['Training-label audit', f"{summary['label_audit']['training_min_deg']:.2f}° to {summary['label_audit']['training_max_deg']:.2f}°; {summary['label_audit']['training_labels_outside_180']} labels outside ±180°"],
            ['Validation labels outside ±180°', str(summary['label_audit']['validation_labels_outside_180'])]]), '',
          'The original `2 × atan` output is bounded to ±180°; the new linear degree readout removes that representational limit. Validation labels are excluded from readout fitting and selection. Historical backbone exposure and driving-data provenance are not established. The display runtime does not load the Huber readout.', '',
          '### Bootstrap intervals', '', summary['bootstrap']['method'], '']
cirows = [['Method', 'MAE 95% interval (°)', 'RMSE 95% interval (°)']]
for name, intervals in summary['bootstrap']['intervals'].items():
    a, b = intervals['mae_95_ci_deg'], intervals['rmse_95_ci_deg']
    cirows.append([name, f'{a[0]:.3f}–{a[1]:.3f}', f'{b[0]:.3f}–{b[1]:.3f}'])
lines += [markdown_table(cirows), '']
paired = summary['bootstrap']['paired_mae_reduction_vs_original_95_ci_deg']
lines += [f'Paired MAE reduction versus the original RGB CNN: **{paired[0]:.3f}°–{paired[1]:.3f}°** (95% interval).', '',
          '### Evidence', '',
          '- [Selected-readout audit](../output/steering_audit/improved_readout_audit.json)',
          '- [Comparison summary](../output/steering_audit/comparison_summary.json)',
          '- [Frame-level predictions](../output/steering_audit/improved_validation_predictions.csv)',
          '- [Experiment report](../output/steering_audit/STEERING_EXPERIMENT.md)',
          '- [Research manuscript source](../research_paper/build_reference_style.py)', '',
          'Numeric precision follows the source tables and JSON records. Nothing here represents an independent rerun of the perception evaluators.', '']
(DOCS / 'RESEARCH_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'Copied {len(manifest)} figures; wrote detailed results and figure provenance.')
