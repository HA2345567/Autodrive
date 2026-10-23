# Reference-inspired paper

The output is `../output/pdf/AutoDrive_Research_Paper_Reference_Style.pdf`.
The input completed paper and the reference PDF remain unchanged.

This edition reflows the completed paper using the reference's A4 format,
mirrored margins, approximately 130 mm text measure, Computer Modern typography,
plain section headings, and horizontal table rules. Following the author's
layout correction, the body extends to 56 pt above the page bottom, and page
numbers are centered on the physical page at a 28 pt bottom baseline. Sections,
the abstract and references flow continuously to use the available page height.
Long tables can continue onto the following page with repeated column headings.
Text, links, rules, and Mermaid diagrams are black; photographic and plotted
figures retain their original colors. The reference's publication metadata and watermark are
not part of this paper.

The original section identifiers, figure/table numbers, citations, author
information, steering results, and first-page repository link are retained.
The proposed ablation matrix is replaced with six measured comparisons extracted
from saved notebook outputs. Its introduction, related error analysis, limitation
and future-work cross-reference are updated. These comparisons concern training
progress, repeated evaluation and validation/test splits; they do not establish
controlled component-ablation effects. `recorded_comparisons.json` records the
source notebook hashes, zero-based cell indices, printed metrics and deltas.
Table VI(b) is transposed for readability; Table VI(c) uses one metric per row.
The lane-label caption describes its original colors. These layout changes
increase the page count from the original 18-page completed paper. The current
page count is recorded in `reference_style_manifest.json`.

## Newly executed steering experiment

Section XVII records a new frozen-backbone steering readout experiment. Original
results remain in Table VI(c). The new study compares heads on all 9,082 original
validation frames, with hyperparameters selected within the training partition.
Squared-error and Huber candidates are retained in the experiment logs. The new
head reduces MAE relative to the original RGB checkpoint, but constant baselines
remain stronger and R² remains negative. Strong or closed-loop steering is not
claimed. The paper also distinguishes the reference DAVE-2 diagram from the
actual 1,595,511-parameter checkpoint and its preprocessing.

`../output/steering_audit/` contains the trained experimental readouts, complete
per-frame predictions, source hashes, method comparisons, bootstrap intervals,
and `STEERING_EXPERIMENT.md`. Run the audit, readout-training, robust-readout,
and summary scripts with the repository virtual environment before rebuilding
the PDF. The optional cached backbone features can be regenerated.

To rebuild with the Codex bundled runtimes:

1. Run `render_monochrome_diagrams.cjs` with Node and bundled Playwright.
2. Run `build_reference_style.py` with Python and ReportLab, PyMuPDF, pdfplumber,
   and Pillow.
3. Render the output with Poppler and review every page.

The `fonts/` files come from Matplotlib's official Computer Modern font
distribution. Unicode mathematical symbols use the local Times New Roman font.
The build records figure and section locations in `reference_style_manifest.json`.
