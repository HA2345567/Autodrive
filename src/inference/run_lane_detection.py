"""
Run the lane detection model over every frame in a driving dataset and
save lane-overlay images into a single merged output folder.

Usage:
    python scripts/run_lane_overlay_merge.py \
        --source data/driving_dataset \
        --model saved_models/lane_detection/best.pt \
        --output data/driving_dataset_merged \
        --conf 0.20

Each output file keeps the same name as its source frame (e.g. "0.jpg"),
so the merged folder lines up 1:1 with the original dataset and can be
dropped straight into the simulator's data_path.
"""

import os
import sys
import argparse
import glob

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.inference.lane_detector import LaneDetector, resolve_lane_model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lane detection over a dataset and save merged overlay images."
    )
    parser.add_argument(
        "--source", type=str, default="data/driving_dataset",
        help="Directory of source frames (default: 'data/driving_dataset')."
    )
    parser.add_argument(
        "--model", type=str, default=resolve_lane_model_path(),
        help="Path to lane detection weights (default: auto-detected from saved_models)."
    )
    parser.add_argument(
        "--output", type=str, default="data/driving_dataset_merged",
        help="Directory to write overlay images into (default: 'data/driving_dataset_merged')."
    )
    parser.add_argument(
        "--conf", type=float, default=0.20,
        help="Lane confidence threshold (default: 0.20)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay blend strength, 0-1 (default: 0.5)."
    )
    # BUG FIX: the original used action="store_true", default=True, which made
    # the flag impossible to turn OFF from the command line (store_true can only
    # push the value to True, and it already defaults to True). Switching to
    # BooleanOptionalAction gives us a real --no-fill-corridor switch.
    parser.add_argument(
        "--fill-corridor", dest="fill_corridor",
        action=argparse.BooleanOptionalAction, default=True,
        help="Fill drivable corridor polygon between lanes (default: True). "
             "Pass --no-fill-corridor to disable."
    )
    parser.add_argument(
        "--ext", type=str, default="jpg",
        help="Source image extension to match, without dot (default: 'jpg')."
    )
    return parser.parse_args()


def _sorted_frames(source_dir: str, ext: str):
    """Sort numerically ('2.jpg' before '10.jpg') when filenames are plain
    integers, falling back to alphabetical order otherwise."""
    paths = glob.glob(os.path.join(source_dir, f"*.{ext}"))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(paths, key=sort_key)


def main():
    args = parse_args()

    if not os.path.isdir(args.source):
        print(f"[Merge] Error: source directory '{args.source}' does not exist.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    frames = _sorted_frames(args.source, args.ext)
    if not frames:
        print(f"[Merge] No .{args.ext} frames found in '{args.source}'.")
        sys.exit(1)

    # Flag a silent fallback to the generic (non-lane) pretrained model before
    # doing any work, so the user isn't left guessing why output looks wrong.
    if os.path.basename(args.model) == "yolo11s-seg.pt" and not os.path.dirname(args.model):
        print(
            "[Merge] Warning: no trained lane-detection weights were found under "
            "'saved_models/'. Falling back to generic 'yolo11s-seg.pt' "
            "(COCO-pretrained) -- lane masks will likely be incorrect. "
            "Pass --model explicitly to point at your trained weights."
        )

    print("=" * 70)
    print("Lane Overlay Merge")
    print(f"  Source frames : {len(frames)} found in '{args.source}'")
    print(f"  Output dir    : {args.output}")
    print(f"  Model path    : {args.model}")
    print(f"  Confidence    : {args.conf}")
    print(f"  Fill corridor : {args.fill_corridor}")
    print("=" * 70)

    detector = LaneDetector(model_name=args.model, conf_threshold=args.conf)

    processed, skipped = 0, 0
    for path in frames:
        image = cv2.imread(path)
        if image is None:
            print(f"[Merge] Warning: could not read '{path}', skipping.")
            skipped += 1
            continue

        merged, lane_info = detector.detect(
            image,
            conf_threshold=args.conf,
            fill_corridor=args.fill_corridor,
        )

        if merged is None:
            print(f"[Merge] Warning: detector returned no output for '{path}', skipping.")
            skipped += 1
            continue

        out_path = os.path.join(args.output, os.path.basename(path))
        cv2.imwrite(out_path, merged)
        processed += 1

        if processed % 100 == 0:
            print(f"[Merge] Processed {processed}/{len(frames)} frames...")

    print("-" * 70)
    print(f"[Merge] Done. {processed} frame(s) written to '{args.output}', {skipped} skipped.")


if __name__ == "__main__":
    main()