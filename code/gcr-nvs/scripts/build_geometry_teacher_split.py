"""Build resumable safe temporal LiDAR teacher artifacts for a sequence split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--split", type=Path, default=Path("configs/splits/pilot_15seq_v1.yaml"))
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="train")
    parser.add_argument("--dynamic-mask-root", type=Path, default=Path("outputs/dynamic_masks"))
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temporal-radius", type=int, default=3)
    parser.add_argument(
        "--temporal-mode", choices=("previous", "next", "both"), default="previous",
        help="dense_route_v1 uses one previous frame; both is legacy compatibility",
    )
    parser.add_argument("--max-interval-seconds", type=float, default=0.75)
    parser.add_argument("--temporal-rmse-gate-m", type=float, default=0.20)
    parser.add_argument("--temporal-inlier-gate", type=float, default=0.40)
    args = parser.parse_args()

    split = yaml.safe_load(args.split.read_text())
    summaries = []
    for sequence_id in split[args.split_name]:
        command = [
            sys.executable,
            "scripts/build_geometry_teacher_sequence.py",
            str(args.root),
            "--sequence", sequence_id,
            "--dynamic-mask-root", str(args.dynamic_mask_root),
            "--distortion", str(args.distortion),
            "--output-root", str(args.output_root),
            "--temporal-radius", str(args.temporal_radius),
            "--temporal-mode", args.temporal_mode,
            "--max-interval-seconds", str(args.max_interval_seconds),
            "--temporal-rmse-gate-m", str(args.temporal_rmse_gate_m),
            "--temporal-inlier-gate", str(args.temporal_inlier_gate),
        ]
        if args.compress:
            command.append("--compress")
        if args.overwrite:
            command.append("--overwrite")
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        summary_path = args.output_root / sequence_id / "sequence_summary.json"
        summary = json.loads(summary_path.read_text())
        row = {
            "sequence_id": sequence_id,
            "record_count": summary["record_count"],
            "saved_centers": summary["saved_centers"],
            "failed_centers": summary["failed_centers"],
            "mean_temporal_points": summary["mean_temporal_points"],
        }
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "split": str(args.split),
        "split_name": args.split_name,
        "sequences": summaries,
        "saved_centers": sum(row["saved_centers"] for row in summaries),
        "failed_centers": sum(row["failed_centers"] for row in summaries),
    }
    report_path = args.output_root / f"{args.split.stem}_{args.split_name}_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"report": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
