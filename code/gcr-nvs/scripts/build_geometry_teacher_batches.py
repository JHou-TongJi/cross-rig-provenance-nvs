"""Resumable dense-route temporal teacher generation by sequence."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _complete_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        expected = max(0, int(payload["record_count"]) - 1)
        rows = payload.get("frames", [])
        return (
            payload.get("contract") == "dense_route_v1"
            and payload.get("temporal_mode") == "previous"
            and int(payload.get("temporal_window", 0)) == 1
            and int(payload.get("saved_centers", -1)) == expected
            and int(payload.get("failed_centers", 1)) == 0
            and len(rows) == expected
            and all("temporal_points" in row and "registration" in row for row in rows)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="train")
    parser.add_argument("--dynamic-mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_root / ".dense_route_teacher.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another dense-route teacher batch is already running: {args.output_root}") from error

    split = yaml.safe_load(args.split.read_text())
    sequence_ids = list(split[args.split_name])
    completed = []
    for sequence in sequence_ids:
        summary = args.dynamic_mask_root / sequence / "summary.json"
        output_summary = args.output_root / sequence / "sequence_summary.json"
        if not summary.exists():
            continue
        if _complete_summary(output_summary):
            completed.append({"sequence": sequence, "reused": True})
            continue
        command = [
            sys.executable, "scripts/build_geometry_teacher_sequence.py", str(args.root),
            "--sequence", sequence, "--dynamic-mask-root", str(args.dynamic_mask_root),
            "--distortion", "camera_intric.yaml", "--output-root", str(args.output_root),
            "--temporal-radius", "1", "--temporal-mode", "previous",
            "--max-interval-seconds", "0.75", "--temporal-rmse-gate-m", "0.20",
            "--temporal-inlier-gate", "0.40",
        ]
        if args.compress:
            command.append("--compress")
        # An incomplete/stale summary means artifacts may have been generated
        # by the old temporal contract.  Rebuild them so the per-frame
        # registration metadata and sequence summary are consistent; merely
        # rewriting a summary from reused files is unsafe.
        command.append("--overwrite")
        subprocess.run(command, check=True)
        if not output_summary.exists():
            raise RuntimeError(f"teacher generation did not produce summary: {output_summary}")
        completed.append({"sequence": sequence, "reused": False})
        if args.max_sequences is not None and len(completed) >= args.max_sequences:
            break
        if len(completed) % max(1, args.batch_size) == 0:
            batch_index = len(completed) // max(1, args.batch_size) - 1
            (args.output_root / f"batch_{batch_index:04d}.json").write_text(
                json.dumps({"split": args.split_name, "sequences": completed[-args.batch_size:]}, ensure_ascii=False, indent=2) + "\n",
            )
            print(json.dumps({"completed": len(completed), "batch": batch_index}, ensure_ascii=False), flush=True)
    print(json.dumps({"split": args.split_name, "completed": completed, "output_root": str(args.output_root)}, ensure_ascii=False, indent=2))
    fcntl.flock(lock_handle, fcntl.LOCK_UN)
    lock_handle.close()


if __name__ == "__main__":
    main()
