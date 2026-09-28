"""Resumable per-sequence batches for dynamic mask generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--mask-scale", type=float, default=0.25)
    parser.add_argument("--dilation-pixels", type=int, default=2)
    args = parser.parse_args()

    sequence_ids = list(yaml.safe_load(args.split.read_text())[args.split_name])
    completed = []
    for start in range(0, len(sequence_ids), args.batch_size):
        batch = sequence_ids[start:start + args.batch_size]
        for sequence in batch:
            summary = args.output_root / sequence / "summary.json"
            if summary.exists():
                completed.append({"sequence": sequence, "reused": True})
                continue
            command = [
                sys.executable, "scripts/generate_dynamic_masks.py", str(args.root),
                "--sequence", sequence, "--output-root", str(args.output_root),
                "--device", args.device, "--mask-scale", str(args.mask_scale),
                "--dilation-pixels", str(args.dilation_pixels), "--all-frames",
            ]
            subprocess.run(command, check=True)
            if not summary.exists():
                raise RuntimeError(f"mask generation did not produce summary: {summary}")
            completed.append({"sequence": sequence, "reused": False})
        (args.output_root / f"batch_{start // args.batch_size:04d}.json").write_text(
            json.dumps({"split": args.split_name, "sequences": completed[-len(batch):]}, ensure_ascii=False, indent=2) + "\n",
        )
        print(json.dumps({"batch_start": start, "batch_size": len(batch), "completed": len(completed)}, ensure_ascii=False), flush=True)
    print(json.dumps({"split": args.split_name, "sequence_count": len(completed), "output_root": str(args.output_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
