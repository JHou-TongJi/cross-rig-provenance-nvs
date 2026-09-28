"""Resumable DA3 cache generation for a sequence split.

The estimator is loaded once per process.  Frame sampling is explicit so a
large dataset cannot accidentally turn into an unbounded GPU job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gcr_nvs.datasets.manifest import build_sequence_manifest
from gcr_nvs.evaluation.build_da3_depth_cache import build_frame
from gcr_nvs.models.monocular_depth import DepthAnything3Estimator


def _complete_cache(manifest_path: Path) -> bool:
    try:
        payload = json.loads(manifest_path.read_text())
        rows = payload["rows"]
        if len(rows) != len(payload["cameras"]):
            return False
        required = (
            "rgb_rectified.jpg", "da3_depth.npy", "da3_confidence.npy",
            "aligned_dense_depth.npy", "fused_depth.npy",
            "fused_confidence.npy", "lidar_sparse_depth.npy",
            "lidar_validity.npy", "lidar_anchor_depth.npy", "report.json",
        )
        return all((manifest_path.parent / row["camera"] / name).exists()
                   for row in rows for name in required)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/heqing/models/depth-anything-3/DA3Metric-Large"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/heqing/Depth-Anything-3"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--sequence", action="append", dest="selected_sequences")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    split = yaml.safe_load(args.split.read_text())
    sequence_ids = list(split[args.split_name])
    if args.selected_sequences:
        allowed = set(sequence_ids)
        unknown = sorted(set(args.selected_sequences) - allowed)
        if unknown:
            raise ValueError(f"selected sequences are not in split {args.split_name}: {unknown}")
        sequence_ids = [name for name in sequence_ids if name in set(args.selected_sequences)]
    if args.max_sequences is not None:
        sequence_ids = sequence_ids[:args.max_sequences]
    estimator = DepthAnything3Estimator(
        model_dir=args.model_dir, source_root=args.source_root, device=args.device,
    )
    cameras = tuple(args.cameras or ())
    rows = []
    for sequence in sequence_ids:
        records = build_sequence_manifest(args.root / sequence, compute_quality=False)
        selected = records[::args.stride]
        # Always include the final frame so a short sequence is represented.
        if records and selected[-1].frame_id != records[-1].frame_id:
            selected = (*selected, records[-1])
        for record in selected:
            manifest_path = args.output_root / sequence / f"{record.frame_id:06d}" / "manifest.json"
            cache_complete = _complete_cache(manifest_path)
            if cache_complete and not args.overwrite:
                rows.append({"sequence": sequence, "frame_id": record.frame_id, "reused": True})
                continue
            chosen = cameras or tuple(record.cameras)
            build_frame(
                root=args.root, sequence=sequence, frame_id=record.frame_id,
                cameras=chosen, output_root=args.output_root,
                distortion=args.distortion, model_dir=args.model_dir,
                source_root=args.source_root, device=args.device,
                output_size=(args.width, args.height),
                overwrite=args.overwrite or not cache_complete,
                estimator=estimator,
            )
            rows.append({"sequence": sequence, "frame_id": record.frame_id, "reused": False})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {
        "contract": "dense_route_v1",
        "split": args.split_name,
        "stride": args.stride,
        "sequences": len(sequence_ids),
        "frames": len(rows),
        "rows": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"{args.split.stem}_{args.split_name}_cache_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: report[k] for k in ("split", "stride", "sequences", "frames")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
