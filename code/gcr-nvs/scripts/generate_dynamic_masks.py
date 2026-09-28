"""Generate SegFormer dynamic masks for a temporal pilot window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from gcr_nvs.datasets.leave_one_out import temporal_window
from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.dynamic_mask import SegFormerDynamicMasker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="train")
    parser.add_argument("--frame-id", type=int, default=11)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dynamic_masks"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--mask-scale", type=float, default=0.25)
    parser.add_argument("--dilation-pixels", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.mask_scale <= 1.0:
        raise ValueError("mask_scale must be in (0,1]")
    sequence_ids = [args.sequence]
    if args.split is not None:
        split = yaml.safe_load(args.split.read_text())
        sequence_ids = list(split[args.split_name])
        if not args.all_frames:
            raise ValueError("split mask generation requires --all-frames")
    masker = SegFormerDynamicMasker(
        device=args.device,
        dilation_pixels=args.dilation_pixels,
    )
    summaries = []
    for sequence_id in sequence_ids:
        sequence_dir = args.root.resolve() / sequence_id
        records = build_sequence_manifest(sequence_dir, compute_quality=False)
        if args.all_frames:
            selected_records = tuple(records)
        else:
            center_index = next(
                index for index, record in enumerate(records)
                if record.frame_id == args.frame_id
            )
            selected_records = temporal_window(
                records, center_index, 3, max_interval_seconds=0.75,
            )
            if not selected_records:
                raise RuntimeError("requested frame has no valid temporal window")
        rows = []
        for record in selected_records:
            output_dir = args.output_root / sequence_id / f"{record.frame_id:06d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            for camera in CAMERA_NAMES:
                image_path = sequence_dir / record.cameras[camera]
                with Image.open(image_path) as image:
                    output_size = (
                        max(1, int(round(image.width * args.mask_scale))),
                        max(1, int(round(image.height * args.mask_scale))),
                    )
                path = output_dir / f"{camera}.dynamic.npy"
                reused = False
                if path.exists() and not args.overwrite:
                    try:
                        existing = np.load(path, mmap_mode="r")
                        if existing.shape == (output_size[1], output_size[0]):
                            # Touch one value so truncated memmaps are rejected
                            # before a partial cache is reused.
                            _ = existing.flat[0]
                            mask = np.asarray(existing)
                            reused = True
                    except (OSError, ValueError, EOFError):
                        path.unlink(missing_ok=True)
                if not reused:
                    mask = masker.predict(image_path, output_size=output_size)
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as handle:
                        np.save(handle, mask)
                    temporary.replace(path)
                rows.append({
                    "frame_id": record.frame_id,
                    "camera": camera,
                    "shape": list(mask.shape),
                    "dynamic_ratio": float(mask.mean()),
                    "reused": reused,
                    "path": str(path),
                })
        summary = {
            "sequence_id": sequence_id,
            "frame_ids": [record.frame_id for record in selected_records],
            "mask_scale": args.mask_scale,
            "dilation_pixels": args.dilation_pixels,
            "masks": rows,
        }
        summary_path = args.output_root / sequence_id / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        summaries.append({
            "sequence_id": sequence_id,
            "frames": len(selected_records),
            "masks": len(rows),
            "reused": sum(bool(row["reused"]) for row in rows),
        })
        print(json.dumps(summaries[-1], ensure_ascii=False))
    print(json.dumps({"sequences": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
