"""Audit every sequence used by the Unified 3D Field pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.pcd import read_pcd


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--split", type=Path, default=Path("configs/splits/pilot_15seq_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audits/unified_3d_field_15seq.json"))
    parser.add_argument("--pcd-samples-per-sequence", type=int, default=5)
    args = parser.parse_args()

    root = args.root.resolve()
    split = yaml.safe_load(args.split.read_text())
    seen: set[str] = set()
    rows = []
    all_offsets_ms: list[float] = []
    total_images = 0
    total_frames = 0
    failures: list[str] = []
    for split_name in ("train", "val", "test"):
        for sequence_id in split[split_name]:
            if sequence_id in seen:
                failures.append(f"duplicate sequence across splits: {sequence_id}")
            seen.add(sequence_id)
            sequence_dir = root / sequence_id
            records = build_sequence_manifest(sequence_dir, compute_quality=False)
            total_frames += len(records)
            total_images += len(records) * len(CAMERA_NAMES)
            offsets = []
            dimensions: dict[str, set[tuple[int, int]]] = {name: set() for name in CAMERA_NAMES}
            missing = []
            for record in records:
                lidar_path = sequence_dir / record.lidar
                if not lidar_path.exists():
                    missing.append(str(lidar_path))
                for camera in CAMERA_NAMES:
                    image_path = sequence_dir / record.cameras[camera]
                    if not image_path.exists():
                        missing.append(str(image_path))
                        continue
                    with Image.open(image_path) as image:
                        dimensions[camera].add((image.width, image.height))
                    offset = (record.camera_timestamps[camera] - record.lidar_timestamp) * 1000.0
                    offsets.append(float(offset))
                    all_offsets_ms.append(float(offset))

            pcd_indices = np.linspace(
                0, len(records) - 1, min(args.pcd_samples_per_sequence, len(records)),
            ).round().astype(np.int64)
            pcd_rows = []
            for index in np.unique(pcd_indices):
                record = records[int(index)]
                points = read_pcd(sequence_dir / record.lidar)
                point_time = points[:, 5] if points.shape[1] >= 6 else np.empty(0)
                finite_time = point_time[np.isfinite(point_time)]
                resets = int(np.count_nonzero(np.diff(finite_time) < -1e-3)) if len(finite_time) else 0
                pcd_rows.append({
                    "frame_id": record.frame_id,
                    "point_count": int(len(points)),
                    "fields": int(points.shape[1]),
                    "time_min": float(finite_time.min()) if len(finite_time) else None,
                    "time_max": float(finite_time.max()) if len(finite_time) else None,
                    "time_resets": resets,
                    "finite_xyz_ratio": float(np.isfinite(points[:, :3]).all(axis=1).mean()),
                })
            if missing:
                failures.extend(missing)
            rows.append({
                "split": split_name,
                "sequence_id": sequence_id,
                "frames": len(records),
                "images": len(records) * len(CAMERA_NAMES),
                "camera_lidar_offset_ms": quantiles(offsets),
                "dimensions": {name: sorted(map(list, values)) for name, values in dimensions.items()},
                "sampled_pcd": pcd_rows,
                "missing_files": len(missing),
            })

    report = {
        "split_file": str(args.split),
        "sequence_count": len(seen),
        "frame_count": total_frames,
        "image_count": total_images,
        "camera_lidar_offset_ms": quantiles(all_offsets_ms),
        "failures": failures,
        "deskew_policy": {
            "enabled": False,
            "reason": "LIDAR_CONCAT point-time has multiple resets; sensor block boundaries and reference convention must be decoded before per-point deskew",
            "current_safe_action": "register complete scans temporally, preserve current-frame exact points, and never treat unverified time as an absolute timestamp",
        },
        "sequences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "sequence_count": report["sequence_count"],
        "frame_count": total_frames,
        "image_count": total_images,
        "camera_lidar_offset_ms": report["camera_lidar_offset_ms"],
        "failures": len(failures),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
