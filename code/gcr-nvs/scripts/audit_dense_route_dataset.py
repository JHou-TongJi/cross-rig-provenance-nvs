"""Audit the deduplicated dense-route manifest without generating model caches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.pcd import read_pcd


def _quantile(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p01": None, "p50": None, "p99": None, "max": None}
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values_array.min()),
        "p01": float(np.quantile(values_array, 0.01)),
        "p50": float(np.quantile(values_array, 0.50)),
        "p99": float(np.quantile(values_array, 0.99)),
        "max": float(values_array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pcd-samples-per-sequence", type=int, default=5)
    parser.add_argument("--full-decode", action="store_true")
    args = parser.parse_args()

    records = [
        SimpleNamespace(**json.loads(line))
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    failures: list[dict[str, str | int]] = []
    dimensions: dict[str, Counter[tuple[int, int]]] = {name: Counter() for name in CAMERA_NAMES}
    offsets: dict[str, list[float]] = {name: [] for name in CAMERA_NAMES}
    all_offsets: list[float] = []
    sequence_frames: dict[str, int] = Counter()
    pcd_rows: list[dict] = []
    pcd_seen: dict[str, int] = Counter()
    sequence_positions: dict[str, int] = Counter()
    sequence_lengths = Counter(record.sequence_id for record in records)
    image_count = 0
    decoded_count = 0
    quality_samples: list[dict] = []

    for record in records:
        sequence_frames[record.sequence_id] += 1
        source_root = Path(record.source_root)
        sequence_dir = source_root / record.sequence_id
        config_path = sequence_dir / record.camera_config
        if not config_path.exists():
            failures.append({"type": "missing_config", "sequence": record.sequence_id, "frame": record.frame_id, "path": str(config_path)})
        lidar_path = sequence_dir / record.lidar
        if not lidar_path.exists():
            failures.append({"type": "missing_lidar", "sequence": record.sequence_id, "frame": record.frame_id, "path": str(lidar_path)})
        else:
            sequence_position = sequence_positions[record.sequence_id]
            if pcd_seen[record.sequence_id] < args.pcd_samples_per_sequence and (
                sequence_position in {0, sequence_lengths[record.sequence_id] // 2, sequence_lengths[record.sequence_id] - 1}
            ):
                try:
                    points = read_pcd(lidar_path)
                    finite_xyz = np.isfinite(points[:, :3]).all(axis=1) if len(points) else np.zeros(0, dtype=bool)
                    time_values = points[:, 5] if points.shape[1] >= 6 else np.empty(0, dtype=np.float32)
                    time_values = time_values[np.isfinite(time_values)]
                    pcd_rows.append({
                        "sequence": record.sequence_id,
                        "frame_id": int(record.frame_id),
                        "point_count": int(len(points)),
                        "fields": int(points.shape[1]),
                        "finite_xyz_ratio": float(finite_xyz.mean()) if len(finite_xyz) else 0.0,
                        "time_min": float(time_values.min()) if len(time_values) else None,
                        "time_max": float(time_values.max()) if len(time_values) else None,
                        "time_resets": int(np.count_nonzero(np.diff(time_values) < -1e-3)) if len(time_values) > 1 else 0,
                    })
                except Exception as error:
                    failures.append({"type": "pcd_decode", "sequence": record.sequence_id, "frame": record.frame_id, "error": repr(error)})
                pcd_seen[record.sequence_id] += 1
        sequence_positions[record.sequence_id] += 1
        if record.camera_timestamps is None or record.lidar_timestamp is None:
            failures.append({"type": "missing_timestamp", "sequence": record.sequence_id, "frame": record.frame_id})
        else:
            for camera in CAMERA_NAMES:
                offset_ms = (float(record.camera_timestamps[camera]) - float(record.lidar_timestamp)) * 1000.0
                offsets[camera].append(offset_ms)
                all_offsets.append(offset_ms)
                image_path = sequence_dir / record.cameras[camera]
                image_count += 1
                if not image_path.exists():
                    failures.append({"type": "missing_image", "sequence": record.sequence_id, "frame": record.frame_id, "camera": camera, "path": str(image_path)})
                    continue
                try:
                    with Image.open(image_path) as image:
                        dimensions[camera][(int(image.width), int(image.height))] += 1
                        if args.full_decode:
                            image.convert("RGB").load()
                            decoded_count += 1
                except Exception as error:
                    failures.append({"type": "image_decode", "sequence": record.sequence_id, "frame": record.frame_id, "camera": camera, "error": repr(error)})

    report = {
        "manifest": str(args.manifest.resolve()),
        "sequence_count": len(sequence_frames),
        "frame_count": len(records),
        "image_count": image_count,
        "decoded_image_count": decoded_count if args.full_decode else None,
        "frames_per_sequence": Counter(sequence_frames.values()),
        "dimensions": {camera: {f"{width}x{height}": count for (width, height), count in sorted(values.items())} for camera, values in dimensions.items()},
        "camera_lidar_offset_ms": {camera: _quantile(values) for camera, values in offsets.items()},
        "camera_lidar_offset_ms_all": _quantile(all_offsets),
        "pcd_samples": pcd_rows,
        "pcd_sample_summary": {
            "point_count": _quantile([float(row["point_count"]) for row in pcd_rows]),
            "fields": sorted({row["fields"] for row in pcd_rows}),
            "finite_xyz_min": min((row["finite_xyz_ratio"] for row in pcd_rows), default=None),
            "time_resets_total": sum(row["time_resets"] for row in pcd_rows),
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sequence_count": report["sequence_count"],
        "frame_count": report["frame_count"],
        "image_count": report["image_count"],
        "failures": len(failures),
        "camera_lidar_offset_ms_all": report["camera_lidar_offset_ms_all"],
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
