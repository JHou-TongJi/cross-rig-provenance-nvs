"""Build true current/temporal LiDAR maps aligned to rectified RGB caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.geometry.lidar_depth_layers import project_lidar_layer
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT


def _rectified_calibration(report_path: Path) -> CameraCalibration:
    payload = json.loads(report_path.read_text())
    return CameraCalibration(
        payload["camera"], np.asarray(payload["intrinsic_newK"], dtype=np.float64),
        np.asarray(payload["distortion_after_rectification"], dtype=np.float64),
        np.asarray(payload["external_world_to_camera"], dtype=np.float64),
        int(payload["output_size"][0]), int(payload["output_size"][1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--rgb-depth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append")
    args = parser.parse_args()
    sequences = args.sequence or sorted(p.name for p in args.teacher_root.iterdir() if p.is_dir())
    rows = []
    for sequence in sequences:
        for teacher_path in sorted((args.teacher_root / sequence).glob("*.npz")):
            frame_id = int(teacher_path.stem)
            with np.load(teacher_path) as payload:
                points = payload["points"].astype(np.float32)
                source_kind = payload["source_kind"].astype(np.uint8)
                confidence = payload["confidence"].astype(np.float32)
            current = points[source_kind == EXACT_CURRENT]
            temporal_mask = source_kind != EXACT_CURRENT
            temporal = points[temporal_mask]
            temporal_confidence = confidence[temporal_mask]
            for camera in CAMERA_NAMES:
                source_dir = args.rgb_depth_root / sequence / f"{frame_id:06d}" / camera
                report = source_dir / "report.json"
                if not report.exists():
                    continue
                calibration = _rectified_calibration(report)
                width, height = calibration.width, calibration.height
                current_layer = project_lidar_layer(current, calibration, width, height)
                temporal_layer = project_lidar_layer(
                    temporal, calibration, width, height, temporal_confidence,
                )
                destination = args.output_root / sequence / f"{frame_id:06d}" / camera
                destination.mkdir(parents=True, exist_ok=True)
                for prefix, layer in (("current", current_layer), ("temporal", temporal_layer)):
                    np.save(destination / f"lidar_{prefix}_z.npy", layer.z_m)
                    np.save(destination / f"lidar_{prefix}_range.npy", layer.range_m)
                    np.save(destination / f"lidar_{prefix}_validity.npy", layer.valid)
                    np.save(destination / f"lidar_{prefix}_point_index.npy", layer.point_index)
                    np.save(destination / f"lidar_{prefix}_confidence.npy", layer.confidence)
                metadata = {
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "camera": camera,
                    "resolution": [width, height],
                    "contract": "true_lidar_layers_separate_from_da3",
                    "current_points": int(len(current)),
                    "temporal_points": int(len(temporal)),
                    "current_coverage": float(current_layer.valid.mean()),
                    "temporal_coverage": float(temporal_layer.valid.mean()),
                    "overlap_coverage": float((current_layer.valid & temporal_layer.valid).mean()),
                    "rgb_or_da3_used_for_projection": False,
                    "da3_role": "independent_dense_rgb_aligned_candidate",
                }
                (destination / "lidar_layers_report.json").write_text(json.dumps(metadata, indent=2))
                rows.append(metadata)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "contract": "true_lidar_layers_separate_from_da3",
        "sequences": sequences,
        "rows": len(rows),
        "mean_current_coverage": float(np.mean([r["current_coverage"] for r in rows])) if rows else 0.0,
        "mean_temporal_coverage": float(np.mean([r["temporal_coverage"] for r in rows])) if rows else 0.0,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

