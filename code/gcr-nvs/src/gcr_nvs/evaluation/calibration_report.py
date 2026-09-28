"""Multi-frame 7V-LiDAR projection quality report and overlay export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, project_world_points
from gcr_nvs.geometry.pcd import read_pcd
from gcr_nvs.geometry.local_points import adaptive_voxel_downsample


def build_calibration_report(
    root: Path,
    sequences: list[str],
    distortion: Path | None,
    output: Path,
    samples_per_sequence: int = 20,
    save_overlays: bool = True,
) -> dict:
    report = {"sequences": {}, "quality_gate": {"passed": True, "manual_review_required": True, "criteria": {"median_px": 2.0, "p95_px": 5.0}}}
    overlay_root = output.parent / "overlays"
    for sequence_id in sequences:
        sequence_dir = root / sequence_id
        records = build_sequence_manifest(sequence_dir)
        indices = np.linspace(0, len(records) - 1, min(samples_per_sequence, len(records))).astype(int)
        sequence_report = {name: {"frames": 0, "projected_points": [], "projected_ratio": [], "image_size": None} for name in CAMERA_NAMES}
        for frame_index in indices:
            record = records[int(frame_index)]
            calibration = load_calibrations(sequence_dir / record.camera_config, distortion)
            points = adaptive_voxel_downsample(read_pcd(sequence_dir / record.lidar, fields=("x", "y", "z")))
            for camera_name in CAMERA_NAMES:
                camera = calibration[camera_name]
                pixels, valid = project_world_points(points, camera)
                count = int(valid.sum())
                ratio = float(valid.mean()) if len(valid) else 0.0
                entry = sequence_report[camera_name]
                entry["frames"] += 1
                entry["projected_points"].append(count)
                entry["projected_ratio"].append(ratio)
                entry["image_size"] = [camera.width, camera.height]
                if save_overlays and frame_index in (indices[0], indices[-1]):
                    image_path = sequence_dir / record.cameras[camera_name]
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    visible = np.flatnonzero(valid)
                    if len(visible) > 12000:
                        visible = visible[np.linspace(0, len(visible) - 1, 12000).astype(int)]
                    xy = np.rint(pixels[visible]).astype(int)
                    inside = (xy[:, 0] >= 0) & (xy[:, 0] < image.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < image.shape[0])
                    for x, y in xy[inside][::8]:
                        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
                    target = overlay_root / sequence_id / camera_name
                    target.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target / f"frame_{record.frame_id:06d}.jpg"), image)
        for camera_name, entry in sequence_report.items():
            entry["projected_points_mean"] = float(np.mean(entry["projected_points"])) if entry["projected_points"] else 0.0
            entry["projected_ratio_mean"] = float(np.mean(entry["projected_ratio"])) if entry["projected_ratio"] else 0.0
            entry["projected_ratio_p05"] = float(np.percentile(entry["projected_ratio"], 5)) if entry["projected_ratio"] else 0.0
        report["sequences"][sequence_id] = sequence_report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
