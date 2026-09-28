"""Estimate and persist local ego-motion quality reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gcr_nvs.datasets.manifest import build_sequence_manifest
from gcr_nvs.geometry.icp import estimate_icp
from gcr_nvs.geometry.local_points import adaptive_voxel_downsample
from gcr_nvs.geometry.pcd import read_pcd


def build_pose_report(root: Path, sequences: list[str], output: Path) -> dict:
    report = {"provider": "local_point_to_point_icp", "sequences": {}}
    for sequence_id in sequences:
        sequence_dir = root / sequence_id
        records = build_sequence_manifest(sequence_dir)
        previous = None
        rows = []
        for record in records:
            points = adaptive_voxel_downsample(read_pcd(sequence_dir / record.lidar, fields=("x", "y", "z")))
            if previous is None:
                rows.append({"frame_id": record.frame_id, "timestamp": record.timestamp, "relative_transform": np.eye(4).tolist(), "valid": True, "rmse_m": 0.0, "inlier_ratio": 1.0, "failure_flag": False})
            else:
                transform, quality = estimate_icp(
                    previous,
                    points,
                    max_points=40_000,
                    iterations=30,
                    max_correspondence_m=1.0,
                    trim_ratio=0.50,
                )
                rows.append({"frame_id": record.frame_id, "timestamp": record.timestamp, "relative_transform": transform.tolist(), "valid": bool(quality["valid"]), "geometry_gate_passed": bool(quality.get("geometry_gate_passed", False)), "rmse_m": float(quality["rmse_m"]), "inlier_ratio": float(quality["inlier_ratio"]), "failure_flag": not bool(quality["valid"]), "quality_gate_failure": not bool(quality.get("geometry_gate_passed", False))})
            previous = points
        valid = [row for row in rows if row["valid"]]
        gate_rows = [row for row in rows[1:] if np.isfinite(row["rmse_m"])]
        gate_passed = bool(gate_rows) and all(row["geometry_gate_passed"] for row in gate_rows)
        report["sequences"][sequence_id] = {
            "frames": rows,
            "valid_ratio": len(valid) / max(1, len(rows)),
            "rmse_median": float(np.median([row["rmse_m"] for row in valid])) if valid else float("inf"),
            "rmse_p95": float(np.percentile([row["rmse_m"] for row in valid], 95)) if valid else float("inf"),
            "failure_count": sum(row["failure_flag"] for row in rows),
            "geometry_gate_passed": gate_passed,
            "temporal_training_allowed": gate_passed,
            "quality_gate": {"inlier_ratio_min": 0.30, "rmse_m_max": 0.15},
        }
    report["temporal_training_allowed"] = bool(report["sequences"]) and all(item["temporal_training_allowed"] for item in report["sequences"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
