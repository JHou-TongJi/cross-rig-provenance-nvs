"""Run a real three-frame KISS-ICP and temporal LiDAR teacher diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcr_nvs.datasets.leave_one_out import temporal_window
from gcr_nvs.datasets.manifest import build_sequence_manifest, sensor_timestamp
from gcr_nvs.geometry.icp import alignment_quality, estimate_kiss_sequence_poses
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.dynamic_mask import dynamic_points_from_camera_masks
from gcr_nvs.geometry.local_points import adaptive_voxel_downsample
from gcr_nvs.geometry.pcd import read_pcd
from gcr_nvs.geometry.temporal_lidar import TemporalLidarTeacher, TemporalNeighbor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-26-11-44-28")
    parser.add_argument("--frame-id", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/geometry_diagnostics/temporal_teacher_pilot.json"),
    )
    parser.add_argument("--dynamic-mask-root", type=Path)
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--teacher-output", type=Path)
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--max-interval-seconds", type=float, default=0.75)
    parser.add_argument("--max-time-offset-seconds", type=float)
    parser.add_argument("--temporal-rmse-gate-m", type=float, default=0.20)
    parser.add_argument("--temporal-inlier-gate", type=float, default=0.40)
    args = parser.parse_args()

    sequence_dir = args.root.resolve() / args.sequence
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    center_index = next(
        index for index, record in enumerate(records)
        if record.frame_id == args.frame_id
    )
    if args.temporal_radius < 1:
        raise ValueError("temporal-radius must be at least one")
    window = temporal_window(
        records, center_index, 2 * args.temporal_radius + 1,
        max_interval_seconds=args.max_interval_seconds,
    )
    if not window:
        raise RuntimeError("requested frame has no valid three-frame temporal window")
    raw_frames = [read_pcd(sequence_dir / record.lidar) for record in window]
    pose_frames = [adaptive_voxel_downsample(frame) for frame in raw_frames]
    poses = estimate_kiss_sequence_poses(pose_frames)
    center_window_index = args.temporal_radius
    center_pose_inverse = np.linalg.inv(poses[center_window_index])
    center_points = pose_frames[center_window_index]
    neighbors = []
    quality_rows = []
    for index in range(len(window)):
        if index == args.temporal_radius:
            continue
        transform = center_pose_inverse @ poses[index]
        quality = alignment_quality(
            pose_frames[index], center_points, transform,
        )
        temporal_gate = bool(
            quality["rmse_m"] <= args.temporal_rmse_gate_m
            and quality["inlier_ratio"] >= args.temporal_inlier_gate
        )
        alignment_confidence = (
            np.exp(-float(quality["rmse_m"]) / 0.15)
            * min(float(quality["inlier_ratio"]) / 0.30, 1.0)
            if np.isfinite(float(quality["rmse_m"])) else 0.0
        )
        dynamic_mask = None
        if args.dynamic_mask_root is not None:
            mask_dir = (
                args.dynamic_mask_root / args.sequence
                / f"{window[index].frame_id:06d}"
            )
            mask_paths = {
                name: mask_dir / f"{name}.dynamic.npy"
                for name in window[index].cameras
            }
            missing = [str(path) for path in mask_paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(f"missing dynamic masks: {missing}")
            masks = {name: np.load(path) for name, path in mask_paths.items()}
            calibrations = load_calibrations(
                sequence_dir / window[index].camera_config,
                args.distortion,
            )
            dynamic_mask = dynamic_points_from_camera_masks(
                pose_frames[index], calibrations, masks,
            )
        neighbors.append(TemporalNeighbor(
            points=pose_frames[index],
            timestamp=sensor_timestamp(window[index]),
            transform_to_center=transform,
            alignment_confidence=float(alignment_confidence),
            alignment_valid=temporal_gate,
            dynamic_mask=dynamic_mask,
        ))
        quality_rows.append({
            "frame_id": window[index].frame_id,
            "time_offset_s": (
                sensor_timestamp(window[index])
                - sensor_timestamp(window[center_window_index])
            ),
            "rmse_m": quality["rmse_m"],
            "inlier_ratio": quality["inlier_ratio"],
            "gate_passed": temporal_gate,
            "strict_gate_passed": quality["valid"],
            "alignment_confidence": alignment_confidence,
            "dynamic_point_ratio": (
                float(dynamic_mask.mean()) if dynamic_mask is not None else None
            ),
            "transform_to_center": transform.tolist(),
        })
    max_time_offset = args.max_time_offset_seconds
    if max_time_offset is None:
        max_time_offset = float(args.temporal_radius * args.max_interval_seconds)
    potential = TemporalLidarTeacher(max_time_offset_s=max_time_offset).fuse(
        center_points,
        sensor_timestamp(window[center_window_index]),
        neighbors,
    )
    time_values = raw_frames[center_window_index][:, 5]
    dynamic_masks_available = args.dynamic_mask_root is not None
    report = {
        "sequence_id": args.sequence,
        "requested_frame_id": args.frame_id,
        "time_ordered_frame_ids": [record.frame_id for record in window],
        "lidar_timestamps": [sensor_timestamp(record) for record in window],
        "pcd_time": {
            "minimum_raw": float(np.nanmin(time_values)),
            "maximum_raw": float(np.nanmax(time_values)),
            "likely_unit": "milliseconds",
            "deskew_enabled": False,
            "reason": "scan reference time and concatenated-sensor reset semantics are not verified",
        },
        "registration": quality_rows,
        "teacher": {
            "exact_current_points": int(potential.exact_mask.sum()),
            "potential_temporal_filled_points": int(potential.temporal_mask.sum()),
            "total_points": int(len(potential.points)),
            "safe_for_distillation": bool(
                dynamic_masks_available
                and all(row["gate_passed"] for row in quality_rows)
            ),
            "reason": (
                "alignment and dynamic-mask gates passed"
                if dynamic_masks_available
                and all(row["gate_passed"] for row in quality_rows)
                else "dynamic masks or alignment gates are incomplete"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if args.teacher_output is not None:
        if not report["teacher"]["safe_for_distillation"]:
            raise RuntimeError("refusing to save an unsafe temporal teacher artifact")
        args.teacher_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.teacher_output,
            points=potential.points.astype(np.float32),
            source_kind=potential.source_kind,
            confidence=potential.confidence.astype(np.float32),
            time_offset_s=potential.time_offset_s.astype(np.float32),
            center_frame_id=np.asarray(args.frame_id, dtype=np.int32),
            center_lidar_timestamp=np.asarray(
                sensor_timestamp(window[center_window_index]), dtype=np.float64,
            ),
        )
        report["teacher"]["artifact"] = str(args.teacher_output)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
