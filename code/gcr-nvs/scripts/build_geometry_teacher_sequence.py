"""Build safe Temporal LiDAR Teacher artifacts for one complete sequence."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import numpy as np

from gcr_nvs.datasets.leave_one_out import temporal_window
from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest, sensor_timestamp
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.dynamic_mask import dynamic_points_from_camera_masks
from gcr_nvs.geometry.icp import alignment_quality, estimate_kiss_sequence_poses
from gcr_nvs.geometry.local_points import adaptive_voxel_downsample
from gcr_nvs.geometry.pcd import read_pcd
from gcr_nvs.geometry.temporal_lidar import TemporalLidarTeacher, TemporalNeighbor


def _dynamic_point_mask(
    sequence_dir: Path,
    mask_root: Path,
    record,
    points: np.ndarray,
    distortion_path: Path,
) -> np.ndarray:
    mask_dir = mask_root / record.sequence_id / f"{record.frame_id:06d}"
    paths = {
        camera: mask_dir / f"{camera}.dynamic.npy"
        for camera in CAMERA_NAMES
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dynamic masks: {missing[:3]}")
    masks = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    calibrations = load_calibrations(
        sequence_dir / record.camera_config,
        distortion_path,
    )
    return dynamic_points_from_camera_masks(points, calibrations, masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--dynamic-mask-root", type=Path, default=Path("outputs/dynamic_masks"))
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--max-centers", type=int)
    parser.add_argument("--start-frame-id", type=int)
    parser.add_argument(
        "--temporal-radius", type=int, default=1,
        help="number of scans on each side of the center (1 gives the legacy 3-frame teacher)",
    )
    parser.add_argument(
        "--temporal-mode", choices=("previous", "next", "both"), default="both",
        help="which adjacent scan(s) to use; dense_route_v1 requires previous",
    )
    parser.add_argument(
        "--max-interval-seconds", type=float, default=0.75,
        help="maximum timestamp gap allowed between adjacent scans",
    )
    parser.add_argument(
        "--max-time-offset-seconds", type=float,
        help="maximum center-to-neighbor offset; defaults to radius * max-interval",
    )
    parser.add_argument("--temporal-rmse-gate-m", type=float, default=0.20)
    parser.add_argument("--temporal-inlier-gate", type=float, default=0.40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()
    if args.temporal_radius < 1:
        raise ValueError("temporal-radius must be at least one")

    sequence_dir = args.root.resolve() / args.sequence
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    point_frames = [
        adaptive_voxel_downsample(read_pcd(sequence_dir / record.lidar))
        for record in records
    ]
    poses = estimate_kiss_sequence_poses(point_frames)
    dynamic_masks: dict[int, np.ndarray] = {}

    def dynamic_mask(index: int) -> np.ndarray:
        if index not in dynamic_masks:
            dynamic_masks[index] = _dynamic_point_mask(
                sequence_dir,
                args.dynamic_mask_root,
                records[index],
                point_frames[index],
                args.distortion,
            )
        return dynamic_masks[index]

    max_time_offset = args.max_time_offset_seconds
    if max_time_offset is None:
        max_time_offset = float(args.temporal_radius * args.max_interval_seconds)
    teacher = TemporalLidarTeacher(max_time_offset_s=max_time_offset)
    output_dir = args.output_root / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    attempted = 0
    radius = int(args.temporal_radius)
    window_length = 2 * radius + 1
    if args.temporal_mode == "previous":
        center_indices = list(range(radius, len(records)))
    elif args.temporal_mode == "next":
        center_indices = list(range(0, len(records) - radius))
    else:
        center_indices = list(range(radius, len(records) - radius))
    if args.start_frame_id is not None:
        start_index = next(
            index for index, record in enumerate(records)
            if record.frame_id == args.start_frame_id
        )
        center_indices = [index for index in center_indices if index >= start_index]
    for center_index in center_indices:
        if args.temporal_mode == "previous":
            window = (records[center_index - 1], records[center_index]) if (
                center_index > 0
                and sensor_timestamp(records[center_index]) > sensor_timestamp(records[center_index - 1])
                and sensor_timestamp(records[center_index]) - sensor_timestamp(records[center_index - 1]) <= args.max_interval_seconds
            ) else []
        elif args.temporal_mode == "next":
            window = (records[center_index], records[center_index + 1]) if (
                center_index + 1 < len(records)
                and sensor_timestamp(records[center_index + 1]) > sensor_timestamp(records[center_index])
                and sensor_timestamp(records[center_index + 1]) - sensor_timestamp(records[center_index]) <= args.max_interval_seconds
            ) else []
        else:
            window = temporal_window(
                records, center_index, window_length,
                max_interval_seconds=args.max_interval_seconds,
            )
        window_invalid = not bool(window)
        if args.max_centers is not None and attempted >= args.max_centers:
            break
        attempted += 1
        center_record = records[center_index]
        artifact = output_dir / f"{center_record.frame_id:06d}.npz"
        if artifact.exists() and not args.overwrite:
            rows.append({
                "frame_id": center_record.frame_id,
                "saved": True,
                "reused": True,
                "artifact": str(artifact),
            })
            continue

        center_pose_inverse = np.linalg.inv(poses[center_index])
        quality_rows = []
        neighbors = []
        temporal_frame_ids = []
        temporal_transforms = []
        temporal_alignment_confidence = []
        temporal_alignment_rmse_m = []
        temporal_alignment_inlier_ratio = []
        if args.temporal_mode == "previous":
            neighbor_indices = (center_index - 1,)
        elif args.temporal_mode == "next":
            neighbor_indices = (center_index + 1,)
        else:
            neighbor_indices = range(center_index - radius, center_index + radius + 1)
        for neighbor_index in neighbor_indices:
            if neighbor_index == center_index:
                continue
            neighbor_dynamic = dynamic_mask(neighbor_index)
            transform = center_pose_inverse @ poses[neighbor_index]
            quality = alignment_quality(
                point_frames[neighbor_index],
                point_frames[center_index],
                transform,
            )
            temporal_gate = bool(
                quality["rmse_m"] <= args.temporal_rmse_gate_m
                and quality["inlier_ratio"] >= args.temporal_inlier_gate
            )
            confidence = (
                np.exp(-float(quality["rmse_m"]) / 0.15)
                * min(float(quality["inlier_ratio"]) / 0.30, 1.0)
                if np.isfinite(float(quality["rmse_m"])) else 0.0
            )
            neighbors.append(TemporalNeighbor(
                points=point_frames[neighbor_index],
                timestamp=sensor_timestamp(records[neighbor_index]),
                transform_to_center=transform,
                alignment_confidence=float(confidence),
                dynamic_mask=neighbor_dynamic,
                alignment_valid=temporal_gate,
            ))
            if temporal_gate:
                temporal_frame_ids.append(records[neighbor_index].frame_id)
                temporal_transforms.append(transform.astype(np.float32))
                temporal_alignment_confidence.append(float(confidence))
                temporal_alignment_rmse_m.append(float(quality["rmse_m"]))
                temporal_alignment_inlier_ratio.append(float(quality["inlier_ratio"]))
            quality_rows.append({
                "frame_id": records[neighbor_index].frame_id,
                "rmse_m": quality["rmse_m"],
                "inlier_ratio": quality["inlier_ratio"],
                "gate_passed": temporal_gate,
                "strict_gate_passed": quality["valid"],
                "dynamic_point_ratio": float(neighbor_dynamic.mean()),
            })
        safe = (not window_invalid) and all(row["gate_passed"] for row in quality_rows)
        # Keep every center frame usable.  If any neighbor fails the strict
        # alignment gate, TemporalLidarTeacher receives the neighbors marked
        # invalid and therefore emits only the exact current scan.  This is a
        # safe degradation: it never injects misregistered temporal points and
        # avoids silently removing RGB/LiDAR training examples.
        result = teacher.fuse(
            point_frames[center_index],
            sensor_timestamp(center_record),
            neighbors if safe else [],
        )
        temporary = artifact.with_name(
            f"{artifact.stem}.tmp.{os.getpid()}.{socket.gethostname()}.npz"
        )
        save = np.savez_compressed if args.compress else np.savez
        with temporary.open("wb") as handle:
            save(
                handle,
            points=result.points.astype(np.float32),
            source_kind=result.source_kind,
            confidence=result.confidence.astype(np.float32),
            time_offset_s=result.time_offset_s.astype(np.float32),
            temporal_frame_ids=np.asarray(temporal_frame_ids, dtype=np.int32),
            temporal_transforms=np.asarray(temporal_transforms, dtype=np.float32).reshape(-1, 4, 4),
            temporal_alignment_confidence=np.asarray(
                temporal_alignment_confidence, dtype=np.float32,
            ),
            temporal_alignment_rmse_m=np.asarray(
                temporal_alignment_rmse_m, dtype=np.float32,
            ),
            temporal_alignment_inlier_ratio=np.asarray(
                temporal_alignment_inlier_ratio, dtype=np.float32,
            ),
            center_frame_id=np.asarray(center_record.frame_id, dtype=np.int32),
            center_lidar_timestamp=np.asarray(
                sensor_timestamp(center_record), dtype=np.float64,
            ),
            )
        os.replace(temporary, artifact)
        rows.append({
            "frame_id": center_record.frame_id,
            "saved": True,
            "reused": False,
            "temporal_fallback": not safe,
            "reason": (
                "invalid_temporal_window_exact_only" if window_invalid
                else "alignment_gate_failed_exact_only" if not safe else None
            ),
            "exact_points": int(result.exact_mask.sum()),
            "temporal_points": int(result.temporal_mask.sum()),
            "total_points": int(len(result.points)),
            "registration": quality_rows,
            "artifact": str(artifact),
        })

    saved = [row for row in rows if row.get("saved")]
    report = {
        "sequence_id": args.sequence,
        "contract": "dense_route_v1",
        "temporal_mode": args.temporal_mode,
        "temporal_radius": int(args.temporal_radius),
        "temporal_window": 1 if args.temporal_mode in ("previous", "next") else window_length,
        "dynamic_mask_root": str(args.dynamic_mask_root),
        "record_count": len(records),
        "attempted_centers": attempted,
        "saved_centers": len(saved),
        "failed_centers": sum(not row.get("saved", False) for row in rows),
        "alignment_gate_failed_centers": sum(
            bool(row.get("temporal_fallback", False)) for row in saved
        ),
        "mean_temporal_points": float(np.mean([
            row["temporal_points"] for row in saved if "temporal_points" in row
        ])) if any("temporal_points" in row for row in saved) else 0.0,
        "frames": rows,
    }
    summary_path = output_dir / "sequence_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
