"""Validate target-pose sparse LiDAR structure plus DA3 RGB reprojection.

This is an audit, not a training entry point.  It keeps true current and
temporal LiDAR layers separate from the DA3 surface and evaluates both on the
same target pose and camera topology.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations
from gcr_nvs.geometry.lidar_depth_layers import project_lidar_layer
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb


def _perturb(calibration: CameraCalibration, translation: tuple[float, float, float]) -> CameraCalibration:
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, dtype=np.float64)
    return replace(calibration, external=np.linalg.inv(camera_to_world))


def _load_cached(root: Path, sequence: str, frame_id: int, camera: str):
    directory = root / sequence / f"{frame_id:06d}" / camera
    metadata = json.loads((directory / "report.json").read_text())
    calibration = CameraCalibration(
        camera, np.asarray(metadata["intrinsic_newK"], np.float64),
        np.asarray(metadata["distortion_after_rectification"], np.float64),
        np.asarray(metadata["external_world_to_camera"], np.float64),
        int(metadata["output_size"][0]), int(metadata["output_size"][1]),
    )
    rgb = np.asarray(Image.open(directory / "rgb_rectified.jpg").convert("RGB"), np.float32) / 255.0
    depth = np.load(directory / "fused_depth.npy").astype(np.float32)
    confidence = np.load(directory / "fused_confidence.npy").astype(np.float32)
    return rgb, depth, confidence, calibration


def _resize_cached(item, width: int, height: int):
    rgb, depth, confidence, calibration = item
    if calibration.width == width and calibration.height == height:
        return item
    matrix = calibration.intrinsic.copy()
    matrix[0] *= width / calibration.width
    matrix[1] *= height / calibration.height
    resized = CameraCalibration(
        calibration.name, matrix, calibration.distortion.copy(),
        calibration.external.copy(), width, height,
    )
    return (
        cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA),
        cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST),
        cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR),
        resized,
    )


def _psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    mse = np.square(prediction[mask] - target[mask]).mean()
    return float(-10.0 * np.log10(max(float(mse), 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()

    records = build_sequence_manifest(args.root / args.sequence, compute_quality=False)
    record = next(row for row in records if row.frame_id == args.frame_id)
    teacher_path = args.teacher_root / args.sequence / f"{args.frame_id:06d}.npz"
    with np.load(teacher_path) as payload:
        points = payload["points"].astype(np.float32)
        kinds = payload["source_kind"].astype(np.uint8)
        confidence = payload["confidence"].astype(np.float32)
    current = points[kinds == EXACT_CURRENT]
    temporal_mask = kinds != EXACT_CURRENT
    temporal = points[temporal_mask]
    temporal_confidence = confidence[temporal_mask]
    cached = {camera: _load_cached(args.cache_root, args.sequence, args.frame_id, camera) for camera in CAMERA_NAMES}
    width = args.width or cached[CAMERA_NAMES[0]][3].width
    height = args.height or cached[CAMERA_NAMES[0]][3].height
    calibrated = {camera: _resize_cached(cached[camera], width, height) for camera in CAMERA_NAMES}
    pose_groups = {
        "5_10cm": [(0.05, 0.05, 0.0), (-0.05, 0.05, 0.05), (0.0, -0.10, 0.0)],
        "10_20cm": [(0.10, 0.10, 0.0), (-0.15, 0.0, 0.10), (0.10, -0.10, -0.15)],
        "20_50cm": [(0.20, 0.20, 0.20), (-0.30, 0.20, 0.20), (0.50, -0.20, -0.50)],
    }
    rows = []
    for group, translations in pose_groups.items():
        for translation in translations:
            for target_camera in CAMERA_NAMES:
                target_rgb, _, _, base_target = calibrated[target_camera]
                target = _perturb(base_target, translation)
                current_layer = project_lidar_layer(current, target, width, height)
                temporal_layer = project_lidar_layer(temporal, target, width, height, temporal_confidence)
                sparse_union = current_layer.valid | temporal_layer.valid
                sparse_new = temporal_layer.valid & ~current_layer.valid
                source_names = appearance_source_cameras(target_camera, include_target=True)
                source = [calibrated[name] for name in source_names]
                rendered, valid, provenance = reproject_dense_rgb(
                    [item[0] for item in source], [item[1] for item in source],
                    [item[3] for item in source], target,
                    source_confidences=[item[2] for item in source], splat_radius=1,
                    exclusive_source_priority=True,
                )
                rows.append({
                    "group": group,
                    "translation_xyz_m": list(translation),
                    "target_camera": target_camera,
                    "source_topology": list(source_names),
                    "current_lidar_coverage": float(current_layer.valid.mean()),
                    "temporal_lidar_coverage": float(temporal_layer.valid.mean()),
                    "temporal_new_coverage": float(sparse_new.mean()),
                    "sparse_union_coverage": float(sparse_union.mean()),
                    "sparse_on_da3_rgb_valid": float((sparse_union & valid).sum() / max(int(sparse_union.sum()), 1)),
                    "dense_rgb_coverage": float(valid.mean()),
                    "dense_rgb_psnr_proxy_db": _psnr(rendered, target_rgb, valid),
                    "temporal_confidence_mean": float(temporal_layer.confidence[temporal_layer.valid].mean()) if temporal_layer.valid.any() else 0.0,
                    "rgb_direct_source_path": True,
                    "lidar_direct_rgb_override": False,
                    "da3_used_as_dense_candidate": True,
                })
                if len(rows) % 7 == 0:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.with_suffix(".partial.json").write_text(
                        json.dumps({"contract": "target_sparse_true_structure_plus_da3_surface", "rows": rows}, ensure_ascii=False)
                    )
                    print(json.dumps({"completed": len(rows), "group": group, "camera": target_camera}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"contract": "target_sparse_true_structure_plus_da3_surface", "sequence": args.sequence, "frame_id": args.frame_id, "rows": rows}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for group in pose_groups:
        selected = [r for r in rows if r["group"] == group]
        print(json.dumps({
            "group": group,
            "count": len(selected),
            "mean_current_lidar": float(np.mean([r["current_lidar_coverage"] for r in selected])),
            "mean_temporal_new": float(np.mean([r["temporal_new_coverage"] for r in selected])),
            "mean_sparse_on_da3_valid": float(np.mean([r["sparse_on_da3_rgb_valid"] for r in selected])),
            "mean_rgb_coverage": float(np.mean([r["dense_rgb_coverage"] for r in selected])),
            "mean_psnr_proxy_db": float(np.mean([r["dense_rgb_psnr_proxy_db"] for r in selected if r["dense_rgb_psnr_proxy_db"] is not None])),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
