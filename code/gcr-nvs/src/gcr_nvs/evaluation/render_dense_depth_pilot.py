"""Render a small-pose shifted view from DA3/LiDAR dense-depth caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations, rectify_image
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb


def _perturb(calibration: CameraCalibration, translation: tuple[float, float, float]) -> CameraCalibration:
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, dtype=np.float64)
    return CameraCalibration(
        calibration.name,
        calibration.intrinsic.copy(),
        calibration.distortion.copy(),
        np.linalg.inv(camera_to_world),
        calibration.width,
        calibration.height,
    )


def _load_cached(cache_root: Path, sequence: str, frame_id: int, camera: str):
    camera_root = cache_root / sequence / f"{frame_id:06d}" / camera
    metadata = json.loads((camera_root / "report.json").read_text())
    contract = metadata.get("contract", {})
    if contract.get("fused_depth_role") != "continuous_da3_aligned_surface":
        raise RuntimeError(
            f"legacy dense-depth cache for {camera}: fused_depth is not marked "
            "as a continuous DA3 surface; rebuild the cache before rendering"
        )
    rgb = np.asarray(Image.open(camera_root / "rgb_rectified.jpg").convert("RGB"), dtype=np.float32) / 255.0
    depth = np.load(camera_root / "fused_depth.npy").astype(np.float32)
    confidence = np.load(camera_root / "fused_confidence.npy").astype(np.float32)
    calibration = CameraCalibration(
        camera,
        np.asarray(metadata["intrinsic_newK"], dtype=np.float64),
        np.asarray(metadata["distortion_after_rectification"], dtype=np.float64),
        np.asarray(metadata["external_world_to_camera"], dtype=np.float64),
        int(metadata["output_size"][0]),
        int(metadata["output_size"][1]),
    )
    return rgb, depth, confidence, calibration, metadata


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.square(pred - target).mean()
    return float(-10.0 * np.log10(max(float(mse), 1e-8)))


def render_target(
    root: Path,
    cache_root: Path,
    sequence: str,
    frame_id: int,
    target_camera: str,
    translation: tuple[float, float, float],
    distortion: Path,
    output: Path,
    splat_radius: int = 0,
    color_invalid: bool = True,
) -> dict:
    records = build_sequence_manifest(root / sequence, compute_quality=False)
    record = next(item for item in records if item.frame_id == frame_id)
    raw_calibrations = load_calibrations(root / sequence / record.camera_config, distortion)
    source_names = appearance_source_cameras(target_camera, include_target=True)
    available = []
    source_rgbs, source_depths, source_confidences, source_calibrations = [], [], [], []
    for name in source_names:
        try:
            rgb, depth, confidence, calibration, metadata = _load_cached(cache_root, sequence, frame_id, name)
        except FileNotFoundError:
            continue
        available.append(name)
        source_rgbs.append(rgb)
        source_depths.append(depth)
        source_confidences.append(confidence)
        source_calibrations.append(calibration)
    if not available:
        raise RuntimeError(f"no dense-depth cache for target topology {target_camera}: {source_names}")
    raw_path = root / sequence / record.cameras[target_camera]
    with Image.open(raw_path) as image:
        raw_rgb = np.asarray(image.convert("RGB"))
    target_rgb, target_rectified = rectify_image(raw_rgb, raw_calibrations[target_camera], output_size=(960, 540), alpha=0.0)
    target_rgb_float = target_rgb.astype(np.float32) / 255.0
    target = _perturb(target_rectified, translation)
    rendered, validity, provenance = reproject_dense_rgb(
        source_rgbs, source_depths, source_calibrations, target,
        source_confidences=source_confidences, splat_radius=splat_radius,
        exclusive_source_priority=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    display = rendered.copy()
    if color_invalid:
        display[~validity] = 1.0, 0.0, 1.0
    source_rgb = target_rgb_float.copy()
    if target_camera in available:
        source_rgb = source_rgbs[available.index(target_camera)].copy()
    Image.fromarray(np.rint(np.clip(source_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output / "source_rgb_rectified.png")
    Image.fromarray(np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output / "reconstruction_rgb.png")
    Image.fromarray(target_rgb).save(output / "target_rectified.png")
    valid_rgb = np.zeros((*validity.shape, 3), dtype=np.uint8)
    valid_rgb[validity] = (55, 205, 85)
    valid_rgb[~validity] = (220, 30, 220)
    Image.fromarray(valid_rgb).save(output / "validity.png")
    panel = np.concatenate([
        np.rint(np.clip(source_rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
        target_rgb,
        np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8),
        valid_rgb,
    ], axis=1)
    Image.fromarray(panel).save(output / "audit_panel.png")
    report = {
        "sequence": sequence,
        "frame_id": frame_id,
        "target_camera": target_camera,
        "translation_xyz_m": list(translation),
        "source_topology": list(source_names),
        "available_sources": available,
        "resolution": [960, 540],
        "coverage": float(validity.mean()),
        "psnr_against_same_camera_current_frame": _psnr(rendered[validity], target_rgb_float[validity]) if np.any(validity) else None,
        "rgb_direct_source_path": True,
        "lidar_direct_rgb_override": False,
        "invalid_display": "magenta" if color_invalid else "raw_renderer_value",
        "provenance_counts": {str(int(index)): int((provenance == index).sum()) for index in np.unique(provenance) if index >= 0},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--camera", choices=CAMERA_NAMES, required=True)
    parser.add_argument("--translate-xyz", type=float, nargs=3, default=(0.0, 0.1, 0.0))
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splat-radius", type=int, default=0)
    parser.add_argument("--no-color-invalid", dest="color_invalid", action="store_false",
                        help="leave invalid pixels at the renderer's raw value instead of coloring them magenta")
    parser.set_defaults(color_invalid=True)
    args = parser.parse_args()
    print(json.dumps(render_target(
        args.root, args.cache_root, args.sequence, args.frame_id, args.camera,
        tuple(args.translate_xyz), args.distortion, args.output, args.splat_radius,
        args.color_invalid,
    ), indent=2))


if __name__ == "__main__":
    main()
