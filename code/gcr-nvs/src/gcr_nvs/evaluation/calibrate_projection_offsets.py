"""Estimate fixed source-image projection offsets from training frames only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.appearance import project_surfaces_to_sources
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0


def _sample(image: np.ndarray, uv: np.ndarray, offset=(0.0, 0.0)) -> np.ndarray:
    return cv2.remap(
        image,
        (uv[:, 0] + offset[0]).astype(np.float32).reshape(-1, 1),
        (uv[:, 1] + offset[1]).astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[:, 0]


def _loss(samples, dx: float, dy: float) -> float:
    total = 0.0
    count = 0
    for source_image, source_uv, target_rgb in samples:
        source_rgb = _sample(source_image, source_uv, (dx, dy))
        error = np.sqrt((source_rgb - target_rgb) ** 2 + 1e-4)
        total += float(error.sum())
        count += error.size
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--target-camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--max-points-per-frame", type=int, default=4000)
    parser.add_argument("--search-radius", type=int, default=24)
    parser.add_argument("--coarse-step", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/calibration_reports/rgb_projection_offsets.json"),
    )
    args = parser.parse_args()
    sequence_dir = Path(args.sequence)
    manifest = {
        row.frame_id: row
        for row in build_sequence_manifest(sequence_dir, compute_quality=False)
    }
    source_names = tuple(name for name in CAMERA_NAMES if name != args.target_camera)
    per_source = {name: [] for name in source_names}
    for frame_id in args.frames:
        record = manifest[frame_id]
        calibrations = load_calibrations(
            sequence_dir / record.camera_config, Path("camera_intric.yaml"),
        )
        target_calibration = calibrations[args.target_camera]
        target_image = _load_rgb(
            sequence_dir / record.cameras[args.target_camera],
            (target_calibration.width, target_calibration.height),
        )
        with np.load(
            Path("outputs/geometry_teacher") / args.sequence / f"{frame_id:06d}.npz"
        ) as payload:
            exact_points = payload["points"][
                payload["source_kind"] == EXACT_CURRENT, :3
            ].astype(np.float32)
        target_uv, target_valid, _, _ = project_surfaces_to_sources(
            exact_points, [target_calibration],
        )
        for source_name in source_names:
            source_calibration = TargetCamera(calibrations[source_name]).resized(
                args.source_width, args.source_height,
            ).calibration
            source_uv, source_valid, _, _ = project_surfaces_to_sources(
                exact_points, [source_calibration],
            )
            valid = target_valid[0] & source_valid[0]
            indices = np.flatnonzero(valid)
            if len(indices) > args.max_points_per_frame:
                indices = indices[np.linspace(
                    0, len(indices) - 1, args.max_points_per_frame,
                ).round().astype(np.int64)]
            if not len(indices):
                continue
            source_image = _load_rgb(
                sequence_dir / record.cameras[source_name],
                (args.source_width, args.source_height),
            )
            target_rgb = _sample(target_image, target_uv[0, indices])
            per_source[source_name].append((
                source_image, source_uv[0, indices], target_rgb,
            ))

    offsets = {}
    for source_name, samples in per_source.items():
        if not samples:
            offsets[source_name] = {"dx": 0.0, "dy": 0.0, "sample_count": 0}
            continue
        center_loss = _loss(samples, 0.0, 0.0)
        best = (center_loss, 0, 0)
        for dx in range(-args.search_radius, args.search_radius + 1, args.coarse_step):
            for dy in range(-args.search_radius, args.search_radius + 1, args.coarse_step):
                best = min(best, (_loss(samples, dx, dy), dx, dy))
        _, coarse_x, coarse_y = best
        for dx in range(coarse_x - args.coarse_step, coarse_x + args.coarse_step + 1):
            for dy in range(coarse_y - args.coarse_step, coarse_y + args.coarse_step + 1):
                best = min(best, (_loss(samples, dx, dy), dx, dy))
        best_loss, dx, dy = best
        offsets[source_name] = {
            "dx": float(dx),
            "dy": float(dy),
            "sample_count": int(sum(len(item[1]) for item in samples)),
            "center_charbonnier": center_loss,
            "corrected_charbonnier": best_loss,
        }
    report = {
        "sequence": args.sequence,
        "training_frames": args.frames,
        "target_camera": args.target_camera,
        "offset_resolution": [args.source_width, args.source_height],
        "target_images_used_as_training_supervision_only": True,
        "offsets": offsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
