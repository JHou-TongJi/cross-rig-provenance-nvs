"""Manifest-backed leave-one-camera-out sample metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gcr_nvs.datasets.leave_one_out import LeaveOneOutSample
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.pcd import read_pcd


@dataclass(frozen=True)
class SamplePaths:
    source_images: tuple[Path, ...]
    target_image: Path
    lidar: Path
    camera_config: Path


def resolve_sample_paths(root: Path, sample: LeaveOneOutSample) -> SamplePaths:
    sequence_dir = root / sample.record.sequence_id
    return SamplePaths(
        source_images=tuple(sequence_dir / sample.record.cameras[name] for name in sample.source_cameras),
        target_image=sequence_dir / sample.record.cameras[sample.target_camera],
        lidar=sequence_dir / sample.record.lidar,
        camera_config=sequence_dir / sample.record.camera_config,
    )


def load_geometry_sample(root: Path, sample: LeaveOneOutSample, distortion: Path | None = None):
    paths = resolve_sample_paths(root, sample)
    calibrations = load_calibrations(paths.camera_config, distortion)
    points = read_pcd(paths.lidar)
    target = TargetCamera(calibrations[sample.target_camera])
    return {
        "paths": paths,
        "calibrations": calibrations,
        "target_camera": target,
        "lidar_points": points,
        "target_ray_map": target.ray_map(),
    }
