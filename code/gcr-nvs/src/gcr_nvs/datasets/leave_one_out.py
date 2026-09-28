"""Strict leave-one-camera-out sample construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcr_nvs.datasets.manifest import CAMERA_NAMES, FrameRecord, sensor_timestamp
from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points


@dataclass(frozen=True)
class LeaveOneOutSample:
    record: FrameRecord
    target_camera: str
    source_cameras: tuple[str, ...]


def rank_source_cameras(
    target_camera: CameraCalibration,
    source_calibrations: dict[str, CameraCalibration],
    points,
    min_sources: int = 3,
    max_sources: int = 5,
) -> tuple[str, ...]:
    """Rank source views by measured point overlap and viewing direction."""
    _, target_valid = project_world_points(points[:, :3], target_camera)
    target_points = points[target_valid, :3]
    if not len(target_points):
        return tuple(list(source_calibrations)[:max_sources])
    target_center = target_camera.camera_center
    target_dirs = target_points - target_center
    target_dirs /= (target_dirs**2).sum(axis=1, keepdims=True).clip(min=1e-8) ** 0.5
    scores = []
    for name, calibration in source_calibrations.items():
        pixels, valid = project_world_points(target_points, calibration)
        source_dirs = target_points - calibration.camera_center
        source_dirs /= (source_dirs**2).sum(axis=1, keepdims=True).clip(min=1e-8) ** 0.5
        similarity = np.maximum((target_dirs * source_dirs).sum(axis=1), 0.0)
        scores.append((float(valid.mean() * similarity.mean()), name))
    scores.sort(reverse=True)
    selected = [name for _, name in scores[:max_sources]]
    return tuple(selected[:max(min_sources, len(selected))])


def make_leave_one_out_samples(
    records: list[FrameRecord], target_camera: str | None = None
) -> list[LeaveOneOutSample]:
    targets = (target_camera,) if target_camera else CAMERA_NAMES
    unknown = set(targets) - set(CAMERA_NAMES)
    if unknown:
        raise ValueError(f"unknown target cameras: {sorted(unknown)}")
    return [
        LeaveOneOutSample(
            record=record,
            target_camera=target,
            source_cameras=tuple(camera for camera in CAMERA_NAMES if camera != target),
        )
        for record in records
        for target in targets
    ]


def temporal_window(
    records: list[FrameRecord],
    frame_index: int,
    length: int = 1,
    max_interval_seconds: float | None = 0.75,
) -> tuple[FrameRecord, ...]:
    """Return a centered, time-valid odd window without crossing gaps.

    Longer windows are used by the LiDAR teacher to recover surfaces that are
    missed by the center scan.  Every interval is checked independently; a
    missing or delayed scan invalidates the complete window instead of being
    silently padded.
    """
    if length < 1 or length % 2 == 0:
        raise ValueError("length must be a positive odd integer")
    if not records:
        raise ValueError("records cannot be empty")
    half = length // 2
    start = frame_index - half
    end = frame_index + half + 1
    if start < 0 or end > len(records):
        return ()
    window = tuple(records[start:end])
    if any(record.sequence_id != window[0].sequence_id for record in window[1:]):
        return ()
    times = np.asarray([sensor_timestamp(record) for record in window], dtype=np.float64)
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("temporal records must be strictly ordered by LiDAR timestamp")
    if max_interval_seconds is not None:
        if max_interval_seconds <= 0.0:
            raise ValueError("max_interval_seconds must be positive or None")
        if np.any(intervals > max_interval_seconds):
            return ()
    return window
