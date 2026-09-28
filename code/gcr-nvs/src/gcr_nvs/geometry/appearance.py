"""Brown-aware source-view visibility for LiDAR-supported surfaces."""

from __future__ import annotations

import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points


def project_surfaces_to_sources(
    points: np.ndarray,
    calibrations: list[CameraCalibration] | tuple[CameraCalibration, ...],
    front_tolerance_m: float = 0.10,
    relative_front_tolerance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project surfaces and retain only the nearest layer in each source pixel."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    uv_per_view = []
    valid_per_view = []
    depth_per_view = []
    centers = []
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float32)]
    for calibration in calibrations:
        uv, valid = project_world_points(points, calibration)
        camera_points = (calibration.external @ homogeneous.T).T[:, :3]
        depth = camera_points[:, 2].astype(np.float32)
        visible = np.zeros(len(points), dtype=bool)
        candidates = np.flatnonzero(valid)
        if len(candidates):
            xy = np.rint(uv[candidates]).astype(np.int64)
            inside = (
                (xy[:, 0] >= 0) & (xy[:, 0] < calibration.width)
                & (xy[:, 1] >= 0) & (xy[:, 1] < calibration.height)
            )
            candidates = candidates[inside]
            xy = xy[inside]
            if len(candidates):
                flat = xy[:, 1] * calibration.width + xy[:, 0]
                z_buffer = np.full(
                    calibration.height * calibration.width,
                    np.inf,
                    dtype=np.float32,
                )
                np.minimum.at(z_buffer, flat, depth[candidates])
                tolerance = np.maximum(
                    front_tolerance_m,
                    relative_front_tolerance * z_buffer[flat],
                )
                visible[candidates] = depth[candidates] <= z_buffer[flat] + tolerance
        clean_uv = np.nan_to_num(
            uv, nan=-1.0, posinf=-1.0, neginf=-1.0,
        ).astype(np.float32)
        clean_uv[~visible] = -1.0
        uv_per_view.append(clean_uv)
        valid_per_view.append(visible)
        depth_per_view.append(depth)
        centers.append(calibration.camera_center.astype(np.float32))
    return (
        np.stack(uv_per_view),
        np.stack(valid_per_view),
        np.stack(depth_per_view),
        np.stack(centers),
    )
