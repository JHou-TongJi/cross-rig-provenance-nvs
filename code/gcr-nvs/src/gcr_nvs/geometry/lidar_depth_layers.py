"""Per-camera true LiDAR depth layers for the dense route.

These maps are deliberately separate from DA3 depth.  ``current`` contains
only the center-frame scan; ``temporal`` contains registered points from the
neighbor scans.  Neither layer contains RGB or monocular predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points


@dataclass(frozen=True)
class LidarDepthLayer:
    z_m: np.ndarray
    range_m: np.ndarray
    valid: np.ndarray
    point_index: np.ndarray
    confidence: np.ndarray


def project_lidar_layer(
    points: np.ndarray,
    calibration: CameraCalibration,
    width: int,
    height: int,
    point_confidence: np.ndarray | None = None,
) -> LidarDepthLayer:
    """Project a point subset with a nearest-surface z-buffer.

    ``calibration`` must describe the image grid being written.  Rectified
    calibrations have zero distortion, so the projection is exactly aligned
    with the rectified RGB/DA3 cache.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N,>=3]")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if point_confidence is None:
        confidence = np.ones(len(points), dtype=np.float32)
    else:
        confidence = np.asarray(point_confidence, dtype=np.float32)
        if confidence.shape != (len(points),):
            raise ValueError("point_confidence must have shape [N]")
    pixels, projected = project_world_points(points[:, :3], calibration, undistort=False)
    homogeneous = np.c_[points[:, :3], np.ones(len(points), dtype=np.float32)]
    camera = (calibration.external @ homogeneous.T).T[:, :3]
    finite = np.isfinite(pixels).all(axis=1) & np.isfinite(camera).all(axis=1)
    valid_points = projected & finite & (camera[:, 2] > 1e-4) & np.isfinite(confidence)
    xy = np.zeros((len(points), 2), dtype=np.int64)
    xy[finite] = np.rint(pixels[finite]).astype(np.int64)
    valid_points &= (
        (xy[:, 0] >= 0) & (xy[:, 0] < width)
        & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    )
    z = np.zeros((height, width), dtype=np.float32)
    ranges = np.zeros_like(z)
    output_confidence = np.zeros_like(z)
    point_index = np.full((height, width), -1, dtype=np.int64)
    candidates = np.flatnonzero(valid_points)
    if len(candidates):
        # Near points win.  Sorting is deterministic for equal-depth points.
        order = candidates[np.argsort(camera[candidates, 2], kind="stable")]
        for index in order:
            x, y = xy[index]
            if point_index[y, x] >= 0:
                continue
            point_index[y, x] = int(index)
            z[y, x] = float(camera[index, 2])
            ranges[y, x] = float(np.linalg.norm(points[index, :3] - calibration.camera_center))
            output_confidence[y, x] = float(np.clip(confidence[index], 0.0, 1.0))
    return LidarDepthLayer(
        z_m=z,
        range_m=ranges,
        valid=point_index >= 0,
        point_index=point_index,
        confidence=output_confidence,
    )

