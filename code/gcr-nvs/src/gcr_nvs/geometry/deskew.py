"""LiDAR deskew with an explicit disabled-until-verified default."""

from __future__ import annotations

import numpy as np

from gcr_nvs.geometry.pose import PoseProvider


def deskew_points(
    points: np.ndarray,
    reference_timestamp: float,
    pose_provider: PoseProvider,
    enabled: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deskew [x,y,z,intensity,ring,time] points when time semantics are verified."""
    metadata = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled_until_time_semantics_verified",
    }
    if not enabled or points.shape[1] < 6:
        return points, metadata
    if not pose_provider.temporal_valid:
        metadata["reason"] = "pose_provider_not_temporal"
        return points, metadata
    result = points.copy()
    for time_value in np.unique(points[:, 5]):
        selected = np.flatnonzero(points[:, 5] == time_value)
        transform = pose_provider.relative(float(time_value), reference_timestamp)
        result[selected, :3] = (transform @ np.c_[points[selected, :3], np.ones(len(selected))].T).T[:, :3]
    metadata.update({"applied": True, "reason": "pose_interpolation"})
    return result, metadata
