"""Small dependency-light rigid ICP for local ego-motion recovery."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from gcr_nvs.geometry.pose import identity_pose


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    covariance = centered_source.T @ centered_target
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = identity_pose()
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def estimate_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_points: int = 12000,
    iterations: int = 12,
    max_correspondence_m: float = 2.0,
    trim_ratio: float = 0.65,
    geometry_rmse_gate_m: float = 0.15,
    geometry_inlier_gate: float = 0.30,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Estimate ``T_target_from_source`` using robust point-to-point ICP."""
    source = np.asarray(source_points, dtype=np.float64)[:, :3]
    target = np.asarray(target_points, dtype=np.float64)[:, :3]
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if len(source) > max_points:
        source = source[np.linspace(0, len(source) - 1, max_points).astype(np.int64)]
    if len(target) > max_points:
        target = target[np.linspace(0, len(target) - 1, max_points).astype(np.int64)]
    if len(source) < 32 or len(target) < 32:
        return identity_pose(), {"valid": False, "rmse_m": float("inf"), "inlier_ratio": 0.0}
    tree = cKDTree(target)
    transform = identity_pose()
    last_rmse = float("inf")
    inlier_ratio = 0.0
    for _ in range(iterations):
        transformed = (transform @ np.c_[source, np.ones(len(source))].T).T[:, :3]
        distances, indices = tree.query(transformed, k=1, workers=1)
        keep = distances < max_correspondence_m
        if keep.sum() < 32:
            break
        candidate_indices = np.flatnonzero(keep)
        keep_count = max(32, int(len(candidate_indices) * trim_ratio))
        trimmed = candidate_indices[np.argsort(distances[candidate_indices])[:keep_count]]
        keep = np.zeros_like(keep)
        keep[trimmed] = True
        inlier_ratio = float(keep.mean())
        update = _rigid_transform(transformed[keep], target[indices[keep]])
        transform = update @ transform
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        if abs(last_rmse - rmse) < 1e-4:
            last_rmse = rmse
            break
        last_rmse = rmse
    registration_valid = bool(inlier_ratio > 0.15 and last_rmse < 0.5)
    geometry_gate_passed = bool(inlier_ratio >= geometry_inlier_gate and last_rmse <= geometry_rmse_gate_m)
    return transform, {
        "valid": registration_valid,
        "registration_valid": registration_valid,
        "geometry_gate_passed": geometry_gate_passed,
        "rmse_m": last_rmse,
        "inlier_ratio": inlier_ratio,
    }


def estimate_kiss_sequence_poses(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    voxel_size: float = 0.20,
    max_range_m: float = 120.0,
    max_points: int = 16_000,
    max_iterations: int = 60,
) -> list[np.ndarray]:
    """Estimate world-from-frame poses with KISS-ICP and deskew disabled."""
    try:
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("kiss-icp is required for temporal pose estimation") from exc
    config = KISSConfig()
    config.data.deskew = False
    config.data.max_range = float(max_range_m)
    config.mapping.voxel_size = float(voxel_size)
    config.registration.max_num_iterations = int(max_iterations)
    odometry = KissICP(config)
    poses = []
    for frame in frames:
        xyz = np.asarray(frame, dtype=np.float64)[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) > max_points:
            xyz = xyz[np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)]
        timestamps = np.zeros(len(xyz), dtype=np.float64)
        odometry.register_frame(xyz, timestamps)
        poses.append(np.asarray(odometry.last_pose, dtype=np.float64).copy())
    return poses


def alignment_quality(
    source_points: np.ndarray,
    target_points: np.ndarray,
    transform: np.ndarray,
    max_points: int = 40_000,
    correspondence_m: float = 0.30,
) -> dict[str, float | bool]:
    """Measure transformed nearest-neighbor RMSE without re-optimizing a pose."""
    source = np.asarray(source_points, dtype=np.float64)[:, :3]
    target = np.asarray(target_points, dtype=np.float64)[:, :3]
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if len(source) > max_points:
        source = source[np.linspace(0, len(source) - 1, max_points).astype(np.int64)]
    if len(target) > max_points:
        target = target[np.linspace(0, len(target) - 1, max_points).astype(np.int64)]
    if len(source) < 32 or len(target) < 32:
        return {"valid": False, "rmse_m": float("inf"), "inlier_ratio": 0.0}
    transformed = (
        np.asarray(transform, dtype=np.float64)
        @ np.c_[source, np.ones(len(source))].T
    ).T[:, :3]
    distances, _ = cKDTree(target).query(transformed, k=1, workers=1)
    inliers = distances <= correspondence_m
    if inliers.sum() < 32:
        return {"valid": False, "rmse_m": float("inf"), "inlier_ratio": float(inliers.mean())}
    rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))
    inlier_ratio = float(inliers.mean())
    return {
        "valid": bool(rmse <= 0.15 and inlier_ratio >= 0.30),
        "rmse_m": rmse,
        "inlier_ratio": inlier_ratio,
    }
