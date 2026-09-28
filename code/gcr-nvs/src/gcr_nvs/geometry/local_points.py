"""Target-view-aware local point cloud construction."""

from __future__ import annotations

import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points
from gcr_nvs.geometry.pcd import voxel_downsample
from gcr_nvs.geometry.pose import PoseProvider


def adaptive_voxel_downsample(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, :3], axis=1)
    result = []
    for lower, upper, voxel in ((0.0, 30.0, 0.08), (30.0, 60.0, 0.16), (60.0, 120.0, 0.32)):
        selected = points[(distances >= lower) & (distances < upper)]
        if len(selected):
            result.append(voxel_downsample(selected, voxel))
    return np.concatenate(result, axis=0) if result else points[:0]


def estimate_normals(points: np.ndarray, neighbors: int = 12) -> np.ndarray:
    """Estimate PCA normals; unstable points receive zero normals."""
    from scipy.spatial import cKDTree

    xyz = points[:, :3].astype(np.float64)
    normals = np.zeros_like(xyz, dtype=np.float32)
    if len(xyz) < 3:
        return normals
    _, indices = cKDTree(xyz).query(xyz, k=min(neighbors, len(xyz)))
    neighborhoods = xyz[np.asarray(indices)]
    local = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local)
    values, vectors = np.linalg.eigh(covariance)
    stable = (values[:, -1] > 1e-8) & ((values[:, 0] / values[:, -1].clip(min=1e-8)) <= 0.35)
    normals[stable] = vectors[stable, :, 0].astype(np.float32)
    return normals


class LocalPointBuilder:
    def __init__(self, max_range_m: float = 120.0, max_points: int = 120_000):
        self.max_range_m = max_range_m
        self.max_points = max_points

    def build(
        self,
        points: np.ndarray,
        target_camera: CameraCalibration,
        pose_provider: PoseProvider | None = None,
        timestamp: float = 0.0,
    ) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape [N, >=3]")
        points = points[np.isfinite(points[:, :3]).all(axis=1)]
        points = points[np.linalg.norm(points[:, :3], axis=1) <= self.max_range_m]
        _, valid = project_world_points(points[:, :3], target_camera)
        points = points[valid]
        if pose_provider and pose_provider.temporal_valid:
            pose = pose_provider.at(timestamp).world_from_vehicle
            transformed = (pose @ np.c_[points[:, :3], np.ones(len(points))].T).T[:, :3]
            points = np.c_[transformed, points[:, 3:]]
        points = adaptive_voxel_downsample(points)
        if len(points) > self.max_points:
            points = points[np.linspace(0, len(points) - 1, self.max_points).astype(np.int64)]
        return points
