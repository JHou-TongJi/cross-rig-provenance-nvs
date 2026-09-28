"""Target-camera rays, resized intrinsics, and sparse depth generation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration


@dataclass(frozen=True)
class TargetCamera:
    calibration: CameraCalibration

    @property
    def intrinsic(self) -> np.ndarray:
        return self.calibration.intrinsic

    def resized(self, width: int, height: int) -> "TargetCamera":
        sx = width / self.calibration.width
        sy = height / self.calibration.height
        matrix = self.intrinsic.copy()
        matrix[0] *= sx
        matrix[1] *= sy
        return TargetCamera(
            CameraCalibration(
                self.calibration.name, matrix, self.calibration.distortion,
                self.calibration.external, width, height
            )
        )

    def ray_map(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        width = width or self.calibration.width
        height = height or self.calibration.height
        camera = self.resized(width, height)
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float64)
        if np.any(camera.calibration.distortion):
            normalized = cv2.undistortPoints(
                pixels[..., :2].reshape(-1, 1, 2),
                camera.intrinsic,
                camera.calibration.distortion,
            ).reshape(height, width, 2)
            rays = np.concatenate(
                [normalized, np.ones((height, width, 1), dtype=np.float64)],
                axis=-1,
            )
        else:
            rays = pixels @ np.linalg.inv(camera.intrinsic).T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-8)
        # T_C_from_E is vehicle-to-camera; map camera rays back to vehicle axes.
        rays = rays @ camera.calibration.external[:3, :3]
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-8)
        return rays.transpose(2, 0, 1).astype(np.float32)

    def ray_map_region(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Return rays for a native-resolution tile with global pixel origin."""
        if min(x0, y0, width, height) < 0:
            raise ValueError("ray tile coordinates must be non-negative")
        camera = self.calibration
        yy, xx = np.meshgrid(
            np.arange(y0, y0 + height),
            np.arange(x0, x0 + width),
            indexing="ij",
        )
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float64)
        if np.any(camera.distortion):
            normalized = cv2.undistortPoints(
                pixels[..., :2].reshape(-1, 1, 2),
                camera.intrinsic,
                camera.distortion,
            ).reshape(height, width, 2)
            rays = np.concatenate([
                normalized, np.ones((height, width, 1), dtype=np.float64),
            ], axis=-1)
        else:
            rays = pixels @ np.linalg.inv(camera.intrinsic).T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-8)
        rays = rays @ camera.external[:3, :3]
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=1e-8)
        return rays.transpose(2, 0, 1).astype(np.float32)


def sparse_depth_from_points(
    points: np.ndarray, calibration: CameraCalibration, output_size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z-buffered depth, validity, and source point index maps."""
    from gcr_nvs.geometry.calibration import project_world_points

    width, height = output_size or (calibration.width, calibration.height)
    pixels, valid = project_world_points(points[:, :3], calibration)
    if output_size and output_size != (calibration.width, calibration.height):
        pixels[:, 0] *= width / calibration.width
        pixels[:, 1] *= height / calibration.height
    depth = np.zeros((height, width), dtype=np.float32)
    point_index = np.full((height, width), -1, dtype=np.int64)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return depth, point_index >= 0, point_index
    xy = np.rint(pixels[valid_indices]).astype(np.int64)
    camera_points = (calibration.external @ np.c_[points[valid_indices, :3], np.ones(len(valid_indices))].T).T
    order = np.argsort(camera_points[:, 2])
    for local in order:
        x, y = xy[local]
        if 0 <= x < width and 0 <= y < height and point_index[y, x] < 0:
            depth[y, x] = camera_points[local, 2]
            point_index[y, x] = valid_indices[local]
    return depth, point_index >= 0, point_index
