"""Calibration loading and pinhole projection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from gcr_nvs.datasets.manifest import CAMERA_NAMES


CALIBRATION_CAMERA_ORDER = (
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_FRONT_WIDE",
    "CAM_FRONT_NARROW",
)


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    external: np.ndarray
    width: int
    height: int

    @property
    def camera_center(self) -> np.ndarray:
        rotation = self.external[:3, :3]
        translation = self.external[:3, 3]
        return -rotation.T @ translation


def rectified_calibration(
    calibration: CameraCalibration,
    output_size: tuple[int, int] | None = None,
    alpha: float = 0.0,
) -> CameraCalibration:
    """Return the pinhole calibration of an explicitly rectified RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("rectification alpha must be in [0,1]")
    width, height = output_size or (calibration.width, calibration.height)
    if width <= 0 or height <= 0:
        raise ValueError("rectified output size must be positive")
    if np.any(calibration.distortion):
        matrix, _ = cv2.getOptimalNewCameraMatrix(
            calibration.intrinsic,
            calibration.distortion,
            (calibration.width, calibration.height),
            alpha,
            (width, height),
            centerPrincipalPoint=False,
        )
    else:
        scale_x = width / calibration.width
        scale_y = height / calibration.height
        matrix = calibration.intrinsic.copy()
        matrix[0] *= scale_x
        matrix[1] *= scale_y
    return CameraCalibration(
        name=calibration.name,
        intrinsic=np.asarray(matrix, dtype=np.float64),
        distortion=np.zeros_like(calibration.distortion, dtype=np.float64),
        external=calibration.external.copy(),
        width=int(width),
        height=int(height),
    )


def rectify_image(
    image: np.ndarray,
    calibration: CameraCalibration,
    output_size: tuple[int, int] | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, CameraCalibration]:
    """Rectify a raw Brown-distorted RGB image and return its new calibration."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    if image.shape[1] != calibration.width or image.shape[0] != calibration.height:
        raise ValueError(
            f"raw image shape {image.shape[:2]} does not match calibration "
            f"{(calibration.height, calibration.width)}"
        )
    rectified = rectified_calibration(calibration, output_size, alpha)
    size = (rectified.width, rectified.height)
    if np.any(calibration.distortion):
        map_x, map_y = cv2.initUndistortRectifyMap(
            calibration.intrinsic,
            calibration.distortion,
            np.eye(3, dtype=np.float64),
            rectified.intrinsic,
            size,
            cv2.CV_32FC1,
        )
        result = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        result = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return result, rectified


def load_distortion(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result = {}
    for key, value in payload.items():
        if key.startswith("camera_image_"):
            result[int(key.rsplit("_", 1)[1])] = np.asarray(value.get("D_manual", []), dtype=np.float64)
    return result


def load_calibrations(
    config_path: Path, distortion_path: Path | None = None
) -> dict[str, CameraCalibration]:
    payload = json.loads(config_path.read_text())
    entries = payload.get("camera_configs", payload)
    distortions = load_distortion(distortion_path) if distortion_path else {}
    result = {}
    for index, name in enumerate(CALIBRATION_CAMERA_ORDER):
        entry = entries[index]
        intrinsic = entry["camera_internal"]
        matrix = np.array(
            [[intrinsic["fx"], 0.0, intrinsic["cx"]], [0.0, intrinsic["fy"], intrinsic["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        external = np.asarray(entry["camera_external"], dtype=np.float64).reshape(4, 4)
        result[name] = CameraCalibration(
            name=name,
            intrinsic=matrix,
            distortion=distortions.get(index, np.zeros(5, dtype=np.float64)),
            external=external,
            width=int(entry["width"]),
            height=int(entry["height"]),
        )
    return result


def project_world_points(
    points: np.ndarray, calibration: CameraCalibration, undistort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Project points in the calibration/world frame into image pixels.

    The supplied external matrix is interpreted as world-to-camera. This is
    the convention used by the captured camera_config files and can be
    overridden by inverting the matrix at the call site for another rig.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, >=3]")
    homogeneous = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
    camera_points = (calibration.external @ homogeneous.T).T[:, :3]
    valid = camera_points[:, 2] > 1e-4
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        xyz = camera_points[valid]
        if undistort and np.any(calibration.distortion):
            projected, _ = cv2.projectPoints(
                xyz.astype(np.float64), np.zeros(3), np.zeros(3), calibration.intrinsic, calibration.distortion
            )
            pixels[valid] = projected[:, 0, :]
        else:
            normalized = xyz[:, :2] / xyz[:, 2:3]
            pixels[valid] = normalized @ calibration.intrinsic[:2, :2].T + calibration.intrinsic[:2, 2]
    finite = np.isfinite(pixels).all(axis=1)
    valid &= finite
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        in_bounds = (
            (pixels[valid_indices, 0] >= 0)
            & (pixels[valid_indices, 0] < calibration.width)
            & (pixels[valid_indices, 1] >= 0)
            & (pixels[valid_indices, 1] < calibration.height)
        )
        valid[valid_indices] = in_bounds
    return pixels, valid
