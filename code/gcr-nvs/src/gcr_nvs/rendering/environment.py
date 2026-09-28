"""Infinity-background rendering for source-observed sky appearance."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.geometry.camera import TargetCamera


@dataclass(frozen=True)
class EnvironmentRenderResult:
    rgb: np.ndarray
    validity: np.ndarray
    source_count: np.ndarray
    source_provenance: np.ndarray


def render_infinite_sky(
    source_images: list[np.ndarray] | tuple[np.ndarray, ...],
    source_sky_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    source_calibrations: list[CameraCalibration] | tuple[CameraCalibration, ...],
    target_calibration: CameraCalibration,
    width: int,
    height: int,
) -> EnvironmentRenderResult:
    if not (
        len(source_images) == len(source_sky_masks) == len(source_calibrations)
    ):
        raise ValueError("source image, sky mask, and calibration counts must match")
    rays = TargetCamera(target_calibration).ray_map(width, height)
    rays = rays.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64)
    color_sum = np.zeros((len(rays), 3), dtype=np.float64)
    weight_sum = np.zeros(len(rays), dtype=np.float64)
    source_count = np.zeros(len(rays), dtype=np.uint8)
    best_weight = np.zeros(len(rays), dtype=np.float64)
    provenance = np.full(len(rays), -1, dtype=np.int16)
    for source_index, (image, sky_mask, calibration) in enumerate(zip(
        source_images, source_sky_masks, source_calibrations,
    )):
        image = np.asarray(image, dtype=np.float32)
        sky_mask = np.asarray(sky_mask, dtype=bool)
        if image.shape[:2] != sky_mask.shape:
            raise ValueError("source image and sky mask shapes must match")
        directions = rays @ calibration.external[:3, :3].T
        in_front = directions[:, 2] > 1e-5
        uv = np.full((len(rays), 2), -1.0, dtype=np.float64)
        if in_front.any():
            projected, _ = cv2.projectPoints(
                directions[in_front],
                np.zeros(3),
                np.zeros(3),
                calibration.intrinsic,
                calibration.distortion,
            )
            uv[in_front] = projected[:, 0]
        map_x = np.clip(
            uv[:, 0], 0.0, image.shape[1] - 1.0,
        ).reshape(height, width).astype(np.float32)
        map_y = np.clip(
            uv[:, 1], 0.0, image.shape[0] - 1.0,
        ).reshape(height, width).astype(np.float32)
        sampled_rgb = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).reshape(-1, 3)
        sampled_sky = cv2.remap(
            sky_mask.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).reshape(-1).astype(bool)
        inside = (
            in_front
            & (uv[:, 0] >= -0.5)
            & (uv[:, 0] <= image.shape[1] - 0.5)
            & (uv[:, 1] >= -0.5)
            & (uv[:, 1] <= image.shape[0] - 0.5)
            & sampled_sky
        )
        # Optical-axis cosine favors the source view with the least angular stretch.
        cosine = np.clip(directions[:, 2] / np.linalg.norm(directions, axis=1).clip(min=1e-8), 0.0, 1.0)
        weight = np.exp(4.0 * cosine) * inside
        color_sum += sampled_rgb * weight[:, None]
        weight_sum += weight
        source_count += inside.astype(np.uint8)
        better = weight > best_weight
        provenance[better] = source_index
        best_weight[better] = weight[better]
    validity = weight_sum > 0
    rgb = np.zeros((len(rays), 3), dtype=np.float32)
    rgb[validity] = (color_sum[validity] / weight_sum[validity, None]).astype(np.float32)
    provenance[~validity] = -1
    return EnvironmentRenderResult(
        rgb=rgb.reshape(height, width, 3),
        validity=validity.reshape(height, width),
        source_count=source_count.reshape(height, width),
        source_provenance=provenance.reshape(height, width),
    )


def render_directional_source_fallback(
    source_images: list[np.ndarray] | tuple[np.ndarray, ...],
    source_calibrations: list[CameraCalibration] | tuple[CameraCalibration, ...],
    target_calibration: CameraCalibration,
    width: int,
    height: int,
) -> EnvironmentRenderResult:
    """Render a sharp, rotation-aligned source-image fallback for missing rays.

    This is deliberately not treated as reconstructed geometry.  It samples the
    source image whose optical axis is closest to each target ray, which is a
    useful visual completion for true disocclusions and a much more honest
    fallback than a constant colour.  The returned validity is kept separate by
    the unified field renderer so metrics can exclude it.
    """
    if len(source_images) != len(source_calibrations) or not source_images:
        raise ValueError("source images and calibrations must be non-empty and aligned")
    rays = TargetCamera(target_calibration).ray_map(width, height)
    rays = rays.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64)
    count = len(rays)
    best_rgb = np.zeros((count, 3), dtype=np.float32)
    best_weight = np.zeros(count, dtype=np.float64)
    provenance = np.full(count, -1, dtype=np.int16)
    source_count = np.zeros(count, dtype=np.uint8)
    target_name = target_calibration.name.split(":", 1)[-1]
    for source_index, (image, calibration) in enumerate(
        zip(source_images, source_calibrations)
    ):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("source images must be HWC RGB arrays")
        directions = rays @ calibration.external[:3, :3].T
        norm = np.linalg.norm(directions, axis=1).clip(min=1e-8)
        cosine = directions[:, 2] / norm
        in_front = cosine > 1e-5
        uv = np.full((count, 2), -1.0, dtype=np.float64)
        if in_front.any():
            projected, _ = cv2.projectPoints(
                directions[in_front], np.zeros(3), np.zeros(3),
                calibration.intrinsic, calibration.distortion,
            )
            uv[in_front] = projected[:, 0]
        inside = (
            in_front
            & (uv[:, 0] >= -0.5) & (uv[:, 0] <= image.shape[1] - 0.5)
            & (uv[:, 1] >= -0.5) & (uv[:, 1] <= image.shape[0] - 0.5)
        )
        source_count += inside.astype(np.uint8)
        map_x = np.clip(uv[:, 0], 0.0, image.shape[1] - 1.0).reshape(height, width).astype(np.float32)
        map_y = np.clip(uv[:, 1], 0.0, image.shape[0] - 1.0).reshape(height, width).astype(np.float32)
        sampled = cv2.remap(
            image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).reshape(count, 3)
        # A high power favours the least stretched source and avoids blurry
        # multi-view averaging across camera seams.
        source_name = calibration.name.split(":", 1)[-1]
        # A neighboring-frame observation from the same physical camera is
        # the most coherent source for a small rig perturbation.  Give it a
        # global priority so per-pixel angular selection cannot cut the image
        # into large camera-shaped pieces. Other views remain available only
        # where the preferred camera is out of bounds.
        is_temporal = ":" in calibration.name
        identity_priority = (
            1_000_000.0
            if source_name == target_name and not is_temporal
            else 1_000.0 if source_name == target_name else 1.0
        )
        weight = np.where(
            inside,
            identity_priority * np.clip(cosine, 0.0, 1.0) ** 8,
            0.0,
        )
        better = weight > best_weight
        best_rgb[better] = sampled[better]
        best_weight[better] = weight[better]
        provenance[better] = source_index
    validity = best_weight > 0.0
    provenance[~validity] = -1
    return EnvironmentRenderResult(
        rgb=best_rgb.reshape(height, width, 3),
        validity=validity.reshape(height, width),
        source_count=source_count.reshape(height, width),
        source_provenance=provenance.reshape(height, width),
    )
