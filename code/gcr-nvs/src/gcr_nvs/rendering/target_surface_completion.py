"""Conservative target-surface repair followed by inverse RGB sampling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from gcr_nvs.geometry.calibration import CameraCalibration


@dataclass(frozen=True)
class SurfaceCompletion:
    depth: np.ndarray
    candidate_mask: np.ndarray
    distance_px: np.ndarray


@dataclass(frozen=True)
class InverseSampling:
    rgb: np.ndarray
    validity: np.ndarray
    provenance: np.ndarray
    log_depth_error: np.ndarray


def _remap_points(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    interpolation: int,
    border_value: float | tuple[float, ...],
    chunk_size: int = 24000,
) -> np.ndarray:
    """Sample arbitrary points in chunks to satisfy OpenCV's 16-bit limit."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    channels = 1 if image.ndim == 2 else image.shape[2]
    shape = (len(x),) if channels == 1 else (len(x), channels)
    output = np.empty(shape, dtype=np.float32)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        sampled = cv2.remap(
            image.astype(np.float32), x[start:stop][None], y[start:stop][None],
            interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        output[start:stop] = sampled[0]
    return output


def complete_nearest_target_surface(
    depth: np.ndarray,
    validity: np.ndarray,
    *,
    maximum_distance_px: float = 2.0,
) -> SurfaceCompletion:
    """Extend only short target-view cracks with their nearest valid surface.

    This is intentionally conservative. It repairs rasterization gaps but does
    not claim to infer the background surface of a large disocclusion.
    """
    depth = np.asarray(depth, dtype=np.float32)
    validity = np.asarray(validity, dtype=bool)
    if depth.ndim != 2 or validity.shape != depth.shape:
        raise ValueError("depth and validity must have matching H,W shapes")
    finite = validity & np.isfinite(depth) & (depth > 1e-4)
    if not finite.any():
        raise ValueError("target surface has no finite seed depth")
    distance, nearest = ndimage.distance_transform_edt(~finite, return_indices=True)
    candidate = (~finite) & (distance <= float(maximum_distance_px))
    completed = depth.copy()
    completed[candidate] = depth[nearest[0][candidate], nearest[1][candidate]]
    return SurfaceCompletion(
        depth=completed,
        candidate_mask=candidate,
        distance_px=distance.astype(np.float32),
    )


def inverse_sample_source_rgb(
    target_depth: np.ndarray,
    target_calibration: CameraCalibration,
    source_rgbs: list[np.ndarray],
    source_depths: list[np.ndarray],
    source_calibrations: list[CameraCalibration],
    candidate_mask: np.ndarray,
    *,
    source_confidences: list[np.ndarray] | None = None,
    maximum_log_depth_error: float = 0.04,
) -> InverseSampling:
    """Pull source RGB through a completed target depth with depth validation.

    A candidate is accepted only when the 3D target point projects inside a
    source image and agrees with that source's own dense depth. This recovers
    forward-splat cracks without inventing RGB for true disocclusions.
    """
    if not (len(source_rgbs) == len(source_depths) == len(source_calibrations)):
        raise ValueError("source RGB, depth, and calibration lists must match")
    if source_confidences is not None and len(source_confidences) != len(source_rgbs):
        raise ValueError("source confidence list must match source views")
    target_depth = np.asarray(target_depth, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    height, width = target_depth.shape
    if candidate_mask.shape != target_depth.shape:
        raise ValueError("candidate mask must match target depth")
    if (width, height) != (target_calibration.width, target_calibration.height):
        raise ValueError("target depth shape must match target calibration")

    rgb_output = np.zeros((height, width, 3), dtype=np.float32)
    validity = np.zeros((height, width), dtype=bool)
    provenance = np.full((height, width), -1, dtype=np.int16)
    error_output = np.full((height, width), np.inf, dtype=np.float32)
    yy, xx = np.nonzero(candidate_mask & np.isfinite(target_depth) & (target_depth > 1e-4))
    if not len(xx):
        return InverseSampling(rgb_output, validity, provenance, error_output)

    z = target_depth[yy, xx]
    x = (xx.astype(np.float32) - target_calibration.intrinsic[0, 2]) \
        / target_calibration.intrinsic[0, 0] * z
    y = (yy.astype(np.float32) - target_calibration.intrinsic[1, 2]) \
        / target_calibration.intrinsic[1, 1] * z
    target_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
    best_error = np.full(len(xx), np.inf, dtype=np.float32)
    best_rgb = np.zeros((len(xx), 3), dtype=np.float32)
    best_source = np.full(len(xx), -1, dtype=np.int16)

    target_to_world = np.linalg.inv(target_calibration.external)
    for source_index, (rgb, source_depth, calibration) in enumerate(zip(
        source_rgbs, source_depths, source_calibrations,
    )):
        rgb = np.asarray(rgb, dtype=np.float32)
        source_depth = np.asarray(source_depth, dtype=np.float32)
        if rgb.shape[:2] != source_depth.shape:
            raise ValueError("each source RGB/depth pair must have matching H,W")
        if (rgb.shape[1], rgb.shape[0]) != (calibration.width, calibration.height):
            raise ValueError("source shape must match source calibration")
        transform = calibration.external @ target_to_world
        source_points = transform @ target_points
        source_z = source_points[2]
        safe_z = np.maximum(source_z, 1e-6)
        source_u = calibration.intrinsic[0, 0] * source_points[0] / safe_z \
            + calibration.intrinsic[0, 2]
        source_v = calibration.intrinsic[1, 1] * source_points[1] / safe_z \
            + calibration.intrinsic[1, 2]
        inside = (
            (source_z > 1e-4)
            & (source_u >= 0.0) & (source_u <= calibration.width - 1.0)
            & (source_v >= 0.0) & (source_v <= calibration.height - 1.0)
        )
        map_x = source_u.astype(np.float32)[:, None]
        map_y = source_v.astype(np.float32)[:, None]
        sampled_depth = _remap_points(
            source_depth, source_u, source_v, interpolation=cv2.INTER_NEAREST,
            border_value=0.0,
        )
        depth_error = np.abs(
            np.log(np.clip(sampled_depth, 0.5, 250.0))
            - np.log(np.clip(source_z, 0.5, 250.0))
        )
        accepted = inside & np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
        accepted &= depth_error <= float(maximum_log_depth_error)
        if source_confidences is not None:
            confidence = _remap_points(
                np.asarray(source_confidences[source_index], dtype=np.float32),
                source_u, source_v, interpolation=cv2.INTER_LINEAR,
                border_value=0.0,
            )
            accepted &= confidence > 0.0
        better = accepted & (depth_error < best_error)
        if not better.any():
            continue
        sampled_rgb = _remap_points(
            rgb, source_u, source_v, interpolation=cv2.INTER_LANCZOS4,
            border_value=(0.0, 0.0, 0.0),
        )
        best_error[better] = depth_error[better]
        best_rgb[better] = sampled_rgb[better]
        best_source[better] = source_index

    accepted = best_source >= 0
    accepted_y, accepted_x = yy[accepted], xx[accepted]
    rgb_output[accepted_y, accepted_x] = best_rgb[accepted]
    validity[accepted_y, accepted_x] = True
    provenance[accepted_y, accepted_x] = best_source[accepted]
    error_output[accepted_y, accepted_x] = best_error[accepted]
    return InverseSampling(rgb_output, validity, provenance, error_output)
