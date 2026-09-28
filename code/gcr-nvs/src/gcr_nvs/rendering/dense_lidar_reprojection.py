"""Native-resolution RGB reprojection over LiDAR-only dense geometry.

The geometry path only consumes calibrated LiDAR surfaces and their normals.
RGB is sampled after the target surface has been fixed, so appearance cannot
move a surface or predict depth.  This avoids the resolution bottleneck of
encoding an entire native image into a few tens of thousands of RGB surfels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points
from gcr_nvs.geometry.camera import TargetCamera


@dataclass(frozen=True)
class DenseLidarGeometry:
    surface_points: np.ndarray
    normal: np.ndarray
    range_m: np.ndarray
    seed_distance_px: np.ndarray
    source_point_index: np.ndarray
    validity: np.ndarray


@dataclass(frozen=True)
class DenseRGBReprojection:
    rgb: np.ndarray
    validity: np.ndarray
    provenance: np.ndarray


def _remap_points(image: np.ndarray, uv: np.ndarray, chunk_size: int = 30_000) -> np.ndarray:
    """Sample an arbitrary point list without OpenCV's SHRT_MAX image limit."""
    chunks = []
    for start in range(0, len(uv), chunk_size):
        selected = uv[start:start + chunk_size].astype(np.float32)
        chunks.append(cv2.remap(
            np.asarray(image, dtype=np.float32),
            selected[:, 0, None],
            selected[:, 1, None],
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[:, 0])
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), np.float32)


def frequency_preserving_rgb_fusion(
    aligned_rgb: np.ndarray,
    aligned_validity: np.ndarray,
    native_reprojection_rgb: np.ndarray,
    native_reprojection_validity: np.ndarray,
    output_validity: np.ndarray,
    detail_gain: float = 0.75,
    detail_sigma_px: float = 1.0,
) -> np.ndarray:
    """Keep aligned low frequencies while restoring real source high frequencies.

    The normalized blur prevents invalid/unknown pixels from creating false
    edges.  This function changes RGB only and receives no mechanism capable
    of changing geometry.
    """
    if aligned_rgb.shape != native_reprojection_rgb.shape:
        raise ValueError("RGB inputs must have matching [H,W,3] shapes")
    if aligned_validity.shape != native_reprojection_validity.shape:
        raise ValueError("validity inputs must have matching [H,W] shapes")
    native_mask = native_reprojection_validity.astype(np.float32)
    native_denominator = cv2.GaussianBlur(
        native_mask, (0, 0), sigmaX=detail_sigma_px, sigmaY=detail_sigma_px,
    ).clip(min=1e-4)
    native_low_frequency = np.stack([
        cv2.GaussianBlur(
            native_reprojection_rgb[..., channel] * native_mask,
            (0, 0), sigmaX=detail_sigma_px, sigmaY=detail_sigma_px,
        ) / native_denominator
        for channel in range(3)
    ], axis=-1)
    aligned_mask = aligned_validity.astype(np.float32)
    aligned_denominator = cv2.GaussianBlur(
        aligned_mask, (0, 0), sigmaX=detail_sigma_px, sigmaY=detail_sigma_px,
    ).clip(min=1e-4)
    aligned_low_frequency = np.stack([
        cv2.GaussianBlur(
            aligned_rgb[..., channel] * aligned_mask,
            (0, 0), sigmaX=detail_sigma_px, sigmaY=detail_sigma_px,
        ) / aligned_denominator
        for channel in range(3)
    ], axis=-1)
    detail = (
        native_reprojection_rgb - native_low_frequency
    ) * native_mask[..., None]
    result = np.zeros_like(aligned_rgb, dtype=np.float32)
    result[aligned_validity] = aligned_rgb[aligned_validity]
    common = aligned_validity & native_reprojection_validity
    result[common] = np.clip(
        aligned_low_frequency[common] + float(detail_gain) * detail[common],
        0.0, 1.0,
    )
    fallback = ~aligned_validity & output_validity
    result[fallback] = native_reprojection_rgb[fallback]
    result[~output_validity] = 0.0
    return result


def _zbuffer_seeds(
    points: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    uv, projected = project_world_points(points, calibration)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float32)]
    camera_depth = (calibration.external @ homogeneous.T).T[:, 2]
    candidates = np.flatnonzero(projected)
    seed_index = np.full((calibration.height, calibration.width), -1, dtype=np.int32)
    if not len(candidates):
        return seed_index, uv
    xy = np.rint(uv[candidates]).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0) & (xy[:, 0] < calibration.width)
        & (xy[:, 1] >= 0) & (xy[:, 1] < calibration.height)
    )
    candidates = candidates[inside]
    xy = xy[inside]
    order = np.argsort(camera_depth[candidates])[::-1]
    # Far-to-near assignment leaves the nearest surface at duplicate pixels.
    seed_index[xy[order, 1], xy[order, 0]] = candidates[order].astype(np.int32)
    return seed_index, uv


def rasterize_lidar_range(
    points: np.ndarray,
    calibration: CameraCalibration,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer measured LiDAR ray range without consulting RGB."""
    width = int(width or calibration.width)
    height = int(height or calibration.height)
    resized = TargetCamera(calibration).resized(width, height).calibration
    points = np.asarray(points, dtype=np.float32)
    seed_index, _ = _zbuffer_seeds(points, resized)
    valid = seed_index >= 0
    safe = seed_index.clip(min=0)
    ranges = np.linalg.norm(
        points[safe] - resized.camera_center.astype(np.float32), axis=-1,
    ).astype(np.float32)
    ranges[~valid] = 0.0
    return ranges, valid


def densify_lidar_surfaces(
    points: np.ndarray,
    normals: np.ndarray,
    calibration: CameraCalibration,
    width: int | None = None,
    height: int | None = None,
    maximum_seed_distance_px: float = 28.0,
    maximum_plane_deviation_m: float = 1.0,
    maximum_relative_plane_deviation: float = 0.08,
    maximum_local_depth_jump_m: float = 3.0,
) -> DenseLidarGeometry:
    """Expand LiDAR surfels into a dense target field by tangent-plane rays.

    Nearest-neighbour assignment happens only in image coordinates.  The
    actual depth is the intersection between the target ray and the local
    LiDAR PCA plane; RGB is never inspected by this function.
    """
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape [N,3]")
    width = int(width or calibration.width)
    height = int(height or calibration.height)
    target = TargetCamera(calibration).resized(width, height)
    resized = target.calibration
    seed_index, _ = _zbuffer_seeds(points, resized)
    seed_mask = seed_index >= 0
    if not seed_mask.any():
        shape = (height, width)
        return DenseLidarGeometry(
            surface_points=np.zeros((*shape, 3), dtype=np.float32),
            normal=np.zeros((*shape, 3), dtype=np.float32),
            range_m=np.zeros(shape, dtype=np.float32),
            seed_distance_px=np.full(shape, np.inf, dtype=np.float32),
            source_point_index=np.full(shape, -1, dtype=np.int32),
            validity=np.zeros(shape, dtype=bool),
        )

    distance_input = (~seed_mask).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    max_label = int(labels.max())
    label_to_point = np.full(max_label + 1, -1, dtype=np.int32)
    seed_labels = labels[seed_mask]
    label_to_point[seed_labels] = seed_index[seed_mask]
    nearest = label_to_point[labels]
    nearest_valid = nearest >= 0
    safe_nearest = nearest.clip(min=0)

    rays = target.ray_map(width, height).transpose(1, 2, 0)
    center = resized.camera_center.astype(np.float32)
    anchor = points[safe_nearest]
    normal = normals[safe_nearest]
    numerator = np.sum(normal * (anchor - center), axis=-1)
    denominator = np.sum(normal * rays, axis=-1)
    plane_range = numerator / np.where(
        np.abs(denominator) > 1e-5, denominator, np.nan,
    )
    anchor_range = np.linalg.norm(anchor - center, axis=-1)
    surface_points = center + rays * plane_range[..., None]
    rotation = resized.external[:3, :3].astype(np.float32)
    translation = resized.external[:3, 3].astype(np.float32)
    anchor_depth = anchor @ rotation[2] + translation[2]
    plane_depth = surface_points @ rotation[2] + translation[2]
    # Reject nearest-surfel assignments that cross a strong LiDAR depth edge.
    # Keeping this gate geometry-only prevents color bleeding across vehicles
    # and road/background boundaries.
    anchor_depth_image = np.where(nearest_valid, anchor_depth, 0.0).astype(np.float32)
    median_depth = cv2.medianBlur(anchor_depth_image, 5)
    local_depth_edge = (
        nearest_valid
        & (median_depth > 0.1)
        & (np.abs(anchor_depth - median_depth) > maximum_local_depth_jump_m)
    )
    allowed_deviation = np.maximum(
        maximum_plane_deviation_m,
        maximum_relative_plane_deviation * np.abs(anchor_depth),
    )
    validity = nearest_valid & np.isfinite(plane_range) & (plane_range > 0.1)
    validity &= distance <= float(maximum_seed_distance_px)
    validity &= np.abs(plane_depth - anchor_depth) <= allowed_deviation
    validity &= ~local_depth_edge
    surface_points[~validity] = 0.0
    normal = normal.copy()
    normal[~validity] = 0.0
    plane_range = np.where(validity, plane_range, 0.0).astype(np.float32)
    nearest = np.where(validity, nearest, -1).astype(np.int32)
    return DenseLidarGeometry(
        surface_points=surface_points.astype(np.float32),
        normal=normal.astype(np.float32),
        range_m=plane_range,
        seed_distance_px=distance.astype(np.float32),
        source_point_index=nearest,
        validity=validity,
    )


def reproject_source_rgb(
    geometry: DenseLidarGeometry,
    source_images: list[np.ndarray] | tuple[np.ndarray, ...],
    source_calibrations: list[CameraCalibration] | tuple[CameraCalibration, ...],
    point_view_weights: np.ndarray,
    point_view_validity: np.ndarray,
    hard_selection: bool = True,
    soft_temperature: float = 1.5,
    spatial_smoothing_sigma: float = 1.0,
) -> DenseRGBReprojection:
    """Sample real source RGB with optional surface-consistent soft fusion."""
    if len(source_images) != len(source_calibrations):
        raise ValueError("each source image requires one calibration")
    view_count = len(source_images)
    weights = np.asarray(point_view_weights, dtype=np.float32)
    view_validity = np.asarray(point_view_validity, dtype=bool)
    if weights.shape != view_validity.shape or weights.shape[1] != view_count:
        raise ValueError("point view tensors must have shape [N,V]")
    height, width = geometry.validity.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    provenance = np.full((height, width), -1, dtype=np.int16)
    point_index = geometry.source_point_index
    geometry_pixels = np.flatnonzero(geometry.validity.reshape(-1))
    if not len(geometry_pixels):
        return DenseRGBReprojection(rgb, provenance >= 0, provenance)
    selected_points = point_index.reshape(-1)[geometry_pixels]
    selected_scores = weights[selected_points].copy()
    selected_scores[~view_validity[selected_points]] = -np.inf
    # Appearance models provide either logits or normalized probabilities.
    # Convert probabilities to log space before the same softmax temperature.
    finite_scores = selected_scores[np.isfinite(selected_scores)]
    probability_rows = (
        len(finite_scores) > 0
        and np.all(finite_scores >= 0.0)
        and np.all(finite_scores <= 1.0)
    )
    if probability_rows:
        selected_scores = np.log(np.clip(selected_scores, 1e-6, 1.0))
    source_choice = np.argmax(selected_scores, axis=1)
    has_source = np.isfinite(selected_scores).any(axis=1)
    flat_points = geometry.surface_points.reshape(-1, 3)
    flat_rgb = rgb.reshape(-1, 3)
    flat_provenance = provenance.reshape(-1)

    if hard_selection:
        source_pixels = [
            geometry_pixels[has_source & (source_choice == source_index)]
            for source_index in range(view_count)
        ]
    else:
        safe_scores = np.where(np.isfinite(selected_scores), selected_scores, -1e9)
        safe_scores = safe_scores - np.max(safe_scores, axis=1, keepdims=True)
        source_weights = np.exp(safe_scores / max(float(soft_temperature), 1e-4))
        source_weights *= np.isfinite(selected_scores)
        source_weights /= source_weights.sum(axis=1, keepdims=True).clip(min=1e-6)
        source_pixels = [
            geometry_pixels[has_source & (source_weights[:, source_index] > 1e-5)]
            for source_index in range(view_count)
        ]
    flat_rgb_accum = np.zeros((height * width, 3), dtype=np.float32)
    flat_weight_accum = np.zeros(height * width, dtype=np.float32)
    flat_soft_weights = np.zeros((height * width, view_count), dtype=np.float32)

    for source_index, (image, calibration) in enumerate(zip(
        source_images, source_calibrations,
    )):
        selected = source_pixels[source_index]
        if not len(selected):
            continue
        uv, projected = project_world_points(flat_points[selected], calibration)
        if not projected.any():
            continue
        destination = selected[projected]
        sample_uv = uv[projected].astype(np.float32)
        sampled = _remap_points(image, sample_uv)
        if hard_selection:
            flat_rgb[destination] = sampled
            flat_provenance[destination] = source_index
            continue
        # ``selected`` is already filtered by the source weight; recover the
        # corresponding rows in the original point list by pixel lookup.
        row_lookup = {int(pixel): row for row, pixel in enumerate(geometry_pixels)}
        rows = np.asarray([row_lookup[int(pixel)] for pixel in destination], dtype=np.int64)
        pixel_weights = source_weights[rows, source_index].astype(np.float32)
        flat_rgb_accum[destination] += sampled * pixel_weights[:, None]
        flat_weight_accum[destination] += pixel_weights
        flat_soft_weights[destination, source_index] += pixel_weights
    if not hard_selection:
        # Smooth only the accumulated contribution and its denominator. This
        # removes one-pixel provenance cracks without mixing unrelated depth
        # surfaces, because all contributions came from the same dense target
        # surface assignment.
        rgb_image = flat_rgb_accum.reshape(height, width, 3)
        weight_image = flat_weight_accum.reshape(height, width)
        if spatial_smoothing_sigma > 0.0:
            rgb_image = np.stack([
                cv2.GaussianBlur(
                    rgb_image[..., channel], (0, 0), spatial_smoothing_sigma,
                )
                for channel in range(3)
            ], axis=-1)
            weight_image = cv2.GaussianBlur(
                weight_image, (0, 0), spatial_smoothing_sigma,
            )
        soft_valid = weight_image > 1e-4
        rgb = np.where(
            soft_valid[..., None],
            rgb_image / weight_image[..., None].clip(min=1e-6),
            0.0,
        ).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        provenance = np.argmax(flat_soft_weights, axis=1).reshape(height, width).astype(np.int16)
        provenance[~soft_valid] = -1
        return DenseRGBReprojection(rgb, soft_valid, provenance)
    return DenseRGBReprojection(rgb, provenance >= 0, provenance)


def reproject_far_field_rgb(
    target: CameraCalibration,
    source_image: np.ndarray,
    source: CameraCalibration,
    width: int | None = None,
    height: int | None = None,
    far_range_m: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate target rays into a real source image for sky/far background."""
    width = int(width or target.width)
    height = int(height or target.height)
    resized_target = TargetCamera(target).resized(width, height)
    rays = resized_target.ray_map(width, height).transpose(1, 2, 0)
    points = resized_target.calibration.camera_center + rays.reshape(-1, 3) * far_range_m
    uv, valid = project_world_points(points, source)
    sampled = cv2.remap(
        np.asarray(source_image, dtype=np.float32),
        uv[:, 0].astype(np.float32).reshape(height, width),
        uv[:, 1].astype(np.float32).reshape(height, width),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled, valid.reshape(height, width)
