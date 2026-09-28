"""Project dense source semantic maps through the same geometry as RGB."""

from __future__ import annotations

import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration


def _project_world(points: np.ndarray, calibration: CameraCalibration):
    camera = (
        calibration.external
        @ np.c_[points, np.ones(len(points), dtype=np.float32)].T
    ).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-4)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    if valid.any():
        normalized = camera[valid, :2] / camera[valid, 2:3]
        uv[valid] = normalized @ calibration.intrinsic[:2, :2].T + calibration.intrinsic[:2, 2]
    valid &= (
        (uv[:, 0] >= 0) & (uv[:, 0] < calibration.width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < calibration.height)
    )
    return uv, camera[:, 2], valid


def reproject_dense_maps(
    source_maps: list[np.ndarray],
    source_depths: list[np.ndarray],
    source_calibrations: list[CameraCalibration],
    target_calibration: CameraCalibration,
    *,
    source_confidences: list[np.ndarray] | None = None,
    splat_radius: int = 0,
    exclusive_source_priority: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject arbitrary dense maps with metric depth and z-buffering.

    ``source_maps`` may contain RGB or DINO/AnyUP features. The function does
    no feature blending across unrelated views; with exclusive priority the
    first source owns observed pixels and later sources only fill holes.
    """
    if not (len(source_maps) == len(source_depths) == len(source_calibrations)):
        raise ValueError("source maps, depths, and calibrations must match")
    if not source_maps:
        raise ValueError("at least one source map is required")
    channels = int(np.asarray(source_maps[0]).shape[-1])
    height, width = target_calibration.height, target_calibration.width
    output = np.zeros((height, width, channels), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    provenance = np.full((height, width), -1, dtype=np.int32)
    for view, (map_value, depth, calibration) in enumerate(
        zip(source_maps, source_depths, source_calibrations)
    ):
        map_value = np.asarray(map_value, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if map_value.ndim != 3 or map_value.shape[:2] != depth.shape:
            raise ValueError("source map and depth must have matching H,W")
        if map_value.shape[-1] != channels:
            raise ValueError("all source maps must have the same channel count")
        if map_value.shape[1] != calibration.width or map_value.shape[0] != calibration.height:
            raise ValueError("source map must match source calibration")
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        z = depth.reshape(-1)
        x = (xx.reshape(-1) - calibration.intrinsic[0, 2]) / calibration.intrinsic[0, 0] * z
        y = (yy.reshape(-1) - calibration.intrinsic[1, 2]) / calibration.intrinsic[1, 1] * z
        source_camera_points = np.stack([x, y, z], axis=-1)
        world = (
            np.linalg.inv(calibration.external)
            @ np.c_[source_camera_points, np.ones(len(source_camera_points), dtype=np.float32)].T
        ).T[:, :3]
        uv, target_depth, projected = _project_world(world, target_calibration)
        valid = (np.isfinite(z) & (z > 1e-4) & projected)
        if source_confidences is not None:
            valid &= np.asarray(source_confidences[view], dtype=np.float32).reshape(-1) > 0
        order = np.flatnonzero(valid)[np.argsort(target_depth[valid])[::-1]]
        pixels = np.rint(uv[order]).astype(np.int64)
        inside = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        order, pixels = order[inside], pixels[inside]
        if not len(order):
            continue
        flat_depth = depth_buffer.reshape(-1)
        if splat_radius == 0:
            flat_pixels = pixels[:, 1] * width + pixels[:, 0]
            if exclusive_source_priority:
                available = np.isinf(flat_depth[flat_pixels])
                flat_pixels, order, pixels = flat_pixels[available], order[available], pixels[available]
                if not len(order):
                    continue
                flat_depth[flat_pixels] = target_depth[order]
                selected = np.ones(len(order), dtype=bool)
            else:
                np.minimum.at(flat_depth, flat_pixels, target_depth[order])
                selected = target_depth[order] <= flat_depth[flat_pixels] + 1e-6
            selected_order = order[selected]
            selected_pixels = pixels[selected]
            output[selected_pixels[:, 1], selected_pixels[:, 0]] = map_value.reshape(-1, channels)[selected_order]
            provenance[selected_pixels[:, 1], selected_pixels[:, 0]] = view
            continue
        radius = int(splat_radius)
        for index, (u, v) in zip(order, pixels):
            y0, y1 = max(0, v - radius), min(height - 1, v + radius)
            x0, x1 = max(0, u - radius), min(width - 1, u + radius)
            region = depth_buffer[y0:y1 + 1, x0:x1 + 1]
            selected = ~np.isfinite(region) if exclusive_source_priority else target_depth[index] <= region
            region[selected] = target_depth[index]
            output[y0:y1 + 1, x0:x1 + 1][selected] = map_value.reshape(-1, channels)[index]
            provenance[y0:y1 + 1, x0:x1 + 1][selected] = view
    return output, np.isfinite(depth_buffer), provenance

