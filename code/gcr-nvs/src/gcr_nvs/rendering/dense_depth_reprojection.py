"""Dense calibrated RGB reprojection from metric source depth maps."""

from __future__ import annotations

import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration


def _project(points: np.ndarray, calibration: CameraCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (calibration.external @ np.c_[points, np.ones(len(points), dtype=np.float32)].T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-4)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    uv[valid] = camera[valid, :2] / camera[valid, 2:3] @ calibration.intrinsic[:2, :2].T + calibration.intrinsic[:2, 2]
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < calibration.width) & (uv[:, 1] >= 0) & (uv[:, 1] < calibration.height)
    return uv, camera[:, 2], valid


def reproject_dense_rgb(
    source_rgbs: list[np.ndarray],
    source_depths: list[np.ndarray],
    source_calibrations: list[CameraCalibration],
    target_calibration: CameraCalibration,
    *,
    source_confidences: list[np.ndarray] | None = None,
    splat_radius: int = 1,
    exclusive_source_priority: bool = False,
    return_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project dense source pixels, SE(3)-transform, and z-buffer RGB.

    All calibrations must describe the same rectified pinhole image grids. A
    source pixel contributes only where its depth is finite; no RGB is invented
    by this function, and true disocclusions remain invalid for a later head.
    """
    if not (len(source_rgbs) == len(source_depths) == len(source_calibrations)):
        raise ValueError("source RGB, depth, and calibration lists must match")
    h, w = int(target_calibration.height), int(target_calibration.width)
    out = np.zeros((h, w, 3), dtype=np.float32)
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
    provenance = np.full((h, w), -1, dtype=np.int32)
    for view, (rgb, depth, calibration) in enumerate(zip(source_rgbs, source_depths, source_calibrations)):
        rgb = np.asarray(rgb, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[:2] != depth.shape:
            raise ValueError("each RGB/depth pair must have matching H,W")
        if (rgb.shape[1], rgb.shape[0]) != (calibration.width, calibration.height):
            raise ValueError("source RGB shape must match source calibration")
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        z = depth
        valid = np.isfinite(z) & (z > 1e-4)
        x = (xx - calibration.intrinsic[0, 2]) / calibration.intrinsic[0, 0] * z
        y = (yy - calibration.intrinsic[1, 2]) / calibration.intrinsic[1, 1] * z
        points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        world = (np.linalg.inv(calibration.external) @ np.c_[points_cam, np.ones(len(points_cam), dtype=np.float32)].T).T[:, :3]
        uv, target_z, projected = _project(world, target_calibration)
        valid = valid.reshape(-1) & projected
        if source_confidences is not None:
            confidence = np.asarray(source_confidences[view], dtype=np.float32).reshape(-1)
            valid &= confidence > 0.0
        candidates = np.flatnonzero(valid)
        if not candidates.size:
            continue
        # Far-to-near ordering gives the nearest surface the final write.
        order = candidates[np.argsort(target_z[candidates])[::-1]]
        px = np.rint(uv[order]).astype(np.int64)
        inside = (
            (px[:, 0] >= 0) & (px[:, 0] < w)
            & (px[:, 1] >= 0) & (px[:, 1] < h)
        )
        order = order[inside]
        px = px[inside]
        if not len(order):
            continue
        if splat_radius == 0:
            flat_pixel = px[:, 1] * w + px[:, 0]
            flat_depth = depth_buffer.reshape(-1)
            if exclusive_source_priority:
                available = np.isinf(flat_depth[flat_pixel])
                flat_pixel = flat_pixel[available]
                order = order[available]
                px = px[available]
                if not len(order):
                    continue
                flat_depth[flat_pixel] = target_z[order]
                winner = np.ones(len(order), dtype=bool)
            else:
                np.minimum.at(flat_depth, flat_pixel, target_z[order])
                winner = target_z[order] <= flat_depth[flat_pixel] + 1e-6
            winner_indices = order[winner]
            winner_pixels = px[winner]
            out[winner_pixels[:, 1], winner_pixels[:, 0]] = rgb.reshape(-1, 3)[winner_indices]
            provenance[winner_pixels[:, 1], winner_pixels[:, 0]] = view
            continue
        for index, (u, v) in zip(order, px):
            radius = int(splat_radius)
            y0, y1 = max(0, v - radius), min(h - 1, v + radius)
            x0, x1 = max(0, u - radius), min(w - 1, u + radius)
            region = depth_buffer[y0:y1 + 1, x0:x1 + 1]
            nearer = ~np.isfinite(region) if exclusive_source_priority else target_z[index] <= region
            region[nearer] = target_z[index]
            out_region = out[y0:y1 + 1, x0:x1 + 1]
            out_region[nearer] = rgb.reshape(-1, 3)[index]
            prov_region = provenance[y0:y1 + 1, x0:x1 + 1]
            prov_region[nearer] = view
    validity = np.isfinite(depth_buffer)
    if return_depth:
        return out, validity, provenance, depth_buffer
    return out, validity, provenance
