"""Deterministic point/surfel projection baseline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    calibration: CameraCalibration,
    provenance: np.ndarray | None = None,
    output_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    width, height = output_size or (calibration.width, calibration.height)
    pixels, valid = project_world_points(points[:, :3], calibration)
    if output_size and output_size != (calibration.width, calibration.height):
        pixels[:, 0] *= width / calibration.width
        pixels[:, 1] *= height / calibration.height
    valid &= np.isfinite(pixels).all(axis=1)
    finite_indices = np.flatnonzero(valid)
    if len(finite_indices):
        valid[finite_indices] &= (
            (pixels[finite_indices, 0] >= 0)
            & (pixels[finite_indices, 0] < width)
            & (pixels[finite_indices, 1] >= 0)
            & (pixels[finite_indices, 1] < height)
        )
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    source_provenance = np.full((height, width), -1, dtype=np.int16)
    if not np.any(valid):
        return {
            "rgb": rgb,
            "depth": depth,
            "valid_mask": confidence.astype(bool),
            "inpaint_mask": confidence == 0,
            "confidence": confidence,
            "source_provenance": source_provenance,
            "dynamic_mask": np.zeros((height, width), dtype=bool),
        }
    xy = np.rint(pixels[valid]).astype(np.int64)
    camera_xyz = (calibration.external @ np.c_[points[valid, :3], np.ones(np.sum(valid))].T).T[:, :3]
    order = np.argsort(camera_xyz[:, 2])
    for index in order:
        x, y = xy[index]
        if 0 <= x < width and 0 <= y < height and confidence[y, x] == 0:
            rgb[y, x] = colors[valid][index, :3]
            depth[y, x] = camera_xyz[index, 2]
            confidence[y, x] = 1.0
            if provenance is not None:
                source_provenance[y, x] = int(provenance[valid][index])
    return {
        "rgb": rgb,
        "depth": depth,
        "valid_mask": confidence > 0,
        "inpaint_mask": confidence == 0,
        "confidence": confidence,
        "source_provenance": source_provenance,
        "dynamic_mask": np.zeros((height, width), dtype=bool),
    }


def load_rgb(path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(size, resampling)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_render(render: dict[str, np.ndarray], output_dir, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(render["rgb"] * 255, 0, 255).astype(np.uint8)).save(output_dir / f"{stem}.jpg", quality=95)
    np.save(output_dir / f"{stem}.depth.npy", render["depth"])
    np.save(output_dir / f"{stem}.valid_mask.npy", render["valid_mask"])
    np.save(output_dir / f"{stem}.inpaint_mask.npy", render["inpaint_mask"])
    np.save(output_dir / f"{stem}.confidence.npy", render["confidence"])
    np.save(output_dir / f"{stem}.source_provenance.npy", render["source_provenance"])
    np.save(output_dir / f"{stem}.dynamic_mask.npy", render["dynamic_mask"])
