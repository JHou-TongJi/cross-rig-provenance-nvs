"""Deterministic geometry renderer used before any learned module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points


@dataclass
class RenderResult:
    channels: np.ndarray
    source_provenance: np.ndarray
    point_index: np.ndarray
    structure_validity: np.ndarray
    appearance_validity: np.ndarray

    @property
    def coarse_rgb(self) -> np.ndarray:
        return self.channels[0:3]

    @property
    def depth(self) -> np.ndarray:
        return self.channels[3:4]

    @property
    def normal(self) -> np.ndarray:
        return self.channels[4:7]

    @property
    def opacity(self) -> np.ndarray:
        return self.channels[7:8]

    @property
    def confidence(self) -> np.ndarray:
        return self.channels[8:9]

    @property
    def view_variance(self) -> np.ndarray:
        return self.channels[9:10]

    @property
    def source_count(self) -> np.ndarray:
        return self.channels[10:11]

    @property
    def dynamic_mask(self) -> np.ndarray:
        return self.channels[11:12]


class FixedSplatRenderer:
    """Z-buffer point splat with confidence and provenance channels.

    The renderer is deliberately non-learned. It is a geometry sanity check
    and the frozen input to the first residual-refinement stage.
    """

    channel_names = (
        "coarse_rgb",
        "depth",
        "normal",
        "opacity",
        "confidence",
        "view_variance",
        "source_count",
        "dynamic_mask",
    )

    def render(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        calibration: CameraCalibration,
        normals: np.ndarray | None = None,
        dynamic_mask: np.ndarray | None = None,
        point_confidence: np.ndarray | None = None,
        source_observation_count: np.ndarray | None = None,
        source_provenance: np.ndarray | None = None,
        color_validity: np.ndarray | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> RenderResult:
        width, height = output_size or (calibration.width, calibration.height)
        pixels, valid = project_world_points(points[:, :3], calibration)
        if output_size and output_size != (calibration.width, calibration.height):
            pixels[:, 0] *= width / calibration.width
            pixels[:, 1] *= height / calibration.height
        valid_indices = np.flatnonzero(valid)
        channels = np.zeros((12, height, width), dtype=np.float32)
        provenance = np.full((height, width), -1, dtype=np.int16)
        point_index = np.full((height, width), -1, dtype=np.int64)
        structure_validity = np.zeros((height, width), dtype=bool)
        appearance_validity = np.zeros((height, width), dtype=bool)
        if color_validity is None:
            color_validity = np.ones(len(points), dtype=bool)
        else:
            color_validity = np.asarray(color_validity, dtype=bool)
            if color_validity.shape != (len(points),):
                raise ValueError("color_validity must have shape [N]")
        if not len(valid_indices):
            channels[4:7] = 0.0
            return RenderResult(
                channels,
                provenance,
                point_index,
                structure_validity,
                appearance_validity,
            )
        xy = np.rint(pixels[valid_indices]).astype(np.int64)
        inside = (
            (xy[:, 0] >= 0) & (xy[:, 0] < width)
            & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        )
        valid_indices = valid_indices[inside]
        xy = xy[inside]
        camera_points = (calibration.external @ np.c_[points[valid_indices, :3], np.ones(len(valid_indices))].T).T[:, :3]
        # Sort by depth, then retain the first point per output pixel. This
        # replaces the slow Python point loop and is the critical training path.
        keys = xy[:, 1] * width + xy[:, 0]
        order = np.argsort(camera_points[:, 2], kind="stable")
        sorted_keys = keys[order]
        _, first = np.unique(sorted_keys, return_index=True)
        selected = order[first]
        pixel_x = xy[selected, 0]
        pixel_y = xy[selected, 1]
        point_ids = valid_indices[selected]
        point_index[pixel_y, pixel_x] = point_ids
        structure_validity[pixel_y, pixel_x] = True
        selected_appearance_valid = color_validity[point_ids]
        appearance_validity[pixel_y, pixel_x] = selected_appearance_valid
        color_x = pixel_x[selected_appearance_valid]
        color_y = pixel_y[selected_appearance_valid]
        color_point_ids = point_ids[selected_appearance_valid]
        channels[0:3, color_y, color_x] = colors[color_point_ids, :3].T
        channels[3, pixel_y, pixel_x] = camera_points[selected, 2]
        if normals is not None:
            channels[4:7, pixel_y, pixel_x] = normals[point_ids, :3].T
        channels[7, pixel_y, pixel_x] = selected_appearance_valid.astype(np.float32)
        channels[8, pixel_y, pixel_x] = point_confidence[point_ids] if point_confidence is not None else 1.0
        channels[10, pixel_y, pixel_x] = source_observation_count[point_ids] if source_observation_count is not None else 1.0
        channels[11, pixel_y, pixel_x] = dynamic_mask[point_ids] if dynamic_mask is not None else 0.0
        if source_provenance is not None:
            provenance[pixel_y, pixel_x] = source_provenance[point_ids]
        provenance[~appearance_validity] = -1
        return RenderResult(
            channels,
            provenance,
            point_index,
            structure_validity,
            appearance_validity,
        )


def merge_observations(
    points: np.ndarray,
    source_observations: list[tuple[np.ndarray, np.ndarray, CameraCalibration]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse source RGB samples and retain count/variance per 3D point."""
    colors = np.zeros((len(points), 3), dtype=np.float32)
    counts = np.zeros(len(points), dtype=np.float32)
    mean_sq = np.zeros((len(points), 3), dtype=np.float32)
    for sampled_colors, valid, _ in source_observations:
        valid = np.asarray(valid, dtype=bool)
        colors[valid] += sampled_colors[valid]
        mean_sq[valid] += sampled_colors[valid] ** 2
        counts[valid] += 1
    safe = counts.clip(min=1)[:, None]
    colors /= safe
    variance = (mean_sq / safe - colors**2).mean(axis=1)
    return colors, counts, variance


def save_render_result(result: RenderResult, output_dir, stem: str) -> None:
    from pathlib import Path
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(result.coarse_rgb.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)).save(
        output_dir / f"{stem}.jpg", quality=95
    )
    np.save(output_dir / f"{stem}.channels.npy", result.channels)
    np.save(output_dir / f"{stem}.source_provenance.npy", result.source_provenance)
    np.save(output_dir / f"{stem}.point_index.npy", result.point_index)
    np.save(output_dir / f"{stem}.valid_mask.npy", result.appearance_validity)
    np.save(output_dir / f"{stem}.structure_validity.npy", result.structure_validity)
    np.save(output_dir / f"{stem}.appearance_validity.npy", result.appearance_validity)
    np.save(output_dir / f"{stem}.inpaint_mask.npy", ~result.appearance_validity)
