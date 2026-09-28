"""Deterministic temporal LiDAR teacher for geometry-only supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EXACT_CURRENT = np.uint8(0)
TEMPORAL_FILLED = np.uint8(1)


@dataclass(frozen=True)
class TemporalNeighbor:
    points: np.ndarray
    timestamp: float
    transform_to_center: np.ndarray
    alignment_confidence: float
    dynamic_mask: np.ndarray | None = None
    alignment_valid: bool = True


@dataclass(frozen=True)
class TemporalFusionResult:
    points: np.ndarray
    source_kind: np.ndarray
    confidence: np.ndarray
    time_offset_s: np.ndarray

    @property
    def exact_mask(self) -> np.ndarray:
        return self.source_kind == EXACT_CURRENT

    @property
    def temporal_mask(self) -> np.ndarray:
        return self.source_kind == TEMPORAL_FILLED


def _valid_points(points: np.ndarray, max_range_m: float) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, >=3]")
    return np.isfinite(points[:, :3]).all(axis=1) & (
        np.linalg.norm(points[:, :3], axis=1) <= max_range_m
    )


def _adaptive_voxel_keys(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, :3], axis=1)
    bands = np.full(len(points), 2, dtype=np.int64)
    bands[distances < 60.0] = 1
    bands[distances < 30.0] = 0
    voxel_sizes = np.asarray([0.08, 0.16, 0.32], dtype=np.float64)
    coordinates = np.floor(
        points[:, :3].astype(np.float64) / voxel_sizes[bands, None]
    ).astype(np.int64)
    return np.column_stack([bands, coordinates])


class TemporalLidarTeacher:
    """Fuse aligned neighbors without ever overwriting current LiDAR points."""

    def __init__(
        self,
        max_time_offset_s: float = 1.0,
        max_range_m: float = 120.0,
        temporal_decay_s: float = 0.75,
        minimum_alignment_confidence: float = 0.25,
    ) -> None:
        self.max_time_offset_s = float(max_time_offset_s)
        self.max_range_m = float(max_range_m)
        self.temporal_decay_s = float(temporal_decay_s)
        self.minimum_alignment_confidence = float(minimum_alignment_confidence)
        if min(
            self.max_time_offset_s,
            self.max_range_m,
            self.temporal_decay_s,
        ) <= 0.0:
            raise ValueError("time, range, and decay limits must be positive")

    def fuse(
        self,
        current_points: np.ndarray,
        current_timestamp: float,
        neighbors: list[TemporalNeighbor] | tuple[TemporalNeighbor, ...],
    ) -> TemporalFusionResult:
        current_points = np.asarray(current_points)
        current_points = current_points[
            _valid_points(current_points, self.max_range_m)
        ].copy()
        occupied = set(map(tuple, _adaptive_voxel_keys(current_points)))

        fused = [current_points]
        source_kind = [np.full(len(current_points), EXACT_CURRENT, dtype=np.uint8)]
        confidence = [np.ones(len(current_points), dtype=np.float32)]
        time_offset = [np.zeros(len(current_points), dtype=np.float32)]

        ordered_neighbors = sorted(
            neighbors,
            key=lambda item: abs(float(item.timestamp) - float(current_timestamp)),
        )
        for neighbor in ordered_neighbors:
            offset = float(neighbor.timestamp) - float(current_timestamp)
            if (
                not neighbor.alignment_valid
                or abs(offset) > self.max_time_offset_s
                or neighbor.alignment_confidence < self.minimum_alignment_confidence
            ):
                continue
            points = np.asarray(neighbor.points)
            valid = _valid_points(points, self.max_range_m)
            if neighbor.dynamic_mask is not None:
                dynamic = np.asarray(neighbor.dynamic_mask, dtype=bool)
                if dynamic.shape != (len(points),):
                    raise ValueError("dynamic_mask must have shape [N]")
                valid &= ~dynamic
            points = points[valid].copy()
            if not len(points):
                continue
            transform = np.asarray(neighbor.transform_to_center, dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("transform_to_center must be finite [4,4]")
            points[:, :3] = (
                transform
                @ np.c_[points[:, :3], np.ones(len(points))].T
            ).T[:, :3]
            points = points[_valid_points(points, self.max_range_m)]
            if not len(points):
                continue

            keys = _adaptive_voxel_keys(points)
            _, unique_indices = np.unique(keys, axis=0, return_index=True)
            unique_indices.sort()
            points = points[unique_indices]
            keys = keys[unique_indices]
            keep = np.fromiter(
                (tuple(key) not in occupied for key in keys),
                dtype=bool,
                count=len(keys),
            )
            points = points[keep]
            keys = keys[keep]
            if not len(points):
                continue
            occupied.update(map(tuple, keys))
            temporal_confidence = float(neighbor.alignment_confidence) * np.exp(
                -abs(offset) / self.temporal_decay_s
            )
            fused.append(points)
            source_kind.append(
                np.full(len(points), TEMPORAL_FILLED, dtype=np.uint8)
            )
            confidence.append(
                np.full(len(points), temporal_confidence, dtype=np.float32)
            )
            time_offset.append(np.full(len(points), offset, dtype=np.float32))

        return TemporalFusionResult(
            points=np.concatenate(fused, axis=0),
            source_kind=np.concatenate(source_kind),
            confidence=np.concatenate(confidence),
            time_offset_s=np.concatenate(time_offset),
        )
