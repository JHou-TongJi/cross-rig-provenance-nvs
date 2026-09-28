"""Sparse geometry distillation data from safe temporal teacher artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT


FREE_SPACE = 2


@dataclass(frozen=True)
class SparseGridSpec:
    minimum_xyz: tuple[float, float, float] = (-120.0, -120.0, -8.0)
    maximum_xyz: tuple[float, float, float] = (120.0, 120.0, 8.0)
    voxel_size_m: float = 0.16

    @property
    def spatial_shape_zyx(self) -> tuple[int, int, int]:
        minimum = np.asarray(self.minimum_xyz)
        maximum = np.asarray(self.maximum_xyz)
        shape_xyz = np.ceil((maximum - minimum) / self.voxel_size_m).astype(int)
        return tuple(int(value) for value in shape_xyz[::-1])

    def coordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(points)[:, :3]
        minimum = np.asarray(self.minimum_xyz, dtype=np.float32)
        maximum = np.asarray(self.maximum_xyz, dtype=np.float32)
        valid = np.isfinite(xyz).all(axis=1) & (xyz >= minimum).all(axis=1) & (xyz < maximum).all(axis=1)
        coordinates_xyz = np.floor(
            (xyz[valid] - minimum) / self.voxel_size_m
        ).astype(np.int32)
        return coordinates_xyz[:, ::-1], valid

    def voxel_centers(self, coordinates_zyx: np.ndarray) -> np.ndarray:
        coordinates_xyz = np.asarray(coordinates_zyx, dtype=np.float32)[:, ::-1]
        minimum = np.asarray(self.minimum_xyz, dtype=np.float32)
        return minimum + (coordinates_xyz + 0.5) * self.voxel_size_m


def _unique_first(coordinates: np.ndarray) -> np.ndarray:
    _, indices = np.unique(coordinates, axis=0, return_index=True)
    return np.sort(indices)


def _sample_indices(length: int, maximum: int | None) -> np.ndarray:
    if maximum is None or length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum).round().astype(np.int64)


def _stratified_distance_indices(
    distances: np.ndarray,
    maximum: int | None,
    boundaries: tuple[float, float] = (20.0, 50.0),
) -> np.ndarray:
    distances = np.asarray(distances)
    if maximum is None or len(distances) <= maximum:
        return np.arange(len(distances), dtype=np.int64)
    lower, upper = boundaries
    bands = (
        np.flatnonzero(distances < lower),
        np.flatnonzero((distances >= lower) & (distances < upper)),
        np.flatnonzero(distances >= upper),
    )
    quota = max(1, maximum // len(bands))
    selected = [band[_sample_indices(len(band), quota)] for band in bands if len(band)]
    chosen = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    if len(chosen) < maximum:
        remaining = np.setdiff1d(
            np.arange(len(distances), dtype=np.int64), chosen, assume_unique=True,
        )
        fill = remaining[_sample_indices(len(remaining), maximum - len(chosen))]
        chosen = np.concatenate([chosen, fill])
    return np.sort(chosen[:maximum])


def _point_features(
    points: np.ndarray,
    measurement_state: np.ndarray | None = None,
) -> np.ndarray:
    feature_dim = 7 if measurement_state is not None else 6
    features = np.zeros((len(points), feature_dim), dtype=np.float32)
    features[:, :3] = points[:, :3] / np.asarray([120.0, 120.0, 8.0], dtype=np.float32)
    if points.shape[1] > 3:
        features[:, 3] = np.clip(points[:, 3] / 255.0, 0.0, 1.0)
    if points.shape[1] > 4:
        features[:, 4] = np.clip(points[:, 4] / 191.0, 0.0, 1.0)
    if points.shape[1] > 5:
        features[:, 5] = np.clip(points[:, 5] / 100.0, 0.0, 1.0)
    if measurement_state is not None:
        features[:, 6] = np.asarray(measurement_state, dtype=np.float32)
    return features


class SparseGeometryDistillationDataset(Dataset):
    def __init__(
        self,
        teacher_root: Path,
        sequence_id: str,
        grid: SparseGridSpec | None = None,
        max_input_points: int | None = 80_000,
        max_positive_queries: int | None = 80_000,
        max_negative_queries: int | None = 40_000,
        ray_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        include_free_input: bool = False,
        include_temporal_input: bool = False,
        temporal_input_fraction: float = 0.40,
    ) -> None:
        self.grid = grid or SparseGridSpec()
        self.max_input_points = max_input_points
        self.max_positive_queries = max_positive_queries
        self.max_negative_queries = max_negative_queries
        self.include_free_input = include_free_input
        self.include_temporal_input = include_temporal_input
        self.temporal_input_fraction = float(temporal_input_fraction)
        if not 0.0 <= self.temporal_input_fraction < 1.0:
            raise ValueError("temporal_input_fraction must be in [0, 1)")
        self.ray_origin_xyz = np.asarray(ray_origin_xyz, dtype=np.float32)
        if self.ray_origin_xyz.shape != (3,) or not np.isfinite(self.ray_origin_xyz).all():
            raise ValueError("ray_origin_xyz must contain three finite coordinates")
        self.artifacts = sorted((teacher_root / sequence_id).glob("*.npz"))
        if not self.artifacts:
            raise ValueError(f"no teacher artifacts found for {sequence_id}")

    def __len__(self) -> int:
        return len(self.artifacts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        artifact = self.artifacts[index]
        with np.load(artifact) as payload:
            points = payload["points"].astype(np.float32)
            source_kind = payload["source_kind"].astype(np.uint8)
            confidence = payload["confidence"].astype(np.float32)
            frame_id = int(payload["center_frame_id"])
            temporal_frame_ids = payload["temporal_frame_ids"].astype(np.int32) if "temporal_frame_ids" in payload else np.empty(0, dtype=np.int32)
            temporal_transforms = payload["temporal_transforms"].astype(np.float32) if "temporal_transforms" in payload else np.empty((0, 4, 4), dtype=np.float32)
            temporal_alignment_confidence = payload["temporal_alignment_confidence"].astype(np.float32) if "temporal_alignment_confidence" in payload else np.ones(len(temporal_frame_ids), dtype=np.float32)
            temporal_alignment_rmse_m = payload["temporal_alignment_rmse_m"].astype(np.float32) if "temporal_alignment_rmse_m" in payload else np.full(len(temporal_frame_ids), np.nan, dtype=np.float32)
            temporal_alignment_inlier_ratio = payload["temporal_alignment_inlier_ratio"].astype(np.float32) if "temporal_alignment_inlier_ratio" in payload else np.full(len(temporal_frame_ids), np.nan, dtype=np.float32)
        if temporal_transforms.shape != (len(temporal_frame_ids), 4, 4):
            raise ValueError("temporal metadata must contain one [4,4] transform per frame id")
        for values, name in (
            (temporal_alignment_confidence, "confidence"),
            (temporal_alignment_rmse_m, "rmse"),
            (temporal_alignment_inlier_ratio, "inlier ratio"),
        ):
            if values.shape != (len(temporal_frame_ids),):
                raise ValueError(f"temporal alignment {name} must match temporal frame ids")

        exact_points = points[source_kind == EXACT_CURRENT]
        input_coordinates, input_valid = self.grid.coordinates(exact_points)
        exact_points = exact_points[input_valid]
        input_unique = _unique_first(input_coordinates)
        input_coordinates = input_coordinates[input_unique]
        exact_points = exact_points[input_unique]
        exact_input_maximum = self.max_input_points
        if self.include_temporal_input and self.max_input_points is not None:
            exact_input_maximum = max(
                1,
                int(round(self.max_input_points * (1.0 - self.temporal_input_fraction))),
            )
        input_selection = _stratified_distance_indices(
            np.linalg.norm(
                exact_points[:, :3] - self.ray_origin_xyz[None], axis=1,
            ),
            exact_input_maximum,
        )
        input_coordinates = input_coordinates[input_selection]
        exact_points = exact_points[input_selection]
        input_points = exact_points
        input_measurement_state = np.ones(len(input_points), dtype=np.float32)
        if self.include_temporal_input:
            temporal = source_kind != EXACT_CURRENT
            temporal_points = points[temporal]
            temporal_confidence = confidence[temporal]
            temporal_coordinates, temporal_valid = self.grid.coordinates(temporal_points)
            temporal_points = temporal_points[temporal_valid]
            temporal_confidence = temporal_confidence[temporal_valid]
            temporal_unique = _unique_first(temporal_coordinates)
            temporal_coordinates = temporal_coordinates[temporal_unique]
            temporal_points = temporal_points[temporal_unique]
            temporal_confidence = temporal_confidence[temporal_unique]
            exact_coordinate_set = set(map(tuple, input_coordinates))
            new_temporal = np.fromiter(
                (tuple(coordinate) not in exact_coordinate_set for coordinate in temporal_coordinates),
                dtype=bool,
                count=len(temporal_coordinates),
            )
            temporal_coordinates = temporal_coordinates[new_temporal]
            temporal_points = temporal_points[new_temporal]
            temporal_confidence = temporal_confidence[new_temporal]
            remaining = None
            if self.max_input_points is not None:
                remaining = max(0, self.max_input_points - len(input_points))
            temporal_selection = _stratified_distance_indices(
                np.linalg.norm(
                    temporal_points[:, :3] - self.ray_origin_xyz[None], axis=1,
                ),
                remaining,
            )
            temporal_coordinates = temporal_coordinates[temporal_selection]
            temporal_points = temporal_points[temporal_selection]
            temporal_confidence = temporal_confidence[temporal_selection]
            input_coordinates = np.concatenate([
                input_coordinates, temporal_coordinates,
            ], axis=0)
            input_points = np.concatenate([
                input_points, temporal_points,
            ], axis=0)
            # +1 is an exact current-frame hit; (0,+1) is aligned static
            # temporal evidence.  -1 remains measured current-frame free space.
            input_measurement_state = np.concatenate([
                input_measurement_state,
                np.clip(temporal_confidence, 0.1, 0.95).astype(np.float32),
            ])

        positive_coordinates, positive_valid = self.grid.coordinates(points)
        positive_confidence = confidence[positive_valid]
        positive_source_kind = source_kind[positive_valid]
        positive_points = points[positive_valid, :3]
        positive_unique = _unique_first(positive_coordinates)
        positive_coordinates = positive_coordinates[positive_unique]
        positive_confidence = positive_confidence[positive_unique]
        positive_source_kind = positive_source_kind[positive_unique]
        positive_points = positive_points[positive_unique]
        positive_selection = _sample_indices(
            len(positive_coordinates), self.max_positive_queries,
        )
        positive_coordinates = positive_coordinates[positive_selection]
        positive_confidence = positive_confidence[positive_selection]
        positive_source_kind = positive_source_kind[positive_selection]
        positive_points = positive_points[positive_selection]

        exact_xyz = exact_points[:, :3]
        free_space_factors = np.asarray([0.25, 0.50, 0.75, 0.90], dtype=np.float32)
        negative_source = exact_xyz
        ray_vectors = negative_source - self.ray_origin_xyz[None]
        source_distance = np.linalg.norm(ray_vectors, axis=1)
        negative_points = np.concatenate([
            self.ray_origin_xyz[None] + ray_vectors * factor
            for factor in free_space_factors
        ], axis=0)
        negative_sdf = np.concatenate([
            source_distance * (1.0 - factor) for factor in free_space_factors
        ]).astype(np.float32)
        negative_coordinates, negative_valid = self.grid.coordinates(negative_points)
        negative_points = negative_points[negative_valid]
        negative_sdf = negative_sdf[negative_valid]
        negative_unique = _unique_first(negative_coordinates)
        negative_coordinates = negative_coordinates[negative_unique]
        negative_points = negative_points[negative_unique]
        negative_sdf = negative_sdf[negative_unique]
        occupied = set(map(tuple, positive_coordinates))
        free = np.fromiter(
            (tuple(coordinate) not in occupied for coordinate in negative_coordinates),
            dtype=bool,
            count=len(negative_coordinates),
        )
        negative_coordinates = negative_coordinates[free]
        negative_points = negative_points[free]
        negative_sdf = negative_sdf[free]
        negative_selection = _stratified_distance_indices(
            np.linalg.norm(
                negative_points - self.ray_origin_xyz[None], axis=1,
            ),
            self.max_negative_queries,
        )
        negative_coordinates = negative_coordinates[negative_selection]
        negative_points = negative_points[negative_selection]
        negative_sdf = negative_sdf[negative_selection]

        if self.include_free_input:
            input_features = np.concatenate([
                _point_features(
                    input_points,
                    input_measurement_state,
                ),
                _point_features(
                    negative_points,
                    -np.ones(len(negative_points), dtype=np.float32),
                ),
            ], axis=0)
            input_coordinates = np.concatenate([
                input_coordinates, negative_coordinates,
            ], axis=0)
        else:
            input_features = _point_features(input_points)

        query_coordinates = np.concatenate([
            positive_coordinates, negative_coordinates,
        ], axis=0)
        query_xyz = self.grid.voxel_centers(query_coordinates)
        occupancy = np.concatenate([
            np.ones(len(positive_coordinates), dtype=np.float32),
            np.zeros(len(negative_coordinates), dtype=np.float32),
        ])
        query_confidence = np.concatenate([
            positive_confidence,
            np.ones(len(negative_coordinates), dtype=np.float32),
        ])
        query_source_kind = np.concatenate([
            positive_source_kind,
            np.full(len(negative_coordinates), FREE_SPACE, dtype=np.uint8),
        ])
        query_distance = np.concatenate([
            np.linalg.norm(
                positive_points - self.ray_origin_xyz[None], axis=1,
            ),
            np.linalg.norm(
                negative_points - self.ray_origin_xyz[None], axis=1,
            ),
        ]).astype(np.float32)
        sdf = np.concatenate([
            np.zeros(len(positive_coordinates), dtype=np.float32),
            np.maximum(0.05, negative_sdf).astype(np.float32),
        ])
        batch_column = np.zeros((len(input_coordinates), 1), dtype=np.int32)
        query_batch_column = np.zeros((len(query_coordinates), 1), dtype=np.int32)
        return {
            "input_features": torch.from_numpy(input_features),
            "input_xyz": torch.from_numpy(exact_points[:, :3].astype(np.float32)),
            "geometry_input_xyz": torch.from_numpy(
                input_points[:, :3].astype(np.float32)
            ),
            "temporal_input_count": torch.tensor(
                len(input_points) - len(exact_points), dtype=torch.int32,
            ),
            "input_indices": torch.from_numpy(np.concatenate([
                batch_column, input_coordinates,
            ], axis=1)),
            "query_indices": torch.from_numpy(np.concatenate([
                query_batch_column, query_coordinates,
            ], axis=1)),
            "query_xyz": torch.from_numpy(query_xyz.astype(np.float32)),
            "occupancy_target": torch.from_numpy(occupancy[:, None]),
            "exact_occupancy": torch.from_numpy(occupancy[:, None]),
            "exact_mask": torch.from_numpy(
                ((query_source_kind == EXACT_CURRENT) | (query_source_kind == FREE_SPACE))[:, None]
            ),
            "sdf_target": torch.from_numpy(sdf[:, None]),
            "confidence_target": torch.from_numpy(query_confidence[:, None]),
            "query_source_kind": torch.from_numpy(query_source_kind[:, None]),
            "query_distance": torch.from_numpy(query_distance[:, None]),
            "spatial_shape": torch.tensor(self.grid.spatial_shape_zyx, dtype=torch.int32),
            "frame_id": torch.tensor(frame_id, dtype=torch.int32),
            "ray_origin": torch.from_numpy(self.ray_origin_xyz.copy()),
            "artifact": str(artifact),
            "temporal_frame_ids": torch.from_numpy(temporal_frame_ids),
            "temporal_transforms": torch.from_numpy(temporal_transforms),
            "temporal_alignment_confidence": torch.from_numpy(temporal_alignment_confidence),
            "temporal_alignment_rmse_m": torch.from_numpy(temporal_alignment_rmse_m),
            "temporal_alignment_inlier_ratio": torch.from_numpy(temporal_alignment_inlier_ratio),
        }
