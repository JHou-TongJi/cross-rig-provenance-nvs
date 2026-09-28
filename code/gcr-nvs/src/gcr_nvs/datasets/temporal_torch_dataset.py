"""Three-frame leave-one-camera-out dataset for the documented temporal stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gcr_nvs.datasets.leave_one_out import LeaveOneOutSample, temporal_window
from gcr_nvs.datasets.manifest import (
    CAMERA_NAMES,
    FrameRecord,
    build_sequence_manifest,
    sensor_timestamp,
)
from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset, _image, _rendered_feature
from gcr_nvs.geometry.icp import estimate_icp
from gcr_nvs.geometry.local_points import adaptive_voxel_downsample
from gcr_nvs.geometry.pcd import read_pcd
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.calibration import project_world_points


class TemporalLeaveOneOutTorchDataset(LeaveOneOutTorchDataset):
    """Return [T,V,...] source data and [T,13,H,W] aligned render inputs.

    Neighboring LiDAR frames are aligned into the center ego frame using the
    local ICP provider. If ICP fails, the neighbor is retained with a zero
    temporal-valid flag and is not treated as trustworthy motion evidence.
    """

    def __init__(
        self,
        *args,
        temporal_window_size: int = 3,
        temporal_max_interval_seconds: float = 0.75,
        **kwargs,
    ):
        if temporal_window_size < 1 or temporal_window_size % 2 == 0:
            raise ValueError("temporal_window_size must be a positive odd integer")
        self.temporal_window_size = int(temporal_window_size)
        requested_max_samples = kwargs.pop("max_samples", None)
        super().__init__(*args, max_samples=None, **kwargs)
        self.temporal_max_interval_seconds = float(temporal_max_interval_seconds)
        records_by_sequence: dict[str, list[FrameRecord]] = {}
        for sequence_id in {sample.record.sequence_id for sample in self.samples}:
            records_by_sequence[sequence_id] = build_sequence_manifest(self.root / sequence_id, compute_quality=False)
        self._records_by_sequence = records_by_sequence
        valid_samples = []
        for sample in self.samples:
            records = records_by_sequence[sample.record.sequence_id]
            index = next(i for i, record in enumerate(records) if record.frame_id == sample.record.frame_id)
            if temporal_window(
                records,
                index,
                self.temporal_window_size,
                max_interval_seconds=self.temporal_max_interval_seconds,
            ):
                valid_samples.append(sample)
        self.samples = valid_samples
        if requested_max_samples is not None:
            self.samples = self.samples[:requested_max_samples]
        self._pose_cache: dict[tuple[str, int], tuple[np.ndarray, dict[str, float | bool]]] = {}
        self._pose_points_cache: dict[tuple[str, int], np.ndarray] = {}

    def _pose_points(self, sample: LeaveOneOutSample) -> np.ndarray:
        key = (sample.record.sequence_id, sample.record.frame_id)
        if key not in self._pose_points_cache:
            sequence_dir = self.root / sample.record.sequence_id
            points = adaptive_voxel_downsample(read_pcd(sequence_dir / sample.record.lidar, fields=("x", "y", "z")))
            if len(points) > 40_000:
                points = points[np.linspace(0, len(points) - 1, 40_000).astype(np.int64)]
            self._pose_points_cache[key] = points.astype(np.float32)
        return self._pose_points_cache[key]

    def _sample_for_record(self, record: FrameRecord, target_camera: str) -> LeaveOneOutSample:
        return LeaveOneOutSample(record, target_camera, tuple(camera for camera in CAMERA_NAMES if camera != target_camera))

    def _aligned_geometry(self, neighbor: LeaveOneOutSample, center: LeaveOneOutSample, target_calibration):
        render_points, neighbor_colors, color_validity = self._sample_colors(neighbor)
        neighbor_points = self._pose_points(neighbor)
        center_points = self._pose_points(center)
        key = (neighbor.record.sequence_id, neighbor.record.frame_id, center.record.frame_id)
        if key not in self._pose_cache:
            transform, quality = estimate_icp(
                neighbor_points,
                center_points,
                max_points=40_000,
                iterations=30,
                max_correspondence_m=1.0,
                trim_ratio=0.50,
            )
            self._pose_cache[key] = transform, quality
        transform, quality = self._pose_cache[key]
        transformed = (transform @ np.c_[render_points[:, :3], np.ones(len(render_points))].T).T
        transformed[:, :3] = transformed[:, :3]
        render = self.renderer.render(
            transformed,
            neighbor_colors,
            target_calibration,
            color_validity=color_validity,
            output_size=(self.width, self.height),
        )
        geometry = render.channels.copy()
        geometry = np.concatenate([geometry, (geometry[7:8] <= 0).astype(np.float32)], axis=0)
        rays = TargetCamera(target_calibration).ray_map(self.width, self.height)
        return (
            geometry,
            _rendered_feature(geometry[:12], rays),
            bool(quality.get("geometry_gate_passed", False)),
            render.structure_validity[None].astype(np.float32),
            render.appearance_validity[None].astype(np.float32),
        )

    def __getitem__(self, index):
        center = self.samples[index]
        records = self._records_by_sequence[center.record.sequence_id]
        center_index = next(i for i, record in enumerate(records) if record.frame_id == center.record.frame_id)
        window = temporal_window(
            records,
            center_index,
            self.temporal_window_size,
            max_interval_seconds=self.temporal_max_interval_seconds,
        )
        if not window:
            raise RuntimeError("sample no longer has a valid temporal window")
        samples = [self._sample_for_record(record, center.target_camera) for record in window]
        center_points, _, _, _, _, _ = self._load_frame(center)
        _, center_calibrations, _, _, source_images, _ = self._load_frame(center)
        target = TargetCamera(center_calibrations[center.target_camera]).resized(self.width, self.height)
        geometries = []
        features = []
        temporal_valid = []
        structure_validity = []
        appearance_validity = []
        center_provenance = None
        center_window_index = self.temporal_window_size // 2
        for frame_index, sample in enumerate(samples):
            if frame_index == center_window_index:
                points, colors, color_validity = self._sample_colors(sample)
                render = self.renderer.render(
                    points,
                    colors,
                    target.calibration,
                    color_validity=color_validity,
                    output_size=(self.width, self.height),
                )
                geometry = np.concatenate([render.channels, (render.channels[7:8] <= 0).astype(np.float32)], axis=0)
                valid = True
                feature = _rendered_feature(geometry[:12], target.ray_map(self.width, self.height))
                center_provenance = render.source_provenance.copy()
                structure = render.structure_validity[None].astype(np.float32)
                appearance = render.appearance_validity[None].astype(np.float32)
            else:
                geometry, feature, valid, structure, appearance = self._aligned_geometry(
                    sample, center, target.calibration,
                )
            geometries.append(geometry)
            features.append(feature)
            temporal_valid.append(float(valid))
            structure_validity.append(structure)
            appearance_validity.append(appearance)
        sequence_dir = self.root / center.record.sequence_id
        valid_height = self.height - 4 if self.height >= 4 else self.height
        target_rgb = _image(sequence_dir / center.record.cameras[center.target_camera], self.width, valid_height)
        if valid_height < self.height:
            target_rgb = np.pad(target_rgb, ((0, 0), (0, self.height - valid_height), (0, 0)), mode="reflect")
        source_indices = [CAMERA_NAMES.index(name) for name in center.source_cameras]
        result = {
            "geometry": torch.from_numpy(np.stack(geometries)).float(),
            "ray_map": torch.from_numpy(np.repeat(target.ray_map(self.width, self.height)[None], 3, axis=0)).float(),
            "rendered_feature": torch.from_numpy(np.stack(features)).float(),
            "target_rgb": torch.from_numpy(target_rgb).float(),
            "valid_mask": torch.ones((1, self.height, self.width), dtype=torch.float32),
            "temporal_valid_mask": torch.tensor(temporal_valid, dtype=torch.float32),
            "temporal_timestamps": torch.tensor(
                [sensor_timestamp(record) for record in window],
                dtype=torch.float64,
            ),
            "temporal_offsets_s": torch.tensor(
                [sensor_timestamp(record) - sensor_timestamp(center.record) for record in window],
                dtype=torch.float32,
            ),
                "observed_mask": (
                    torch.from_numpy(geometries[center_window_index][7:8]) > 0
                ).float(),
            "structure_validity": torch.from_numpy(np.stack(structure_validity)).float(),
            "appearance_validity": torch.from_numpy(np.stack(appearance_validity)).float(),
            "disocclusion_mask": torch.from_numpy(
                np.stack(structure_validity) * (1.0 - np.stack(appearance_validity))
            ).float(),
            "unknown_structure_mask": torch.from_numpy(
                1.0 - np.stack(structure_validity)
            ).float(),
            "source_provenance": torch.from_numpy(center_provenance).long(),
            "sequence_id": center.record.sequence_id,
            "frame_id": center.record.frame_id,
            "target_camera": center.target_camera,
        }
        if source_images is not None:
            result["source_images"] = torch.from_numpy(
                source_images[source_indices][None].repeat(
                    self.temporal_window_size, axis=0,
                )
            ).float()
        return result
