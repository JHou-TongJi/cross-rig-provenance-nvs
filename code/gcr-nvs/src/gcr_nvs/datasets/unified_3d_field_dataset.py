"""Rectified RGB and LiDAR sparse-field samples across sequence-level splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from gcr_nvs.datasets.manifest import (
    CAMERA_FOCAL_ROLES,
    CAMERA_NAMES,
    appearance_source_cameras,
    build_sequence_manifest,
    sensor_timestamp,
)
from gcr_nvs.datasets.sparse_geometry_dataset import (
    SparseGeometryDistillationDataset,
    SparseGridSpec,
)
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations, rectify_image
from gcr_nvs.geometry.camera import sparse_depth_from_points


def temporal_camera_in_center_frame(
    calibration: CameraCalibration,
    center_from_neighbor: np.ndarray,
) -> CameraCalibration:
    """Express a neighboring camera as a projection from center-frame xyz."""
    center_from_neighbor = np.asarray(center_from_neighbor, dtype=np.float64)
    if center_from_neighbor.shape != (4, 4) or not np.isfinite(center_from_neighbor).all():
        raise ValueError("center_from_neighbor must be a finite [4,4] transform")
    neighbor_from_center = np.linalg.inv(center_from_neighbor)
    return CameraCalibration(
        name=calibration.name,
        intrinsic=calibration.intrinsic,
        distortion=calibration.distortion,
        external=calibration.external @ neighbor_from_center,
        width=calibration.width,
        height=calibration.height,
    )


class Unified3DFieldDataset(Dataset):
    """Join safe temporal geometry supervision with explicitly rectified RGB."""

    def __init__(
        self,
        root: Path,
        teacher_root: Path,
        split_file: Path,
        split_name: str,
        distortion_path: Path,
        source_size: tuple[int, int] = (960, 540),
        appearance_rgb_size: tuple[int, int] | None = None,
        rectification_alpha: float = 0.0,
        grid: SparseGridSpec | None = None,
        max_input_points: int | None = 80_000,
        max_positive_queries: int | None = 80_000,
        max_negative_queries: int | None = 40_000,
        include_free_input: bool = True,
        include_temporal_input: bool = False,
        temporal_input_fraction: float = 0.40,
        temporal_rgb: bool = False,
        temporal_rgb_max_views: int | None = None,
        temporal_rgb_max_frames: int = 1,
        temporal_rgb_include_target: bool = True,
        include_target_source: bool = False,
        target_camera: str | None = None,
        sequence_ids: list[str] | None = None,
    ) -> None:
        if not distortion_path.exists():
            raise FileNotFoundError(
                f"Unified3DField requires explicit Brown calibration: {distortion_path}"
            )
        if split_name not in {"train", "val", "test"}:
            raise ValueError("split_name must be train, val, or test")
        if target_camera is not None and target_camera not in CAMERA_NAMES:
            raise ValueError(f"unknown target camera: {target_camera}")
        self.root = root.resolve()
        self.teacher_root = teacher_root
        self.distortion_path = distortion_path
        self.source_size = tuple(map(int, source_size))
        self.appearance_rgb_size = (
            self.source_size
            if appearance_rgb_size is None
            else tuple(map(int, appearance_rgb_size))
        )
        self.rectification_alpha = float(rectification_alpha)
        self.temporal_rgb = bool(temporal_rgb)
        self.temporal_rgb_max_views = temporal_rgb_max_views
        self.temporal_rgb_max_frames = int(temporal_rgb_max_frames)
        if self.temporal_rgb_max_frames <= 0:
            raise ValueError("temporal_rgb_max_frames must be positive")
        self.temporal_rgb_include_target = bool(temporal_rgb_include_target)
        self.include_target_source = bool(include_target_source)
        self.target_camera = target_camera
        split = yaml.safe_load(split_file.read_text())
        self.sequence_ids = list(sequence_ids) if sequence_ids is not None else list(split[split_name])
        self.geometry_datasets: list[SparseGeometryDistillationDataset] = []
        self.records = {}
        self.samples: list[tuple[int, int]] = []
        missing_teacher = []
        for sequence_id in self.sequence_ids:
            sequence_records = build_sequence_manifest(
                self.root / sequence_id, compute_quality=False,
            )
            self.records[sequence_id] = {
                record.frame_id: record for record in sequence_records
            }
            teacher_dir = teacher_root / sequence_id
            if not list(teacher_dir.glob("*.npz")):
                missing_teacher.append(sequence_id)
                continue
            geometry_dataset = SparseGeometryDistillationDataset(
                teacher_root,
                sequence_id,
                grid=grid,
                max_input_points=max_input_points,
                max_positive_queries=max_positive_queries,
                max_negative_queries=max_negative_queries,
                include_free_input=include_free_input,
                include_temporal_input=include_temporal_input,
                temporal_input_fraction=temporal_input_fraction,
            )
            dataset_index = len(self.geometry_datasets)
            self.geometry_datasets.append(geometry_dataset)
            self.samples.extend(
                (dataset_index, sample_index)
                for sample_index in range(len(geometry_dataset))
            )
        if missing_teacher:
            raise FileNotFoundError(
                "missing safe geometry teacher for split sequences: "
                + ", ".join(missing_teacher)
            )
        if not self.samples:
            raise ValueError("no Unified3DField samples found")

    def __len__(self) -> int:
        return len(self.samples) * (1 if self.target_camera else len(CAMERA_NAMES))

    @staticmethod
    def _focal_role(camera_names: tuple[str, ...]) -> np.ndarray:
        result = np.zeros((len(camera_names), 2), dtype=np.float32)
        for index, name in enumerate(camera_names):
            role = CAMERA_FOCAL_ROLES.get(name)
            if role == "wide":
                result[index, 0] = 1.0
            elif role == "narrow":
                result[index, 1] = 1.0
        return result

    def __getitem__(self, index: int):
        if self.target_camera is None:
            target_camera = CAMERA_NAMES[index % len(CAMERA_NAMES)]
            sample_index = index // len(CAMERA_NAMES)
        else:
            target_camera = self.target_camera
            sample_index = index
        dataset_index, local_index = self.samples[sample_index]
        geometry = self.geometry_datasets[dataset_index][local_index]
        sequence_id = Path(geometry["artifact"]).parent.name
        frame_id = int(geometry["frame_id"])
        record = self.records[sequence_id][frame_id]
        sequence_dir = self.root / sequence_id
        raw_calibrations = load_calibrations(
            sequence_dir / record.camera_config,
            self.distortion_path,
        )
        for name, calibration in raw_calibrations.items():
            if calibration.distortion.shape[0] < 4 or not np.isfinite(calibration.distortion).all():
                raise ValueError(f"invalid Brown coefficients for {name}")

        current_source_names = appearance_source_cameras(
            target_camera, include_target=self.include_target_source,
        )
        source_names = list(current_source_names)
        source_images = []
        source_intrinsics = []
        source_extrinsics = []
        source_depths = []
        source_rectified_calibrations = []
        source_rgb_images = []
        source_rgb_intrinsics = []
        source_time_offsets = [0.0] * len(source_names)
        source_alignment_confidence = [1.0] * len(source_names)
        width, height = self.source_size
        exact_points = geometry["input_xyz"].numpy()
        geometry_points = geometry["geometry_input_xyz"].numpy()
        for name in source_names:
            image_path = sequence_dir / record.cameras[name]
            with Image.open(image_path) as image:
                raw_rgb = np.asarray(image.convert("RGB"))
            rectified_rgb, rectified = rectify_image(
                raw_rgb,
                raw_calibrations[name],
                output_size=self.source_size,
                alpha=self.rectification_alpha,
            )
            depth, _, _ = sparse_depth_from_points(
                geometry_points,
                rectified,
                self.source_size,
            )
            source_images.append(
                rectified_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            )
            if self.appearance_rgb_size == self.source_size:
                appearance_rgb = rectified_rgb
                appearance_calibration = rectified
            else:
                appearance_rgb, appearance_calibration = rectify_image(
                    raw_rgb,
                    raw_calibrations[name],
                    output_size=self.appearance_rgb_size,
                    alpha=self.rectification_alpha,
                )
            source_rgb_images.append(
                appearance_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            )
            source_rgb_intrinsics.append(
                appearance_calibration.intrinsic.astype(np.float32)
            )
            source_intrinsics.append(rectified.intrinsic.astype(np.float32))
            source_extrinsics.append(rectified.external.astype(np.float32))
            source_depths.append(depth[None].astype(np.float32))
            source_rectified_calibrations.append(rectified)

        # Optional temporal RGB views.  Teacher metadata stores transforms
        # T_center_from_neighbor; center points are projected with the inverse
        # transform into each neighboring camera.
        temporal_frame_ids = geometry.get("temporal_frame_ids")
        temporal_transforms = geometry.get("temporal_transforms")
        temporal_alignment_confidence = geometry.get("temporal_alignment_confidence")
        if self.temporal_rgb and temporal_frame_ids is not None and temporal_transforms is not None:
            count = min(int(temporal_frame_ids.numel()), int(temporal_transforms.shape[0]))
            if temporal_alignment_confidence is None:
                temporal_alignment_confidence = torch.ones(count, dtype=torch.float32)
            if int(temporal_alignment_confidence.numel()) < count:
                raise ValueError("temporal alignment confidence is shorter than transforms")
            added_views = 0
            temporal_order = sorted(
                range(count),
                key=lambda temporal_index: abs(
                    sensor_timestamp(self.records[sequence_id][int(temporal_frame_ids[temporal_index])])
                    - sensor_timestamp(record)
                ),
            )
            added_frames = 0
            for temporal_index in temporal_order:
                if added_frames >= self.temporal_rgb_max_frames:
                    break
                if self.temporal_rgb_max_views is not None and added_views >= self.temporal_rgb_max_views:
                    break
                neighbor_frame_id = int(temporal_frame_ids[temporal_index])
                neighbor_record = self.records[sequence_id][neighbor_frame_id]
                neighbor_calibrations = load_calibrations(
                    sequence_dir / neighbor_record.camera_config,
                    self.distortion_path,
                )
                center_from_neighbor = temporal_transforms[temporal_index].numpy()
                temporal_names = appearance_source_cameras(
                    target_camera,
                    include_target=self.temporal_rgb_include_target,
                )
                for name in temporal_names:
                    if name == target_camera and not self.temporal_rgb_include_target:
                        continue
                    if self.temporal_rgb_max_views is not None and added_views >= self.temporal_rgb_max_views:
                        break
                    raw_path = sequence_dir / neighbor_record.cameras[name]
                    with Image.open(raw_path) as image:
                        neighbor_raw = np.asarray(image.convert("RGB"))
                    rectified_neighbor, neighbor_calibration = rectify_image(
                        neighbor_raw,
                        neighbor_calibrations[name],
                        output_size=self.source_size,
                        alpha=self.rectification_alpha,
                    )
                    neighbor_calibration = temporal_camera_in_center_frame(
                        neighbor_calibration, center_from_neighbor,
                    )
                    temporal_depth, _, _ = sparse_depth_from_points(
                        geometry_points, neighbor_calibration, self.source_size,
                    )
                    if self.appearance_rgb_size == self.source_size:
                        appearance_neighbor = rectified_neighbor
                        appearance_calibration = neighbor_calibration
                    else:
                        appearance_neighbor, appearance_calibration = rectify_image(
                            neighbor_raw,
                            neighbor_calibrations[name],
                            output_size=self.appearance_rgb_size,
                            alpha=self.rectification_alpha,
                        )
                        appearance_calibration = CameraCalibration(
                            name=appearance_calibration.name,
                            intrinsic=appearance_calibration.intrinsic,
                            distortion=appearance_calibration.distortion,
                            external=neighbor_calibration.external,
                            width=appearance_calibration.width,
                            height=appearance_calibration.height,
                        )
                    source_names.append(f"T{neighbor_frame_id}:{name}")
                    source_time_offsets.append(
                        sensor_timestamp(neighbor_record) - sensor_timestamp(record)
                    )
                    source_alignment_confidence.append(
                        float(temporal_alignment_confidence[temporal_index])
                    )
                    source_images.append(
                        rectified_neighbor.astype(np.float32).transpose(2, 0, 1) / 255.0
                    )
                    source_rgb_images.append(
                        appearance_neighbor.astype(np.float32).transpose(2, 0, 1) / 255.0
                    )
                    source_intrinsics.append(neighbor_calibration.intrinsic.astype(np.float32))
                    source_rgb_intrinsics.append(appearance_calibration.intrinsic.astype(np.float32))
                    source_extrinsics.append(neighbor_calibration.external.astype(np.float32))
                    source_depths.append(temporal_depth[None].astype(np.float32))
                    source_rectified_calibrations.append(neighbor_calibration)
                    added_views += 1
                added_frames += 1

        target_path = sequence_dir / record.cameras[target_camera]
        with Image.open(target_path) as image:
            target_raw = np.asarray(image.convert("RGB"))
        target_rgb, target_calibration = rectify_image(
            target_raw,
            raw_calibrations[target_camera],
            output_size=self.source_size,
            alpha=self.rectification_alpha,
        )
        target_depth, _, _ = sparse_depth_from_points(
            exact_points,
            target_calibration,
            self.source_size,
        )
        target_geometry_depth, _, _ = sparse_depth_from_points(
            geometry_points,
            target_calibration,
            self.source_size,
        )
        if self.appearance_rgb_size == self.source_size:
            target_rgb_high = target_rgb
            target_calibration_high = target_calibration
        else:
            target_rgb_high, target_calibration_high = rectify_image(
                target_raw,
                raw_calibrations[target_camera],
                output_size=self.appearance_rgb_size,
                alpha=self.rectification_alpha,
            )
        target_depth_high, _, _ = sparse_depth_from_points(
            exact_points,
            target_calibration_high,
            self.appearance_rgb_size,
        )
        target_geometry_depth_high, _, _ = sparse_depth_from_points(
            geometry_points,
            target_calibration_high,
            self.appearance_rgb_size,
        )
        result = dict(geometry)
        result.update({
            "source_images": torch.from_numpy(np.stack(source_images)),
            "source_rgb_images": torch.from_numpy(np.stack(source_rgb_images)),
            "source_intrinsics": torch.from_numpy(np.stack(source_intrinsics)),
            "source_rgb_intrinsics": torch.from_numpy(
                np.stack(source_rgb_intrinsics)
            ),
            "source_rgb_distortions": torch.zeros(
                len(source_names), 5, dtype=torch.float32,
            ),
            "source_extrinsics": torch.from_numpy(np.stack(source_extrinsics)),
            "source_distortions": torch.zeros(len(source_names), 5, dtype=torch.float32),
            "source_lidar_depth": torch.from_numpy(np.stack(source_depths)),
            "source_validity": torch.ones(len(source_names), dtype=torch.bool),
            "source_time_offsets": torch.tensor(
                source_time_offsets, dtype=torch.float32,
            ),
            "source_alignment_confidence": torch.tensor(
                source_alignment_confidence, dtype=torch.float32,
            ),
            "source_camera_indices": torch.tensor([
                CAMERA_NAMES.index(name.split(":", 1)[-1])
                for name in source_names
            ], dtype=torch.long),
            "focal_role": torch.from_numpy(self._focal_role(tuple(
                name.split(":", 1)[-1] for name in source_names
            ))),
            "target_rgb": torch.from_numpy(
                target_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            ),
            "target_intrinsic": torch.from_numpy(target_calibration.intrinsic.astype(np.float32)),
            "target_extrinsic": torch.from_numpy(target_calibration.external.astype(np.float32)),
            "target_distortion": torch.zeros(5, dtype=torch.float32),
            "target_lidar_depth": torch.from_numpy(target_depth[None].astype(np.float32)),
            "target_geometry_depth": torch.from_numpy(
                target_geometry_depth[None].astype(np.float32)
            ),
            "target_rgb_high": torch.from_numpy(
                target_rgb_high.astype(np.float32).transpose(2, 0, 1) / 255.0
            ),
            "target_intrinsic_high": torch.from_numpy(
                target_calibration_high.intrinsic.astype(np.float32)
            ),
            "target_lidar_depth_high": torch.from_numpy(
                target_depth_high[None].astype(np.float32)
            ),
            "target_geometry_depth_high": torch.from_numpy(
                target_geometry_depth_high[None].astype(np.float32)
            ),
            "sequence_id": sequence_id,
            "target_camera": target_camera,
            "target_camera_index": torch.tensor(
                CAMERA_NAMES.index(target_camera), dtype=torch.long,
            ),
            "target_focal_role": torch.from_numpy(
                self._focal_role((target_camera,))[0]
            ),
            "source_cameras": "|".join(source_names),
            "current_source_count": torch.tensor(
                len(current_source_names), dtype=torch.int32,
            ),
            "include_target_source": torch.tensor(
                self.include_target_source, dtype=torch.bool,
            ),
            "image_geometry_mode": "rectified_pinhole",
            "image_width": torch.tensor(width, dtype=torch.int32),
            "image_height": torch.tensor(height, dtype=torch.int32),
        })
        return result
