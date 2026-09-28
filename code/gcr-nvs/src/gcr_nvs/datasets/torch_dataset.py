"""PyTorch dataset for the first single-frame leave-one-out stage."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gcr_nvs.datasets.leave_one_out import LeaveOneOutSample, make_leave_one_out_samples
from gcr_nvs.datasets.manifest import (
    CAMERA_FAMILIES,
    CAMERA_FOCAL_ROLES,
    CAMERA_NAMES,
    FrameRecord,
    build_sequence_manifest,
)
from gcr_nvs.geometry.camera import TargetCamera, sparse_depth_from_points
from gcr_nvs.geometry.calibration import load_calibrations, project_world_points
from gcr_nvs.geometry.local_points import adaptive_voxel_downsample, estimate_normals
from gcr_nvs.geometry.pcd import read_pcd
from gcr_nvs.geometry.rig import load_target_rig, resolve_target_calibration
from gcr_nvs.rendering.fixed_splat import FixedSplatRenderer


def _image(path: Path, width: int, height: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = Image.open(path).convert("RGB").resize((width, height), resampling)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _dynamic_mask(mask_root: Path | None, sequence_id: str, frame_id: int, camera_name: str, width: int, height: int) -> np.ndarray:
    """Load an optional precomputed dynamic mask using the documented layouts."""
    if mask_root is None:
        return np.zeros((height, width), dtype=bool)
    candidates = (
        mask_root / sequence_id / f"{frame_id:06d}_{camera_name}.dynamic.npy",
        mask_root / sequence_id / f"{frame_id:06d}" / f"{camera_name}.dynamic.npy",
        mask_root / f"{sequence_id}_{frame_id:06d}_{camera_name}.dynamic.npy",
    )
    for path in candidates:
        if path.exists():
            mask = np.load(path).astype(np.float32)
            mask = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
            return mask > 0.5
    return np.zeros((height, width), dtype=bool)


def _fuse_point_colors(
    observed_colors: np.ndarray,
    observed_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average observed RGB without inventing colors for unobserved points."""
    counts = observed_valid.sum(axis=0).astype(np.float32)
    colors = observed_colors.sum(axis=0) / counts.clip(min=1.0)[:, None]
    color_validity = counts > 0.0
    colors[~color_validity] = 0.0
    return colors.astype(np.float32), color_validity, counts


def _rendered_feature(geometry: np.ndarray, ray_map: np.ndarray) -> np.ndarray:
    """Build the fixed-render feature contract for the Lite refiner.

    The first implementation keeps this renderer-side and deterministic: it
    exposes geometry and target-ray cues without smuggling a full source view
    into the RGB output path. Learned source-view anchor fusion is a later
    stage and must consume these anchors explicitly.
    """
    depth = np.log1p(geometry[3:4].clip(min=0.0)) / np.log1p(120.0)
    blocks = [
        geometry,
        ray_map,
        geometry[:3] * geometry[7:8],
        geometry[4:7] * geometry[7:8],
        depth,
        geometry[8:9],
        geometry[10:11] / 3.0,
        geometry[11:12],
    ]
    feature = np.concatenate(blocks, axis=0)
    if feature.shape[0] < 32:
        feature = np.concatenate([feature, np.zeros((32 - feature.shape[0], *feature.shape[1:]), dtype=feature.dtype)], axis=0)
    return feature[:32].astype(np.float32)


def _plane_sweep_uv(target: TargetCamera, source_calibrations, source_names: tuple[str, ...], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project target rays at several depths into source cameras for dense feature conditioning."""
    rays = target.ray_map(width, height).transpose(1, 2, 0)
    depths = np.asarray((4.0, 8.0, 16.0, 32.0, 64.0), dtype=np.float32)
    target_center = target.calibration.camera_center.astype(np.float32)
    uv_maps, valid_maps = [], []
    for source_name in source_names:
        points = []
        for depth in depths:
            xyz_vehicle = (target_center[None, None, :] + rays * depth).reshape(-1, 3)
            uv, valid = project_world_points(xyz_vehicle, source_calibrations[source_name])
            uv[:, 0] *= width / source_calibrations[source_name].width
            uv[:, 1] *= height / source_calibrations[source_name].height
            uv[~valid] = -1.0
            points.append(uv.reshape(height, width, 2).astype(np.float32))
            valid_maps.append(valid.reshape(height, width))
        uv_maps.append(np.stack(points))
    return np.stack(uv_maps), np.stack(valid_maps).reshape(len(source_names), len(depths), height, width)


class LeaveOneOutTorchDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sequence_ids: list[str],
        distortion: Path | None = None,
        width: int = 960,
        height: int = 544,
        max_points: int = 120_000,
        max_geometry_points: int = 120_000,
        max_samples: int | None = None,
        target_rig: Path | None = None,
        dynamic_mask_root: Path | None = None,
        supervision_root: Path | None = None,
        target_camera: str | None = None,
        include_foundation_inputs: bool = True,
        foundation_minimal_inputs: bool = False,
        include_plane_sweep_grids: bool = True,
        balance_target_cameras: bool = False,
        include_target_source: bool = False,
        drop_nearest_source_probability: float = 0.0,
        front_narrow_drop_nearest_source_probability: float | None = None,
        exclude_nearest_source: bool = False,
        cache_root: Path | None = None,
        max_samples_per_sequence: int | None = None,
    ):
        self.root = root
        self.width = width
        self.height = height
        self.distortion = distortion
        self.max_points = max_points
        self.max_geometry_points = max(max_geometry_points, max_points)
        self.target_rig = load_target_rig(target_rig) if target_rig else None
        self.dynamic_mask_root = dynamic_mask_root
        self.supervision_root = supervision_root
        self.target_camera = target_camera
        self.include_foundation_inputs = include_foundation_inputs
        self.foundation_minimal_inputs = foundation_minimal_inputs
        self.include_plane_sweep_grids = include_plane_sweep_grids
        self.balance_target_cameras = balance_target_cameras
        self.include_target_source = include_target_source
        self.drop_nearest_source_probability = float(drop_nearest_source_probability)
        self.front_narrow_drop_nearest_source_probability = (
            None
            if front_narrow_drop_nearest_source_probability is None
            else float(front_narrow_drop_nearest_source_probability)
        )
        self.exclude_nearest_source = exclude_nearest_source
        self.cache_root = cache_root
        records: list[FrameRecord] = []
        for sequence_id in sequence_ids:
            records.extend(build_sequence_manifest(root / sequence_id, compute_quality=False))
        self.samples = make_leave_one_out_samples(records, target_camera)
        if max_samples_per_sequence is not None:
            balanced = []
            for sequence_id in sequence_ids:
                target_names = CAMERA_NAMES if balance_target_cameras and target_camera is None else (target_camera,)
                for target_name in target_names:
                    sequence_samples = [
                        sample for sample in self.samples
                        if sample.record.sequence_id == sequence_id
                        and (target_name is None or sample.target_camera == target_name)
                    ]
                    if len(sequence_samples) > max_samples_per_sequence:
                        indices = np.linspace(0, len(sequence_samples) - 1, max_samples_per_sequence).round().astype(np.int64)
                        sequence_samples = [sequence_samples[index] for index in indices]
                    balanced.extend(sequence_samples)
            self.samples = balanced
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.renderer = FixedSplatRenderer()
        self._cached_key = None
        self._cached_frame = None
        self._plane_sweep_cache: dict[tuple[str, str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self.samples)

    def _source_cameras(self, sample: LeaveOneOutSample) -> tuple[str, ...]:
        return CAMERA_NAMES if self.include_target_source else sample.source_cameras

    def _maybe_drop_nearest_source(
        self,
        source_cameras: tuple[str, ...],
        calibrations: dict,
        target_calibration,
    ) -> tuple[str, ...]:
        probability = self.drop_nearest_source_probability
        target_name = target_calibration.name
        if (
            CAMERA_FOCAL_ROLES.get(target_name) == "narrow"
            and self.front_narrow_drop_nearest_source_probability is not None
        ):
            probability = self.front_narrow_drop_nearest_source_probability
        random_drop = probability > 0.0 and np.random.random() < probability
        should_drop = self.exclude_nearest_source or random_drop
        if not should_drop or len(source_cameras) <= 1:
            return source_cameras
        target_rotation = target_calibration.external[:3, :3]
        target_forward = target_rotation.T[:, 2]
        target_center = target_calibration.camera_center
        candidates = []
        for name in source_cameras:
            calibration = calibrations[name]
            source_forward = calibration.external[:3, :3].T[:, 2]
            similarity = float(np.dot(source_forward, target_forward))
            baseline = float(np.linalg.norm(calibration.camera_center - target_center))
            candidates.append((similarity, baseline, name))
        if random_drop and CAMERA_FOCAL_ROLES.get(target_name) == "narrow":
            target_family = CAMERA_FAMILIES.get(target_name)
            candidates = [
                item for item in candidates if CAMERA_FAMILIES.get(item[2]) != target_family
            ]
            if not candidates:
                return source_cameras
        near_facing = [item for item in candidates if item[0] >= 0.9]
        pool = near_facing or candidates
        _, _, dropped = max(pool, key=lambda item: (item[0], -item[1]))
        return tuple(name for name in source_cameras if name != dropped)

    def _load_frame(self, sample: LeaveOneOutSample):
        sequence_dir = self.root / sample.record.sequence_id
        cache_key = (sample.record.sequence_id, sample.record.frame_id)
        if cache_key == self._cached_key:
            return self._cached_frame
        calibrations = load_calibrations(sequence_dir / sample.record.camera_config, self.distortion)
        points = read_pcd(sequence_dir / sample.record.lidar)
        points = adaptive_voxel_downsample(points)
        if len(points) > self.max_geometry_points:
            points = points[
                np.linspace(0, len(points) - 1, self.max_geometry_points).astype(np.int64)
            ]
        observed_colors = np.zeros((len(CAMERA_NAMES), len(points), 3), dtype=np.float32)
        observed_valid = np.zeros((len(CAMERA_NAMES), len(points)), dtype=bool)
        observed_dynamic = np.zeros((len(CAMERA_NAMES), len(points)), dtype=bool)
        source_images = np.zeros((len(CAMERA_NAMES), 3, self.height, self.width), dtype=np.float32) if self.include_foundation_inputs else None
        for camera_index, source_name in enumerate(CAMERA_NAMES):
            source_calibration = calibrations[source_name]
            image_path = sequence_dir / sample.record.cameras[source_name]
            if self.include_foundation_inputs:
                image = _image(image_path, source_calibration.width, source_calibration.height)
                image_height, image_width = image.shape[1:]
            else:
                image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                image_height, image_width = image.shape[:2]
            if source_images is not None:
                source_images[camera_index] = _image(image_path, self.width, self.height)
            pixels, valid = project_world_points(points[:, :3], source_calibration)
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            xy = np.rint(pixels[indices]).astype(np.int64)
            inside = (
                (xy[:, 0] >= 0) & (xy[:, 0] < image_width)
                & (xy[:, 1] >= 0) & (xy[:, 1] < image_height)
            )
            if inside.any():
                valid_indices = indices[inside]
                xy_inside = xy[inside]
                camera_points = (
                    source_calibration.external
                    @ np.c_[points[valid_indices, :3], np.ones(len(valid_indices))].T
                ).T[:, :3]
                flat = xy_inside[:, 1] * image_width + xy_inside[:, 0]
                order = np.argsort(camera_points[:, 2], kind="stable")
                sorted_flat = flat[order]
                _, first = np.unique(sorted_flat, return_index=True)
                z_buffer = np.full(image_height * image_width, np.inf, dtype=np.float32)
                nearest = order[first]
                z_buffer[flat[nearest]] = camera_points[nearest, 2]
                front_tolerance = np.maximum(0.10, 0.02 * z_buffer[flat])
                visible = camera_points[:, 2] <= z_buffer[flat] + front_tolerance
                if not visible.any():
                    continue
                valid_indices = valid_indices[visible]
                xy_inside = xy_inside[visible]
                if self.include_foundation_inputs:
                    sampled = image[:, xy_inside[:, 1], xy_inside[:, 0]].T
                else:
                    sampled = image[xy_inside[:, 1], xy_inside[:, 0]].astype(np.float32) / 255.0
                observed_colors[camera_index, valid_indices] = sampled
                observed_valid[camera_index, valid_indices] = True
                dynamic = _dynamic_mask(self.dynamic_mask_root, sample.record.sequence_id, sample.record.frame_id, source_name, image_width, image_height)
                observed_dynamic[camera_index, valid_indices] = dynamic[xy_inside[:, 1], xy_inside[:, 0]]
        self._cached_key = cache_key
        self._cached_frame = (points, calibrations, observed_colors, observed_valid, source_images, observed_dynamic)
        return self._cached_frame

    def _sample_colors(
        self,
        sample: LeaveOneOutSample,
        source_cameras: tuple[str, ...] | None = None,
    ):
        points, _, observed_colors, observed_valid, _, _ = self._load_frame(sample)
        source_cameras = source_cameras or self._source_cameras(sample)
        source_indices = [CAMERA_NAMES.index(name) for name in source_cameras]
        valid = observed_valid[source_indices]
        colors, color_validity, _ = _fuse_point_colors(
            observed_colors[source_indices], valid,
        )
        return points, colors, color_validity

    def __getitem__(self, index):
        sample = self.samples[index]
        cache_path = None
        cache_enabled = (
            self.cache_root is not None
            and self.supervision_root is None
            and self.drop_nearest_source_probability <= 0.0
            and (
                self.front_narrow_drop_nearest_source_probability is None
                or self.front_narrow_drop_nearest_source_probability <= 0.0
            )
            and not self.exclude_nearest_source
        )
        if cache_enabled:
            if self.include_foundation_inputs:
                if self.foundation_minimal_inputs:
                    cache_kind = "foundation_minimal"
                elif not self.include_plane_sweep_grids:
                    cache_kind = "foundation_anchors"
                else:
                    cache_kind = "foundation"
            else:
                cache_kind = "lite"
            cache_path = self.cache_root / f"{cache_kind}_source_depth_v3" / sample.record.sequence_id / (
                f"{sample.record.frame_id:06d}_{sample.target_camera}_{self.width}x{self.height}"
                f"_{self.max_points}a_{self.max_geometry_points}g.pt"
            )
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                height, width = cached["geometry"].shape[-2:]
                cached.setdefault("supervision_visibility", torch.ones(1, height, width))
                cached.setdefault("supervision_confidence", torch.ones(1, height, width))
                cached.setdefault("structure_validity", cached["geometry"][7:8].gt(0).float())
                cached.setdefault("appearance_validity", cached["geometry"][7:8].gt(0).float())
                cached.setdefault(
                    "disocclusion_mask",
                    cached["structure_validity"] * (1.0 - cached["appearance_validity"]),
                )
                cached.setdefault("unknown_structure_mask", 1.0 - cached["structure_validity"])
                return {key: (value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value) for key, value in cached.items()}
        sequence_dir = self.root / sample.record.sequence_id
        _, calibrations, _, _, source_images, observed_dynamic = self._load_frame(sample)
        target_calibration = calibrations[sample.target_camera]
        if self.target_rig:
            target_calibration = resolve_target_calibration(self.target_rig.camera(sample.target_camera), calibrations)
        target = TargetCamera(target_calibration).resized(self.width, self.height)
        source_cameras = self._maybe_drop_nearest_source(
            self._source_cameras(sample), calibrations, target_calibration,
        )
        points, colors, color_validity = self._sample_colors(sample, source_cameras)
        source_indices = [CAMERA_NAMES.index(name) for name in source_cameras]
        _, _, observed_colors, observed_valid, _, observed_dynamic = self._load_frame(sample)
        source_valid = observed_valid[source_indices]
        source_counts = source_valid.sum(axis=0).astype(np.float32)
        source_mean = observed_colors[source_indices].sum(axis=0) / source_counts.clip(min=1)[:, None]
        source_variance = (
            ((observed_colors[source_indices] - source_mean[None]) ** 2 * source_valid[..., None])
            .sum(axis=(0, 2))
            / source_counts.clip(min=1)
        )
        point_confidence = np.minimum(source_counts / 2.0, 1.0) * np.exp(-source_variance)
        provenance = np.full(len(points), -1, dtype=np.int16)
        point_dynamic = np.zeros(len(points), dtype=np.float32)
        for camera_id in source_indices:
            valid_points = observed_valid[camera_id] & (provenance < 0)
            provenance[valid_points] = camera_id
            point_dynamic = np.maximum(point_dynamic, observed_dynamic[camera_id].astype(np.float32))
        render = self.renderer.render(
            points,
            colors,
            target.calibration,
            normals=estimate_normals(points),
            point_confidence=point_confidence,
            source_observation_count=source_counts,
            source_provenance=provenance,
            dynamic_mask=point_dynamic,
            color_validity=color_validity,
            output_size=(self.width, self.height),
        )
        if len(points) > self.max_points:
            anchor_indices = np.linspace(0, len(points) - 1, self.max_points).astype(np.int64)
        else:
            anchor_indices = np.arange(len(points), dtype=np.int64)
        anchor_points = points[anchor_indices]
        anchor_observed_valid = observed_valid[:, anchor_indices]
        valid_height = self.height - 4 if self.height >= 4 else self.height
        target_rgb = _image(sequence_dir / sample.record.cameras[sample.target_camera], self.width, valid_height)
        if valid_height < self.height:
            target_rgb = np.pad(target_rgb, ((0, 0), (0, self.height - valid_height), (0, 0)), mode="reflect")
        supervision_visibility = np.ones((1, self.height, self.width), dtype=np.float32)
        supervision_confidence = np.ones((1, self.height, self.width), dtype=np.float32)
        if self.supervision_root is not None:
            rig_id = self.target_rig.rig_id if self.target_rig is not None else "source_rig"
            supervision_dir = (
                self.supervision_root / rig_id / sample.record.sequence_id
                / f"{sample.record.frame_id:06d}"
            )
            rgb_path = supervision_dir / f"{sample.target_camera}.rgb.png"
            if not rgb_path.exists():
                raise FileNotFoundError(f"missing external target supervision: {rgb_path}")
            target_rgb = _image(rgb_path, self.width, self.height)
            for name, destination in (
                ("visibility", supervision_visibility),
                ("confidence", supervision_confidence),
            ):
                mask_path = supervision_dir / f"{sample.target_camera}.{name}.npy"
                if mask_path.exists():
                    mask = np.load(mask_path).astype(np.float32)
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    mask_image = Image.fromarray(mask, mode="F").resize(
                        (self.width, self.height),
                        getattr(Image, "Resampling", Image).BILINEAR,
                    )
                    destination[0] = np.asarray(mask_image, dtype=np.float32).clip(0.0, 1.0)
        geometry_np = render.channels.copy()
        hole = (geometry_np[7:8] <= 0).astype(np.float32)
        geometry_np = np.concatenate([geometry_np, hole], axis=0)
        geometry = torch.from_numpy(geometry_np)
        rays = torch.from_numpy(target.ray_map(self.width, self.height))
        rendered_feature = None
        if not (
            self.include_foundation_inputs
            and (self.foundation_minimal_inputs or not self.include_plane_sweep_grids)
        ):
            rendered_feature = torch.from_numpy(_rendered_feature(geometry_np[:12], rays.numpy())).float()
        structure_validity = torch.from_numpy(render.structure_validity[None]).float()
        appearance_validity = torch.from_numpy(render.appearance_validity[None]).float()
        observed = appearance_validity > 0
        disocclusion = structure_validity * (1.0 - appearance_validity)
        unknown_structure = 1.0 - structure_validity
        target_valid = torch.ones((1, self.height, self.width), dtype=torch.float32)
        result = {
            "geometry": geometry.float(),
            "ray_map": rays.float(),
            "target_rgb": torch.from_numpy(target_rgb).float(),
            "supervision_visibility": torch.from_numpy(supervision_visibility).float(),
            "supervision_confidence": torch.from_numpy(supervision_confidence).float(),
            "sequence_id": sample.record.sequence_id,
            "frame_id": sample.record.frame_id,
            "target_camera": sample.target_camera,
            "source_cameras": "|".join(source_cameras),
        }
        if not (self.include_foundation_inputs and self.foundation_minimal_inputs):
            result.update({
                "rendered_feature": rendered_feature,
                "valid_mask": target_valid,
                "observed_mask": observed.float(),
                "structure_validity": structure_validity,
                "appearance_validity": appearance_validity,
                "disocclusion_mask": disocclusion,
                "unknown_structure_mask": unknown_structure,
                "source_provenance": torch.from_numpy(render.source_provenance.copy()).long(),
            })
            if rendered_feature is None:
                result.pop("rendered_feature")
        if self.include_foundation_inputs:
            source_input_calibrations = [
                TargetCamera(calibrations[name]).resized(
                    self.width, self.height,
                ).calibration
                for name in source_cameras
            ]
            source_lidar_depths = []
            for input_calibration in source_input_calibrations:
                source_depth, _, _ = sparse_depth_from_points(
                    points, input_calibration, (self.width, self.height),
                )
                source_lidar_depths.append(source_depth[None])
            result.update({
                "source_images": torch.from_numpy(source_images[source_indices]).float(),
                "source_lidar_depths": torch.from_numpy(np.stack(source_lidar_depths)).float(),
                "source_intrinsics": torch.from_numpy(np.stack([
                    calibration.intrinsic for calibration in source_input_calibrations
                ]).astype(np.float32)),
                "source_extrinsics": torch.from_numpy(np.stack([calibrations[name].external for name in source_cameras]).astype(np.float32)),
                "source_distortions": torch.from_numpy(np.stack([
                    calibrations[name].distortion for name in source_cameras
                ]).astype(np.float32)),
                "target_intrinsic": torch.from_numpy(target.calibration.intrinsic.astype(np.float32)),
                "target_extrinsic": torch.from_numpy(target.calibration.external.astype(np.float32)),
                "target_distortion": torch.from_numpy(target.calibration.distortion.astype(np.float32)),
            })
            if not self.foundation_minimal_inputs:
                source_uv = []
                source_valid = []
                for source_name, source_input_calibration in zip(
                    source_cameras, source_input_calibrations,
                ):
                    pixels, valid = project_world_points(
                        anchor_points[:, :3], source_input_calibration,
                    )
                    pixels[~valid] = -1.0
                    source_uv.append(np.nan_to_num(pixels, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32))
                    source_valid.append(
                        valid & anchor_observed_valid[CAMERA_NAMES.index(source_name)]
                    )
                target_uv, target_valid_points = project_world_points(anchor_points[:, :3], target.calibration)
                target_uv[:, 0] *= self.width / target.calibration.width
                target_uv[:, 1] *= self.height / target.calibration.height
                target_uv[~target_valid_points] = -1.0
                result.update({
                    "anchor_xyz": torch.from_numpy(anchor_points[:, :3].astype(np.float32)),
                    "source_uv": torch.from_numpy(np.stack(source_uv)).float(),
                    "source_valid": torch.from_numpy(np.stack(source_valid)).bool(),
                    "target_uv": torch.from_numpy(np.nan_to_num(target_uv, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)),
                    "target_anchor_valid": torch.from_numpy(target_valid_points).bool(),
                })
                if self.include_plane_sweep_grids:
                    plane_key = (sample.record.sequence_id, sample.target_camera, source_cameras)
                    if plane_key not in self._plane_sweep_cache:
                        self._plane_sweep_cache[plane_key] = _plane_sweep_uv(
                            target, calibrations, source_cameras, self.width, self.height,
                        )
                    plane_uv, plane_valid = self._plane_sweep_cache[plane_key]
                    result.update({
                        "plane_sweep_uv": torch.from_numpy(plane_uv).float(),
                        "plane_sweep_valid": torch.from_numpy(plane_valid).bool(),
                    })
        if cache_enabled and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            compact = {key: (value.half() if isinstance(value, torch.Tensor) and value.is_floating_point() else value) for key, value in result.items()}
            temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(compact, temporary)
            os.replace(temporary, cache_path)
        return result
