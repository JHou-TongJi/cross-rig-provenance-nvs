from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gcr_nvs.datasets.leave_one_out import make_leave_one_out_samples
from gcr_nvs.datasets.manifest import build_sequence_manifest, sensor_timestamp
from gcr_nvs.geometry.pcd import read_pcd, voxel_downsample
from gcr_nvs.geometry.calibration import CameraCalibration, CALIBRATION_CAMERA_ORDER, load_calibrations
from gcr_nvs.geometry.camera import sparse_depth_from_points, TargetCamera
from gcr_nvs.geometry.local_points import LocalPointBuilder
from gcr_nvs.datasets.leave_one_out import temporal_window
from gcr_nvs.datasets.torch_dataset import _fuse_point_colors
from gcr_nvs.rendering.fixed_splat import FixedSplatRenderer


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = Path(os.environ.get("GCR_NVS_TEST_SEQUENCE", ROOT / "2026-05-22-13-45-50"))
DATASET_REQUIRED = pytest.mark.skipif(
    not (SEQUENCE / "Key_frames/camera_config").is_dir(),
    reason="requires an external sequence; set GCR_NVS_TEST_SEQUENCE to its path",
)


@DATASET_REQUIRED
def test_manifest_references_existing_sensor_files():
    records = build_sequence_manifest(SEQUENCE, compute_quality=False)
    assert len(records) == 80
    first = records[0]
    assert (SEQUENCE / first.lidar).exists()
    assert all((SEQUENCE / path).exists() for path in first.cameras.values())


@DATASET_REQUIRED
def test_manifest_is_sorted_by_precise_lidar_time_not_frame_id():
    records = build_sequence_manifest(SEQUENCE, compute_quality=False)
    times = np.asarray([sensor_timestamp(record) for record in records])
    assert np.all(np.diff(times) > 0.0)
    assert [record.frame_id for record in records] != sorted(
        record.frame_id for record in records
    )


@DATASET_REQUIRED
def test_camera_config_index_order_matches_physical_camera_directions():
    assert CALIBRATION_CAMERA_ORDER == (
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_FRONT_WIDE",
        "CAM_FRONT_NARROW",
    )
    calibration = load_calibrations(SEQUENCE / "Key_frames/camera_config/000001.json")
    expected_xy = {
        "CAM_BACK_LEFT": (-1.0, 1.0),
        "CAM_FRONT_LEFT": (1.0, 1.0),
        "CAM_FRONT_RIGHT": (1.0, -1.0),
        "CAM_BACK_RIGHT": (-1.0, -1.0),
        "CAM_BACK": (-1.0, 0.0),
        "CAM_FRONT_WIDE": (1.0, 0.0),
        "CAM_FRONT_NARROW": (1.0, 0.0),
    }
    for name, expected in expected_xy.items():
        forward = calibration[name].external[:3, :3].T[:, 2]
        expected_direction = np.asarray(expected, dtype=np.float64)
        expected_direction /= np.linalg.norm(expected_direction)
        assert np.dot(forward[:2], expected_direction) > 0.85, name


@DATASET_REQUIRED
def test_pcd_reader_and_voxel_downsample():
    path = SEQUENCE / "Key_frames/LIDAR_CONCAT/1779426718.550012.pcd"
    points = read_pcd(path, fields=("x", "y", "z"))
    assert points.ndim == 2 and points.shape[1] == 3
    reduced = voxel_downsample(points, 0.2)
    assert 0 < len(reduced) < len(points)


@DATASET_REQUIRED
def test_leave_one_out_hides_target_camera():
    record = build_sequence_manifest(SEQUENCE, compute_quality=False)[0]
    samples = make_leave_one_out_samples([record], "CAM_FRONT_WIDE")
    assert len(samples) == 1
    assert samples[0].target_camera not in samples[0].source_cameras
    assert set(samples[0].source_cameras) == {
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_NARROW",
        "CAM_FRONT_RIGHT",
    }


@DATASET_REQUIRED
def test_nearest_source_dropout_removes_pose_nearest_remaining_camera():
    from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset

    dataset = LeaveOneOutTorchDataset.__new__(LeaveOneOutTorchDataset)
    dataset.drop_nearest_source_probability = 1.0
    dataset.front_narrow_drop_nearest_source_probability = None
    dataset.exclude_nearest_source = False
    calibrations = load_calibrations(SEQUENCE / "Key_frames/camera_config/000001.json")
    sources = (
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_NARROW",
        "CAM_FRONT_RIGHT",
    )
    remaining = dataset._maybe_drop_nearest_source(
        sources,
        calibrations,
        calibrations["CAM_FRONT_WIDE"],
    )
    assert len(remaining) == len(sources) - 1
    assert "CAM_FRONT_NARROW" not in remaining

    dataset.drop_nearest_source_probability = 0.0
    assert dataset._maybe_drop_nearest_source(
        sources, calibrations, calibrations["CAM_FRONT_WIDE"],
    ) == sources


@DATASET_REQUIRED
def test_front_narrow_can_preserve_wide_source_with_camera_specific_dropout():
    from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset

    dataset = LeaveOneOutTorchDataset.__new__(LeaveOneOutTorchDataset)
    dataset.drop_nearest_source_probability = 1.0
    dataset.front_narrow_drop_nearest_source_probability = 1.0
    dataset.exclude_nearest_source = False
    calibrations = load_calibrations(SEQUENCE / "Key_frames/camera_config/000001.json")
    sources = tuple(name for name in calibrations if name != "CAM_FRONT_NARROW")
    remaining = dataset._maybe_drop_nearest_source(
        sources, calibrations, calibrations["CAM_FRONT_NARROW"],
    )
    assert len(remaining) == len(sources) - 1
    assert "CAM_FRONT_WIDE" in remaining


@DATASET_REQUIRED
def test_target_rays_and_sparse_depth_are_geometry_consistent():
    config = SEQUENCE / "Key_frames/camera_config/000001.json"
    calibration = load_calibrations(config)["CAM_FRONT_WIDE"]
    camera = TargetCamera(calibration)
    rays = camera.ray_map(32, 18)
    assert rays.shape == (3, 18, 32)
    assert np.allclose(np.linalg.norm(rays, axis=0), 1.0, atol=1e-5)
    points = read_pcd(SEQUENCE / "Key_frames/LIDAR_CONCAT/1779426718.550012.pcd", fields=("x", "y", "z"))
    depth, mask, point_index = sparse_depth_from_points(points, calibration, (320, 180))
    assert depth.shape == mask.shape == point_index.shape == (180, 320)
    assert np.array_equal(mask, point_index >= 0)


def test_target_ray_map_undistorts_pixels():
    intrinsic = np.array([
        [80.0, 0.0, 31.5],
        [0.0, 80.0, 23.5],
        [0.0, 0.0, 1.0],
    ])
    distortion = np.array([-0.30, 0.10, 0.001, -0.0005, 0.0])
    camera = TargetCamera(CameraCalibration(
        "synthetic", intrinsic, distortion, np.eye(4), 64, 48,
    ))
    rays = camera.ray_map()
    distorted_corner = np.array([(0.0 - 31.5) / 80.0, (0.0 - 23.5) / 80.0])
    ideal_corner = rays[:2, 0, 0] / rays[2, 0, 0]
    assert np.linalg.norm(ideal_corner) > np.linalg.norm(distorted_corner)


@DATASET_REQUIRED
def test_local_point_builder_limits_and_preserves_attributes():
    config = SEQUENCE / "Key_frames/camera_config/000001.json"
    calibration = load_calibrations(config)["CAM_FRONT_WIDE"]
    points = read_pcd(SEQUENCE / "Key_frames/LIDAR_CONCAT/1779426718.550012.pcd")
    selected = LocalPointBuilder(max_points=5000).build(points, calibration)
    assert len(selected) <= 5000
    assert selected.shape[1] == points.shape[1]


@DATASET_REQUIRED
def test_temporal_window_skips_sequence_boundaries():
    records = build_sequence_manifest(SEQUENCE, compute_quality=False)
    assert temporal_window(records, 0, 3) == ()
    assert len(temporal_window(records, 1, 3)) == 3
    assert len(temporal_window(records, 2, 5)) == 5


@DATASET_REQUIRED
def test_temporal_window_rejects_gaps_and_non_monotonic_input():
    records = build_sequence_manifest(SEQUENCE, compute_quality=False)[:3]
    with_gap = [
        replace(records[0], lidar_timestamp=0.0),
        replace(records[1], lidar_timestamp=0.5),
        replace(records[2], lidar_timestamp=1.5),
    ]
    assert temporal_window(with_gap, 1, 3, max_interval_seconds=0.75) == ()
    unordered = [with_gap[1], with_gap[0], with_gap[2]]
    try:
        temporal_window(unordered, 1, 3, max_interval_seconds=0.75)
    except ValueError as error:
        assert "strictly ordered" in str(error)
    else:
        raise AssertionError("non-monotonic temporal records were accepted")


def test_unobserved_point_colors_stay_zero_and_invalid():
    observed_colors = np.asarray([
        [[0.2, 0.4, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.4, 0.6, 0.8], [0.7, 0.5, 0.3], [0.0, 0.0, 0.0]],
    ], dtype=np.float32)
    observed_valid = np.asarray([
        [True, False, False],
        [True, True, False],
    ])
    colors, validity, counts = _fuse_point_colors(
        observed_colors, observed_valid,
    )
    assert np.allclose(colors[0], [0.3, 0.5, 0.7])
    assert np.allclose(colors[1], [0.7, 0.5, 0.3])
    assert np.array_equal(colors[2], np.zeros(3, dtype=np.float32))
    assert np.array_equal(validity, [True, True, False])
    assert np.array_equal(counts, [2.0, 1.0, 0.0])


def test_renderer_separates_lidar_structure_from_rgb_appearance():
    calibration = CameraCalibration(
        "synthetic",
        np.asarray([[20.0, 0.0, 8.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        16,
        12,
    )
    render = FixedSplatRenderer().render(
        np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        calibration,
        color_validity=np.asarray([False]),
    )
    assert render.structure_validity.sum() == 1
    assert render.appearance_validity.sum() == 0
    assert render.depth.max() == 5.0
    assert render.opacity.max() == 0.0
    assert render.coarse_rgb.max() == 0.0


@DATASET_REQUIRED
def test_temporal_dataset_can_skip_foundation_source_images():
    from gcr_nvs.datasets.temporal_torch_dataset import TemporalLeaveOneOutTorchDataset

    dataset = TemporalLeaveOneOutTorchDataset(
        SEQUENCE.parent,
        [SEQUENCE.name],
        width=32,
        height=18,
        max_points=2000,
        max_geometry_points=2000,
        max_samples=1,
        target_camera="CAM_BACK_LEFT",
        include_foundation_inputs=False,
    )
    sample = dataset[0]
    assert "source_images" not in sample
    assert sample["geometry"].shape == (3, 13, 18, 32)
    assert torch.all(sample["temporal_timestamps"][1:] > sample["temporal_timestamps"][:-1])
    assert sample["temporal_offsets_s"][1] == 0.0
    assert sample["structure_validity"].shape == (3, 1, 18, 32)
    assert sample["appearance_validity"].shape == (3, 1, 18, 32)


def test_local_icp_returns_rigid_pose_contract():
    from gcr_nvs.geometry.icp import estimate_icp

    rng = np.random.default_rng(7)
    source = rng.normal(size=(500, 3)).astype(np.float32)
    target = source + np.array([0.2, -0.1, 0.05], dtype=np.float32)
    transform, quality = estimate_icp(source, target, max_points=500)
    assert transform.shape == (4, 4)
    assert np.isfinite(transform).all()
    assert "valid" in quality and "rmse_m" in quality


@DATASET_REQUIRED
def test_target_rig_is_explicit_and_loadable():
    from gcr_nvs.geometry.rig import load_target_rig, resolve_target_calibration

    rig = load_target_rig(ROOT / "configs/target_rigs/l4_base.yaml")
    assert rig.rig_id == "l4_base"
    assert rig.camera("CAM_FRONT_WIDE").inherit_source_camera == "CAM_FRONT_WIDE"
    source = load_calibrations(SEQUENCE / "Key_frames/camera_config/000001.json")
    resolved = resolve_target_calibration(rig.camera("CAM_FRONT_WIDE"), source)
    assert resolved.width == 1920
    assert resolved.height == 1080
    assert np.allclose(resolved.intrinsic[0], source["CAM_FRONT_WIDE"].intrinsic[0] * 0.5)
    assert np.allclose(resolved.intrinsic[1], source["CAM_FRONT_WIDE"].intrinsic[1] * 0.5)
