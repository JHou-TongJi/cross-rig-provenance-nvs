import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.dense_lidar_reprojection import (
    densify_lidar_surfaces,
    frequency_preserving_rgb_fusion,
    reproject_source_rgb,
)


def _calibration(width=32, height=24):
    return CameraCalibration(
        "test",
        np.array([[20.0, 0.0, (width - 1) / 2], [0.0, 20.0, (height - 1) / 2], [0.0, 0.0, 1.0]]),
        np.zeros(5), np.eye(4), width, height,
    )


def test_dense_geometry_uses_lidar_plane_without_rgb():
    calibration = _calibration()
    points = np.array([
        [-1.0, -1.0, 10.0], [1.0, -1.0, 10.0],
        [-1.0, 1.0, 10.0], [1.0, 1.0, 10.0],
    ], dtype=np.float32)
    normals = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (4, 1))
    dense = densify_lidar_surfaces(
        points, normals, calibration, maximum_seed_distance_px=20.0,
    )
    assert dense.validity.mean() > 0.8
    assert np.allclose(dense.surface_points[dense.validity, 2], 10.0, atol=1e-4)
    assert np.all(dense.source_point_index[dense.validity] >= 0)


def test_dense_rgb_is_sampled_from_selected_real_view():
    calibration = _calibration()
    points = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    dense = densify_lidar_surfaces(
        points, normals, calibration, maximum_seed_distance_px=20.0,
    )
    red = np.zeros((24, 32, 3), dtype=np.float32)
    red[..., 0] = 1.0
    green = np.zeros_like(red)
    green[..., 1] = 1.0
    render = reproject_source_rgb(
        dense, [red, green], [calibration, calibration],
        np.array([[0.1, 0.9]], dtype=np.float32),
        np.array([[True, True]]),
    )
    assert render.validity.any()
    assert np.allclose(render.rgb[render.validity], np.array([0.0, 1.0, 0.0]))
    assert np.all(render.provenance[render.validity] == 1)


def test_dense_rgb_soft_fusion_blends_only_valid_real_views():
    calibration = _calibration()
    points = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    dense = densify_lidar_surfaces(
        points, normals, calibration, maximum_seed_distance_px=20.0,
    )
    red = np.zeros((24, 32, 3), dtype=np.float32)
    red[..., 0] = 1.0
    green = np.zeros_like(red)
    green[..., 1] = 1.0
    render = reproject_source_rgb(
        dense, [red, green], [calibration, calibration],
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.array([[True, True]]),
        hard_selection=False,
        spatial_smoothing_sigma=0.0,
    )
    assert render.validity.any()
    assert np.allclose(
        render.rgb[render.validity], np.array([0.5, 0.5, 0.0]), atol=1e-5,
    )


def test_frequency_fusion_preserves_base_and_never_fills_invalid_unknown():
    aligned = np.full((12, 16, 3), 0.4, dtype=np.float32)
    native = aligned.copy()
    native[:, 1::2] += 0.1
    aligned_valid = np.ones((12, 16), dtype=bool)
    native_valid = aligned_valid.copy()
    output_valid = aligned_valid.copy()
    output_valid[:2, :3] = False
    fused = frequency_preserving_rgb_fusion(
        aligned, aligned_valid, native, native_valid, output_valid,
        detail_gain=1.0,
    )
    assert fused[:, 1::2].mean() > fused[:, ::2].mean()
    assert np.all(fused[:2, :3] == 0.0)
