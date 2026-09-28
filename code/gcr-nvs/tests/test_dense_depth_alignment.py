import numpy as np

from gcr_nvs.geometry.dense_depth_alignment import align_monocular_depth
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb


def _calibration(width=8, height=6, tx=0.0):
    external = np.eye(4, dtype=np.float64)
    external[0, 3] = tx
    return CameraCalibration(
        "test",
        np.array([[12.0, 0.0, (width - 1) / 2], [0.0, 12.0, (height - 1) / 2], [0.0, 0.0, 1.0]]),
        np.zeros(5), external, width, height,
    )


def test_alignment_keeps_continuous_da3_surface_and_reports_lidar_holdout():
    lidar = np.zeros((6, 8), np.float32)
    lidar[1::2, 1::2] = 10.0
    # Monocular prediction has an inverse-depth affine distortion.
    mono = np.full_like(lidar, 8.0)
    mono[lidar > 0] = 6.0
    result = align_monocular_depth(mono, lidar, holdout_fraction=0.25, min_fit_points=2)
    assert result.fit_count >= 2
    assert result.holdout["count"] > 0
    assert np.all(result.lidar_anchor_depth[lidar > 0] == lidar[lidar > 0])
    assert np.allclose(result.depth, result.aligned_monocular_depth)
    assert np.all(result.depth[lidar == 0] > 0)
    assert result.holdout["rmse_m"] < 1e-4


def test_alignment_fills_monocular_holes_when_confidence_is_missing():
    lidar = np.zeros((4, 4), np.float32)
    lidar[1, 1] = 10.0
    mono = np.full((4, 4), 10.0, np.float32)
    result = align_monocular_depth(mono, lidar, None, min_fit_points=1)
    assert np.all(result.depth > 0)
    assert np.all(result.confidence[lidar == 0] == 1.0)


def test_dense_reprojection_respects_nearest_surface():
    source = np.zeros((6, 8, 3), np.float32)
    source[..., 0] = 1.0
    source2 = np.zeros_like(source)
    source2[..., 1] = 1.0
    depth = np.full((6, 8), 10.0, np.float32)
    depth2 = np.full((6, 8), 5.0, np.float32)
    rgb, valid, provenance = reproject_dense_rgb(
        [source, source2], [depth, depth2], [_calibration(), _calibration()], _calibration(), splat_radius=0,
    )
    assert valid.all()
    assert np.all(provenance == 1)
    assert np.allclose(rgb[valid], np.array([0.0, 1.0, 0.0]))


def test_dense_reprojection_applies_se3_translation():
    source = np.zeros((6, 8, 3), np.float32)
    source[..., 2] = 1.0
    depth = np.full((6, 8), 10.0, np.float32)
    rgb, valid, _ = reproject_dense_rgb(
        [source], [depth], [_calibration()], _calibration(tx=-0.5), splat_radius=1,
    )
    assert valid.mean() > 0.5
    assert np.allclose(rgb[valid, 2], 1.0)


def test_dense_reprojection_can_keep_first_view_as_appearance_anchor():
    first = np.zeros((6, 8, 3), np.float32)
    first[..., 0] = 1.0
    second = np.zeros_like(first)
    second[..., 1] = 1.0
    depth = np.full((6, 8), 10.0, np.float32)
    rgb, valid, provenance = reproject_dense_rgb(
        [first, second], [depth, depth], [_calibration(), _calibration()], _calibration(),
        exclusive_source_priority=True, splat_radius=0,
    )
    assert valid.all()
    assert np.all(provenance == 0)
    assert np.allclose(rgb[valid], np.array([1.0, 0.0, 0.0]))
