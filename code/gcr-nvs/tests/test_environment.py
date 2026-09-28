import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.environment import (
    render_directional_source_fallback,
    render_infinite_sky,
)


def test_infinite_sky_identity_warp_only_fills_semantic_sky():
    calibration = CameraCalibration(
        "camera",
        np.asarray([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        8,
        8,
    )
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:4] = np.asarray([0.2, 0.4, 0.8], dtype=np.float32)
    sky = np.zeros((8, 8), dtype=bool)
    sky[:4] = True
    result = render_infinite_sky(
        [image], [sky], [calibration], calibration, width=8, height=8,
    )
    assert result.validity[:4].all()
    assert not result.validity[4:].any()
    assert np.allclose(result.rgb[:4], [0.2, 0.4, 0.8], atol=1e-5)
    assert np.all(result.rgb[4:] == 0.0)


def test_directional_source_fallback_preserves_identity_rgb():
    calibration = CameraCalibration(
        "camera",
        np.asarray([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        8,
        8,
    )
    yy, xx = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    image = np.stack([xx / 7.0, yy / 7.0, np.full_like(xx, 0.5)], axis=-1).astype(np.float32)
    result = render_directional_source_fallback(
        [image], [calibration], calibration, width=8, height=8,
    )
    assert result.validity.all()
    assert np.all(result.source_provenance == 0)
    assert np.allclose(result.rgb, image, atol=1e-5)


def test_directional_fallback_prefers_same_physical_camera():
    calibration = CameraCalibration(
        "CAM_FRONT_NARROW", np.eye(3), np.zeros(5), np.eye(4), 4, 4,
    )
    preferred = np.full((4, 4, 3), [0.1, 0.2, 0.3], dtype=np.float32)
    competing = np.full((4, 4, 3), [0.9, 0.8, 0.7], dtype=np.float32)
    other = CameraCalibration(
        "CAM_FRONT_WIDE", np.eye(3), np.zeros(5), np.eye(4), 4, 4,
    )
    result = render_directional_source_fallback(
        [competing, preferred], [other, calibration], calibration, 4, 4,
    )
    assert np.allclose(result.rgb, preferred)
    assert np.all(result.source_provenance == 1)


def test_directional_fallback_prefers_current_over_temporal_same_camera():
    current_calibration = CameraCalibration(
        "CAM_FRONT_NARROW", np.eye(3), np.zeros(5), np.eye(4), 4, 4,
    )
    temporal_calibration = CameraCalibration(
        "T3:CAM_FRONT_NARROW", np.eye(3), np.zeros(5), np.eye(4), 4, 4,
    )
    current = np.full((4, 4, 3), [0.1, 0.2, 0.3], dtype=np.float32)
    temporal = np.full((4, 4, 3), [0.9, 0.8, 0.7], dtype=np.float32)
    result = render_directional_source_fallback(
        [temporal, current],
        [temporal_calibration, current_calibration],
        current_calibration,
        4,
        4,
    )
    assert np.allclose(result.rgb, current)
    assert np.all(result.source_provenance == 1)
