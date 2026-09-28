import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.dense_semantic_reprojection import reproject_dense_maps


def _calibration(width=8, height=6):
    return CameraCalibration(
        "test",
        np.array([[12.0, 0.0, (width - 1) / 2], [0.0, 12.0, (height - 1) / 2], [0.0, 0.0, 1.0]]),
        np.zeros(5), np.eye(4), width, height,
    )


def test_semantic_map_uses_same_source_priority_as_rgb():
    first = np.zeros((6, 8, 4), np.float32)
    first[..., 0] = 1.0
    second = np.zeros_like(first)
    second[..., 1] = 1.0
    depth = np.full((6, 8), 10.0, np.float32)
    output, valid, provenance = reproject_dense_maps(
        [first, second], [depth, depth], [_calibration(), _calibration()], _calibration(),
        exclusive_source_priority=True,
    )
    assert valid.all()
    assert np.all(provenance == 0)
    assert np.allclose(output[valid, 0], 1.0)
    assert np.allclose(output[valid, 1:], 0.0)

