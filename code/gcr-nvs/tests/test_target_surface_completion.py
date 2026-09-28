import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.target_surface_completion import (
    complete_nearest_target_surface,
    inverse_sample_source_rgb,
)


def _calibration(width: int, height: int) -> CameraCalibration:
    intrinsic = np.array([
        [80.0, 0.0, width / 2.0],
        [0.0, 80.0, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    return CameraCalibration(
        "CAM_TEST", intrinsic, np.zeros(5), np.eye(4), width, height,
    )


def test_nearest_surface_completion_only_repairs_short_cracks() -> None:
    depth = np.full((9, 11), 12.0, np.float32)
    valid = np.ones_like(depth, dtype=bool)
    valid[4, 5] = False
    valid[1:4, 8:11] = False
    depth[~valid] = np.inf
    result = complete_nearest_target_surface(depth, valid, maximum_distance_px=1.0)
    assert result.candidate_mask[4, 5]
    assert result.depth[4, 5] == 12.0
    assert not result.candidate_mask[2, 9]


def test_inverse_sampling_recovers_rgb_only_when_depth_agrees() -> None:
    height, width = 8, 10
    calibration = _calibration(width, height)
    rgb = np.zeros((height, width, 3), np.float32)
    rgb[..., 1] = 0.75
    source_depth = np.full((height, width), 10.0, np.float32)
    target_depth = source_depth.copy()
    candidate = np.zeros((height, width), bool)
    candidate[3, 4] = True
    candidate[5, 7] = True
    source_depth[5, 7] = 20.0
    result = inverse_sample_source_rgb(
        target_depth, calibration, [rgb], [source_depth], [calibration],
        candidate, maximum_log_depth_error=0.04,
    )
    assert result.validity[3, 4]
    assert np.allclose(result.rgb[3, 4], [0.0, 0.75, 0.0])
    assert not result.validity[5, 7]
