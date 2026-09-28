import numpy as np

from gcr_nvs.evaluation.report_lidar_depth_se3_stress import (
    _fragmentation_metrics,
    translated_calibration,
)
from gcr_nvs.geometry.calibration import CameraCalibration


def test_combined_translation_moves_camera_center_in_vehicle_axes():
    calibration = CameraCalibration(
        "test", np.eye(3), np.zeros(5), np.eye(4), 32, 24,
    )
    shifted = translated_calibration(calibration, (0.35, -0.30, 0.25))
    assert np.allclose(shifted.camera_center, [0.35, -0.30, 0.25])
    assert np.allclose(shifted.external[:3, :3], calibration.external[:3, :3])


def test_fragmentation_metrics_detect_provenance_and_depth_cracks():
    rgb = np.zeros((4, 6, 3), dtype=np.float32)
    rgb[:, 3:] = 1.0
    depth = np.full((4, 6), 10.0, dtype=np.float32)
    depth[:, 3:] = 20.0
    valid = np.ones((4, 6), dtype=bool)
    provenance = np.zeros((4, 6), dtype=np.int16)
    provenance[:, 3:] = 1
    metrics = _fragmentation_metrics(rgb, depth, valid, provenance)
    assert metrics["provenance_neighbor_jump_ratio"] > 0.0
    assert metrics["depth_discontinuity_ratio"] > 0.0
    assert metrics["rgb_l1_at_provenance_boundary"] == 1.0
