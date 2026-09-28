import numpy as np

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.geometry.lidar_depth_layers import project_lidar_layer


def test_lidar_layers_are_true_zbuffered_and_confidence_aware():
    calibration = CameraCalibration(
        "test", np.array([[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]]),
        np.zeros(5), np.eye(4), 20, 20,
    )
    points = np.array([[0., 0., 10.], [0., 0., 5.], [1., 0., 10.]], np.float32)
    layer = project_lidar_layer(points, calibration, 20, 20, np.array([.2, .8, .6], np.float32))
    assert layer.valid[10, 10]
    assert layer.z_m[10, 10] == 5.0
    assert layer.confidence[10, 10] == .8
    assert layer.valid[10, 10] and layer.valid[10, 11]

