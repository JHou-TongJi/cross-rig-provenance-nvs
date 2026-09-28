import numpy as np
import pytest
import torch

from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.rendering.lidar_gaussians import (
    estimate_adaptive_pixel_radius,
    render_fixed_lidar_features,
    render_fixed_lidar_gaussians,
)


def test_fixed_lidar_gaussian_cuda_render_is_nonblank_and_finite():
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by gsplat")
    calibration = CameraCalibration(
        "test",
        np.asarray([[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        32,
        24,
    )
    result = render_fixed_lidar_gaussians(
        torch.tensor([[0.0, 0.0, 5.0]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([True], device="cuda"),
        calibration,
        width=32,
        height=24,
        pixel_radius=1.5,
    )
    assert result.point_count == 1
    assert result.appearance_validity.any()
    assert torch.isfinite(result.rgb).all()
    assert torch.isfinite(result.depth).all()
    assert result.rgb[0].max() > 0.5
    assert result.depth[result.appearance_validity].min() > 0.0


def test_fixed_lidar_gaussian_renders_learned_features_with_gradients():
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by gsplat")
    calibration = CameraCalibration(
        "test",
        np.asarray([[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        32,
        24,
    )
    features = torch.randn(1, 32, device="cuda", requires_grad=True)
    result = render_fixed_lidar_features(
        torch.tensor([[0.0, 0.0, 5.0]], device="cuda"),
        features,
        torch.tensor([True], device="cuda"),
        calibration,
        width=32,
        height=24,
        pixel_radius=1.5,
    )
    assert result.features.shape == (32, 24, 32)
    result.features.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_adaptive_pixel_radius_is_finite_and_per_point():
    calibration = CameraCalibration(
        "test",
        np.asarray([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        320,
        240,
    )
    points = torch.tensor([
        [-0.5, 0.0, 5.0], [0.0, 0.0, 5.0], [0.5, 0.0, 5.0],
    ])
    radii = estimate_adaptive_pixel_radius(points, calibration)
    assert radii.shape == (3,)
    assert torch.isfinite(radii).all()
    assert radii.min() >= 3.0
    assert radii.max() <= 18.0
