import torch
import numpy as np

from gcr_nvs.models.geometry_appearance_nvs import (
    GeometryAppearanceNVS,
    TargetRGBDecoder,
)
from gcr_nvs.evaluation.build_native_surface_cache import estimate_lidar_normals


def _inputs():
    geometry = torch.randn(1, 5, 32, requires_grad=True)
    features = torch.randn(1, 5, 3, 64)
    rgb = torch.rand(1, 5, 3, 3)
    valid = torch.tensor([[[True, True, False]] * 5])
    prior = torch.zeros(1, 5, 3)
    return geometry, features, rgb, valid, prior


def test_rgb_reconstruction_gradient_cannot_modify_lidar_geometry_tokens():
    model = GeometryAppearanceNVS(detach_geometry_for_rgb=True)
    geometry, features, rgb, valid, prior = _inputs()
    field = model.build_surface_field(geometry, features, rgb, valid, prior)
    field.features.square().mean().backward()
    assert geometry.grad is None
    assert any(
        parameter.grad is not None
        for parameter in model.appearance_encoder.parameters()
    )


def test_appearance_branch_learns_rgb_but_never_invents_invalid_surfaces():
    model = GeometryAppearanceNVS(detach_geometry_for_rgb=True).eval()
    geometry, features, rgb, valid, prior = _inputs()
    valid[:, 2] = False
    with torch.no_grad():
        first = model.build_surface_field(geometry, features, rgb, valid, prior)
        changed = model.build_surface_field(
            geometry, features, rgb + 0.25, valid, prior,
        )
    assert not torch.allclose(first.features[:, :2], changed.features[:, :2])
    assert torch.equal(first.features[:, 2], torch.zeros_like(first.features[:, 2]))
    assert not first.appearance_validity[:, 2].any()


def test_target_decoder_has_rgb_output_only_and_masks_unknown():
    decoder = TargetRGBDecoder(feature_dim=32).eval()
    height, width = 8, 12
    valid = torch.ones(1, 1, height, width, dtype=torch.bool)
    valid[..., :2, :3] = False
    with torch.no_grad():
        rgb = decoder(
            torch.randn(1, 32, height, width),
            torch.rand(1, 1, height, width) * 40.0,
            torch.randn(1, 3, height, width),
            torch.rand(1, 1, height, width),
            torch.rand(1, 1, height, width),
            torch.rand(1, 1, height, width),
            torch.randn(1, 3, height, width),
            valid,
        )
    assert rgb.shape == (1, 3, height, width)
    assert torch.equal(rgb[..., :2, :3], torch.zeros_like(rgb[..., :2, :3]))
    assert not any("depth" in name for name, _ in decoder.named_modules())


def test_lidar_normals_are_geometry_only_and_oriented_to_camera():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    points = np.stack([x.reshape(-1), y.reshape(-1), np.zeros(x.size)], axis=1)
    normals = estimate_lidar_normals(points, np.asarray([0.0, 0.0, 2.0]), neighbors=8)
    assert normals.shape == (len(points), 3)
    assert np.isfinite(normals).all()
    assert np.mean(normals[:, 2]) > 0.9
