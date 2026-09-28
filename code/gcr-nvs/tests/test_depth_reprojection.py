import torch

from gcr_nvs.rendering.depth_reprojection import (
    align_inverse_depth_to_lidar,
    align_metric_depth_to_lidar,
    forward_splat_rgb,
)


def test_inverse_depth_alignment_recovers_metric_scale_and_shift():
    metric_depth = torch.linspace(2.0, 40.0, 64).reshape(1, 1, 8, 8)
    relative_inverse_depth = 2.5 * metric_depth.reciprocal() + 0.07
    aligned = align_inverse_depth_to_lidar(relative_inverse_depth, metric_depth)
    assert torch.allclose(aligned, metric_depth, atol=1e-3)


def test_metric_depth_alignment_recovers_scale():
    metric_depth = torch.linspace(2.0, 40.0, 64).reshape(1, 1, 8, 8)
    predicted_metric_depth = metric_depth / 1.8
    aligned = align_metric_depth_to_lidar(predicted_metric_depth, metric_depth)
    assert torch.allclose(aligned, metric_depth, atol=1e-3)


def test_forward_splat_identity_preserves_rgb():
    height, width = 32, 48
    source = torch.rand(1, 3, height, width)
    depth = torch.full((1, 1, height, width), 10.0)
    intrinsic = torch.tensor([
        [80.0, 0.0, width / 2],
        [0.0, 80.0, height / 2],
        [0.0, 0.0, 1.0],
    ]).unsqueeze(0)
    external = torch.eye(4).unsqueeze(0)
    rendered, rendered_depth, valid = forward_splat_rgb(
        source, depth, intrinsic, external, intrinsic, external,
    )
    assert torch.all(valid == 1)
    assert torch.allclose(rendered, source, atol=1e-5)
    assert torch.allclose(rendered_depth, depth, atol=1e-5)


def test_forward_splat_identity_preserves_rgb_with_brown_distortion():
    height, width = 32, 48
    source = torch.rand(1, 3, height, width)
    depth = torch.full((1, 1, height, width), 10.0)
    intrinsic = torch.tensor([[
        [40.0, 0.0, (width - 1) / 2],
        [0.0, 40.0, (height - 1) / 2],
        [0.0, 0.0, 1.0],
    ]])
    distortion = torch.tensor([[-0.25, 0.08, 0.001, -0.0005, 0.0]])
    external = torch.eye(4).unsqueeze(0)
    rendered, _, valid = forward_splat_rgb(
        source, depth, intrinsic, external, intrinsic, external,
        source_distortion=distortion,
        target_distortion=distortion,
    )
    mask = valid.bool().expand_as(source)
    assert valid.float().mean() > 0.98
    assert torch.allclose(rendered[mask], source[mask], atol=2e-4)


def test_forward_splat_ignores_zero_source_depth():
    source = torch.rand(1, 3, 8, 12)
    depth = torch.zeros(1, 1, 8, 12)
    intrinsic = torch.tensor([[
        [20.0, 0.0, 5.5], [0.0, 20.0, 3.5], [0.0, 0.0, 1.0],
    ]])
    external = torch.eye(4).unsqueeze(0)
    rendered, rendered_depth, valid = forward_splat_rgb(
        source, depth, intrinsic, external, intrinsic, external,
    )
    assert torch.all(valid == 0)
    assert torch.all(rendered == 0)
    assert torch.all(rendered_depth == 0)
