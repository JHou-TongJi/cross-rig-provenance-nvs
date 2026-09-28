import torch

from gcr_nvs.models.semantic_surface_completion import SemanticSurfaceCompletionNet


def test_surface_completion_preserves_observed_depth_and_trains_holes() -> None:
    model = SemanticSurfaceCompletionNet(semantic_dim=8, channels=8)
    shape = (1, 1, 24, 32)
    depth = torch.full(shape, 12.0)
    validity = torch.ones(shape)
    validity[..., 8:16, 10:20] = 0.0
    lidar = torch.zeros(shape)
    lidar_validity = torch.zeros(shape)
    distance = torch.zeros(shape)
    rays = torch.randn(1, 3, 24, 32)
    edges = torch.randn(1, 3, 24, 32)
    semantics = torch.randn(1, 8, 24, 32)
    output = model(
        depth, validity, lidar, lidar_validity, distance,
        rays, edges, semantics,
    )
    assert torch.equal(output.depth_m[validity.bool()], depth[validity.bool()])
    loss = output.depth_m[~validity.bool()].mean()
    loss.backward()
    assert model.head.weight.grad is not None
