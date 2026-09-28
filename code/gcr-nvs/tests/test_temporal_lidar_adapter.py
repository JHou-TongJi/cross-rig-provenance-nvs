import torch

from gcr_nvs.models.temporal_lidar_adapter import TemporalLidarGeometryAdapter


def test_temporal_adapter_is_geometry_only_and_preserves_current_returns():
    model = TemporalLidarGeometryAdapter(channels=8).eval()
    shape = (1, 1, 32, 48)
    scalar = [torch.full(shape, 12.0), torch.full(shape, 8.0), torch.ones(shape),
              torch.full(shape, 10.0), torch.ones(shape), torch.full(shape, 0.8),
              torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)]
    rays = torch.randn(1, 3, 32, 48)
    scalar[1][..., 10, 12] = 7.0
    scalar[2][..., 10, 12] = 1.0
    with torch.no_grad():
        output = model(*scalar, rays)
    assert output.depth_m.shape == shape
    assert output.depth_m[..., 10, 12] == 7.0
    assert output.visibility[..., 10, 12] == 1.0
    assert all("rgb" not in name.lower() and "dino" not in name.lower() for name, _ in model.named_modules())


def test_temporal_adapter_is_identity_without_previous_frame_support():
    model = TemporalLidarGeometryAdapter(channels=8).eval()
    shape = (1, 1, 16, 24)
    values = [torch.full(shape, 12.0), torch.zeros(shape), torch.zeros(shape),
              torch.zeros(shape), torch.zeros(shape), torch.zeros(shape),
              torch.zeros(shape), torch.full(shape, 4.0), torch.zeros(shape)]
    rays = torch.randn(1, 3, 16, 24)
    with torch.no_grad():
        output = model(*values, rays)
    assert torch.allclose(output.depth_m, values[0])


def test_temporal_adapter_preserves_current_support_even_with_temporal_support():
    model = TemporalLidarGeometryAdapter(channels=8).eval()
    shape = (1, 1, 16, 24)
    current = torch.zeros(shape)
    current[..., 5, 7] = 8.0
    current_valid = current > 0
    values = [torch.full(shape, 12.0), current, current_valid,
              torch.full(shape, 10.0), torch.ones(shape), torch.ones(shape),
              torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)]
    rays = torch.randn(1, 3, 16, 24)
    with torch.no_grad():
        output = model(*values, rays)
    assert output.depth_m[..., 5, 7] == 8.0
