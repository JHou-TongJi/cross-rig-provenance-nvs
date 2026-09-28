import torch

from gcr_nvs.models.t0_structure_constraint import T0StructureConstraintNet


def test_initial_network_is_exact_da3_identity():
    model = T0StructureConstraintNet(channels=8)
    h, w = 33, 65
    scalar = [torch.ones(1, 1, h, w) for _ in range(10)]
    scalar[0] *= 20.0
    scalar[2] *= 18.0
    scalar[4] *= 22.0
    output = model(*scalar, torch.zeros(1, 3, h, w), torch.zeros(1, 3, h, w), torch.zeros(1, 32, h, w))
    assert output.depth_range_m.shape == (1, 1, h, w)
    assert torch.equal(output.depth_range_m, scalar[0])
    assert torch.count_nonzero(output.residual_log_range) == 0


def test_far_from_lidar_is_always_identity():
    model = T0StructureConstraintNet(channels=8)
    h, w = 32, 64
    scalar = [torch.ones(1, 1, h, w) for _ in range(10)]
    scalar[0] *= 20.0
    scalar[8].fill_(128.0)
    with torch.no_grad():
        model.head.bias[0] = 10.0
    output = model(*scalar, torch.zeros(1, 3, h, w), torch.zeros(1, 3, h, w), torch.zeros(1, 32, h, w))
    assert float((output.depth_range_m - scalar[0]).abs().max()) < 0.03


def test_transformer_accepts_multiple_resolutions():
    model = T0StructureConstraintNet(channels=8)
    for h, w in ((32, 64), (45, 81)):
        scalar = [torch.ones(1, 1, h, w) for _ in range(10)]
        scalar[0] *= 20.0
        output = model(*scalar, torch.zeros(1, 3, h, w), torch.zeros(1, 3, h, w), torch.zeros(1, 32, h, w))
        assert output.depth_range_m.shape[-2:] == (h, w)
