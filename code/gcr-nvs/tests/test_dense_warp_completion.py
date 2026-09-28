import torch

from gcr_nvs.models.dense_warp_completion import DenseWarpCompletionGenerator


def test_dense_completion_preserves_observed_shape_and_range():
    model = DenseWarpCompletionGenerator(semantic_dim=4, decoder_dim=16).eval()
    inputs = [
        torch.rand(1, 3, 16, 24),
        torch.rand(1, 4, 16, 24),
        torch.rand(1, 1, 16, 24),
        torch.ones(1, 1, 16, 24),
        torch.ones(1, 1, 16, 24),
        torch.rand(1, 3, 16, 24),
    ]
    with torch.no_grad():
        output = model(*inputs)
    assert output["rgb"].shape == inputs[0].shape
    assert torch.isfinite(output["rgb"]).all()
    assert output["rgb"].min() >= 0 and output["rgb"].max() <= 1
    assert output["alpha"].shape == (1, 1, 16, 24)

