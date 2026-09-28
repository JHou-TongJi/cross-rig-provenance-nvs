import torch

from gcr_nvs.models.target_view_generator import TargetViewGenerator


def test_target_view_generator_preserves_contract_at_non_eighth_resolution():
    model = TargetViewGenerator(geometry_dim=4, appearance_dim=6)
    inputs = {
        "geometry_features": torch.randn(1, 4, 17, 19),
        "appearance_features": torch.randn(1, 6, 17, 19),
        "source_rgb": torch.rand(1, 3, 17, 19),
        "depth": torch.rand(1, 1, 17, 19),
        "normal": torch.randn(1, 3, 17, 19),
        "structure_confidence": torch.rand(1, 1, 17, 19),
        "appearance_confidence": torch.rand(1, 1, 17, 19),
        "ray_map": torch.randn(1, 3, 17, 19),
        "source_validity": torch.ones(1, 1, 17, 19),
        "source_weight": torch.rand(1, 1, 17, 19),
    }
    output = model(**inputs)
    assert output["rgb"].shape == (1, 3, 17, 19)
    assert output["alpha"].shape == (1, 1, 17, 19)
    assert output["logvar"].shape == (1, 1, 17, 19)
    assert torch.isfinite(output["rgb"]).all()
    assert output["rgb"].min() >= 0.0
    assert output["rgb"].max() <= 1.0


def test_target_view_generator_accepts_camera_and_focal_role_conditioning():
    model = TargetViewGenerator(
        geometry_dim=4, appearance_dim=6, camera_conditioning=True,
    )
    inputs = {
        "geometry_features": torch.randn(1, 4, 9, 11),
        "appearance_features": torch.randn(1, 6, 9, 11),
        "source_rgb": torch.rand(1, 3, 9, 11),
        "depth": torch.rand(1, 1, 9, 11),
        "normal": torch.randn(1, 3, 9, 11),
        "structure_confidence": torch.rand(1, 1, 9, 11),
        "appearance_confidence": torch.rand(1, 1, 9, 11),
        "ray_map": torch.randn(1, 3, 9, 11),
        "source_validity": torch.ones(1, 1, 9, 11),
        "source_weight": torch.rand(1, 1, 9, 11),
        "target_camera_index": torch.tensor(4),
        "target_focal_role": torch.tensor([[0.0, 1.0]]),
    }
    output = model(**inputs)
    assert output["rgb"].shape == (1, 3, 9, 11)
    assert torch.isfinite(output["rgb"]).all()
