import numpy as np
import torch

from gcr_nvs.geometry.appearance import project_surfaces_to_sources
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.models.lidar_appearance import (
    LearnedViewWeighter,
    LiDARSurfaceAppearanceLift,
)


def _camera():
    return CameraCalibration(
        "test",
        np.asarray([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        np.eye(4),
        8,
        8,
    )


def test_source_projection_z_buffer_keeps_front_surface():
    points = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    uv, valid, depth, centers = project_surfaces_to_sources(points, [_camera()])
    assert uv.shape == (1, 2, 2)
    assert np.array_equal(valid, [[True, False]])
    assert np.allclose(depth, [[2.0, 5.0]])
    assert np.allclose(centers, [[0.0, 0.0, 0.0]])


def test_appearance_lift_never_invents_unobserved_rgb_or_features():
    lift = LiDARSurfaceAppearanceLift(
        feature_dim=8,
        top_k=2,
        use_dino=False,
        dino_pretrained=False,
    ).eval()
    images = torch.zeros(1, 2, 3, 8, 8)
    images[:, 0, 0] = 1.0
    images[:, 1, 1] = 1.0
    xyz = torch.tensor([[[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]])
    uv = torch.tensor([[[[4.0, 4.0], [4.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]])
    valid = torch.tensor([[[True, False], [True, False]]])
    centers = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    with torch.no_grad():
        result = lift(images, xyz, uv, valid, centers)
    assert torch.equal(result.appearance_validity, torch.tensor([[True, False]]))
    assert torch.allclose(result.rgb[0, 0], torch.tensor([0.5, 0.5, 0.0]), atol=1e-5)
    assert torch.equal(result.rgb[0, 1], torch.zeros(3))
    assert torch.equal(result.features[0, 1], torch.zeros(8))
    assert result.source_provenance[0, 1] == -1


def test_learned_view_weighter_only_blends_valid_source_rgb():
    model = LearnedViewWeighter(feature_dim=4).eval()
    features = torch.zeros(1, 2, 3, 4)
    rgb = torch.tensor([[[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]]])
    valid = torch.tensor([[[True, True, False], [False, False, False]]])
    prior = torch.zeros(1, 2, 3)
    with torch.no_grad():
        output, weights = model(features, rgb, valid, prior)
    assert weights[0, 0, 2] == 0
    assert torch.allclose(output[0, 0].sum(), torch.tensor(1.0))
    assert torch.equal(output[0, 1], torch.zeros(3))
