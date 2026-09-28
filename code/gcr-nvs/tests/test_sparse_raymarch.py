import torch

from gcr_nvs.rendering.sparse_geometry_raymarch import (
    coordinate_membership,
    first_surface_indices,
    _nested_depth_samples,
)


def test_first_surface_uses_nearest_valid_occupied_sample():
    occupancy = torch.tensor([
        [0.1, 0.8, 0.9, 0.2],
        [0.9, 0.8, 0.1, 0.1],
        [0.1, 0.2, 0.3, 0.4],
    ])
    validity = torch.tensor([
        [True, True, True, True],
        [False, True, True, True],
        [True, True, True, True],
    ])
    first, hit = first_surface_indices(occupancy, validity, threshold=0.5)
    assert torch.equal(first, torch.tensor([1, 1, 0]))
    assert torch.equal(hit, torch.tensor([True, True, False]))


def test_first_surface_rejects_shape_mismatch():
    occupancy = torch.zeros(2, 3)
    validity = torch.zeros(2, 2, dtype=torch.bool)
    try:
        first_surface_indices(occupancy, validity, threshold=0.5)
    except ValueError as error:
        assert "matching" in str(error)
    else:
        raise AssertionError("shape mismatch must fail")


def test_coordinate_membership_requires_nearby_occupied_support():
    query = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 4],
    ], dtype=torch.int32)
    occupied = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    assert torch.equal(
        coordinate_membership(query, occupied, stride=4),
        torch.tensor([True, True, False]),
    )


def test_nested_depth_samples_keep_32_layer_grid():
    base = _nested_depth_samples(1.0, 80.0, 32, torch.device("cpu"))
    refined = _nested_depth_samples(1.0, 80.0, 64, torch.device("cpu"))
    assert len(refined) == 64
    assert torch.allclose(refined[:1], base[:1])
    assert torch.allclose(refined[-1:], base[-1:])
    assert torch.all(torch.isin(base, refined))
