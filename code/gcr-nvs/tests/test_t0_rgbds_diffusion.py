import numpy as np
import torch

from gcr_nvs.datasets.t0_rgbds_diffusion import (
    assert_no_hole_leakage,
    build_rgbds_conditions,
    hole_inner_boundary_mask,
    hole_topology_sampling_weight,
    masked_rgb_gradient_l1,
    resize_hole_mask,
)


def test_rgbds_conditions_hide_gt_and_block_dynamic_extension() -> None:
    height, width = 32, 48
    rgb = np.random.default_rng(4).random((height, width, 3), dtype=np.float32)
    depth = np.full((height, width), 12.0, np.float32)
    dynamic = np.zeros((height, width), bool)
    dynamic[10:20, 12:20] = True
    hole = np.zeros((height, width), bool)
    hole[8:24, 20:34] = True
    semantics = np.random.default_rng(7).normal(size=(8, height, width)).astype(np.float32)
    first = build_rgbds_conditions(rgb, depth, dynamic, semantics, hole)
    changed = rgb.copy()
    changed[hole] = np.random.default_rng(9).random((hole.sum(), 3))
    second = build_rgbds_conditions(changed, depth, dynamic, semantics, hole)
    assert_no_hole_leakage(first, second)
    assert np.all(first["masked_rgb"][hole] == 0.0)
    assert np.all(first["control"][:, hole][4:12] == 0.0)
    assert first["forbid_foreground_extension"].any()


def test_rgbds_observed_values_are_preserved() -> None:
    rgb = np.full((12, 16, 3), 0.25, np.float32)
    depth = np.full((12, 16), 8.0, np.float32)
    hole = np.zeros((12, 16), bool)
    hole[:, -3:] = True
    result = build_rgbds_conditions(
        rgb, depth, np.zeros_like(hole), np.zeros((8, 12, 16), np.float32), hole,
    )
    assert np.array_equal(result["masked_rgb"][~hole], rgb[~hole])
    assert not result["forbid_foreground_extension"].any()


def test_hole_resize_preserves_single_pixel_high_resolution_cracks() -> None:
    hole = np.zeros((2160, 3840), bool)
    hole[1100, 1900] = True
    reduced = resize_hole_mask(hole, (512, 288))
    assert reduced.any()


def test_topology_sampling_prioritizes_large_contiguous_edge_holes() -> None:
    small = np.zeros((64, 96), bool)
    small[30:32, 40:42] = True
    large_edge = np.zeros_like(small)
    large_edge[:20, :36] = True
    assert hole_topology_sampling_weight(large_edge) > hole_topology_sampling_weight(small)


def test_inner_boundary_stays_inside_hole() -> None:
    hole = torch.zeros(1, 1, 20, 30)
    hole[:, :, 4:16, 8:24] = 1.0
    boundary = hole_inner_boundary_mask(hole, width=2)
    assert torch.all(boundary <= hole)
    assert boundary.sum() > 0
    assert boundary[:, :, 8:12, 12:20].sum() == 0


def test_masked_gradient_loss_is_zero_for_matching_images() -> None:
    image = torch.rand(2, 3, 16, 24)
    mask = torch.zeros(2, 1, 16, 24)
    mask[:, :, 5:11, 7:17] = 1.0
    assert masked_rgb_gradient_l1(image, image.clone(), mask).item() == 0.0


def test_masked_gradient_loss_detects_boundary_discontinuity() -> None:
    target = torch.zeros(1, 3, 12, 20)
    prediction = target.clone()
    mask = torch.zeros(1, 1, 12, 20)
    mask[:, :, :, 10:14] = 1.0
    prediction[:, :, :, 10:14] = 1.0
    assert masked_rgb_gradient_l1(prediction, target, mask).item() > 0.0
