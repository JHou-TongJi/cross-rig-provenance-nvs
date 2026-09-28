"""Target-camera first-surface queries over the LiDAR sparse geometry field."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gcr_nvs.datasets.sparse_geometry_dataset import SparseGridSpec
from gcr_nvs.geometry.camera import TargetCamera


@dataclass(frozen=True)
class SparseRaymarchResult:
    depth: torch.Tensor
    surface_points: torch.Tensor
    occupancy: torch.Tensor
    confidence: torch.Tensor
    normal: torch.Tensor
    surface_features: torch.Tensor
    structure_validity: torch.Tensor
    query_coverage: torch.Tensor
    occupied_support: torch.Tensor
    unknown_structure_mask: torch.Tensor


def first_surface_indices(
    occupancy: torch.Tensor,
    validity: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first valid occupied sample index and per-ray hit mask."""
    if occupancy.shape != validity.shape or occupancy.ndim != 2:
        raise ValueError("occupancy and validity must have matching [rays, samples] shapes")
    candidates = validity & (occupancy >= threshold)
    hit = candidates.any(dim=1)
    first = candidates.to(torch.int64).argmax(dim=1)
    return first, hit


def _points_to_query_indices(
    points: torch.Tensor,
    grid: SparseGridSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    minimum = points.new_tensor(grid.minimum_xyz)
    maximum = points.new_tensor(grid.maximum_xyz)
    valid = torch.isfinite(points).all(dim=1)
    valid &= (points >= minimum).all(dim=1) & (points < maximum).all(dim=1)
    coordinates_xyz = torch.floor(
        (points[valid] - minimum) / grid.voxel_size_m
    ).to(torch.int32)
    coordinates_zyx = coordinates_xyz[:, [2, 1, 0]]
    batch = torch.zeros(
        len(coordinates_zyx), 1, dtype=torch.int32, device=points.device,
    )
    return torch.cat([batch, coordinates_zyx], dim=1), valid


def coordinate_membership(
    query_indices: torch.Tensor,
    reference_indices: torch.Tensor,
    stride: int = 1,
) -> torch.Tensor:
    """Test whether query and reference coordinates share a coarse voxel."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    query = query_indices.to(torch.int64).clone()
    reference = reference_indices.to(torch.int64).clone()
    query[:, 1:] = torch.div(query[:, 1:], stride, rounding_mode="floor")
    reference[:, 1:] = torch.div(reference[:, 1:], stride, rounding_mode="floor")
    extent = torch.maximum(query.amax(dim=0), reference.amax(dim=0)) + 2
    multipliers = torch.stack([
        extent[1] * extent[2] * extent[3],
        extent[2] * extent[3],
        extent[3],
        extent.new_tensor(1),
    ])
    query_keys = (query * multipliers).sum(dim=1)
    reference_keys = torch.unique((reference * multipliers).sum(dim=1)).sort().values
    positions = torch.searchsorted(reference_keys, query_keys)
    clamped = positions.clamp(max=max(len(reference_keys) - 1, 0))
    return (positions < len(reference_keys)) & (
        reference_keys[clamped] == query_keys
    )


def _nested_depth_samples(
    near_m: float, far_m: float, sample_count: int, device: torch.device,
) -> torch.Tensor:
    """Build samples that retain the proven 32-layer locations.

    A plain ``linspace`` changes every sample location when its count changes.
    For a sparse occupancy field that can turn a hit into a miss.  Keeping the
    original 32-layer grid as a subset makes higher-resolution raymarching
    monotonic in support while adding interleaved samples for thin surfaces.
    """
    if sample_count <= 32:
        return torch.linspace(near_m, far_m, sample_count, device=device)
    base = torch.linspace(near_m, far_m, 32, device=device)
    remaining = sample_count - len(base)
    candidates = []
    # Interleave progressively finer strata inside the original intervals.
    for fraction in (0.5, 0.25, 0.75, 0.125, 0.875, 0.375, 0.625):
        candidates.append(
            base[:-1] + (base[1:] - base[:-1]) * fraction
        )
    pool = torch.cat(candidates)
    selection = torch.linspace(
        0, len(pool) - 1, remaining, device=device,
    ).round().to(torch.long)
    return torch.sort(torch.cat([base, pool[selection]])).values


def raymarch_sparse_geometry(
    model,
    input_features: torch.Tensor,
    input_indices: torch.Tensor,
    spatial_shape: tuple[int, int, int] | list[int],
    target: TargetCamera,
    grid: SparseGridSpec,
    width: int,
    height: int,
    near_m: float = 1.0,
    far_m: float = 80.0,
    sample_count: int = 96,
    occupancy_threshold: float = 0.5,
    occupied_support_stride: int = 4,
    query_chunk_size: int | None = 262_144,
    refine_sample_count: int = 0,
    target_depth_prior: torch.Tensor | None = None,
    pixel_origin: tuple[int, int] = (0, 0),
    encoded_geometry: dict[str, torch.Tensor] | None = None,
) -> SparseRaymarchResult:
    if sample_count < 2 or not (0.0 < near_m < far_m):
        raise ValueError("raymarch requires sample_count >= 2 and 0 < near_m < far_m")
    if refine_sample_count not in (0,) and refine_sample_count < 3:
        raise ValueError("refine_sample_count must be zero or at least three")
    device = input_features.device
    if pixel_origin == (0, 0):
        ray_map = target.ray_map(width, height)
    else:
        ray_map = target.ray_map_region(
            pixel_origin[0], pixel_origin[1], width, height,
        )
    rays = torch.from_numpy(ray_map).to(device=device)
    rays = rays.permute(1, 2, 0).reshape(-1, 3)
    origin = input_features.new_tensor(target.calibration.camera_center)
    depths = _nested_depth_samples(near_m, far_m, sample_count, device)
    effective_sample_count = len(depths)
    points = origin[None, None] + rays[:, None] * depths[None, :, None]
    flat_points = points.reshape(-1, 3)
    query_indices, in_grid = _points_to_query_indices(flat_points, grid)

    unique_indices, inverse = torch.unique(
        query_indices, dim=0, return_inverse=True,
    )
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive or None")
    encoded = encoded_geometry
    if encoded is None and hasattr(model, "encode_sparse") and hasattr(model, "query_encoded"):
        encoded = model.encode_sparse(
            input_features, input_indices, spatial_shape, batch_size=1,
        )

    def query_unique(indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if encoded is None:
            return model(
                input_features,
                input_indices,
                spatial_shape,
                batch_size=1,
                query_indices=indices,
            )
        chunk_size = query_chunk_size or max(len(indices), 1)
        chunk_outputs = [
            model.query_encoded(encoded, indices[start:start + chunk_size])
            for start in range(0, len(indices), chunk_size)
        ]
        return {
            key: torch.cat([chunk[key] for chunk in chunk_outputs], dim=0)
            for key in (
                "indices", "surface_features", "query_validity", "occupancy",
                "sdf", "normal", "visibility", "confidence",
            )
        }

    outputs = query_unique(unique_indices)
    flat_count = len(flat_points)
    occupancy = input_features.new_zeros(flat_count)
    confidence = input_features.new_zeros(flat_count)
    normal = input_features.new_zeros(flat_count, 3)
    feature_dim = outputs["surface_features"].shape[-1]
    surface_features = input_features.new_zeros(flat_count, feature_dim)
    validity = torch.zeros(flat_count, dtype=torch.bool, device=device)
    occupancy[in_grid] = outputs["occupancy"][inverse, 0].to(occupancy.dtype)
    confidence[in_grid] = outputs["confidence"][inverse, 0].to(confidence.dtype)
    normal[in_grid] = outputs["normal"][inverse].to(normal.dtype)
    surface_features[in_grid] = outputs["surface_features"][inverse].to(surface_features.dtype)
    validity[in_grid] = outputs["query_validity"][inverse, 0] > 0.5
    if input_features.shape[1] >= 7:
        occupied_inputs = input_indices[input_features[:, 6] > 0]
    else:
        occupied_inputs = input_indices
    unique_support = coordinate_membership(
        unique_indices, occupied_inputs, stride=occupied_support_stride,
    )
    support = torch.zeros(flat_count, dtype=torch.bool, device=device)
    support[in_grid] = unique_support[inverse]

    ray_count = height * width
    occupancy = occupancy.reshape(ray_count, effective_sample_count)
    confidence = confidence.reshape(ray_count, effective_sample_count)
    normal = normal.reshape(ray_count, effective_sample_count, 3)
    surface_features = surface_features.reshape(ray_count, effective_sample_count, feature_dim)
    validity = validity.reshape(ray_count, effective_sample_count)
    support = support.reshape(ray_count, effective_sample_count)
    first, hit = first_surface_indices(
        occupancy, validity & support, occupancy_threshold,
    )
    rows = torch.arange(ray_count, device=device)
    hit_depth = depths[first]
    hit_occupancy = occupancy[rows, first]
    hit_confidence = confidence[rows, first]
    hit_normal = normal[rows, first]
    hit_features = surface_features[rows, first]
    if refine_sample_count and hit.any():
        hit_rows = hit.nonzero(as_tuple=False)[:, 0]
        coarse_step = (far_m - near_m) / (sample_count - 1)
        offsets = torch.linspace(
            -coarse_step, coarse_step, refine_sample_count, device=device,
        )
        refine_depths = (
            depths[first[hit_rows], None] + offsets[None]
        ).clamp(near_m, far_m)
        refine_points = (
            origin[None, None]
            + rays[hit_rows, None] * refine_depths[..., None]
        )
        refine_flat = refine_points.reshape(-1, 3)
        refine_indices, refine_in_grid = _points_to_query_indices(refine_flat, grid)
        refine_unique, refine_inverse = torch.unique(
            refine_indices, dim=0, return_inverse=True,
        )
        refine_outputs = query_unique(refine_unique)
        refine_count = len(refine_flat)
        refine_occupancy = input_features.new_zeros(refine_count)
        refine_confidence = input_features.new_zeros(refine_count)
        refine_normal = input_features.new_zeros(refine_count, 3)
        refine_features = input_features.new_zeros(refine_count, feature_dim)
        refine_validity = torch.zeros(refine_count, dtype=torch.bool, device=device)
        refine_occupancy[refine_in_grid] = refine_outputs["occupancy"][refine_inverse, 0].to(refine_occupancy.dtype)
        refine_confidence[refine_in_grid] = refine_outputs["confidence"][refine_inverse, 0].to(refine_confidence.dtype)
        refine_normal[refine_in_grid] = refine_outputs["normal"][refine_inverse].to(refine_normal.dtype)
        refine_features[refine_in_grid] = refine_outputs["surface_features"][refine_inverse].to(refine_features.dtype)
        refine_validity[refine_in_grid] = (
            refine_outputs["query_validity"][refine_inverse, 0] > 0.5
        )
        refine_unique_support = coordinate_membership(
            refine_unique, occupied_inputs, stride=occupied_support_stride,
        )
        refine_support = torch.zeros(
            refine_count, dtype=torch.bool, device=device,
        )
        refine_support[refine_in_grid] = refine_unique_support[refine_inverse]
        refine_occupancy = refine_occupancy.reshape(-1, refine_sample_count)
        refine_confidence = refine_confidence.reshape(-1, refine_sample_count)
        refine_normal = refine_normal.reshape(-1, refine_sample_count, 3)
        refine_features = refine_features.reshape(-1, refine_sample_count, feature_dim)
        refine_validity = refine_validity.reshape(-1, refine_sample_count)
        refine_support = refine_support.reshape(-1, refine_sample_count)
        refine_first, refine_hit = first_surface_indices(
            refine_occupancy,
            refine_validity & refine_support,
            occupancy_threshold,
        )
        refine_rows = torch.arange(len(hit_rows), device=device)
        accepted_rows = hit_rows[refine_hit]
        accepted_first = refine_first[refine_hit]
        accepted_local_rows = refine_rows[refine_hit]
        hit_depth[accepted_rows] = refine_depths[accepted_local_rows, accepted_first]
        hit_occupancy[accepted_rows] = refine_occupancy[accepted_local_rows, accepted_first]
        hit_confidence[accepted_rows] = refine_confidence[accepted_local_rows, accepted_first]
        hit_normal[accepted_rows] = refine_normal[accepted_local_rows, accepted_first]
        hit_features[accepted_rows] = refine_features[accepted_local_rows, accepted_first]
    if target_depth_prior is not None:
        prior = target_depth_prior.to(device=device, dtype=hit_depth.dtype).squeeze()
        if prior.shape != (height, width):
            raise ValueError("target_depth_prior must match render height and width")
        prior = prior.reshape(-1)
        camera_rotation = input_features.new_tensor(
            target.calibration.external[:3, :3]
        )
        camera_ray_z = (rays @ camera_rotation.T)[:, 2]
        prior_valid = (
            torch.isfinite(prior)
            & (prior > 1e-4)
            & (camera_ray_z > 1e-4)
        )
        prior_range = prior / camera_ray_z.clamp(min=1e-4)
        prior_valid &= (prior_range >= near_m) & (prior_range <= far_m)
        prior_rows = prior_valid.nonzero(as_tuple=False)[:, 0]
        if len(prior_rows):
            prior_points = (
                origin[None] + rays[prior_rows] * prior_range[prior_rows, None]
            )
            prior_indices, prior_in_grid = _points_to_query_indices(prior_points, grid)
            if not prior_in_grid.any():
                prior_indices = prior_indices[:0]
            else:
                prior_unique, prior_inverse = torch.unique(
                    prior_indices, dim=0, return_inverse=True,
                )
                prior_outputs = query_unique(prior_unique)
                prior_support = coordinate_membership(
                    prior_unique, occupied_inputs, stride=occupied_support_stride,
                )[prior_inverse]
                accepted = prior_in_grid.clone()
                accepted[prior_in_grid] &= prior_support
                accepted_rows = prior_rows[accepted]
                accepted_inverse = prior_inverse[accepted[prior_in_grid]]
                hit[accepted_rows] = True
                hit_depth[accepted_rows] = prior_range[accepted_rows]
                hit_occupancy[accepted_rows] = 1.0
                hit_confidence[accepted_rows] = prior_outputs["confidence"][accepted_inverse, 0].to(hit_confidence.dtype)
                hit_normal[accepted_rows] = prior_outputs["normal"][accepted_inverse].to(hit_normal.dtype)
                hit_features[accepted_rows] = prior_outputs["surface_features"][accepted_inverse].to(hit_features.dtype)
    hit_depth = torch.where(hit, hit_depth, torch.zeros_like(hit_depth))
    hit_occupancy = torch.where(hit, hit_occupancy, torch.zeros_like(hit_occupancy))
    hit_confidence = torch.where(hit, hit_confidence, torch.zeros_like(hit_confidence))
    hit_normal = torch.where(hit[:, None], hit_normal, torch.zeros_like(hit_normal))
    hit_features = torch.where(hit[:, None], hit_features, torch.zeros_like(hit_features))
    surface_points = origin[None] + rays * hit_depth[:, None]
    surface_points = torch.where(
        hit[:, None], surface_points, torch.zeros_like(surface_points),
    )
    coverage = validity.float().mean(dim=1)

    return SparseRaymarchResult(
        depth=hit_depth.reshape(height, width),
        surface_points=surface_points.reshape(height, width, 3),
        occupancy=hit_occupancy.reshape(height, width),
        confidence=hit_confidence.reshape(height, width),
        normal=hit_normal.reshape(height, width, 3),
        surface_features=hit_features.reshape(height, width, feature_dim),
        structure_validity=hit.reshape(height, width),
        query_coverage=coverage.reshape(height, width),
        occupied_support=support.float().mean(dim=1).reshape(height, width),
        unknown_structure_mask=(~hit).reshape(height, width),
    )
