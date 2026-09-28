"""RGB rendering from LiDAR geometry and source-observed 3D appearance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gcr_nvs.datasets.sparse_geometry_dataset import SparseGridSpec
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.rendering.environment import EnvironmentRenderResult
from gcr_nvs.rendering.sparse_geometry_raymarch import (
    SparseRaymarchResult,
    raymarch_sparse_geometry,
)


@dataclass(frozen=True)
class UnifiedFieldRenderResult:
    rgb: torch.Tensor
    surface_rgb: torch.Tensor
    generated_rgb: torch.Tensor
    generation_alpha: torch.Tensor
    generation_logvar: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    structure_confidence: torch.Tensor
    appearance_confidence: torch.Tensor
    uncertainty: torch.Tensor
    structure_validity: torch.Tensor
    appearance_validity: torch.Tensor
    surface_validity: torch.Tensor
    environment_validity: torch.Tensor
    fallback_validity: torch.Tensor
    output_validity: torch.Tensor
    unknown_structure: torch.Tensor
    true_disocclusion: torch.Tensor
    source_provenance: torch.Tensor
    source_weight: torch.Tensor
    geometry: SparseRaymarchResult


def _environment_tensors(
    environment: EnvironmentRenderResult | None,
    height: int,
    width: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if environment is None:
        return (
            reference.new_zeros(height, width, 3),
            torch.zeros(height, width, dtype=torch.bool, device=reference.device),
            torch.full(
                (height, width), -1, dtype=torch.long, device=reference.device,
            ),
        )
    if environment.rgb.shape != (height, width, 3):
        raise ValueError("environment resolution must match the target render")
    rgb = torch.as_tensor(
        np.asarray(environment.rgb), dtype=reference.dtype, device=reference.device,
    )
    valid = torch.as_tensor(
        np.asarray(environment.validity), dtype=torch.bool, device=reference.device,
    )
    provenance = torch.as_tensor(
        np.asarray(environment.source_provenance),
        dtype=torch.long,
        device=reference.device,
    )
    return rgb, valid, provenance


def raymarch_unified_field(
    model,
    input_features: torch.Tensor,
    input_indices: torch.Tensor,
    spatial_shape: tuple[int, int, int] | list[int],
    target: TargetCamera,
    grid: SparseGridSpec,
    source_images: torch.Tensor,
    source_intrinsics: torch.Tensor,
    source_extrinsics: torch.Tensor,
    source_distortions: torch.Tensor,
    width: int,
    height: int,
    source_lidar_depth: torch.Tensor | None = None,
    source_validity: torch.Tensor | None = None,
    source_time_offsets: torch.Tensor | None = None,
    source_alignment_confidence: torch.Tensor | None = None,
    focal_role: torch.Tensor | None = None,
    target_camera_index: torch.Tensor | int | None = None,
    target_focal_role: torch.Tensor | None = None,
    environment: EnvironmentRenderResult | None = None,
    fallback_environment: EnvironmentRenderResult | None = None,
    near_m: float = 1.0,
    far_m: float = 80.0,
    sample_count: int = 96,
    occupancy_threshold: float = 0.5,
    occupied_support_stride: int = 4,
    appearance_chunk_size: int = 65_536,
    geometry_query_chunk_size: int | None = 262_144,
    geometry_refine_sample_count: int = 0,
    target_depth_prior: torch.Tensor | None = None,
    source_rgb_images: torch.Tensor | None = None,
    source_rgb_intrinsics: torch.Tensor | None = None,
    source_rgb_distortions: torch.Tensor | None = None,
    source_camera_indices: torch.Tensor | None = None,
    source_features: torch.Tensor | None = None,
    pixel_origin: tuple[int, int] = (0, 0),
    encoded_geometry: dict[str, torch.Tensor] | None = None,
) -> UnifiedFieldRenderResult:
    """Render a target view while keeping geometry and appearance contracts separate.

    A missing surface appearance is left invalid.  The environment may fill only
    rays with no finite LiDAR-supported surface, so sky RGB can never overwrite
    an opaque but unobserved foreground surface.
    """
    if source_images.ndim != 5 or source_images.shape[0] != 1:
        raise ValueError("source_images must be [1,V,3,H,W]")
    if appearance_chunk_size <= 0:
        raise ValueError("appearance_chunk_size must be positive")
    geometry = raymarch_sparse_geometry(
        model.geometry,
        input_features,
        input_indices,
        spatial_shape,
        target,
        grid,
        width=width,
        height=height,
        near_m=near_m,
        far_m=far_m,
        sample_count=sample_count,
        occupancy_threshold=occupancy_threshold,
        occupied_support_stride=occupied_support_stride,
        query_chunk_size=geometry_query_chunk_size,
        refine_sample_count=geometry_refine_sample_count,
        target_depth_prior=target_depth_prior,
        pixel_origin=pixel_origin,
        encoded_geometry=encoded_geometry,
    )
    if source_features is None:
        source_features = model.encode_sources(source_images)
    flat_structure = geometry.structure_validity.reshape(-1)
    hit_indices = flat_structure.nonzero(as_tuple=False)[:, 0]
    pixel_count = height * width
    surface_rgb = input_features.new_zeros(pixel_count, 3)
    appearance_confidence = input_features.new_zeros(pixel_count)
    appearance_validity = torch.zeros(
        pixel_count, dtype=torch.bool, device=input_features.device,
    )
    provenance = torch.full(
        (pixel_count,), -1, dtype=torch.long, device=input_features.device,
    )
    source_weight = input_features.new_zeros(pixel_count)
    flat_points = geometry.surface_points.reshape(-1, 3)
    flat_geometry_features = geometry.surface_features.reshape(
        pixel_count, -1,
    )
    appearance_features = input_features.new_zeros(
        pixel_count, source_features.shape[2] if source_features.ndim == 5 else 64,
    )
    geometry_features = input_features.new_zeros(
        pixel_count, flat_geometry_features.shape[-1],
    )
    camera_center = input_features.new_tensor(target.calibration.camera_center)
    for start in range(0, len(hit_indices), appearance_chunk_size):
        selected = hit_indices[start:start + appearance_chunk_size]
        points = flat_points[selected][None]
        features = flat_geometry_features[selected][None]
        view_direction = F.normalize(
            camera_center[None, None] - points, dim=-1, eps=1e-6,
        )
        appearance, field_rgb = model.fuse_at_points(
            features,
            points,
            source_features,
            source_images,
            source_intrinsics,
            source_extrinsics,
            source_distortions,
            source_rgb_images=source_rgb_images,
            source_rgb_intrinsics=source_rgb_intrinsics,
            source_rgb_distortions=source_rgb_distortions,
            source_lidar_depth=source_lidar_depth,
            source_validity=source_validity,
            source_time_offsets=source_time_offsets,
            source_alignment_confidence=source_alignment_confidence,
            focal_role=focal_role,
            source_camera_indices=source_camera_indices,
            target_view_direction=view_direction,
            target_camera_index=target_camera_index,
            target_focal_role=target_focal_role,
        )
        valid = appearance.validity[0]
        weights = appearance.view_weights[0]
        best_weight, best_source = weights.max(dim=-1)
        surface_rgb[selected] = field_rgb[0].to(surface_rgb.dtype)
        appearance_features[selected] = appearance.features[0].to(
            appearance_features.dtype,
        )
        geometry_features[selected] = features[0]
        appearance_confidence[selected] = appearance.confidence[0, :, 0].to(
            appearance_confidence.dtype,
        )
        appearance_validity[selected] = valid
        provenance[selected] = torch.where(
            valid, best_source, torch.full_like(best_source, -1),
        )
        source_weight[selected] = torch.where(
            valid, best_weight, torch.zeros_like(best_weight),
        ).to(source_weight.dtype)

    structure_validity = geometry.structure_validity
    appearance_validity = appearance_validity.reshape(height, width)
    surface_validity = structure_validity & appearance_validity
    surface_rgb = surface_rgb.reshape(height, width, 3)
    appearance_features = appearance_features.reshape(
        height, width, appearance_features.shape[-1],
    )
    geometry_features = geometry_features.reshape(
        height, width, geometry_features.shape[-1],
    )
    environment_rgb, raw_environment_validity, environment_provenance = (
        _environment_tensors(environment, height, width, input_features)
    )
    fallback_rgb, raw_fallback_validity, fallback_provenance = _environment_tensors(
        fallback_environment, height, width, input_features,
    )
    environment_validity = raw_environment_validity & ~structure_validity
    fallback_validity = (
        raw_fallback_validity & ~surface_validity & ~environment_validity
    )
    # Geometry-supported pixels use RGB sampled through their 3D surface
    # points. Unknown rays use the sharp directional fallback. Same-camera
    # temporal sources receive a strong prior inside MultiViewVoxelFusion, so
    # surface RGB stays coherent without ignoring translational parallax.
    generator_source_rgb = torch.where(
        surface_validity[..., None], surface_rgb, fallback_rgb,
    )
    generator_source_validity = surface_validity | raw_fallback_validity
    generated_rgb = surface_rgb
    generation_alpha = input_features.new_zeros(height, width, 1)
    generation_logvar = input_features.new_zeros(height, width, 1)
    if hasattr(model, "generate_target_view"):
        ray_map = torch.from_numpy(target.ray_map(width, height)).to(
            device=input_features.device, dtype=input_features.dtype,
        )
        generator_output = model.generate_target_view(
            geometry_features=geometry_features.permute(2, 0, 1)[None],
            appearance_features=appearance_features.permute(2, 0, 1)[None],
            source_rgb=generator_source_rgb.permute(2, 0, 1)[None],
            depth=geometry.depth[None, None],
            normal=geometry.normal.permute(2, 0, 1)[None],
            structure_confidence=geometry.confidence[None, None],
            appearance_confidence=appearance_confidence.reshape(1, 1, height, width),
            ray_map=ray_map[None],
            source_validity=generator_source_validity[None, None].to(
                input_features.dtype,
            ),
            source_weight=source_weight.reshape(1, 1, height, width),
            target_camera_index=target_camera_index,
            target_focal_role=target_focal_role,
        )
        generated_rgb = generator_output["rgb"][0].permute(1, 2, 0)
        generation_alpha = generator_output["alpha"][0].permute(1, 2, 0)
        generation_logvar = generator_output["logvar"][0].permute(1, 2, 0)
    generator_active = hasattr(model, "generate_target_view")
    generated_route = generated_rgb if generator_active else surface_rgb
    generator_validity = structure_validity | fallback_validity if generator_active else surface_validity
    rgb = torch.where(
        generator_validity[..., None], generated_route, environment_rgb,
    )
    if not generator_active:
        rgb = torch.where(fallback_validity[..., None], fallback_rgb, rgb)
    output_validity = (
        generator_validity
        | environment_validity | fallback_validity
    )
    rgb = torch.where(output_validity[..., None], rgb, torch.zeros_like(rgb))
    provenance = provenance.reshape(height, width)
    provenance = torch.where(
        environment_validity, environment_provenance, provenance,
    )
    provenance = torch.where(
        fallback_validity, fallback_provenance, provenance,
    )
    structure_confidence = geometry.confidence
    appearance_confidence = appearance_confidence.reshape(height, width)
    combined_confidence = structure_confidence * appearance_confidence
    uncertainty = torch.where(
        surface_validity,
        1.0 - combined_confidence.clamp(0.0, 1.0),
        torch.where(environment_validity, torch.full_like(combined_confidence, 0.5),
                    torch.where(fallback_validity, torch.ones_like(combined_confidence),
                                torch.ones_like(combined_confidence))),
    )
    return UnifiedFieldRenderResult(
        rgb=rgb,
        surface_rgb=surface_rgb,
        generated_rgb=generated_rgb,
        generation_alpha=generation_alpha,
        generation_logvar=generation_logvar,
        depth=geometry.depth,
        normal=geometry.normal,
        structure_confidence=structure_confidence,
        appearance_confidence=appearance_confidence,
        uncertainty=uncertainty,
        structure_validity=structure_validity,
        appearance_validity=appearance_validity,
        surface_validity=surface_validity,
        environment_validity=environment_validity,
        fallback_validity=fallback_validity,
        output_validity=output_validity,
        unknown_structure=~structure_validity,
        true_disocclusion=structure_validity & ~appearance_validity,
        source_provenance=provenance,
        source_weight=source_weight.reshape(height, width),
        geometry=geometry,
    )
