"""Brown-aware multi-view RGB feature fusion at persistent 3D voxels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VoxelAppearance:
    features: torch.Tensor
    rgb: torch.Tensor
    validity: torch.Tensor
    confidence: torch.Tensor
    view_weights: torch.Tensor
    projected_uv: torch.Tensor
    projected_depth: torch.Tensor
    projection_validity: torch.Tensor
    depth_consistency: torch.Tensor


def brown_distort(normalized_xy: torch.Tensor, distortion: torch.Tensor) -> torch.Tensor:
    """Apply Brown-Conrady distortion to normalized camera coordinates."""
    if normalized_xy.shape[-1] != 2:
        raise ValueError("normalized_xy must end in two coordinates")
    if distortion.shape[-1] < 4:
        raise ValueError("distortion must contain at least k1,k2,p1,p2")
    coefficients = F.pad(distortion, (0, max(0, 5 - distortion.shape[-1])))
    k1, k2, p1, p2, k3 = coefficients[..., :5].unbind(dim=-1)
    x, y = normalized_xy.unbind(dim=-1)
    r2 = x.square() + y.square()
    radial = 1.0 + k1 * r2 + k2 * r2.square() + k3 * r2.pow(3)
    xy = x * y
    distorted_x = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x.square())
    distorted_y = y * radial + p1 * (r2 + 2.0 * y.square()) + 2.0 * p2 * xy
    return torch.stack([distorted_x, distorted_y], dim=-1)


def project_voxels(
    voxel_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    distortion: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project [B,N,3] voxels into V Brown-distorted source images."""
    if voxel_xyz.ndim != 3 or intrinsics.ndim != 4 or world_to_camera.ndim != 4:
        raise ValueError("expected voxel_xyz [B,N,3], intrinsics [B,V,3,3], extrinsics [B,V,4,4]")
    batch, points, _ = voxel_xyz.shape
    views = intrinsics.shape[1]
    homogeneous = torch.cat([voxel_xyz, voxel_xyz.new_ones(batch, points, 1)], dim=-1)
    camera_xyz = torch.einsum("bvij,bnj->bvni", world_to_camera, homogeneous)[..., :3]
    depth = camera_xyz[..., 2]
    safe_depth = torch.where(
        depth.abs() > 1e-6,
        depth,
        torch.ones_like(depth),
    )
    normalized = camera_xyz[..., :2] / safe_depth[..., None]
    distorted = brown_distort(normalized, distortion[:, :, None, :])
    ones = torch.ones_like(distorted[..., :1])
    pixels_h = torch.einsum(
        "bvij,bvnj->bvni", intrinsics, torch.cat([distorted, ones], dim=-1),
    )
    uv = pixels_h[..., :2]
    height, width = image_size
    valid = (
        torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 1e-4)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= width - 1)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= height - 1)
    )
    return uv, depth, valid


def sample_view_maps(maps: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B,V,C,H,W] maps at [B,V,N,2] pixel locations."""
    batch, views, channels, height, width = maps.shape
    points = uv.shape[2]
    grid = torch.stack([
        uv[..., 0] / max(width - 1, 1) * 2.0 - 1.0,
        uv[..., 1] / max(height - 1, 1) * 2.0 - 1.0,
    ], dim=-1)
    grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    grid = grid.reshape(batch * views, points, 1, 2)
    sampled = F.grid_sample(
        maps.reshape(batch * views, channels, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(batch, views, channels, points).permute(0, 1, 3, 2)


class GatedCrossViewVoxelAttention(nn.Module):
    """Fuse views with learned attention multiplied by geometric validity priors."""

    def __init__(
        self,
        geometry_dim: int,
        appearance_dim: int,
        hidden_dim: int = 96,
        heads: int = 4,
        focal_role_dim: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(geometry_dim, hidden_dim)
        self.key = nn.Linear(appearance_dim + 5 + focal_role_dim, hidden_dim)
        self.value = nn.Linear(appearance_dim + 3, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + geometry_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, appearance_dim),
        )
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim + geometry_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        geometry: torch.Tensor,
        appearance: torch.Tensor,
        rgb: torch.Tensor,
        view_direction: torch.Tensor,
        relative_depth_error: torch.Tensor,
        depth_observed: torch.Tensor,
        focal_role: torch.Tensor,
        prior: torch.Tensor,
        validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, views, points, _ = appearance.shape
        geometry_query = self.query(geometry).view(batch, points, self.heads, self.head_dim)
        role = focal_role[:, :, None].expand(-1, -1, points, -1)
        key_input = torch.cat([
            appearance, view_direction, relative_depth_error[..., None], depth_observed[..., None], role,
        ], dim=-1)
        value_input = torch.cat([appearance, rgb], dim=-1)
        keys = self.key(key_input).view(batch, views, points, self.heads, self.head_dim)
        values = self.value(value_input).view(batch, views, points, self.heads, self.head_dim)
        logits = (geometry_query[:, None] * keys).sum(dim=-1) / self.head_dim**0.5
        logits = logits + prior.clamp(min=1e-6).log()[..., None]
        valid = validity[..., None]
        logits = logits.masked_fill(~valid, -torch.inf)
        any_valid = validity.any(dim=1)
        logits = torch.where(any_valid[:, None, :, None], logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attended = (values * weights[..., None]).sum(dim=1).reshape(batch, points, -1)
        fused_input = torch.cat([attended, geometry], dim=-1)
        features = self.output(fused_input)
        confidence = self.confidence(fused_input)
        mean_weights = weights.mean(dim=-1).permute(0, 2, 1)
        fused_rgb = (rgb.permute(0, 2, 1, 3) * mean_weights[..., None]).sum(dim=2)
        features = torch.where(any_valid[..., None], features, torch.zeros_like(features))
        fused_rgb = torch.where(any_valid[..., None], fused_rgb, torch.zeros_like(fused_rgb))
        confidence = confidence * any_valid[..., None]
        return features, fused_rgb, confidence, mean_weights


class MultiViewVoxelFusion(nn.Module):
    """Lift source RGB features into candidate voxels without changing geometry."""

    def __init__(
        self,
        geometry_dim: int,
        appearance_dim: int = 64,
        hidden_dim: int = 96,
        heads: int = 4,
        unknown_depth_prior: float = 0.35,
        rgb_weight_sharpness: float = 1.0,
    ) -> None:
        super().__init__()
        self.unknown_depth_prior = float(unknown_depth_prior)
        self.rgb_weight_sharpness = float(rgb_weight_sharpness)
        if self.rgb_weight_sharpness < 1.0:
            raise ValueError("rgb_weight_sharpness must be at least one")
        self.attention = GatedCrossViewVoxelAttention(
            geometry_dim=geometry_dim,
            appearance_dim=appearance_dim,
            hidden_dim=hidden_dim,
            heads=heads,
        )

    def forward(
        self,
        geometry_features: torch.Tensor,
        voxel_xyz: torch.Tensor,
        source_features: torch.Tensor,
        source_rgb: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_extrinsics: torch.Tensor,
        source_distortions: torch.Tensor,
        source_rgb_intrinsics: torch.Tensor | None = None,
        source_rgb_distortions: torch.Tensor | None = None,
        source_lidar_depth: torch.Tensor | None = None,
        source_validity: torch.Tensor | None = None,
        source_time_offsets: torch.Tensor | None = None,
        source_alignment_confidence: torch.Tensor | None = None,
        focal_role: torch.Tensor | None = None,
        source_camera_indices: torch.Tensor | None = None,
        target_camera_index: torch.Tensor | int | None = None,
    ) -> VoxelAppearance:
        height, width = source_features.shape[-2:]
        uv, depth, projection_valid = project_voxels(
            voxel_xyz, source_intrinsics, source_extrinsics, source_distortions, (height, width),
        )
        sampled_features = sample_view_maps(source_features, uv)
        if source_rgb_intrinsics is None:
            rgb_uv = uv
        else:
            rgb_uv, _, _ = project_voxels(
                voxel_xyz,
                source_rgb_intrinsics,
                source_extrinsics,
                (
                    source_distortions
                    if source_rgb_distortions is None
                    else source_rgb_distortions
                ),
                source_rgb.shape[-2:],
            )
        sampled_rgb = sample_view_maps(source_rgb, rgb_uv)
        if source_lidar_depth is None:
            observed_depth = depth.new_zeros(depth.shape)
        else:
            observed_depth = sample_view_maps(source_lidar_depth, uv)[..., 0]
        has_depth = observed_depth > 1e-4
        tolerance = torch.maximum(observed_depth * 0.02, observed_depth.new_tensor(0.15))
        relative_error = torch.where(
            has_depth,
            (depth - observed_depth).abs() / tolerance.clamp(min=1e-4),
            torch.zeros_like(depth),
        )
        consistency = torch.where(
            has_depth,
            torch.exp(-relative_error),
            torch.full_like(depth, self.unknown_depth_prior),
        )
        if source_time_offsets is not None:
            if source_time_offsets.shape != depth.shape[:2]:
                raise ValueError("source_time_offsets must have shape [B,V]")
            temporal_prior = torch.exp(
                -source_time_offsets.abs() / 0.75
            )[:, :, None]
            consistency = consistency * temporal_prior
        if source_alignment_confidence is not None:
            if source_alignment_confidence.shape != depth.shape[:2]:
                raise ValueError("source_alignment_confidence must have shape [B,V]")
            consistency = consistency * source_alignment_confidence.clamp(
                min=0.05, max=1.0,
            )[:, :, None]
        if source_camera_indices is not None and target_camera_index is not None:
            source_indices = source_camera_indices.to(
                device=depth.device, dtype=torch.long,
            )
            if source_indices.shape != depth.shape[:2]:
                raise ValueError("source_camera_indices must have shape [B,V]")
            target_index = torch.as_tensor(
                target_camera_index, device=depth.device, dtype=torch.long,
            ).reshape(-1)
            if target_index.numel() == 1 and depth.shape[0] > 1:
                target_index = target_index.expand(depth.shape[0])
            if target_index.numel() != depth.shape[0]:
                raise ValueError("target_camera_index must match batch size")
            same_camera = source_indices == target_index[:, None]
            if source_time_offsets is None:
                current_frame = torch.ones_like(same_camera)
            else:
                current_frame = source_time_offsets.abs() < 1e-6
            # For a shifted target, the current image from the same physical
            # camera is the immutable appearance anchor. A temporal image from
            # that camera may fill only unsupported regions; it must not win a
            # softmax competition over the current source.
            identity_prior = torch.where(
                same_camera & current_frame,
                consistency.new_tensor(1_000_000.0),
                torch.where(
                    same_camera,
                    consistency.new_tensor(100.0),
                    consistency.new_tensor(1.0),
                ),
            )
            consistency = consistency * identity_prior[:, :, None]
        valid = projection_valid
        if source_validity is not None:
            valid = valid & source_validity[:, :, None].bool()
        centers = -torch.einsum(
            "bvji,bvj->bvi",
            source_extrinsics[..., :3, :3],
            source_extrinsics[..., :3, 3],
        )
        directions = F.normalize(voxel_xyz[:, None] - centers[:, :, None], dim=-1, eps=1e-6)
        if focal_role is None:
            focal_role = geometry_features.new_zeros(
                geometry_features.shape[0], source_features.shape[1], 2,
            )
        features, rgb, confidence, weights = self.attention(
            geometry_features,
            sampled_features,
            sampled_rgb,
            directions,
            relative_error.clamp(max=10.0),
            has_depth.to(geometry_features.dtype),
            focal_role,
            consistency,
            valid,
        )
        if self.rgb_weight_sharpness != 1.0:
            rgb_weights = weights.clamp(min=0.0).pow(self.rgb_weight_sharpness)
            rgb_weights = rgb_weights / rgb_weights.sum(
                dim=-1, keepdim=True,
            ).clamp(min=1e-6)
            rgb = (
                sampled_rgb.permute(0, 2, 1, 3)
                * rgb_weights[..., None]
            ).sum(dim=2)
            rgb = torch.where(
                valid.any(dim=1)[..., None], rgb, torch.zeros_like(rgb),
            )
        return VoxelAppearance(
            features=features,
            rgb=rgb,
            validity=valid.any(dim=1),
            confidence=confidence,
            view_weights=weights,
            projected_uv=uv,
            projected_depth=depth,
            projection_validity=projection_valid,
            depth_consistency=consistency,
        )
