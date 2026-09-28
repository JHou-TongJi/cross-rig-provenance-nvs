"""LiDAR-structure/RGB-appearance novel-view synthesis.

The modality contract is structural: RGB is accepted only by the appearance
encoder, while the geometry encoder is called only with LiDAR voxel tensors.
The RGB reconstruction loss uses detached geometry tokens by default, so it
cannot turn the geometry encoder into an implicit RGB depth estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=dilation, dilation=dilation,
            ),
            nn.GroupNorm(max(1, channels // 8), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(max(1, channels // 8), channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.block(inputs))


class RGBAppearanceEncoder(nn.Module):
    """Encode sampled RGB observations and DINO/NAF features into tokens.

    This branch has no depth, occupancy, SDF, or position output.
    """

    def __init__(
        self,
        source_feature_dim: int = 64,
        geometry_dim: int = 32,
        token_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_encoder = nn.Sequential(
            nn.Linear(source_feature_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, token_dim),
        )
        self.geometry_query = nn.Sequential(
            nn.Linear(geometry_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.view_score = nn.Sequential(
            nn.Linear(token_dim * 2 + 1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        per_source_features: torch.Tensor,
        per_source_rgb: torch.Tensor,
        per_source_validity: torch.Tensor,
        prior_scores: torch.Tensor,
        geometry_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if per_source_features.ndim != 4:
            raise ValueError("per_source_features must be [B,N,V,C]")
        if per_source_rgb.shape[:-1] != per_source_features.shape[:-1]:
            raise ValueError("per_source_rgb must share [B,N,V] dimensions")
        if per_source_rgb.shape[-1] != 3:
            raise ValueError("RGB observations must have three channels")
        valid = per_source_validity.bool()
        observation = self.observation_encoder(torch.cat([
            per_source_features, per_source_rgb,
        ], dim=-1))
        query = self.geometry_query(geometry_tokens)[:, :, None]
        query = query.expand(-1, -1, observation.shape[2], -1)
        logits = self.view_score(torch.cat([
            observation, query, prior_scores[..., None],
        ], dim=-1))[..., 0] + prior_scores
        logits = logits.masked_fill(~valid, -torch.inf)
        any_valid = valid.any(dim=-1)
        safe_logits = torch.where(
            any_valid[..., None], logits, torch.zeros_like(logits),
        )
        weights = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        fused = (observation * weights[..., None]).sum(dim=-2)
        fused = self.output(fused)
        fused = torch.where(any_valid[..., None], fused, torch.zeros_like(fused))
        return fused, weights, any_valid


class GeometryAppearanceSurfaceFusion(nn.Module):
    """Fuse learned LiDAR structure and RGB appearance on fixed 3D surfaces."""

    def __init__(
        self,
        geometry_dim: int = 32,
        appearance_dim: int = 64,
        output_dim: int = 32,
        detach_geometry_for_rgb: bool = True,
    ) -> None:
        super().__init__()
        self.detach_geometry_for_rgb = detach_geometry_for_rgb
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, appearance_dim * 2),
            nn.LayerNorm(appearance_dim * 2),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(appearance_dim * 2, appearance_dim),
            nn.LayerNorm(appearance_dim),
            nn.SiLU(inplace=True),
            nn.Linear(appearance_dim, output_dim),
        )

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        appearance_tokens: torch.Tensor,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        geometry = (
            geometry_tokens.detach()
            if self.detach_geometry_for_rgb
            else geometry_tokens
        )
        scale, bias = self.geometry(geometry).chunk(2, dim=-1)
        conditioned = appearance_tokens * (1.0 + 0.1 * torch.tanh(scale)) + bias
        features = self.fusion(torch.cat([conditioned, appearance_tokens], dim=-1))
        return torch.where(validity[..., None], features, torch.zeros_like(features))


class TargetRGBDecoder(nn.Module):
    """Decode a rendered feature field to RGB; deliberately has no depth head."""

    def __init__(self, feature_dim: int = 32, channels: int = 64) -> None:
        super().__init__()
        # feature + depth + normal + visibility + confidence + alpha + ray
        input_dim = feature_dim + 1 + 3 + 1 + 1 + 1 + 3
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, channels, 3, padding=1),
            nn.GroupNorm(max(1, channels // 8), channels),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            ConvResidualBlock(channels, dilation=1),
            ConvResidualBlock(channels, dilation=2),
            ConvResidualBlock(channels, dilation=4),
            ConvResidualBlock(channels, dilation=1),
        )
        self.rgb_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 3, 1),
        )

    def forward(
        self,
        rendered_features: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        alpha: torch.Tensor,
        ray_map: torch.Tensor,
        appearance_validity: torch.Tensor,
    ) -> torch.Tensor:
        depth_scaled = torch.log1p(depth.clamp(min=0.0)) / torch.log(
            depth.new_tensor(121.0)
        )
        inputs = torch.cat([
            rendered_features,
            depth_scaled,
            normal,
            visibility,
            confidence,
            alpha,
            ray_map,
        ], dim=1)
        rgb = torch.sigmoid(self.rgb_head(self.context(self.stem(inputs))))
        valid = appearance_validity.bool()
        return torch.where(valid, rgb, torch.zeros_like(rgb))


@dataclass(frozen=True)
class SurfaceFieldOutput:
    features: torch.Tensor
    appearance_tokens: torch.Tensor
    view_weights: torch.Tensor
    appearance_validity: torch.Tensor


class GeometryAppearanceNVS(nn.Module):
    """Unified dual-branch model with an explicit modality firewall.

    ``geometry_encoder`` is a LiDAR-only SparseGeometryStudent supplied by the
    caller. It is intentionally not invoked from ``encode_appearance``.
    """

    def __init__(
        self,
        geometry_encoder: nn.Module | None = None,
        source_feature_dim: int = 64,
        geometry_dim: int = 32,
        appearance_dim: int = 64,
        rendered_feature_dim: int = 32,
        detach_geometry_for_rgb: bool = True,
    ) -> None:
        super().__init__()
        self.geometry_encoder = geometry_encoder
        self.appearance_encoder = RGBAppearanceEncoder(
            source_feature_dim=source_feature_dim,
            geometry_dim=geometry_dim,
            token_dim=appearance_dim,
        )
        self.surface_fusion = GeometryAppearanceSurfaceFusion(
            geometry_dim=geometry_dim,
            appearance_dim=appearance_dim,
            output_dim=rendered_feature_dim,
            detach_geometry_for_rgb=detach_geometry_for_rgb,
        )
        # Training-only probe: forces the 3D feature to retain appearance
        # information. Its RGB is never used as the renderer's main output.
        self.surface_rgb_probe = nn.Sequential(
            nn.Linear(rendered_feature_dim, appearance_dim),
            nn.SiLU(inplace=True),
            nn.Linear(appearance_dim, 3),
        )
        self.target_decoder = TargetRGBDecoder(feature_dim=rendered_feature_dim)

    def encode_geometry(
        self,
        lidar_features: torch.Tensor,
        lidar_indices: torch.Tensor,
        spatial_shape: tuple[int, int, int] | list[int],
        batch_size: int,
        query_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.geometry_encoder is None:
            raise RuntimeError("a LiDAR geometry encoder was not supplied")
        return self.geometry_encoder(
            lidar_features,
            lidar_indices,
            spatial_shape,
            batch_size,
            query_indices=query_indices,
        )

    def build_surface_field(
        self,
        geometry_tokens: torch.Tensor,
        per_source_features: torch.Tensor,
        per_source_rgb: torch.Tensor,
        per_source_validity: torch.Tensor,
        prior_scores: torch.Tensor,
    ) -> SurfaceFieldOutput:
        geometry_for_query = geometry_tokens.detach()
        appearance, weights, validity = self.appearance_encoder(
            per_source_features,
            per_source_rgb,
            per_source_validity,
            prior_scores,
            geometry_for_query,
        )
        features = self.surface_fusion(geometry_tokens, appearance, validity)
        return SurfaceFieldOutput(features, appearance, weights, validity)
