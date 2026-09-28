"""Attach source-view RGB and DINO/NAF features to fixed LiDAR surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from gcr_nvs.models.source_encoder import SourceImageEncoder


@dataclass(frozen=True)
class SurfaceAppearance:
    features: torch.Tensor
    rgb: torch.Tensor
    appearance_validity: torch.Tensor
    source_count: torch.Tensor
    source_provenance: torch.Tensor
    view_weights: torch.Tensor
    sampled_features: torch.Tensor
    sampled_rgb: torch.Tensor
    observation_validity: torch.Tensor
    per_source_features: torch.Tensor
    per_source_rgb: torch.Tensor
    per_source_validity: torch.Tensor
    per_source_scores: torch.Tensor


class LiDARSurfaceAppearanceLift(nn.Module):
    """Fuse appearance only; this module never predicts or moves 3D positions."""

    def __init__(
        self,
        feature_dim: int = 64,
        top_k: int = 3,
        use_dino: bool = True,
        dino_pretrained: bool = True,
        hard_rgb_selection: bool = False,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.hard_rgb_selection = hard_rgb_selection
        self.encoder = SourceImageEncoder(
            feature_dim=feature_dim,
            use_dino=use_dino,
            dino_pretrained=dino_pretrained,
        )

    @staticmethod
    def _sample_maps(maps: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = maps.shape
        anchors = uv.shape[2]
        flattened = maps.reshape(batch * views, channels, height, width)
        grid = uv.reshape(batch * views, anchors, 1, 2)
        normalized = torch.stack([
            grid[..., 0] / max(width - 1, 1) * 2.0 - 1.0,
            grid[..., 1] / max(height - 1, 1) * 2.0 - 1.0,
        ], dim=-1)
        sampled = F.grid_sample(
            flattened,
            normalized,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(batch, views, channels, anchors).permute(0, 1, 3, 2)

    def forward(
        self,
        source_images: torch.Tensor,
        surface_xyz: torch.Tensor,
        source_uv: torch.Tensor,
        source_valid: torch.Tensor,
        source_camera_centers: torch.Tensor,
        target_camera_center: torch.Tensor | None = None,
        source_optical_axes: torch.Tensor | None = None,
        target_optical_axis: torch.Tensor | None = None,
        rgb_source_images: torch.Tensor | None = None,
        rgb_source_uv: torch.Tensor | None = None,
    ) -> SurfaceAppearance:
        if source_images.ndim != 5 or surface_xyz.ndim != 3:
            raise ValueError("source_images must be [B,V,3,H,W] and surface_xyz [B,N,3]")
        source_features = self.encoder(source_images)
        sampled_features = self._sample_maps(source_features, source_uv)
        sampled_rgb = self._sample_maps(
            rgb_source_images if rgb_source_images is not None else source_images,
            rgb_source_uv if rgb_source_uv is not None else source_uv,
        )
        valid = source_valid.bool()
        directions = surface_xyz[:, None] - source_camera_centers[:, :, None]
        distances = directions.norm(dim=-1).clamp(min=1e-3)
        score = -distances / 50.0
        if target_camera_center is not None:
            source_rays = F.normalize(directions, dim=-1)
            target_rays = F.normalize(
                surface_xyz - target_camera_center[:, None], dim=-1,
            )
            ray_similarity = (source_rays * target_rays[:, None]).sum(dim=-1)
            score = score + 4.0 * ray_similarity
        if source_optical_axes is not None and target_optical_axis is not None:
            source_axes = F.normalize(source_optical_axes, dim=-1)
            target_axis = F.normalize(target_optical_axis, dim=-1)
            axis_similarity = (
                source_axes * target_axis[:, None]
            ).sum(dim=-1)
            score = score + 6.0 * axis_similarity[:, :, None]
        score = score.masked_fill(~valid, -torch.inf)
        count = min(self.top_k, source_images.shape[1])
        top_score, top_index = score.topk(count, dim=1)
        gather_index = top_index.permute(0, 2, 1)
        selected_valid = valid.permute(0, 2, 1).gather(2, gather_index)

        feature_index = gather_index[..., None].expand(
            -1, -1, -1, sampled_features.shape[-1],
        )
        rgb_index = gather_index[..., None].expand(-1, -1, -1, 3)
        selected_features = sampled_features.permute(0, 2, 1, 3).gather(2, feature_index)
        selected_rgb = sampled_rgb.permute(0, 2, 1, 3).gather(2, rgb_index)
        safe_score = torch.where(selected_valid, top_score.permute(0, 2, 1), -torch.inf)
        any_valid = selected_valid.any(dim=2)
        safe_score = torch.where(
            any_valid[..., None], safe_score, torch.zeros_like(safe_score),
        )
        weights = torch.softmax(safe_score, dim=2) * selected_valid.to(safe_score.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        fused_features = (selected_features * weights[..., None]).sum(dim=2)
        if self.hard_rgb_selection:
            fused_rgb = selected_rgb[:, :, 0]
        else:
            fused_rgb = (selected_rgb * weights[..., None]).sum(dim=2)
        fused_features = torch.where(
            any_valid[..., None], fused_features, torch.zeros_like(fused_features),
        )
        fused_rgb = torch.where(any_valid[..., None], fused_rgb, torch.zeros_like(fused_rgb))
        provenance = gather_index[:, :, 0].to(torch.int64)
        provenance = torch.where(any_valid, provenance, provenance.new_full((), -1))
        return SurfaceAppearance(
            features=fused_features,
            rgb=fused_rgb,
            appearance_validity=any_valid,
            source_count=valid.sum(dim=1),
            source_provenance=provenance,
            view_weights=weights,
            sampled_features=selected_features,
            sampled_rgb=selected_rgb,
            observation_validity=selected_valid,
            per_source_features=sampled_features.permute(0, 2, 1, 3),
            per_source_rgb=sampled_rgb.permute(0, 2, 1, 3),
            per_source_validity=valid.permute(0, 2, 1),
            per_source_scores=torch.where(
                valid, score, torch.zeros_like(score),
            ).permute(0, 2, 1),
        )


class LearnedViewWeighter(nn.Module):
    """Learn view weights while keeping source RGB and geometry immutable."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 96) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        per_source_features: torch.Tensor,
        per_source_rgb: torch.Tensor,
        per_source_validity: torch.Tensor,
        prior_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([
            per_source_features,
            per_source_rgb,
            prior_scores[..., None],
        ], dim=-1)
        logits = self.scorer(inputs)[..., 0] + prior_scores
        valid = per_source_validity.bool()
        logits = logits.masked_fill(~valid, -torch.inf)
        any_valid = valid.any(dim=-1)
        safe_logits = torch.where(
            any_valid[..., None], logits, torch.zeros_like(logits),
        )
        weights = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        rgb = (per_source_rgb * weights[..., None]).sum(dim=-2)
        rgb = torch.where(any_valid[..., None], rgb, torch.zeros_like(rgb))
        return rgb, weights
