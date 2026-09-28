"""LiDAR-anchored source feature fusion and target feature rendering."""

from __future__ import annotations

import torch
from torch import nn

from gcr_nvs.models.fusion import AnchorObservationBuilder, ViewFusionNetwork
from gcr_nvs.models.source_encoder import SourceImageEncoder


class AnchorFeatureRenderer(nn.Module):
    """Scatter fused anchor features into the target camera raster."""

    def __init__(self, anchor_dim: int = 64, rendered_dim: int = 32, render_scale: int = 2):
        super().__init__()
        if render_scale < 1:
            raise ValueError("render_scale must be positive")
        self.render_scale = render_scale
        self.project = nn.Sequential(
            nn.Linear(anchor_dim, rendered_dim),
            nn.SiLU(inplace=True),
            nn.Linear(rendered_dim, rendered_dim),
        )

    def forward(
        self,
        anchor_features: torch.Tensor,
        target_uv: torch.Tensor,
        target_valid: torch.Tensor,
        target_size: tuple[int, int],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, anchors, _ = anchor_features.shape
        height, width = target_size
        projected = self.project(anchor_features)
        render_height = max(1, (height + self.render_scale - 1) // self.render_scale)
        render_width = max(1, (width + self.render_scale - 1) // self.render_scale)
        render_uv = target_uv / float(self.render_scale)
        if weights is None:
            weights = target_valid.float()
        weights = weights.to(projected.dtype) * target_valid.to(projected.dtype)
        output = projected.new_zeros(batch, render_height * render_width, projected.shape[-1])
        normalizer = projected.new_zeros(batch, render_height * render_width, 1)
        xy = render_uv.round().long()
        inside = (
            target_valid
            & (xy[..., 0] >= 0)
            & (xy[..., 0] < render_width)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < render_height)
        )
        flat_index = (xy[..., 1].clamp(0, render_height - 1) * render_width + xy[..., 0].clamp(0, render_width - 1)).unsqueeze(-1)
        for batch_index in range(batch):
            valid = inside[batch_index]
            index = flat_index[batch_index, valid].expand(-1, projected.shape[-1])
            output[batch_index].scatter_add_(0, index, projected[batch_index, valid] * weights[batch_index, valid, None])
            normalizer[batch_index].scatter_add_(0, flat_index[batch_index, valid], weights[batch_index, valid, None])
        output = output / normalizer.clamp(min=1e-6)
        output = output.transpose(1, 2).reshape(batch, projected.shape[-1], render_height, render_width)
        if (render_height, render_width) != (height, width):
            output = torch.nn.functional.interpolate(output, (height, width), mode="bilinear", align_corners=False)
        return output


class LiDARAnchoredViewFusion(nn.Module):
    """Encode source views, fuse observations at fixed LiDAR anchors, render features."""

    def __init__(
        self,
        feature_dim: int = 64,
        observation_dim: int = 144,
        anchor_dim: int = 64,
        rendered_dim: int = 32,
        top_k: int = 3,
        use_dino: bool = True,
    ):
        super().__init__()
        self.encoder = SourceImageEncoder(feature_dim=feature_dim, use_dino=use_dino, dino_pretrained=use_dino)
        self.observations = AnchorObservationBuilder(feature_dim, observation_dim, top_k)
        self.fusion = ViewFusionNetwork(observation_dim, anchor_dim)
        self.renderer = AnchorFeatureRenderer(anchor_dim, rendered_dim, render_scale=2)
        self.dense_context = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.GroupNorm(8, feature_dim), nn.SiLU(inplace=True))

    def forward(
        self,
        source_images: torch.Tensor,
        anchor_xyz: torch.Tensor,
        source_uv: torch.Tensor,
        target_uv: torch.Tensor,
        target_valid: torch.Tensor,
        source_valid: torch.Tensor,
        target_size: tuple[int, int],
        anchor_weights: torch.Tensor | None = None,
        plane_sweep_uv: torch.Tensor | None = None,
        plane_sweep_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if source_images.ndim != 5:
            raise ValueError("source_images must have shape [B,V,3,H,W]")
        source_features = self.encoder(source_images)
        dense_context = source_features.mean(dim=1)
        dense_rgb = source_images.mean(dim=1)
        dense_valid = dense_rgb.new_ones(dense_rgb.shape[0], 1, dense_rgb.shape[2], dense_rgb.shape[3])
        if plane_sweep_uv is not None and plane_sweep_valid is not None:
            batch, views, samples, height, width, _ = plane_sweep_uv.shape
            feature_height, feature_width = source_features.shape[-2:]
            maps = source_features[:, :, None].expand(-1, -1, samples, -1, -1, -1).reshape(batch * views * samples, source_features.shape[2], feature_height, feature_width)
            uv = plane_sweep_uv.reshape(batch * views * samples, height, width, 2)
            normalized = torch.stack((uv[..., 0] / max(feature_width - 1, 1) * 2 - 1, uv[..., 1] / max(feature_height - 1, 1) * 2 - 1), dim=-1)
            sampled = torch.nn.functional.grid_sample(maps, normalized, align_corners=True).reshape(batch, views, samples, source_features.shape[2], height, width)
            valid = plane_sweep_valid[:, :, :, None].to(sampled.dtype)
            rgb_maps = source_images[:, :, None].expand(-1, -1, samples, -1, -1, -1).reshape(batch * views * samples, 3, source_images.shape[-2], source_images.shape[-1])
            sampled_rgb = torch.nn.functional.grid_sample(rgb_maps, normalized, align_corners=True).reshape(batch, views, samples, 3, height, width)
            view_count = valid.sum(dim=1).clamp(min=1.0)
            rgb_per_depth = (sampled_rgb * valid).sum(dim=1) / view_count
            feature_per_depth = (sampled * valid).sum(dim=1) / view_count
            rgb_variance = (((sampled_rgb - rgb_per_depth[:, None]) ** 2) * valid).sum(dim=1) / view_count
            score = rgb_variance.mean(dim=2) + (view_count[:, :, 0] < 2).to(sampled.dtype) * 1e3
            best_depth = score.argmin(dim=1, keepdim=True)
            dense_rgb = rgb_per_depth.gather(1, best_depth[:, :, None].expand(-1, -1, 3, -1, -1)).squeeze(1)
            dense_context = feature_per_depth.gather(1, best_depth[:, :, None].expand(-1, -1, source_features.shape[2], -1, -1)).squeeze(1)
            dense_valid = (view_count[:, :, 0].gather(1, best_depth) >= 2).to(dense_rgb.dtype)
        dense_context = self.dense_context(dense_context)
        observations, scores, selected_valid = self.observations(
            anchor_xyz,
            source_features,
            source_uv,
            source_valid,
        )
        observations = observations * selected_valid.unsqueeze(-1)
        safe_valid = selected_valid.bool()
        empty = ~safe_valid.any(dim=-1)
        if empty.any():
            safe_valid = safe_valid.clone()
            empty_indices = empty.nonzero(as_tuple=False)
            safe_valid[empty_indices[:, 0], empty_indices[:, 1], 0] = True
        anchor_features = self.fusion(observations, scores, safe_valid)
        rendered = self.renderer(anchor_features, target_uv, target_valid, target_size, anchor_weights)
        return {
            "source_features": source_features,
            "anchor_observations": observations,
            "observation_scores": scores,
            "observation_valid": selected_valid,
            "anchor_features": anchor_features,
            "rendered_feature": rendered,
            "global_context": source_features.mean(dim=(1, 3, 4)),
            "dense_context": dense_context,
            "dense_rgb": dense_rgb,
            "dense_valid": dense_valid,
        }
