"""Geometry-aware Top-K observation builder and view fusion network."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AnchorObservationBuilder(nn.Module):
    def __init__(self, feature_dim: int = 64, observation_dim: int = 144, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.projection = nn.Linear(feature_dim + 8, observation_dim)

    def forward(
        self,
        anchor_xyz,
        source_features,
        projected_uv,
        source_valid: torch.Tensor | None = None,
        metadata=None,
    ):
        """Sample source features and return [B,N,K,D] observations and scores."""
        batch, views, _, height, width = source_features.shape
        _, anchors, _ = anchor_xyz.shape
        feature_maps = source_features.reshape(batch * views, source_features.shape[2], height, width)
        uv = projected_uv.reshape(batch * views, anchors, 1, 2)
        normalized = torch.stack((uv[..., 0] / max(width - 1, 1) * 2 - 1, uv[..., 1] / max(height - 1, 1) * 2 - 1), dim=-1)
        sampled = F.grid_sample(feature_maps, normalized, align_corners=True).reshape(batch, views, -1, anchors).permute(0, 1, 3, 2)
        valid = (projected_uv[..., 0] >= 0) & (projected_uv[..., 0] < width) & (projected_uv[..., 1] >= 0) & (projected_uv[..., 1] < height)
        if source_valid is not None:
            valid = valid & source_valid.bool()
        direction = F.normalize(anchor_xyz, dim=-1).unsqueeze(1).expand(-1, views, -1, -1)
        valid_float = valid.float().unsqueeze(-1)
        depth = anchor_xyz[..., 2].abs().unsqueeze(1).expand(-1, views, -1).unsqueeze(-1)
        features = torch.cat([sampled, direction, valid_float, depth, torch.zeros_like(depth), torch.zeros_like(depth), torch.zeros_like(depth)], dim=-1)
        scores = valid_float.squeeze(-1) - depth.squeeze(-1) * 1e-3
        top_scores, top_indices = scores.topk(min(self.top_k, views), dim=1)
        gathered = features.permute(0, 2, 1, 3).gather(2, top_indices.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]))
        observations = self.projection(gathered)
        selected_valid = valid.permute(0, 2, 1).gather(2, top_indices.permute(0, 2, 1)).float()
        return observations, top_scores.permute(0, 2, 1), selected_valid


class ViewFusionNetwork(nn.Module):
    def __init__(self, observation_dim: int = 144, anchor_dim: int = 64, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(observation_dim)
        self.attention = nn.MultiheadAttention(observation_dim, heads, batch_first=True)
        self.output = nn.Sequential(nn.Linear(observation_dim, anchor_dim), nn.SiLU(), nn.Linear(anchor_dim, anchor_dim))

    def forward(
        self,
        observations: torch.Tensor,
        scores: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, anchors, top_k, dim = observations.shape
        x = observations.reshape(batch * anchors, top_k, dim)
        x = self.norm(x)
        key_padding_mask = None if valid is None else ~valid.reshape(batch * anchors, top_k).bool()
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        if valid is None:
            x = x.mean(dim=1)
        else:
            weights = valid.reshape(batch * anchors, top_k).float().unsqueeze(-1)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.output(x).reshape(batch, anchors, -1)
