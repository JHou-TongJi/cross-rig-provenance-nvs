"""Semantic completion of missing target-view depth surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = max(1, output_channels // 8)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity() if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.skip(inputs) + self.layers(inputs)


class _SurfaceTransformer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=8, dim_feedforward=channels * 3,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs + self.position(inputs)
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        tokens = self.norm(self.layers(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


@dataclass(frozen=True)
class SemanticSurfaceOutput:
    depth_m: torch.Tensor
    confidence: torch.Tensor
    boundary_probability: torch.Tensor
    residual_log_depth: torch.Tensor


class SemanticSurfaceCompletionNet(nn.Module):
    """Predict depth only where the transformed DA3 target surface is absent.

    Observed target depth is hard-composited at the output. Semantic features
    may guide a missing surface, but can never modify an already observed one.
    """

    def __init__(
        self,
        semantic_dim: int = 32,
        channels: int = 32,
        maximum_log_residual: float = 1.0,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.maximum_log_residual = float(maximum_log_residual)
        # base depth, validity, LiDAR depth/validity, seed support, rays(3),
        # RGB edges(3), dense semantics.
        input_channels = 11 + self.semantic_dim
        self.enc1 = _Block(input_channels, channels)
        self.enc2 = _Block(channels, channels * 2)
        self.enc3 = _Block(channels * 2, channels * 4)
        self.bottleneck = _Block(channels * 4, channels * 4)
        self.transformer = _SurfaceTransformer(channels * 4)
        self.dec3 = _Block(channels * 8, channels * 4)
        self.dec2 = _Block(channels * 6, channels * 2)
        self.dec1 = _Block(channels * 3, channels)
        self.head = nn.Conv2d(channels, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _safe_log(depth: torch.Tensor) -> torch.Tensor:
        return torch.log(depth.clamp(0.5, 250.0)) / torch.log(depth.new_tensor(251.0))

    @staticmethod
    def _down(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, 2, ceil_mode=True)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        base_depth_m: torch.Tensor,
        observed_validity: torch.Tensor,
        lidar_depth_m: torch.Tensor,
        lidar_validity: torch.Tensor,
        seed_distance_px: torch.Tensor,
        target_rays: torch.Tensor,
        rgb_edges: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> SemanticSurfaceOutput:
        scalars = (
            base_depth_m, observed_validity, lidar_depth_m,
            lidar_validity, seed_distance_px,
        )
        if any(value.ndim != 4 or value.shape[1] != 1 for value in scalars):
            raise ValueError("surface scalar inputs must be B1HW")
        if target_rays.ndim != 4 or target_rays.shape[1] != 3:
            raise ValueError("target_rays must be B3HW")
        if rgb_edges.ndim != 4 or rgb_edges.shape[1] != 3:
            raise ValueError("rgb_edges must be B3HW")
        if semantic_features.ndim != 4 or semantic_features.shape[1] != self.semantic_dim:
            raise ValueError(f"semantic_features must be B{self.semantic_dim}HW")
        safe_base = base_depth_m.clamp(0.5, 250.0)
        safe_lidar = torch.where(
            lidar_validity > 0.5, lidar_depth_m, safe_base,
        ).clamp(0.5, 250.0)
        support = torch.exp(-seed_distance_px.clamp(0.0, 128.0) / 16.0)
        inputs = torch.cat([
            self._safe_log(safe_base), observed_validity.float(),
            self._safe_log(safe_lidar), lidar_validity.float(), support,
            target_rays.float().clamp(-1.0, 1.0),
            rgb_edges.float().clamp(-1.0, 1.0),
            semantic_features.float(),
        ], dim=1)
        e1 = self.enc1(inputs)
        e2 = self.enc2(self._down(e1))
        e3 = self.enc3(self._down(e2))
        latent = self.transformer(self.bottleneck(self._down(e3)))
        d3 = self.dec3(torch.cat([self._up(latent, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        raw = self.head(d1)
        residual = torch.tanh(raw[:, :1]) * self.maximum_log_residual
        predicted = torch.exp(torch.log(safe_base) + residual).clamp(0.5, 250.0)
        completed = torch.where(observed_validity > 0.5, base_depth_m, predicted)
        return SemanticSurfaceOutput(
            depth_m=completed,
            confidence=torch.sigmoid(raw[:, 1:2]),
            boundary_probability=torch.sigmoid(raw[:, 2:3]),
            residual_log_depth=residual,
        )
