"""Learned metric structure constraint for the deterministic T0 renderer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, channels // 8)
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class Stage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, blocks: int = 2) -> None:
        super().__init__()
        self.project = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(output_channels) for _ in range(blocks)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.project(inputs))


class GlobalStructureTransformer(nn.Module):
    """Global low-resolution attention for long continuous surfaces."""

    def __init__(self, channels: int, layers: int = 3) -> None:
        super().__init__()
        heads = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        # Conditional positional encoding keeps the calibrated 2D layout
        # visible to attention without a fixed training resolution.
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        block = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positioned = inputs + self.position(inputs)
        batch, channels, height, width = positioned.shape
        tokens = positioned.flatten(2).transpose(1, 2)
        tokens = self.norm(self.encoder(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


@dataclass(frozen=True)
class T0StructureOutput:
    depth_range_m: torch.Tensor
    residual_log_range: torch.Tensor
    correction_gate: torch.Tensor
    geometry_confidence: torch.Tensor
    boundary_probability: torch.Tensor


class T0StructureConstraintNet(nn.Module):
    """Correct an RGB-aligned DA3 surface using sparse metric evidence.

    The network never emits RGB and never hard-copies a LiDAR depth into the
    dense output. Current and registered temporal LiDAR are metric conditions;
    RGB enters only as three fixed edge maps used to stop propagation across
    appearance boundaries.
    """

    base_input_channels = 16

    def __init__(self, channels: int = 48, maximum_log_residual: float = 0.22,
                 semantic_dim: int = 32) -> None:
        super().__init__()
        self.maximum_log_residual = float(maximum_log_residual)
        self.semantic_dim = int(semantic_dim)
        self.input_channels = self.base_input_channels + self.semantic_dim
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(self.semantic_dim, self.semantic_dim, 1),
            nn.GroupNorm(max(1, self.semantic_dim // 8), self.semantic_dim),
            nn.SiLU(inplace=True),
        )
        self.enc1 = Stage(self.input_channels, channels)
        self.enc2 = Stage(channels, channels * 2)
        self.enc3 = Stage(channels * 2, channels * 4)
        self.enc4 = Stage(channels * 4, channels * 6)
        self.bottleneck = Stage(channels * 6, channels * 8, blocks=3)
        self.global_structure = GlobalStructureTransformer(channels * 8, layers=3)
        self.dec4 = Stage(channels * 8 + channels * 6, channels * 6)
        self.dec3 = Stage(channels * 6 + channels * 4, channels * 4)
        self.dec2 = Stage(channels * 4 + channels * 2, channels * 2)
        self.dec1 = Stage(channels * 2 + channels, channels)
        # residual, correction gate, confidence, depth-boundary probability
        self.head = nn.Conv2d(channels, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _log_range(value: torch.Tensor) -> torch.Tensor:
        safe = torch.where(value > 0.1, value, value.new_full((), 40.0))
        return torch.log(safe.clamp(0.5, 160.0)) / torch.log(value.new_tensor(161.0))

    @staticmethod
    def _down(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, 2, ceil_mode=True)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        da3_range_m: torch.Tensor,
        da3_confidence: torch.Tensor,
        current_range_m: torch.Tensor,
        current_validity: torch.Tensor,
        temporal_range_m: torch.Tensor,
        temporal_validity: torch.Tensor,
        temporal_confidence: torch.Tensor,
        temporal_time_offset_s: torch.Tensor,
        seed_distance_px: torch.Tensor,
        dynamic_mask: torch.Tensor,
        rgb_edges: torch.Tensor,
        target_rays: torch.Tensor,
        semantic_features: torch.Tensor | None = None,
    ) -> T0StructureOutput:
        scalar = (
            da3_range_m, da3_confidence, current_range_m, current_validity,
            temporal_range_m, temporal_validity, temporal_confidence,
            temporal_time_offset_s, seed_distance_px, dynamic_mask,
        )
        if any(value.ndim != 4 or value.shape[1] != 1 for value in scalar):
            raise ValueError("all scalar structure inputs must be [B,1,H,W]")
        if rgb_edges.ndim != 4 or rgb_edges.shape[1] != 3:
            raise ValueError("rgb_edges must be [B,3,H,W]")
        if target_rays.ndim != 4 or target_rays.shape[1] != 3:
            raise ValueError("target_rays must be [B,3,H,W]")
        if semantic_features is None:
            semantic_features = da3_range_m.new_zeros(
                da3_range_m.shape[0], self.semantic_dim, *da3_range_m.shape[-2:]
            )
        if semantic_features.ndim != 4 or semantic_features.shape[1] != self.semantic_dim:
            raise ValueError(f"semantic_features must be [B,{self.semantic_dim},H,W]")
        if semantic_features.shape[0] != da3_range_m.shape[0] or semantic_features.shape[-2:] != da3_range_m.shape[-2:]:
            raise ValueError("semantic_features must match depth spatial shape")

        support = torch.exp(-seed_distance_px.float().clamp(0.0, 128.0) / 24.0)
        condition = torch.cat([
            self._log_range(da3_range_m),
            da3_confidence.float().clamp(0.0, 1.0),
            self._log_range(current_range_m),
            current_validity.float().clamp(0.0, 1.0),
            self._log_range(temporal_range_m),
            temporal_validity.float().clamp(0.0, 1.0),
            temporal_confidence.float().clamp(0.0, 1.0),
            temporal_time_offset_s.float().clamp(-1.0, 1.0),
            support,
            dynamic_mask.float().clamp(0.0, 1.0),
            rgb_edges.float().clamp(-1.0, 1.0),
            target_rays.float().clamp(-1.0, 1.0),
            self.semantic_projection(semantic_features.float()),
        ], dim=1)
        if condition.shape[1] != self.input_channels:
            raise RuntimeError(f"expected {self.input_channels} channels, got {condition.shape[1]}")

        e1 = self.enc1(condition)
        e2 = self.enc2(self._down(e1))
        e3 = self.enc3(self._down(e2))
        e4 = self.enc4(self._down(e3))
        latent = self.global_structure(self.bottleneck(self._down(e4)))
        d4 = self.dec4(torch.cat([self._up(latent, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        raw = self.head(d1)

        # Geometry evidence may propagate locally, but the model is forced to
        # become the identity far away from all current/temporal LiDAR seeds.
        dynamic_gate = 1.0 - 0.75 * dynamic_mask.float().clamp(0.0, 1.0)
        gate = torch.sigmoid(raw[:, 1:2]) * support * dynamic_gate
        residual = torch.tanh(raw[:, :1]) * self.maximum_log_residual * gate
        safe_da3 = torch.where(da3_range_m > 0.1, da3_range_m, da3_range_m.new_full((), 40.0))
        depth = torch.exp(torch.log(safe_da3.clamp(0.5, 160.0)) + residual).clamp(0.5, 160.0)
        return T0StructureOutput(
            depth_range_m=depth,
            residual_log_range=residual,
            correction_gate=gate,
            geometry_confidence=torch.sigmoid(raw[:, 2:3]),
            boundary_probability=torch.sigmoid(raw[:, 3:4]),
        )
