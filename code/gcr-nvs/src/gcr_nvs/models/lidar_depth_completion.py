"""LiDAR-only dense metric range completion for target camera rays."""

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


@dataclass(frozen=True)
class DenseDepthOutput:
    range_m: torch.Tensor
    confidence: torch.Tensor
    residual_log_range: torch.Tensor


class LidarDenseDepthCompletion(nn.Module):
    """Complete target range from LiDAR geometry, never from RGB features.

    Inputs are base metric range, LiDAR support, LiDAR PCA normals, distance to
    a measured return, and calibrated target rays.  The network predicts only
    a bounded log-range residual and confidence.
    """

    def __init__(self, channels: int = 32, maximum_log_residual: float = 0.7) -> None:
        super().__init__()
        self.maximum_log_residual = float(maximum_log_residual)
        # range + support + distance + normal(3) + ray(3)
        self.enc1 = _Block(9, channels)
        self.enc2 = _Block(channels, channels * 2)
        self.enc3 = _Block(channels * 2, channels * 4)
        self.bottleneck = _Block(channels * 4, channels * 8)
        self.dec3 = _Block(channels * 8 + channels * 4, channels * 4)
        self.dec2 = _Block(channels * 4 + channels * 2, channels * 2)
        self.dec1 = _Block(channels * 2 + channels, channels)
        self.head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        base_range_m: torch.Tensor,
        lidar_support: torch.Tensor,
        lidar_normal: torch.Tensor,
        seed_distance_px: torch.Tensor,
        target_rays: torch.Tensor,
        measured_range_m: torch.Tensor | None = None,
        measured_validity: torch.Tensor | None = None,
    ) -> DenseDepthOutput:
        if base_range_m.ndim != 4 or base_range_m.shape[1] != 1:
            raise ValueError("base_range_m must have shape [B,1,H,W]")
        safe_base = torch.where(
            base_range_m > 0.1, base_range_m, base_range_m.new_full((), 40.0),
        ).clamp(1.0, 120.0)
        normalized_range = torch.log(safe_base) / torch.log(safe_base.new_tensor(121.0))
        normalized_distance = torch.exp(-seed_distance_px.clamp(min=0.0) / 12.0)
        inputs = torch.cat([
            normalized_range,
            lidar_support.float(),
            normalized_distance,
            lidar_normal,
            target_rays,
        ], dim=1)
        e1 = self.enc1(inputs)
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, 2, ceil_mode=True))
        latent = self.bottleneck(F.avg_pool2d(e3, 2, ceil_mode=True))
        d3 = self.dec3(torch.cat([self._up(latent, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        raw = self.head(d1)
        residual = torch.tanh(raw[:, :1]) * self.maximum_log_residual
        completed = torch.exp(torch.log(safe_base) + residual).clamp(1.0, 120.0)
        if measured_range_m is not None and measured_validity is not None:
            completed = torch.where(
                measured_validity.bool(), measured_range_m, completed,
            )
        return DenseDepthOutput(
            range_m=completed,
            confidence=torch.sigmoid(raw[:, 1:2]),
            residual_log_range=residual,
        )
