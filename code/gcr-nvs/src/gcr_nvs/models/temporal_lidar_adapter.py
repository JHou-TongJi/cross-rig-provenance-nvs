"""Geometry-only adapter for current and previous-frame LiDAR evidence.

The adapter refines a frozen, RGB-aligned dense depth candidate (DA3) with
registered LiDAR support.  It has no RGB or semantic inputs by construction:
LiDAR can change metric geometry and visibility, but cannot paint appearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
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
class TemporalGeometryOutput:
    """The complete T1 geometry contract consumed by the RGB projector."""

    depth_m: torch.Tensor
    visibility: torch.Tensor
    temporal_confidence: torch.Tensor
    dynamic_suppression: torch.Tensor
    residual_log_depth: torch.Tensor


class TemporalLidarGeometryAdapter(nn.Module):
    """Predict bounded geometry corrections from LiDAR-only image planes.

    Inputs are ``[B,1,H,W]`` unless noted otherwise:

    ``da3_depth_m``
        Continuous metric candidate from frozen DA3 after LiDAR scale/shift.
    ``current_range_m/current_validity``
        Current-frame measured LiDAR z-buffer and support mask.
    ``temporal_range_m/temporal_validity``
        The previous-frame points after SE(3) registration to the center frame.
    ``temporal_alignment_confidence/time_offset_s``
        Per-pixel registered-neighbor quality maps.
    ``seed_distance_px``
        Distance to the nearest current/temporal measured return.
    ``dynamic_mask``
        Current-frame dynamic suppression mask (one means dynamic).
    ``target_rays``
        Calibrated camera rays, shape ``[B,3,H,W]``.

    Current measured returns are immutable at the output boundary.  The
    adapter only changes the continuous surface between measurements.
    """

    def __init__(
        self,
        channels: int = 32,
        maximum_log_residual: float = 0.25,
    ) -> None:
        super().__init__()
        self.maximum_log_residual = float(maximum_log_residual)
        if self.maximum_log_residual <= 0.0:
            raise ValueError("maximum_log_residual must be positive")
        # DA3, current range/support, temporal range/support/confidence/time,
        # seed distance, dynamic mask, and three calibrated ray components.
        self.input_channels = 12
        self.enc1 = _ConvBlock(self.input_channels, channels)
        self.enc2 = _ConvBlock(channels, channels * 2)
        self.enc3 = _ConvBlock(channels * 2, channels * 4)
        self.bottleneck = _ConvBlock(channels * 4, channels * 8)
        self.dec3 = _ConvBlock(channels * 8 + channels * 4, channels * 4)
        self.dec2 = _ConvBlock(channels * 4 + channels * 2, channels * 2)
        self.dec1 = _ConvBlock(channels * 2 + channels, channels)
        # residual, visibility, temporal confidence, dynamic suppression
        self.head = nn.Conv2d(channels, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _safe_log_depth(depth_m: torch.Tensor) -> torch.Tensor:
        safe = torch.where(depth_m > 0.1, depth_m, depth_m.new_full((), 40.0))
        return torch.log(safe.clamp(1.0, 120.0)) / torch.log(depth_m.new_tensor(121.0))

    def forward(
        self,
        da3_depth_m: torch.Tensor,
        current_range_m: torch.Tensor,
        current_validity: torch.Tensor,
        temporal_range_m: torch.Tensor,
        temporal_validity: torch.Tensor,
        temporal_alignment_confidence: torch.Tensor,
        temporal_time_offset_s: torch.Tensor,
        seed_distance_px: torch.Tensor,
        dynamic_mask: torch.Tensor,
        target_rays: torch.Tensor,
    ) -> TemporalGeometryOutput:
        inputs = (
            da3_depth_m, current_range_m, current_validity,
            temporal_range_m, temporal_validity, temporal_alignment_confidence,
            temporal_time_offset_s, seed_distance_px, dynamic_mask,
        )
        if any(value.ndim != 4 or value.shape[1] != 1 for value in inputs):
            raise ValueError("all scalar inputs must have shape [B,1,H,W]")
        if target_rays.ndim != 4 or target_rays.shape[1] != 3:
            raise ValueError("target_rays must have shape [B,3,H,W]")
        shape = da3_depth_m.shape[-2:]
        if any(value.shape[0] != da3_depth_m.shape[0] or value.shape[-2:] != shape for value in inputs):
            raise ValueError("all scalar inputs must share batch and spatial dimensions")
        if target_rays.shape[0] != da3_depth_m.shape[0] or target_rays.shape[-2:] != shape:
            raise ValueError("target_rays must match DA3 batch and spatial dimensions")

        # Keep all quality signals bounded before they enter the network.  The
        # signed time offset is useful for diagnostics but cannot become a
        # large geometry correction by itself.
        normalized = [
            self._safe_log_depth(da3_depth_m),
            self._safe_log_depth(current_range_m),
            current_validity.float().clamp(0.0, 1.0),
            self._safe_log_depth(temporal_range_m),
            temporal_validity.float().clamp(0.0, 1.0),
            temporal_alignment_confidence.float().clamp(0.0, 1.0),
            temporal_time_offset_s.float().clamp(-1.0, 1.0),
            torch.exp(-seed_distance_px.float().clamp(min=0.0) / 12.0),
            dynamic_mask.float().clamp(0.0, 1.0),
            target_rays.float().clamp(-1.0, 1.0),
        ]
        encoded = self.enc1(torch.cat(normalized, dim=1))
        e2 = self.enc2(F.avg_pool2d(encoded, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, 2, ceil_mode=True))
        latent = self.bottleneck(F.avg_pool2d(e3, 2, ceil_mode=True))
        d3 = self.dec3(torch.cat([self._up(latent, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, encoded), encoded], dim=1))
        raw = self.head(d1)

        # Temporal evidence is local.  Outside pixels supported by the
        # registered previous scan the adapter must be an exact identity on
        # the frozen DA3 surface; otherwise it becomes an RGB-independent
        # monocular depth replacement, which is not its contract.
        temporal_gate = (
            temporal_validity.float().clamp(0.0, 1.0)
            * temporal_alignment_confidence.float().clamp(0.0, 1.0)
            * (1.0 - current_validity.float().clamp(0.0, 1.0))
        )
        residual = torch.tanh(raw[:, :1]) * self.maximum_log_residual * temporal_gate
        safe_da3 = torch.where(da3_depth_m > 0.1, da3_depth_m, da3_depth_m.new_full((), 40.0))
        depth = torch.exp(torch.log(safe_da3.clamp(1.0, 120.0)) + residual).clamp(1.0, 120.0)
        # Dynamic returns may inform geometry during training, but never claim
        # reliable visibility in the final contract.
        dynamic_suppression = torch.sigmoid(raw[:, 3:4])
        visibility = torch.sigmoid(raw[:, 1:2]) * (1.0 - dynamic_mask.float().clamp(0.0, 1.0))
        temporal_confidence = torch.sigmoid(raw[:, 2:3]) * temporal_alignment_confidence.float().clamp(0.0, 1.0)
        if current_validity.dtype != torch.bool:
            current_validity = current_validity > 0.5
        depth = torch.where(current_validity, current_range_m, depth)
        visibility = torch.where(current_validity, torch.ones_like(visibility), visibility)
        return TemporalGeometryOutput(
            depth_m=depth,
            visibility=visibility,
            temporal_confidence=temporal_confidence,
            dynamic_suppression=dynamic_suppression,
            residual_log_depth=residual,
        )
