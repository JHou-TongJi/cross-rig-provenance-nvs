"""GCR-NVS Lite residual refiner with optional bidirectional ConvGRU."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.projection = (
            nn.Identity() if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1)
        )
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(max(1, output_channels // 8), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(max(1, output_channels // 8), output_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.projection(x))


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
        combined = torch.cat([x, hidden], dim=1)
        reset, update = self.gates(combined).chunk(2, dim=1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class BidirectionalConvGRU(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.forward_cell = ConvGRUCell(input_channels, hidden_channels)
        self.backward_cell = ConvGRUCell(input_channels, hidden_channels)
        self.merge = nn.Conv2d(hidden_channels * 2, input_channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        forward = []
        hidden = None
        for frame in sequence.unbind(dim=1):
            hidden = self.forward_cell(frame, hidden)
            forward.append(hidden)
        backward = []
        hidden = None
        for frame in reversed(sequence.unbind(dim=1)):
            hidden = self.backward_cell(frame, hidden)
            backward.append(hidden)
        backward.reverse()
        merged = [self.merge(torch.cat([left, right], dim=1)) for left, right in zip(forward, backward)]
        return torch.stack(merged, dim=1)


class LiteFrameEncoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 48):
        super().__init__()
        self.stem = ResidualBlock(input_channels, base_channels)
        self.level0 = nn.Sequential(ResidualBlock(base_channels, base_channels), ResidualBlock(base_channels, base_channels))
        self.down1 = nn.Conv2d(base_channels, 64, 3, stride=2, padding=1)
        self.level1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.down2 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.level2 = nn.Sequential(ResidualBlock(96, 96), ResidualBlock(96, 96))
        self.down3 = nn.Conv2d(96, 160, 3, stride=2, padding=1)
        self.level3 = nn.Sequential(ResidualBlock(160, 160), ResidualBlock(160, 160), ResidualBlock(160, 160))
        self.down4 = nn.Conv2d(160, 192, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(ResidualBlock(192, 192), ResidualBlock(192, 192))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level0 = self.level0(self.stem(x))
        level1 = self.level1(self.down1(level0))
        level2 = self.level2(self.down2(level1))
        level3 = self.level3(self.down3(level2))
        bottleneck = self.bottleneck(self.down4(level3))
        return level0, level1, level2, level3, bottleneck


class LiteTemporalRefiner(nn.Module):
    """Documented B0-style residual refiner.

    Inputs are [B,T,48,H,W], with 13 geometry channels, 3 target-ray
    channels, and 32 rendered learned-feature channels. The target frame is
    the centered frame for T=3 and the only frame for T=1.
    """

    def __init__(self, input_channels: int = 48, base_channels: int = 48, max_rgb_residual: float = 0.25):
        super().__init__()
        if input_channels != 48:
            raise ValueError("GCR-NVS Lite expects the documented 48-channel refiner input")
        self.max_rgb_residual = max_rgb_residual
        self.encoder = LiteFrameEncoder(input_channels, base_channels)
        self.temporal_level2 = BidirectionalConvGRU(96, 48)
        self.temporal_level3 = BidirectionalConvGRU(160, 80)
        self.up3 = nn.Conv2d(192, 160, 3, padding=1)
        self.dec3 = nn.Sequential(ResidualBlock(320, 160), ResidualBlock(160, 160))
        self.up2 = nn.Conv2d(160, 96, 3, padding=1)
        self.dec2 = nn.Sequential(ResidualBlock(192, 96), ResidualBlock(96, 96))
        self.up1 = nn.Conv2d(96, 64, 3, padding=1)
        self.dec1 = nn.Sequential(ResidualBlock(128, 64), ResidualBlock(64, 64))
        self.up0 = nn.Conv2d(64, base_channels, 3, padding=1)
        self.dec0 = nn.Sequential(ResidualBlock(base_channels * 2, base_channels), ResidualBlock(base_channels, base_channels))
        self.rgb_head = nn.Sequential(nn.Conv2d(base_channels, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1))
        self.alpha_head = nn.Sequential(nn.Conv2d(base_channels, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))
        self.logvar_head = nn.Sequential(nn.Conv2d(base_channels, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))

    def _frame_forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level0, level1, level2, level3, bottleneck = self.encoder(inputs)
        return level0, level1, level2, level3, bottleneck

    @staticmethod
    def _multiband_blend(
        observed: torch.Tensor,
        generated: torch.Tensor,
        opacity: torch.Tensor,
        levels: int = 4,
    ) -> torch.Tensor:
        observed_pyramid = [observed]
        generated_pyramid = [generated]
        opacity_pyramid = [opacity]
        for _ in range(levels - 1):
            observed_pyramid.append(F.avg_pool2d(
                observed_pyramid[-1], 5, stride=2, padding=2, count_include_pad=False,
            ))
            generated_pyramid.append(F.avg_pool2d(
                generated_pyramid[-1], 5, stride=2, padding=2, count_include_pad=False,
            ))
            opacity_pyramid.append(F.avg_pool2d(
                opacity_pyramid[-1], 5, stride=2, padding=2, count_include_pad=False,
            ))

        blended = (
            opacity_pyramid[-1] * observed_pyramid[-1]
            + (1.0 - opacity_pyramid[-1]) * generated_pyramid[-1]
        )
        for level in range(levels - 2, -1, -1):
            target_size = observed_pyramid[level].shape[-2:]
            observed_low = F.interpolate(observed_pyramid[level + 1], target_size, mode="bilinear", align_corners=False)
            generated_low = F.interpolate(generated_pyramid[level + 1], target_size, mode="bilinear", align_corners=False)
            observed_laplacian = observed_pyramid[level] - observed_low
            generated_laplacian = generated_pyramid[level] - generated_low
            mask = opacity_pyramid[level]
            blended = F.interpolate(blended, target_size, mode="bilinear", align_corners=False)
            blended = blended + mask * observed_laplacian + (1.0 - mask) * generated_laplacian
        return blended

    def forward(
        self,
        geometry: torch.Tensor,
        ray_map: torch.Tensor,
        rendered_feature: torch.Tensor,
        temporal_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if geometry.ndim == 4:
            geometry = geometry.unsqueeze(1)
            ray_map = ray_map.unsqueeze(1)
            rendered_feature = rendered_feature.unsqueeze(1)
        inputs = torch.cat([geometry, ray_map, rendered_feature], dim=2)
        batch, frames, channels, height, width = inputs.shape
        if temporal_valid_mask is not None:
            temporal_valid_mask = temporal_valid_mask.to(inputs.dtype).view(batch, frames, 1, 1, 1)
        flat = inputs.reshape(batch * frames, channels, height, width)
        encoded = [self._frame_forward(flattened) for flattened in flat.reshape(batch, frames, channels, height, width)]
        levels = [torch.stack([item[index] for item in encoded], dim=0) for index in range(5)]
        if temporal_valid_mask is not None:
            levels = [level * temporal_valid_mask for level in levels]
        levels[2] = self.temporal_level2(levels[2])
        levels[3] = self.temporal_level3(levels[3])
        target_index = frames // 2
        level0, level1, level2, level3, bottleneck = [level[:, target_index] for level in levels]
        x = torch.nn.functional.interpolate(bottleneck, size=level3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([self.up3(x), level3], dim=1))
        x = torch.nn.functional.interpolate(x, size=level2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([self.up2(x), level2], dim=1))
        x = torch.nn.functional.interpolate(x, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([self.up1(x), level1], dim=1))
        x = torch.nn.functional.interpolate(x, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec0(torch.cat([self.up0(x), level0], dim=1))
        raw_residual = torch.tanh(self.rgb_head(x))
        alpha_pred = torch.sigmoid(self.alpha_head(x))
        log_uncertainty = self.logvar_head(x).clamp(-6.0, 2.0)
        target_geometry = geometry[:, target_index]
        coarse = target_geometry[:, :3]
        opacity = target_geometry[:, 7:8].clamp(0.0, 1.0)
        confidence = target_geometry[:, 8:9].clamp(0.0, 1.0)
        blend_opacity = target_geometry[:, 10:11].clamp(0.0, 1.0)
        hole = 1.0 - opacity
        modify_gate = (hole + (1.0 - confidence)).clamp(0.0, 1.0)
        alpha_effective = alpha_pred * modify_gate
        proposal = (coarse + self.max_rgb_residual * raw_residual).clamp(0.0, 1.0)
        observed_rgb = (1.0 - alpha_effective) * coarse + alpha_effective * proposal
        rgb = self._multiband_blend(observed_rgb, proposal, blend_opacity).clamp(0.0, 1.0)
        return {
            "rgb": rgb,
            "final_rgb": rgb,
            "coarse_rgb": coarse,
            "proposal": proposal,
            "rgb_residual": raw_residual,
            "alpha": alpha_pred,
            "alpha_effective": alpha_effective,
            "log_uncertainty": log_uncertainty,
            "depth": target_geometry[:, 3:4],
            "target_geometry": target_geometry,
        }


class SingleFrameRefiner(LiteTemporalRefiner):
    """Compatibility alias for the documented single-frame B0 stage."""

    def forward(self, geometry, ray_map=None, context=None, rendered_feature=None, **kwargs):
        if ray_map is None:
            raise ValueError("ray_map is required")
        if rendered_feature is None:
            rendered_feature = geometry.new_zeros(*geometry.shape[:-3], 32, geometry.shape[-2], geometry.shape[-1])
        return super().forward(geometry, ray_map, rendered_feature, **kwargs)
