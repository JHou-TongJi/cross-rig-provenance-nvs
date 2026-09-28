"""Source-view semantic and detail encoder with a DPT/FPN-style pyramid."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from gcr_nvs.models.backbones import DINOv2Encoder
from gcr_nvs.models.feature_upsampler import FeatureUpsampler


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = inputs.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)
        self.depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, 1))
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.gate(self.depthwise(self.conv1(self.norm1(inputs))))
        features = self.conv2(features * self.channel_attention(features))
        residual = inputs + features * self.beta
        features = self.ffn2(self.gate(self.ffn1(self.norm2(residual))))
        return residual + features * self.gamma


class SourceImageEncoder(nn.Module):
    """Build dense multi-scale source features without copying RGB to output.

    DINO patch tokens provide semantic context while the shallow branch keeps
    high-frequency edges. A top-down FPN path brings both signals back to the
    image grid used by LiDAR anchor sampling.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        use_dino: bool = False,
        dino_model_name: str = "dinov2_vitb14",
        dino_pretrained: bool = True,
        dino_input_size: tuple[int, int] = (476, 840),
        semantic_upsampling: str = "bilinear",
        anyup_checkpoint=None,
        anyup_root=None,
        anyup_q_chunk_size: int | None = 8192,
    ):
        super().__init__()
        self.use_dino = use_dino
        self.dino_input_size = dino_input_size
        self.semantic = (
            DINOv2Encoder(model_name=dino_model_name, pretrained=dino_pretrained, freeze=True)
            if use_dino else None
        )
        upsampler_args = {
            "mode": semantic_upsampling,
            "checkpoint": anyup_checkpoint,
            "q_chunk_size": anyup_q_chunk_size,
        }
        if anyup_root is not None:
            upsampler_args["anyup_root"] = anyup_root
        self.semantic_upsampler = FeatureUpsampler(**upsampler_args)
        semantic_channels = self.semantic.feature_dim if self.semantic is not None else 768
        self.semantic_laterals = nn.ModuleList([
            nn.Conv2d(semantic_channels, feature_dim, 1) for _ in range(4)
        ])
        self.semantic_fuse = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(inplace=True),
            NAFBlock(feature_dim),
        )
        self.detail_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(inplace=True),
        )
        self.detail_down2 = nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1)
        self.detail_half = nn.Sequential(NAFBlock(feature_dim), NAFBlock(feature_dim))
        self.detail_down4 = nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1)
        self.detail_quarter = nn.Sequential(NAFBlock(feature_dim), NAFBlock(feature_dim))
        self.fuse_quarter = nn.Sequential(nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1), NAFBlock(feature_dim))
        self.fuse_half = nn.Sequential(nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1), NAFBlock(feature_dim))
        self.fuse_full = nn.Sequential(nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1), NAFBlock(feature_dim))

    def _semantic_pyramid(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detail = self.detail_stem(images)
        detail_half = self.detail_half(self.detail_down2(detail))
        detail_quarter = self.detail_quarter(self.detail_down4(detail_half))
        if self.semantic is None:
            semantic_quarter = torch.zeros_like(detail_quarter)
        else:
            normalized = (images - images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            semantic_input = F.interpolate(normalized, self.dino_input_size, mode="bilinear", align_corners=False)
            intermediate = self.semantic.forward_intermediate(semantic_input)
            projected = [
                self.semantic_upsampler(
                    normalized,
                    lateral(feature),
                    detail_quarter.shape[-2:],
                )
                for lateral, feature in zip(self.semantic_laterals, intermediate)
            ]
            semantic_quarter = self.semantic_fuse(torch.cat(projected, dim=1))
        pyramid_quarter = self.fuse_quarter(torch.cat([semantic_quarter, detail_quarter], dim=1))
        semantic_half = F.interpolate(pyramid_quarter, size=detail_half.shape[-2:], mode="bilinear", align_corners=False)
        pyramid_half = self.fuse_half(torch.cat([semantic_half, detail_half], dim=1))
        semantic_full = F.interpolate(pyramid_half, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        pyramid_full = self.fuse_full(torch.cat([semantic_full, detail], dim=1))
        return pyramid_full, pyramid_half, pyramid_quarter

    def forward_pyramid(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flatten = images.ndim == 5
        if not flatten:
            return self._semantic_pyramid(images)
        batch, views, _, _, _ = images.shape
        per_view = [self._semantic_pyramid(images[:, view]) for view in range(views)]
        return tuple(torch.stack([features[level] for features in per_view], dim=1) for level in range(3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_pyramid(images)[0]
