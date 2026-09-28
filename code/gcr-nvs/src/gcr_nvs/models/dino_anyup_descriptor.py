"""Stable frozen DINOv2 descriptors with AnyUP dense upsampling."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from gcr_nvs.models.backbones import DINOv2Encoder
from gcr_nvs.models.feature_upsampler import FeatureUpsampler


class FrozenDinoAnyUpDescriptor(nn.Module):
    """Return normalized dense descriptors without trainable random adapters."""

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vitl14",
        output_channels: int = 64,
        dino_input_size: tuple[int, int] = (574, 1022),
        anyup_checkpoint: Path,
        anyup_root: Path,
        projection_seed: int = 20260824,
        q_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.dino = DINOv2Encoder(model_name=model_name, pretrained=True, freeze=True)
        self.upsampler = FeatureUpsampler(
            mode="anyup", checkpoint=anyup_checkpoint, anyup_root=anyup_root,
            q_chunk_size=q_chunk_size, freeze=True,
        )
        self.dino_input_size = tuple(int(value) for value in dino_input_size)
        generator = torch.Generator(device="cpu").manual_seed(int(projection_seed))
        projection = torch.randn(output_channels, self.dino.feature_dim, generator=generator)
        self.register_buffer(
            "projection",
            F.normalize(projection, dim=1)[:, :, None, None],
            persistent=True,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.inference_mode()
    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINO descriptor input must have shape B,3,H,W")
        normalized = (
            images - images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        ) / images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        dino_input = F.interpolate(
            normalized, size=self.dino_input_size, mode="bilinear", align_corners=False,
        )
        device_type = images.device.type
        with torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda",
        ):
            patches = self.dino.forward_intermediate(
                dino_input, layers=(self.dino.intermediate_layers[-1],),
            )[0]
            compressed = F.conv2d(patches, self.projection.to(patches.dtype))
        return F.normalize(compressed.float(), dim=1, eps=1e-6)

    @torch.inference_mode()
    def upsample_tokens(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        normalized = (
            images - images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        ) / images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        device_type = images.device.type
        with torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda",
        ):
            dense = self.upsampler(normalized, tokens.to(normalized.dtype), output_size)
        return F.normalize(dense.float(), dim=1, eps=1e-6)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return self.upsample_tokens(images, self.encode_tokens(images), output_size)
