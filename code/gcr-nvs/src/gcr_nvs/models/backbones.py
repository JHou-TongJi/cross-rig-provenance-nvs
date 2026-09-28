"""Optional frozen DINOv2 source encoder with a dependency-safe fallback."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Unable to load DINOv2. Retry with network access or use pretrained=False."
                ) from exc
            self.backbone = nn.Identity()
        self.feature_dim = int(
            getattr(self.backbone, "embed_dim", {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
            }.get(model_name, 768))
        )
        depth = int(getattr(self.backbone, "n_blocks", {
            "dinov2_vits14": 12,
            "dinov2_vitb14": 12,
            "dinov2_vitl14": 24,
            "dinov2_vitg14": 40,
        }.get(model_name, 12)))
        self.intermediate_layers = tuple(
            max(0, round(depth * fraction) - 1)
            for fraction in (0.25, 0.50, 0.75, 1.0)
        )
        self.frozen = freeze
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if all(not parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(self.backbone, nn.Identity):
            return images
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            return self.backbone.forward_features(images)["x_norm_patchtokens"]

    def forward_intermediate(
        self,
        images: torch.Tensor,
        layers: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        layers = self.intermediate_layers if layers is None else layers
        if isinstance(self.backbone, nn.Identity):
            return (images,) * len(layers)
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            return tuple(
                self.backbone.get_intermediate_layers(
                    images,
                    n=layers,
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
            )
