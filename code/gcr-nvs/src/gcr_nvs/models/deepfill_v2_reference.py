"""Reference-conditioned DeepFill v2 style completion for residual T0 holes.

This is an independent implementation of the gated coarse/refinement design
from Yu et al. It does not copy code from third-party PyTorch ports.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from gcr_nvs.models.semantic_reference_completion import strict_mask_composite


class GatedConv2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.features = nn.Conv2d(
            input_channels, output_channels, kernel_size, stride,
            padding=padding, dilation=dilation,
        )
        self.gates = nn.Conv2d(
            input_channels, output_channels, kernel_size, stride,
            padding=padding, dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(self.features(inputs)) * torch.sigmoid(self.gates(inputs))


class GatedUpBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = GatedConv2d(input_channels, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(inputs, scale_factor=2.0, mode="bilinear", align_corners=False))


class GatedEncoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            GatedConv2d(input_channels, base_channels, 5),
            GatedConv2d(base_channels, base_channels * 2, 3, stride=2),
            GatedConv2d(base_channels * 2, base_channels * 2),
            GatedConv2d(base_channels * 2, base_channels * 4, 3, stride=2),
            GatedConv2d(base_channels * 4, base_channels * 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DilatedBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[
            GatedConv2d(channels, channels, dilation=dilation)
            for dilation in (1, 2, 4, 8, 16, 1)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class GatedDecoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            GatedUpBlock(input_channels, base_channels * 2),
            GatedConv2d(base_channels * 2, base_channels * 2),
            GatedUpBlock(base_channels * 2, base_channels),
            GatedConv2d(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(inputs))


class LocalReferenceContextAttention(nn.Module):
    """Patch attention from the target branch into retrieved source features."""

    def __init__(self, channels: int, radius: int = 3, temperature: float = 10.0) -> None:
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = GatedConv2d(channels, channels)

    def forward(self, target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if target.shape != reference.shape:
            raise ValueError("target and reference attention features must match")
        batch, channels, height, width = target.shape
        kernel = self.radius * 2 + 1
        query = F.normalize(self.query(target), dim=1, eps=1e-6)
        keys = F.unfold(self.key(reference), kernel, padding=self.radius)
        values = F.unfold(self.value(reference), kernel, padding=self.radius)
        keys = keys.view(batch, channels, kernel * kernel, height, width)
        values = values.view(batch, channels, kernel * kernel, height, width)
        keys = F.normalize(keys, dim=1, eps=1e-6)
        logits = (query[:, :, None] * keys).sum(dim=1) * self.temperature
        weights = logits.softmax(dim=1)
        attended = (values * weights[:, None]).sum(dim=2)
        return self.output(attended)


class SemanticDeepFillV2(nn.Module):
    """Two-stage residual-hole generator with a strict T0 output contract."""

    def __init__(self, semantic_channels: int = 32, geometry_channels: int = 5,
                 base_channels: int = 48, attention_radius: int = 3) -> None:
        super().__init__()
        # masked RGB, mask, retrieved RGB/confidence, semantic, geometry
        self.condition_channels = 3 + 1 + 3 + 1 + semantic_channels + geometry_channels
        self.coarse_encoder = GatedEncoder(self.condition_channels, base_channels)
        self.coarse_bottleneck = DilatedBottleneck(base_channels * 4)
        self.coarse_decoder = GatedDecoder(base_channels * 4, base_channels)

        self.refine_channels = self.condition_channels + 3
        self.refine_encoder = GatedEncoder(self.refine_channels, base_channels)
        self.hallucination = DilatedBottleneck(base_channels * 4)
        self.reference_encoder = GatedEncoder(self.refine_channels, base_channels)
        self.context_attention = LocalReferenceContextAttention(
            base_channels * 4, radius=attention_radius,
        )
        self.refine_decoder = GatedDecoder(base_channels * 8, base_channels)

    @staticmethod
    def _pad(inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = inputs.shape[-2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        return F.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate"), (height, width)

    def forward(
        self,
        t0_rgb: torch.Tensor,
        residual_hole_mask: torch.Tensor,
        retrieved_rgb: torch.Tensor,
        retrieval_confidence: torch.Tensor,
        semantic_condition: torch.Tensor,
        geometry_condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if residual_hole_mask.shape != t0_rgb[:, :1].shape:
            raise ValueError("residual hole mask must have shape B1HW")
        masked = t0_rgb * (1.0 - residual_hole_mask)
        condition = torch.cat([
            masked, residual_hole_mask, retrieved_rgb, retrieval_confidence,
            semantic_condition, geometry_condition,
        ], dim=1)
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} condition channels, got {condition.shape[1]}"
            )
        padded, size = self._pad(condition)
        coarse = self.coarse_decoder(self.coarse_bottleneck(self.coarse_encoder(padded)))

        hole = F.pad(
            residual_hole_mask,
            (0, padded.shape[-1] - size[1], 0, padded.shape[-2] - size[0]),
            mode="replicate",
        )
        coarse_composite = strict_mask_composite(padded[:, :3], coarse, hole)
        refine_input = torch.cat([padded, coarse_composite], dim=1)
        target_features = self.refine_encoder(refine_input)
        hallucinated = self.hallucination(target_features)

        reference_input = refine_input.clone()
        reference_input[:, :3] = padded[:, 4:7]
        reference_features = self.reference_encoder(reference_input)
        attended = self.context_attention(target_features, reference_features)
        refined = self.refine_decoder(torch.cat([hallucinated, attended], dim=1))

        coarse = coarse[..., :size[0], :size[1]]
        refined = refined[..., :size[0], :size[1]]
        final = strict_mask_composite(t0_rgb, refined, residual_hole_mask)
        return {"rgb": final, "coarse": coarse, "proposal": refined}


class DeepFillPatchDiscriminator(nn.Module):
    """Spectral-normalized PatchGAN used by DeepFill v2 training."""

    def __init__(self, input_channels: int = 4, base_channels: int = 48) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)
        layers: list[nn.Module] = []
        current = input_channels
        for output in channels:
            layers.extend([
                nn.utils.spectral_norm(nn.Conv2d(current, output, 5, stride=2, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            current = output
        layers.append(nn.utils.spectral_norm(nn.Conv2d(current, 1, 3, padding=1)))
        self.layers = nn.Sequential(*layers)

    def forward(self, rgb: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        if hole_mask.shape != rgb[:, :1].shape:
            raise ValueError("discriminator mask must have shape B1HW")
        return self.layers(torch.cat([rgb, hole_mask], dim=1))


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
