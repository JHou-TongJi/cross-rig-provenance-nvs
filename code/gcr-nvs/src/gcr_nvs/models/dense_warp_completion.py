"""Controlled RGB completion on top of a dense geometric warp."""

from __future__ import annotations

import torch
from torch import nn

from gcr_nvs.models.restormer import RestormerSmallFeatureDecoder


class DenseWarpCompletionGenerator(nn.Module):
    """Refine a dense RGB warp without repainting reliable observed pixels.

    The network receives source RGB, AnyUP semantic features, dense depth,
    visibility/confidence and target rays. Geometry is already fixed before
    this module; its output is only an RGB residual and an uncertainty head.
    """

    def __init__(
        self,
        semantic_dim: int = 64,
        decoder_dim: int = 64,
        max_observed_change: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_observed_change = float(max_observed_change)
        self.input_channels = 3 + semantic_dim + 1 + 1 + 1 + 3
        self.decoder = RestormerSmallFeatureDecoder(
            input_channels=self.input_channels,
            output_channels=decoder_dim,
            dim=decoder_dim,
            blocks=(2, 2, 3, 3),
            refinement_blocks=2,
            heads=(1, 2, 4, 8),
            use_checkpoint=True,
        )
        self.rgb_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_dim, 3, 3, padding=1),
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, 1, 3, padding=1),
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, 1, 3, padding=1),
        )

    @staticmethod
    def _pad(inputs: torch.Tensor):
        height, width = inputs.shape[-2:]
        padded_height = (height + 7) // 8 * 8
        padded_width = (width + 7) // 8 * 8
        return torch.nn.functional.pad(
            inputs, (0, padded_width - width, 0, padded_height - height), mode="replicate",
        ), (height, width)

    def forward(
        self,
        warped_rgb: torch.Tensor,
        semantic_features: torch.Tensor,
        depth: torch.Tensor,
        validity: torch.Tensor,
        confidence: torch.Tensor,
        ray_map: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = (warped_rgb, semantic_features, depth, validity, confidence, ray_map)
        if any(value.ndim != 4 for value in inputs):
            raise ValueError("dense completion inputs must be BCHW")
        condition = torch.cat(inputs, dim=1)
        if condition.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} channels, got {condition.shape[1]}")
        padded, size = self._pad(condition)
        decoded = self.decoder(padded)[..., :size[0], :size[1]]
        proposal = torch.sigmoid(self.rgb_head(decoded))
        alpha = torch.sigmoid(self.alpha_head(decoded))
        logvar = self.logvar_head(decoded).clamp(-6.0, 2.0)
        observed = (
            warped_rgb
            + self.max_observed_change * torch.tanh(proposal - 0.5) * alpha
        ).clamp(0.0, 1.0)
        rgb = torch.where(validity > 0.5, observed, proposal)
        return {
            "rgb": rgb,
            "proposal": proposal,
            "alpha": alpha,
            "logvar": logvar,
        }

