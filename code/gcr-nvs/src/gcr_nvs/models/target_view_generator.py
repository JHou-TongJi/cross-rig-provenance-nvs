"""Generative target-view decoder conditioned on 3D geometry and RGB tokens."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from gcr_nvs.models.restormer import RestormerSmallFeatureDecoder


class TargetViewGenerator(nn.Module):
    """Generate target RGB while preserving reliable source observations.

    The decoder receives rasterized 3D surface tokens, not a source image
    concatenation alone.  Source RGB is a color anchor on observed surfaces;
    geometry-supported holes are generated without a source-color shortcut.
    """

    def __init__(
        self,
        geometry_dim: int = 32,
        appearance_dim: int = 64,
        decoder_dim: int = 48,
        max_observed_change: float = 0.20,
        camera_count: int = 7,
        camera_embedding_dim: int = 16,
        camera_conditioning: bool = False,
    ) -> None:
        super().__init__()
        # geometry + appearance + depth + normal + confidences + ray + RGB
        # validity/weight = 32 + 64 + 1 + 3 + 2 + 3 + 3 + 2 = 110.
        self.camera_conditioning = bool(camera_conditioning)
        self.unknown_camera_index = int(camera_count)
        camera_channels = camera_embedding_dim + 2 if self.camera_conditioning else 0
        self.input_channels = (
            geometry_dim + appearance_dim + 1 + 3 + 2 + 3 + 3 + 2
            + camera_channels
        )
        self.target_camera_embedding = (
            nn.Embedding(camera_count + 1, camera_embedding_dim)
            if self.camera_conditioning else None
        )
        self.decoder = RestormerSmallFeatureDecoder(
            input_channels=self.input_channels,
            output_channels=decoder_dim,
            dim=decoder_dim,
            blocks=(2, 3, 3, 4),
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
        self.max_observed_change = float(max_observed_change)

    @staticmethod
    def _pad_to_eighth(inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = inputs.shape[-2:]
        padded_height = (height + 7) // 8 * 8
        padded_width = (width + 7) // 8 * 8
        padding = (0, padded_width - width, 0, padded_height - height)
        return F.pad(inputs, padding, mode="replicate"), (height, width)

    def forward(
        self,
        geometry_features: torch.Tensor,
        appearance_features: torch.Tensor,
        source_rgb: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
        structure_confidence: torch.Tensor,
        appearance_confidence: torch.Tensor,
        ray_map: torch.Tensor,
        source_validity: torch.Tensor,
        source_weight: torch.Tensor,
        target_camera_index: torch.Tensor | int | None = None,
        target_focal_role: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tensors = (
            geometry_features, appearance_features, source_rgb, depth, normal,
            structure_confidence, appearance_confidence, ray_map,
            source_validity, source_weight,
        )
        if any(value.ndim != 4 for value in tensors):
            raise ValueError("target generator inputs must be BCHW tensors")
        condition_parts = [
            geometry_features,
            appearance_features,
            depth,
            normal,
            structure_confidence,
            appearance_confidence,
            ray_map,
            source_rgb,
            source_validity,
            source_weight,
        ]
        if self.camera_conditioning:
            batch, _, height, width = geometry_features.shape
            if target_camera_index is None:
                camera_index = torch.full(
                    (batch,), self.unknown_camera_index,
                    dtype=torch.long, device=geometry_features.device,
                )
            else:
                camera_index = torch.as_tensor(
                    target_camera_index,
                    dtype=torch.long,
                    device=geometry_features.device,
                ).reshape(-1)
                if camera_index.numel() == 1 and batch > 1:
                    camera_index = camera_index.expand(batch)
                if camera_index.numel() != batch:
                    raise ValueError("target_camera_index must match batch size")
                camera_index = camera_index.clamp(0, self.unknown_camera_index)
            camera_map = self.target_camera_embedding(camera_index).to(
                geometry_features.dtype,
            )[:, :, None, None].expand(-1, -1, height, width)
            if target_focal_role is None:
                focal_role = geometry_features.new_zeros(batch, 2)
            else:
                focal_role = target_focal_role.to(
                    device=geometry_features.device,
                    dtype=geometry_features.dtype,
                ).reshape(batch, 2)
            focal_map = focal_role[:, :, None, None].expand(-1, -1, height, width)
            condition_parts.extend((camera_map, focal_map))
        condition = torch.cat(condition_parts, dim=1)
        if condition.shape[1] != self.input_channels:
            raise ValueError(
                f"target generator expected {self.input_channels} channels, "
                f"got {condition.shape[1]}"
            )
        padded, original_size = self._pad_to_eighth(condition)
        decoded = self.decoder(padded)
        height, width = original_size
        decoded = decoded[..., :height, :width]
        generated_rgb = torch.sigmoid(self.rgb_head(decoded))
        alpha = torch.sigmoid(self.alpha_head(decoded))
        logvar = self.logvar_head(decoded).clamp(-6.0, 2.0)

        # Keep observed source colors as the default.  The generator has a
        # bounded correction budget on observed surfaces and full authority
        # only where geometry has no RGB observation.
        source_validity = source_validity.clamp(0.0, 1.0)
        effective_alpha = alpha * (
            1.0 - source_validity + 0.20 * source_validity
        )
        observed = (
            source_rgb + self.max_observed_change
            * torch.tanh(generated_rgb - 0.5) * effective_alpha
        ).clamp(0.0, 1.0)
        rgb = torch.where(
            source_validity > 0.5,
            observed,
            generated_rgb,
        )
        return {
            "rgb": rgb,
            "generated_rgb": generated_rgb,
            "alpha": effective_alpha,
            "logvar": logvar,
        }
