"""GCR-NVS Foundation single-frame model."""

from __future__ import annotations

import torch
from torch import nn

from gcr_nvs.models.anchor_pipeline import LiDARAnchoredViewFusion
from gcr_nvs.models.refiner import LiteTemporalRefiner
from gcr_nvs.models.restormer import RestormerFeatureAdapter
from gcr_nvs.models.neural_field import NeuralRadianceCompletion


class GCRNVSFoundationSingleFrame(nn.Module):
    """DINO/source-view anchor fusion followed by controlled residual repair."""

    def __init__(self, use_dino: bool = True, base_channels: int = 48):
        super().__init__()
        self.view_fusion = LiDARAnchoredViewFusion(use_dino=use_dino)
        self.restormer = RestormerFeatureAdapter(115, 32, channels=48, blocks=2)
        self.refiner = LiteTemporalRefiner(48, base_channels, max_rgb_residual=0.25)
        self.radiance = NeuralRadianceCompletion(context_dim=64)
        self.completion_gate = nn.Sequential(nn.Conv2d(32 + 3, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 1, 3, padding=1))
        self.completion_rgb = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())

    def forward(
        self,
        geometry: torch.Tensor,
        ray_map: torch.Tensor,
        source_images: torch.Tensor,
        anchor_xyz: torch.Tensor,
        source_uv: torch.Tensor,
        target_uv: torch.Tensor,
        target_anchor_valid: torch.Tensor,
        source_valid: torch.Tensor,
        plane_sweep_uv: torch.Tensor | None = None,
        plane_sweep_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fusion = self.view_fusion(
            source_images,
            anchor_xyz,
            source_uv,
            target_uv,
            target_anchor_valid,
            source_valid,
            geometry.shape[-2:],
            plane_sweep_uv=plane_sweep_uv,
            plane_sweep_valid=plane_sweep_valid,
        )
        dense_source_context = fusion["dense_context"]
        restormer_input = torch.cat([geometry, ray_map, fusion["rendered_feature"], dense_source_context, fusion["dense_rgb"]], dim=1)
        restored_feature = self.restormer(restormer_input)
        refined = self.refiner(geometry, ray_map, restored_feature)
        field = self.radiance(ray_map, geometry, fusion["global_context"])
        learned_completion = self.completion_rgb(torch.cat((field["rgb"], fusion["dense_rgb"]), dim=1))
        projected_completion = (0.75 * fusion["dense_rgb"] + 0.25 * learned_completion).clamp(0.0, 1.0)
        completion_rgb = fusion["dense_valid"] * projected_completion + (1.0 - fusion["dense_valid"]) * field["rgb"]
        hole = 1.0 - geometry[:, 7:8].clamp(0.0, 1.0)
        gate = (0.35 + 0.65 * torch.sigmoid(self.completion_gate(torch.cat((restored_feature, geometry[:, :3]), dim=1)))) * hole
        rgb = ((1.0 - gate) * refined["rgb"] + gate * completion_rgb).clamp(0.0, 1.0)
        return {**fusion, "restored_feature": restored_feature, **refined, "radiance_rgb": completion_rgb, "radiance_log_uncertainty": field["log_uncertainty"], "completion_gate": gate, "rgb": rgb, "final_rgb": rgb}
