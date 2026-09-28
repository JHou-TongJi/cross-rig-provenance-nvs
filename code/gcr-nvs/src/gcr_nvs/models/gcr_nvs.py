"""Unified GCR-NVS Lite model output contract."""

from __future__ import annotations

import torch
from torch import nn

from gcr_nvs.models.refiner import SingleFrameRefiner


class GCRNVSModel(nn.Module):
    def __init__(self, geometry_channels: int = 12, ray_channels: int = 3, base_channels: int = 48):
        super().__init__()
        self.refiner = SingleFrameRefiner(48, base_channels)

    def forward(self, geometry, target_ray_map, target_rgb=None, source_provenance=None):
        if geometry.shape[1] == 12:
            geometry = torch.cat([geometry, (geometry[:, 7:8] <= 0).to(geometry.dtype)], dim=1)
        rendered_feature = geometry.new_zeros(geometry.shape[0], 32, geometry.shape[2], geometry.shape[3])
        refined = self.refiner(geometry, target_ray_map, rendered_feature=rendered_feature)
        output = {
            "coarse_rgb": geometry[:, :3],
            "depth": geometry[:, 3:4],
            "target_depth": geometry[:, 3:4],
            "normal": geometry[:, 4:7],
            "target_normal": geometry[:, 4:7],
            "opacity": geometry[:, 7:8],
            "confidence": geometry[:, 8:9],
            "view_variance": geometry[:, 9:10],
            "source_count": geometry[:, 10:11],
            "dynamic_mask": geometry[:, 11:12],
            **refined,
        }
        if source_provenance is not None:
            output["source_provenance"] = source_provenance
        return output
