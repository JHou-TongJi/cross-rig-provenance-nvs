"""Geometry-conditioned neural radiance completion for unobserved target rays."""

from __future__ import annotations

import torch
from torch import nn


class NeuralRadianceCompletion(nn.Module):
    """Small conditional radiance field used only as a hole proposal."""

    def __init__(self, context_dim: int = 64, hidden_dim: int = 128, frequencies: int = 6):
        super().__init__()
        self.frequencies = frequencies
        ray_dim = 3 * (1 + 2 * frequencies)
        input_dim = ray_dim + context_dim + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.rgb = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 3))
        self.logvar = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, ray_map: torch.Tensor, geometry: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = ray_map.shape
        rays = ray_map.permute(0, 2, 3, 1)
        encoded = [rays]
        for index in range(self.frequencies):
            frequency = 2.0 ** index
            encoded.extend((torch.sin(frequency * rays), torch.cos(frequency * rays)))
        context_map = context[:, None, None, :].expand(batch, height, width, -1)
        geometry_cues = torch.cat((geometry[:, 3:4], geometry[:, 7:8], geometry[:, 8:9]), dim=1).permute(0, 2, 3, 1)
        features = self.mlp(torch.cat((*encoded, context_map, geometry_cues), dim=-1))
        return {
            "rgb": torch.sigmoid(self.rgb(features)).permute(0, 3, 1, 2),
            "log_uncertainty": self.logvar(features).clamp(-6.0, 2.0).permute(0, 3, 1, 2),
        }
