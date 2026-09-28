"""Trainable confidence calibration for frozen DINO/AnyUP retrieval."""

from __future__ import annotations

import torch
from torch import nn


class SemanticCorrespondenceHead(nn.Module):
    """Predict whether a geometry/semantic source match is safe to copy.

    The backbone and deterministic search remain frozen. The head only sees
    retrieval diagnostics, so it cannot invent RGB or alter observed T0 data.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start near the hand-designed gate while allowing calibration.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dimension {self.input_dim}, got {features.shape[-1]}")
        return self.network(features).squeeze(-1)
