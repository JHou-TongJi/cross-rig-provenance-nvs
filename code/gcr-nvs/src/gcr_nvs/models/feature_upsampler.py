"""Optional AnyUp adapter for edge-aware semantic feature upsampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class FeatureUpsampler(nn.Module):
    """Select bilinear interpolation or frozen AnyUp with an explicit fallback."""

    def __init__(
        self,
        mode: str = "bilinear",
        checkpoint: Path | None = None,
        anyup_root: Path = Path("third_party/anyup"),
        q_chunk_size: int | None = 8192,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"bilinear", "anyup"}:
            raise ValueError("feature upsampling mode must be bilinear or anyup")
        self.mode = mode
        self.q_chunk_size = q_chunk_size
        self.anyup = None
        if mode == "anyup":
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError(f"AnyUp checkpoint does not exist: {checkpoint}")
            absolute_root = anyup_root.resolve()
            if not (absolute_root / "anyup/model.py").exists():
                raise FileNotFoundError(f"AnyUp source does not exist: {absolute_root}")
            sys.path.insert(0, str(absolute_root))
            try:
                from anyup.model import AnyUp
            finally:
                sys.path.pop(0)
            self.anyup = AnyUp()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.anyup.load_state_dict(state)
            if freeze:
                self.anyup.requires_grad_(False)
                self.anyup.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.anyup is not None and not any(parameter.requires_grad for parameter in self.anyup.parameters()):
            self.anyup.eval()
        return self

    def forward(
        self,
        normalized_rgb: torch.Tensor,
        low_resolution_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if self.mode == "bilinear":
            return F.interpolate(
                low_resolution_features,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.anyup(
            normalized_rgb,
            low_resolution_features,
            output_size=output_size,
            q_chunk_size=self.q_chunk_size,
        )
