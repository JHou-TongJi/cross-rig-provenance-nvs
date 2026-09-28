"""Semantic source retrieval for completing recoverable T0 holes.

The retriever never hallucinates RGB. It finds a topology-approved source
feature at each target location and copies the corresponding source RGB.
Uncertain matches remain invalid for the generative completion stage.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def strict_mask_composite(
    observed_rgb: torch.Tensor,
    proposal_rgb: torch.Tensor,
    editable_mask: torch.Tensor,
) -> torch.Tensor:
    """Composite without allowing the proposal to alter observed pixels."""
    if observed_rgb.shape != proposal_rgb.shape:
        raise ValueError("observed and proposal RGB tensors must have the same shape")
    if editable_mask.shape != observed_rgb[:, :1].shape:
        raise ValueError("editable mask must have shape B,1,H,W")
    return torch.where(editable_mask.bool(), proposal_rgb, observed_rgb)


def _shift(values: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``values[..., y + dy, x + dx]`` and return its valid support."""
    batch, _, height, width = values.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=values.dtype),
        torch.arange(width, device=values.device, dtype=values.dtype),
        indexing="ij",
    )
    grid_x = (grid_x + dx) * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = (grid_y + dy) * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(batch, -1, -1, -1)
    shifted = F.grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )[:, None]
    return shifted, valid


class LocalSemanticReferenceRetriever(nn.Module):
    """Search topology-approved references in a bounded semantic window.

    DINO descriptors should be supplied directly or through a deterministic
    projection. Search is intentionally local because the T0 target pose is a
    small SE(3) perturbation rather than an unrelated scene.
    """

    def __init__(
        self,
        radius: int = 8,
        stride: int = 2,
        minimum_confidence: float = 0.58,
        margin_scale: float = 12.0,
    ) -> None:
        super().__init__()
        if radius < 0 or stride < 1:
            raise ValueError("radius must be non-negative and stride must be positive")
        self.radius = int(radius)
        self.stride = int(stride)
        self.minimum_confidence = float(minimum_confidence)
        self.margin_scale = float(margin_scale)

    def forward(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
        reference_rgb: torch.Tensor,
        hole_mask: torch.Tensor,
        reference_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if query_features.ndim != 4 or reference_features.ndim != 5:
            raise ValueError("features must have BCHW and BVCHW shapes")
        batch, views, channels, height, width = reference_features.shape
        if query_features.shape != (batch, channels, height, width):
            raise ValueError("query and reference feature shapes do not match")
        if reference_rgb.shape != (batch, views, 3, height, width):
            raise ValueError("reference RGB must have shape BV3HW")
        if hole_mask.shape != (batch, 1, height, width):
            raise ValueError("hole mask must have shape B1HW")
        if reference_validity is None:
            reference_validity = torch.ones(
                (batch, views, 1, height, width),
                device=query_features.device,
                dtype=torch.bool,
            )
        if reference_validity.shape != (batch, views, 1, height, width):
            raise ValueError("reference validity must have shape BV1HW")

        query = F.normalize(query_features.float(), dim=1, eps=1e-6)
        references = F.normalize(reference_features.float(), dim=2, eps=1e-6)
        best = torch.full((batch, 1, height, width), -2.0, device=query.device)
        second = torch.full_like(best, -2.0)
        best_view = torch.full((batch, 1, height, width), -1, device=query.device, dtype=torch.long)
        best_dx = torch.zeros_like(best_view)
        best_dy = torch.zeros_like(best_view)
        retrieved = torch.zeros((batch, 3, height, width), device=query.device, dtype=reference_rgb.dtype)

        offsets = range(-self.radius, self.radius + 1, self.stride)
        for view in range(views):
            for dy in offsets:
                for dx in offsets:
                    shifted_feature, inside = _shift(references[:, view], dx, dy)
                    shifted_valid, _ = _shift(reference_validity[:, view].float(), dx, dy)
                    candidate_valid = inside & (shifted_valid > 0.5)
                    score = (query * shifted_feature).sum(dim=1, keepdim=True)
                    score = torch.where(candidate_valid, score, score.new_full((), -2.0))
                    replace = score > best
                    second = torch.where(replace, best, torch.maximum(second, score))
                    best = torch.where(replace, score, best)
                    if replace.any():
                        shifted_rgb, _ = _shift(reference_rgb[:, view], dx, dy)
                        retrieved = torch.where(replace.expand_as(retrieved), shifted_rgb, retrieved)
                        best_view = torch.where(replace, best_view.new_full((), view), best_view)
                        best_dx = torch.where(replace, best_dx.new_full((), dx), best_dx)
                        best_dy = torch.where(replace, best_dy.new_full((), dy), best_dy)

        similarity = ((best + 1.0) * 0.5).clamp(0.0, 1.0)
        margin = torch.sigmoid((best - second) * self.margin_scale)
        confidence = similarity * margin
        recoverable = hole_mask.bool() & (best_view >= 0) & (confidence >= self.minimum_confidence)
        return {
            "retrieved_rgb": retrieved,
            "confidence": confidence,
            "similarity": similarity,
            "source_index": best_view,
            "offset_x": best_dx,
            "offset_y": best_dy,
            "recoverable_mask": recoverable,
            "residual_mask": hole_mask.bool() & ~recoverable,
        }
