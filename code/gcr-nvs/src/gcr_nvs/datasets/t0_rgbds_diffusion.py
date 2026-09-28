"""Leakage-safe data contracts for residual T0 RGB-D-S inpainting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RGBDSReference:
    sequence: str
    frame: int
    camera: str
    rgb_path: Path
    depth_path: Path
    dynamic_path: Path
    semantic_path: Path
    hole_path: Path


def load_rgbds_references(manifest: Path) -> list[RGBDSReference]:
    root = manifest.parent
    rows = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    return [RGBDSReference(
        sequence=row["sequence"], frame=int(row["frame"]), camera=row["camera"],
        rgb_path=root / row["rgb"], depth_path=root / row["depth"],
        dynamic_path=root / row["dynamic"], semantic_path=root / row["semantic"],
        hole_path=root / row["hole"],
    ) for row in rows]


def resize_hole_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Keep any high-resolution invalid support when reducing to training size."""
    fraction = cv2.resize(
        np.asarray(mask, np.float32), size, interpolation=cv2.INTER_AREA,
    )
    return fraction > 0.0


def hole_topology_sampling_weight(
    mask: np.ndarray,
    *,
    area_weight: float = 3.0,
    component_weight: float = 2.0,
    edge_weight: float = 1.5,
) -> float:
    """Prioritize large, contiguous and image-edge holes during training."""
    hole = np.asarray(mask, bool)
    if not hole.any():
        return 1.0
    height, width = hole.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        hole.astype(np.uint8), connectivity=8,
    )
    largest = float(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0.0
    hole_pixels = float(hole.sum())
    area_score = min(hole_pixels / max(height * width * 0.04, 1.0), 1.0)
    component_score = largest / max(hole_pixels, 1.0)
    margin = max(2, min(height, width) // 32)
    edge = np.zeros_like(hole)
    edge[:margin] = True
    edge[-margin:] = True
    edge[:, :margin] = True
    edge[:, -margin:] = True
    edge_score = float((hole & edge).sum()) / max(hole_pixels, 1.0)
    return float(
        1.0
        + area_weight * area_score
        + component_weight * component_score
        + edge_weight * edge_score
    )


def hole_inner_boundary_mask(hole: torch.Tensor, width: int = 4) -> torch.Tensor:
    """Return only unknown pixels within ``width`` pixels of observed support."""
    if hole.ndim != 4 or hole.shape[1] != 1:
        raise ValueError("hole must have shape B,1,H,W")
    if width < 1:
        raise ValueError("boundary width must be positive")
    value = hole.float().clamp(0.0, 1.0)
    observed_nearby = F.max_pool2d(
        1.0 - value, kernel_size=2 * width + 1, stride=1, padding=width,
    )
    return value * (observed_nearby > 0.0).to(value.dtype)


def masked_rgb_gradient_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Measure RGB gradient disagreement on pixel pairs touching ``mask``."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share B,C,H,W")
    if mask.shape != (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]):
        raise ValueError("mask must have shape B,1,H,W")
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_x = torch.maximum(mask[..., :, 1:], mask[..., :, :-1])
    mask_y = torch.maximum(mask[..., 1:, :], mask[..., :-1, :])
    loss_x = ((pred_dx - target_dx).abs() * mask_x).sum() / (
        mask_x.sum() * prediction.shape[1]
    ).clamp_min(1.0)
    loss_y = ((pred_dy - target_dy).abs() * mask_y).sum() / (
        mask_y.sum() * prediction.shape[1]
    ).clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def build_rgbds_conditions(
    target_rgb: np.ndarray,
    target_depth_m: np.ndarray,
    dynamic_mask: np.ndarray,
    semantic_features: np.ndarray,
    hole_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build inference-available conditions without exposing hole contents."""
    rgb = np.asarray(target_rgb, np.float32)
    depth = np.asarray(target_depth_m, np.float32)
    dynamic = np.asarray(dynamic_mask, bool)
    semantic = np.asarray(semantic_features, np.float32)
    hole = np.asarray(hole_mask, bool)
    height, width = depth.shape
    if rgb.shape != (height, width, 3) or dynamic.shape != depth.shape:
        raise ValueError("RGB, depth and dynamic mask must share H,W")
    if semantic.ndim != 3 or semantic.shape[1:] != depth.shape:
        raise ValueError("semantic features must have shape C,H,W")

    observed = ~hole
    masked_rgb = rgb.copy()
    masked_rgb[hole] = 0.0
    valid_depth = observed & np.isfinite(depth) & (depth > 0.1)
    log_depth = np.zeros_like(depth, np.float32)
    log_depth[valid_depth] = (
        np.log1p(np.clip(depth[valid_depth], 0.0, 160.0)) / np.log(161.0)
    )
    gx = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, 3) / 4.0
    gy = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, 3) / 4.0
    gx[hole] = 0.0
    gy[hole] = 0.0

    semantic = semantic.copy()
    semantic[:, hole] = 0.0
    observed_dynamic = dynamic & observed
    # A dynamic foreground may not spread into an adjacent unknown region.
    dynamic_dilated = cv2.dilate(
        observed_dynamic.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1,
    ).astype(bool)
    forbid_foreground_extension = hole & dynamic_dilated
    yy, xx = np.indices((height, width), dtype=np.float32)
    ray_x = xx / max(width - 1, 1) * 2.0 - 1.0
    ray_y = yy / max(height - 1, 1) * 2.0 - 1.0
    control = np.concatenate([
        log_depth[None], valid_depth[None].astype(np.float32),
        gx[None], gy[None], semantic,
        observed_dynamic[None].astype(np.float32),
        forbid_foreground_extension[None].astype(np.float32),
        ray_x[None], ray_y[None],
    ], axis=0).astype(np.float32)
    return {
        "target_rgb": rgb,
        "masked_rgb": masked_rgb,
        "hole_mask": hole.astype(np.float32),
        "control": control,
        "forbid_foreground_extension": forbid_foreground_extension.astype(np.float32),
        "target_dynamic_mask": dynamic.astype(np.float32),
    }


def assert_no_hole_leakage(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray],
) -> None:
    """Prove that changing GT inside a hole cannot change model conditions."""
    for key in ("masked_rgb", "hole_mask", "control", "forbid_foreground_extension"):
        if not np.array_equal(first[key], second[key]):
            raise AssertionError(f"hole GT leaked into condition tensor: {key}")


class T0RGBDSDiffusionDataset(Dataset):
    """Read fixed pilot artifacts produced from sequence-disjoint references."""

    def __init__(self, references: list[RGBDSReference]) -> None:
        self.references = list(references)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        reference = self.references[index]
        rgb = np.asarray(Image.open(reference.rgb_path).convert("RGB"), np.float32) / 255.0
        depth = np.load(reference.depth_path).astype(np.float32)
        dynamic = np.load(reference.dynamic_path).astype(bool)
        semantic = np.load(reference.semantic_path).astype(np.float32)
        hole = np.load(reference.hole_path).astype(bool)
        conditions = build_rgbds_conditions(rgb, depth, dynamic, semantic, hole)
        return {
            "target_rgb": torch.from_numpy(conditions["target_rgb"].transpose(2, 0, 1).copy()),
            "masked_rgb": torch.from_numpy(conditions["masked_rgb"].transpose(2, 0, 1).copy()),
            "hole_mask": torch.from_numpy(conditions["hole_mask"][None].copy()),
            "control": torch.from_numpy(conditions["control"].copy()),
            "forbid_foreground_extension": torch.from_numpy(
                conditions["forbid_foreground_extension"][None].copy()
            ),
            "target_dynamic_mask": torch.from_numpy(
                conditions["target_dynamic_mask"][None].copy()
            ),
            "sequence": reference.sequence,
            "frame": reference.frame,
            "camera": reference.camera,
        }
