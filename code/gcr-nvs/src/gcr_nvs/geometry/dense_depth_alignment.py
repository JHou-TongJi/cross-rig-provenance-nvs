"""Metric alignment and fusion of DA3 dense depth with sparse LiDAR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthAlignmentResult:
    depth: np.ndarray
    confidence: np.ndarray
    aligned_monocular_depth: np.ndarray
    lidar_validity: np.ndarray
    monocular_validity: np.ndarray
    scale: float
    shift: float
    fit_count: int
    holdout: dict[str, float | int]
    lidar_anchor_depth: np.ndarray | None = None


def _robust_affine(x: np.ndarray, y: np.ndarray, iterations: int = 8) -> tuple[float, float]:
    A = np.column_stack([x, np.ones_like(x)])
    weights = np.ones_like(x)
    scale = 1.0
    for _ in range(iterations):
        Aw = A * np.sqrt(weights[:, None])
        yw = y * np.sqrt(weights)
        a, b = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        residual = y - (a * x + b)
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        scale = max(float(1.4826 * mad), 1e-4)
        cutoff = 1.5 * scale
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        large = absolute > cutoff
        weights[large] = cutoff / np.maximum(absolute[large], 1e-8)
    return float(a), float(b)


def align_monocular_depth(
    monocular_depth: np.ndarray,
    lidar_depth: np.ndarray,
    monocular_confidence: np.ndarray | None = None,
    *,
    holdout_fraction: float = 0.1,
    min_fit_points: int = 24,
    confidence_threshold: float = 0.02,
    seed: int = 20260820,
    hard_insert_lidar: bool = False,
) -> DepthAlignmentResult:
    """Align a continuous monocular depth map using sparse LiDAR constraints.

    The default output stays on the continuous DA3 surface. LiDAR is used for
    metric scale/shift fitting and is returned separately for visibility and
    diagnostics. ``hard_insert_lidar`` is retained only for legacy ablations.
    """
    mono = np.asarray(monocular_depth, dtype=np.float32)
    lidar = np.asarray(lidar_depth, dtype=np.float32)
    if mono.ndim != 2 or lidar.shape != mono.shape:
        raise ValueError("monocular_depth and lidar_depth must have matching [H,W] shapes")
    mono_valid = np.isfinite(mono) & (mono > 1e-4)
    lidar_valid = np.isfinite(lidar) & (lidar > 1e-4)
    if monocular_confidence is None:
        confidence = np.ones_like(mono, dtype=np.float32)
    else:
        confidence = np.asarray(monocular_confidence, dtype=np.float32)
        if confidence.shape != mono.shape:
            raise ValueError("monocular_confidence must match depth shape")
        confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = np.clip(confidence, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    anchors = np.flatnonzero(mono_valid & lidar_valid)
    if anchors.size:
        rng.shuffle(anchors)
    holdout_n = int(round(anchors.size * float(np.clip(holdout_fraction, 0.0, 0.9))))
    holdout = anchors[:holdout_n]
    fit = anchors[holdout_n:]
    if fit.size < min_fit_points:
        fit = anchors
        holdout = np.empty(0, dtype=np.int64)
    if fit.size < min_fit_points:
        aligned = mono.copy()
        continuous = np.where(mono_valid, aligned, 0.0).astype(np.float32)
        fused = np.where(lidar_valid, lidar, continuous).astype(np.float32) if hard_insert_lidar else continuous
        return DepthAlignmentResult(
            fused, confidence * mono_valid, aligned, lidar_valid, mono_valid,
            1.0, 0.0, int(fit.size), {"count": int(holdout.size), "mae_m": float("nan"), "rmse_m": float("nan")},
            lidar.copy(),
        )
    inv_mono = 1.0 / np.maximum(mono.reshape(-1)[fit], 1e-4)
    inv_lidar = 1.0 / np.maximum(lidar.reshape(-1)[fit], 1e-4)
    scale, shift = _robust_affine(inv_mono.astype(np.float64), inv_lidar.astype(np.float64))
    aligned_inv = scale / np.maximum(mono, 1e-4) + shift
    aligned = np.where(aligned_inv > 1e-6, 1.0 / np.maximum(aligned_inv, 1e-6), 0.0).astype(np.float32)
    aligned[~mono_valid] = 0.0
    holdout_pred = aligned.reshape(-1)[holdout] if holdout.size else np.empty(0, np.float32)
    holdout_gt = lidar.reshape(-1)[holdout] if holdout.size else np.empty(0, np.float32)
    if holdout.size:
        error = holdout_pred.astype(np.float64) - holdout_gt.astype(np.float64)
        metrics = {
            "count": int(holdout.size),
            "mae_m": float(np.mean(np.abs(error))),
            "rmse_m": float(np.sqrt(np.mean(error * error))),
            "absrel": float(np.mean(np.abs(error) / np.maximum(holdout_gt, 1e-6))),
        }
    else:
        metrics = {"count": 0, "mae_m": float("nan"), "rmse_m": float("nan"), "absrel": float("nan")}
    fill = mono_valid & (confidence >= float(confidence_threshold))
    continuous = np.where(fill, aligned, 0.0).astype(np.float32)
    fused = np.where(lidar_valid, lidar, continuous).astype(np.float32) if hard_insert_lidar else continuous
    fused_confidence = np.where(lidar_valid, 1.0, np.where(fill, confidence, 0.0)).astype(np.float32)
    return DepthAlignmentResult(
        fused, fused_confidence, aligned, lidar_valid, mono_valid,
        float(scale), float(shift), int(fit.size), metrics, lidar.copy(),
    )
