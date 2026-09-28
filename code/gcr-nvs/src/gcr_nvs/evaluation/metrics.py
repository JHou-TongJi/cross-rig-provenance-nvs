"""Image metrics used for coarse-vs-final comparisons."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    error = (pred - target) ** 2
    if mask is not None:
        expanded_mask = mask.to(error.dtype).expand_as(error)
        error = error * expanded_mask
        mse = error.sum() / expanded_mask.sum().clamp(min=1.0)
    else:
        mse = error.mean()
    return float(10.0 * math.log10(1.0 / max(float(mse), 1e-10)))


def ssim_proxy(pred: torch.Tensor, target: torch.Tensor) -> float:
    mu_x = F.avg_pool2d(pred, 11, 1, 5)
    mu_y = F.avg_pool2d(target, 11, 1, 5)
    sigma_x = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return float(score.mean())


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return psnr(pred, target, mask.expand_as(pred))


def depth_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = (mask > 0) & (pred > 0) & (target > 0)
    if not valid.any():
        return {"mae": float("nan"), "rmse": float("nan"), "count": 0.0}
    difference = pred[valid] - target[valid]
    return {"mae": float(difference.abs().mean()), "rmse": float(torch.sqrt((difference**2).mean())), "count": float(valid.sum())}


def lpips_score(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    try:
        import lpips
    except ImportError:
        return None
    network = lpips.LPIPS(net="alex").to(pred.device).eval()
    with torch.no_grad():
        value = network(pred * 2 - 1, target * 2 - 1)
    return float(value.mean())


def uncertainty_metrics(pred: torch.Tensor, target: torch.Tensor, log_variance: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    """Report calibrated Gaussian residual statistics for the uncertainty head."""
    error = (pred - target).pow(2).mean(dim=1, keepdim=True)
    if mask is not None:
        mask = mask.to(error.dtype)
        error = error * mask
        count = mask.sum().clamp(min=1.0)
    else:
        count = torch.tensor(error.numel(), device=error.device, dtype=error.dtype)
    variance = log_variance.exp().clamp(min=1e-6, max=100.0)
    nll = 0.5 * (log_variance + error / variance)
    if mask is not None:
        nll = nll * mask
    return {
        "uncertainty_nll": float(nll.sum() / count),
        "uncertainty_abs_error": float(torch.sqrt(error).sum() / count),
        "uncertainty_mean_variance": float((variance * (mask if mask is not None else 1.0)).sum() / count),
    }


def temporal_warp_error(current: torch.Tensor, previous: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    """Measure frame-to-frame RGB change after an already aligned render."""
    difference = (current - previous).abs().mean(dim=1, keepdim=True)
    if valid is not None:
        valid = valid.to(difference.dtype)
        return float((difference * valid).sum() / valid.sum().clamp(min=1.0))
    return float(difference.mean())
