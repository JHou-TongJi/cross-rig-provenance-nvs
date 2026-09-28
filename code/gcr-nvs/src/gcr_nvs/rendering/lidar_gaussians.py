"""Deterministic gsplat rendering for fixed LiDAR-supported surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from gcr_nvs.geometry.calibration import CameraCalibration, project_world_points

try:
    from gsplat import rasterization
except ImportError:  # pragma: no cover
    rasterization = None


@dataclass(frozen=True)
class GaussianRenderResult:
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    appearance_validity: torch.Tensor
    point_count: int


@dataclass(frozen=True)
class GaussianFeatureRenderResult:
    features: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    appearance_validity: torch.Tensor
    point_count: int


def _distorted_to_ideal_grid(
    calibration: CameraCalibration,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not np.any(calibration.distortion):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1)[None]
    resized = calibration.intrinsic.copy()
    resized[0] *= width / calibration.width
    resized[1] *= height / calibration.height
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    distorted = np.stack([xx, yy], axis=-1).astype(np.float64).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(
        distorted,
        resized,
        calibration.distortion,
        P=resized,
    ).reshape(height, width, 2)
    grid = np.stack([
        ideal[..., 0] / max(width - 1, 1) * 2.0 - 1.0,
        ideal[..., 1] / max(height - 1, 1) * 2.0 - 1.0,
    ], axis=-1)
    return torch.from_numpy(grid).to(device=device, dtype=dtype)[None]


def render_fixed_lidar_gaussians(
    points: torch.Tensor,
    colors: torch.Tensor,
    appearance_validity: torch.Tensor,
    calibration: CameraCalibration,
    width: int,
    height: int,
    source_count: torch.Tensor | None = None,
    pixel_radius: float = 0.85,
    alpha_threshold: float = 0.02,
) -> GaussianRenderResult:
    result = render_fixed_lidar_features(
        points=points,
        features=colors.clamp(0.0, 1.0),
        appearance_validity=appearance_validity,
        calibration=calibration,
        width=width,
        height=height,
        source_count=source_count,
        pixel_radius=pixel_radius,
        alpha_threshold=alpha_threshold,
    )
    return GaussianRenderResult(
        rgb=result.features,
        depth=result.depth,
        alpha=result.alpha,
        appearance_validity=result.appearance_validity,
        point_count=result.point_count,
    )


def render_fixed_lidar_features(
    points: torch.Tensor,
    features: torch.Tensor,
    appearance_validity: torch.Tensor,
    calibration: CameraCalibration,
    width: int,
    height: int,
    source_count: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
    geometry_confidence: torch.Tensor | None = None,
    pixel_radius: float | torch.Tensor = 0.85,
    alpha_threshold: float = 0.02,
) -> GaussianFeatureRenderResult:
    """Rasterize a learned surface feature field at immutable LiDAR centers."""
    if rasterization is None:
        raise RuntimeError("gsplat is required for Gaussian rendering")
    if isinstance(pixel_radius, torch.Tensor):
        if pixel_radius.shape != (len(points),):
            raise ValueError("per-point pixel_radius must have shape [N]")
        radius_values = pixel_radius[appearance_validity.bool()].float()
    else:
        radius_values = float(pixel_radius)
    valid = appearance_validity.bool().clone()
    valid = valid & torch.isfinite(points).all(dim=1) & torch.isfinite(features).all(dim=1)
    if isinstance(pixel_radius, torch.Tensor):
        radius_values = pixel_radius[valid].float()
    points = points[valid].float()
    features = features[valid].float()
    if not len(points):
        shape = (height, width)
        zeros = features.new_zeros
        return GaussianFeatureRenderResult(
            features=zeros(features.shape[-1], *shape),
            depth=zeros(1, *shape),
            alpha=zeros(1, *shape),
            appearance_validity=torch.zeros(1, *shape, dtype=torch.bool, device=features.device),
            point_count=0,
        )
    device = points.device
    external = points.new_tensor(calibration.external)
    homogeneous = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)
    camera_points = (external @ homogeneous.T).T[:, :3]
    resized_intrinsic = calibration.intrinsic.copy()
    resized_intrinsic[0] *= width / calibration.width
    resized_intrinsic[1] *= height / calibration.height
    focal = float(0.5 * (resized_intrinsic[0, 0] + resized_intrinsic[1, 1]))
    radius = (
        camera_points[:, 2].clamp(min=0.1)
        / max(focal, 1e-6)
        * radius_values
    )
    radius = radius.clamp(min=0.025, max=0.50)
    scales = torch.stack([radius, radius, radius * 0.5], dim=1)
    quaternions = points.new_zeros(len(points), 4)
    quaternions[:, 0] = 1.0
    if source_count is None:
        opacities = points.new_full((len(points),), 0.92)
    else:
        counts = source_count[valid].float()
        opacities = (0.70 + 0.08 * counts).clamp(max=0.96)
    if visibility is not None:
        opacities = opacities * visibility[valid].float().clamp(0.05, 1.0)
    if geometry_confidence is not None:
        opacities = opacities * geometry_confidence[valid].float().clamp(0.05, 1.0)
    renders, alpha, _ = rasterization(
        means=points,
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        colors=features,
        viewmats=external[None],
        Ks=points.new_tensor(resized_intrinsic)[None],
        width=width,
        height=height,
        near_plane=0.1,
        far_plane=120.0,
        render_mode="RGB+ED",
        packed=True,
        rasterize_mode="antialiased",
    )
    feature_dim = features.shape[-1]
    rendered_features = renders[..., :feature_dim].permute(0, 3, 1, 2)
    depth = renders[..., feature_dim:feature_dim + 1].permute(0, 3, 1, 2)
    alpha = alpha.permute(0, 3, 1, 2)
    grid = _distorted_to_ideal_grid(
        calibration, width, height, device, rendered_features.dtype,
    )
    rendered_features = F.grid_sample(
        rendered_features, grid, align_corners=True, padding_mode="zeros",
    )
    depth = F.grid_sample(depth, grid, align_corners=True, padding_mode="zeros")
    alpha = F.grid_sample(alpha, grid, align_corners=True, padding_mode="zeros")
    validity = alpha > alpha_threshold
    # gsplat features are alpha-premultiplied. Keep alpha as a separate
    # visibility signal and recover the surface feature before decoding RGB.
    rendered_features = rendered_features / alpha.clamp(min=1e-6)
    rendered_features = torch.where(
        validity, rendered_features, torch.zeros_like(rendered_features),
    )
    depth = torch.where(validity, depth, torch.zeros_like(depth))
    return GaussianFeatureRenderResult(
        features=rendered_features[0],
        depth=depth[0],
        alpha=alpha[0],
        appearance_validity=validity[0],
        point_count=len(points),
    )


def estimate_adaptive_pixel_radius(
    points: torch.Tensor,
    calibration: CameraCalibration,
    neighbor_rank: int = 3,
    spacing_scale: float = 0.60,
    minimum: float = 3.0,
    maximum: float = 18.0,
) -> torch.Tensor:
    """Estimate a per-surfel footprint from native-image point spacing."""
    if neighbor_rank < 1 or not (0.0 < minimum <= maximum):
        raise ValueError("invalid adaptive footprint parameters")
    points_np = points.detach().cpu().numpy().astype(np.float32)
    uv, projected = project_world_points(points_np, calibration, undistort=False)
    radii = np.full(len(points_np), maximum, dtype=np.float32)
    valid_indices = np.flatnonzero(projected)
    if len(valid_indices) > neighbor_rank:
        distances, _ = cKDTree(uv[valid_indices]).query(
            uv[valid_indices], k=neighbor_rank + 1,
        )
        local_spacing = distances[:, neighbor_rank]
        radii[valid_indices] = np.clip(
            local_spacing * spacing_scale, minimum, maximum,
        ).astype(np.float32)
    return torch.from_numpy(radii).to(device=points.device, dtype=points.dtype)
