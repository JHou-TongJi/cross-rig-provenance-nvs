"""Learned plane-sweep stereo for dense RGB reconstruction.

This module implements a full plane-sweep cost volume approach:
- 32-64 inverse depth hypothesis layers
- Multi-view photometric consistency
- Learnable 3D cost regularization
- Soft depth probability (not argmin)
- LiDAR depth prior integration
- Source view occlusion handling
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _distortion_coefficients(
    distortion: torch.Tensor | None,
    reference: torch.Tensor,
    batch: int,
) -> torch.Tensor:
    if distortion is None:
        return reference.new_zeros(batch, 5)
    coefficients = distortion.to(device=reference.device, dtype=torch.float32)
    if coefficients.ndim == 1:
        coefficients = coefficients[None]
    if coefficients.shape[0] == 1 and batch > 1:
        coefficients = coefficients.expand(batch, -1)
    if coefficients.shape[1] < 5:
        coefficients = F.pad(coefficients, (0, 5 - coefficients.shape[1]))
    return coefficients[:, :5]


def distort_normalized_points(
    points: torch.Tensor,
    distortion: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Brown-Conrady distortion to normalized camera coordinates."""
    batch = points.shape[0]
    coefficients = _distortion_coefficients(distortion, points, batch)
    k1, k2, p1, p2, k3 = coefficients.unbind(dim=1)
    x, y = points[:, 0], points[:, 1]
    r2 = x.square() + y.square()
    radial = 1.0 + k1[:, None] * r2 + k2[:, None] * r2.square() + k3[:, None] * r2.pow(3)
    delta_x = 2.0 * p1[:, None] * x * y + p2[:, None] * (r2 + 2.0 * x.square())
    delta_y = p1[:, None] * (r2 + 2.0 * y.square()) + 2.0 * p2[:, None] * x * y
    return torch.stack([x * radial + delta_x, y * radial + delta_y], dim=1)


def undistort_normalized_points(
    distorted_points: torch.Tensor,
    distortion: torch.Tensor | None,
    iterations: int = 10,
) -> torch.Tensor:
    """Invert Brown-Conrady distortion with a vectorized fixed-point solve."""
    batch = distorted_points.shape[0]
    coefficients = _distortion_coefficients(distortion, distorted_points, batch)
    k1, k2, p1, p2, k3 = coefficients.unbind(dim=1)
    distorted_x, distorted_y = distorted_points[:, 0], distorted_points[:, 1]
    x, y = distorted_x.clone(), distorted_y.clone()
    for _ in range(iterations):
        r2 = x.square() + y.square()
        radial = (
            1.0
            + k1[:, None] * r2
            + k2[:, None] * r2.square()
            + k3[:, None] * r2.pow(3)
        ).clamp(min=1e-4)
        delta_x = 2.0 * p1[:, None] * x * y + p2[:, None] * (r2 + 2.0 * x.square())
        delta_y = p1[:, None] * (r2 + 2.0 * y.square()) + 2.0 * p2[:, None] * x * y
        x = (distorted_x - delta_x) / radial
        y = (distorted_y - delta_y) / radial
    return torch.stack([x, y], dim=1)


def pixels_to_camera_rays(
    pixels: torch.Tensor,
    intrinsic: torch.Tensor,
    distortion: torch.Tensor | None,
) -> torch.Tensor:
    """Backproject distorted image pixels to ideal pinhole camera rays."""
    batch = intrinsic.shape[0]
    homogeneous = torch.linalg.inv(intrinsic.float()) @ pixels[None].expand(batch, -1, -1)
    distorted = homogeneous[:, :2] / homogeneous[:, 2:3].clamp(min=1e-6)
    ideal = undistort_normalized_points(distorted, distortion)
    return torch.cat([ideal, torch.ones_like(ideal[:, :1])], dim=1)


def project_camera_points(
    camera_points: torch.Tensor,
    intrinsic: torch.Tensor,
    distortion: torch.Tensor | None,
) -> torch.Tensor:
    """Project camera-space points through Brown distortion into pixels."""
    depth = camera_points[:, 2:3]
    normalized = camera_points[:, :2] / depth.clamp(min=1e-4)
    projectable = (
        (depth > 1e-4)
        & torch.isfinite(normalized).all(dim=1, keepdim=True)
        & (normalized.abs() <= 4.0).all(dim=1, keepdim=True)
    )
    safe_normalized = torch.where(projectable.expand_as(normalized), normalized, torch.zeros_like(normalized))
    distorted = distort_normalized_points(safe_normalized, distortion)
    distorted = torch.where(
        projectable.expand_as(distorted),
        distorted,
        torch.full_like(distorted, 1e6),
    )
    homogeneous = torch.cat([distorted, torch.ones_like(distorted[:, :1])], dim=1)
    return intrinsic.float() @ homogeneous


def inverse_depth_sampling(depth_min: float, depth_max: float, num_layers: int = 48) -> torch.Tensor:
    """Sample inverse depth to get denser sampling at far distances."""
    inv_depth_min = 1.0 / depth_max
    inv_depth_max = 1.0 / depth_min
    inv_depths = torch.linspace(inv_depth_min, inv_depth_max, num_layers)
    depths = 1.0 / inv_depths
    # Flip to get ascending depth order
    depths = torch.flip(depths, [0])
    return depths


def reliable_lidar_occlusion_depth(
    sparse_depth: torch.Tensor,
    kernel_size: int = 5,
    minimum_support: int = 3,
    maximum_relative_span: float = 0.08,
) -> torch.Tensor:
    """Build conservative source-view occlusion evidence from sparse LiDAR.

    An isolated LiDAR return is valid metric structure, but it is not enough
    evidence to invalidate a whole RGB sample. Only locally supported,
    depth-consistent front surfaces participate in source visibility tests.
    """
    depth = sparse_depth.float()
    valid = depth > 0
    padding = kernel_size // 2
    support = F.avg_pool2d(
        valid.to(depth.dtype), kernel_size, stride=1, padding=padding,
        count_include_pad=False,
    ) * float(kernel_size * kernel_size)
    local_mean = F.avg_pool2d(
        depth, kernel_size, stride=1, padding=padding, count_include_pad=False,
    ) * float(kernel_size * kernel_size) / support.clamp(min=1.0)
    local_max = F.max_pool2d(depth, kernel_size, stride=1, padding=padding)
    local_min = -F.max_pool2d(
        torch.where(valid, -depth, torch.full_like(depth, -float("inf"))),
        kernel_size,
        stride=1,
        padding=padding,
    )
    relative_span = (local_max - local_min) / local_mean.clamp(min=1.0)
    reliable = (
        (support >= minimum_support)
        & torch.isfinite(relative_span)
        & (relative_span <= maximum_relative_span)
    )
    return torch.where(reliable, local_mean, torch.zeros_like(local_mean))


def fuse_lidar_metric_depth(
    stereo_depth: torch.Tensor,
    stereo_confidence: torch.Tensor,
    lidar_depth: torch.Tensor | None,
    lidar_dynamic_mask: torch.Tensor | None = None,
    propagation_steps: int = 4,
    minimum_support: int = 2,
    maximum_relative_span: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use metric LiDAR as primary structure and stereo only as fallback.

    Exact static LiDAR z-buffer hits are hard constraints. Sparse depth is
    propagated only across locally consistent surfaces; depth discontinuities
    and dynamic pixels fall back to stereo.
    """
    if lidar_depth is None:
        zeros = torch.zeros_like(stereo_depth)
        return stereo_depth, stereo_confidence, zeros, zeros
    lidar = F.interpolate(
        lidar_depth.float(), stereo_depth.shape[-2:], mode="nearest",
    )
    static_valid = lidar > 0
    if lidar_dynamic_mask is not None:
        dynamic = F.interpolate(
            lidar_dynamic_mask.float(), stereo_depth.shape[-2:], mode="nearest",
        ) > 0.5
        static_valid = static_valid & ~dynamic
    completed = torch.where(static_valid, lidar, torch.zeros_like(lidar))
    completed_valid = static_valid.clone()
    lidar_reliability = static_valid.to(stereo_depth.dtype)
    propagation_distance = torch.zeros_like(stereo_depth)
    for distance in range(1, propagation_steps + 1):
        valid_float = completed_valid.to(stereo_depth.dtype)
        patches = F.unfold(completed, 3, padding=1).view(
            completed.shape[0], 9, *completed.shape[-2:],
        )
        patch_valid = F.unfold(valid_float, 3, padding=1).view_as(patches) > 0.5
        support = patch_valid.sum(dim=1, keepdim=True)
        positive = torch.where(patch_valid, patches, torch.full_like(patches, float("inf")))
        negative = torch.where(patch_valid, patches, torch.full_like(patches, -float("inf")))
        local_min = positive.amin(dim=1, keepdim=True)
        local_max = negative.amax(dim=1, keepdim=True)
        local_mean = (patches * patch_valid).sum(dim=1, keepdim=True) / support.clamp(min=1)
        relative_span = (local_max - local_min) / local_mean.clamp(min=1.0)
        new_valid = (
            ~completed_valid
            & (support >= minimum_support)
            & torch.isfinite(relative_span)
            & (relative_span <= maximum_relative_span)
        )
        if not new_valid.any():
            break
        stability = (1.0 - relative_span / maximum_relative_span).clamp(0.0, 1.0)
        distance_confidence = max(0.15, 1.0 - distance / float(propagation_steps + 1))
        completed = torch.where(new_valid, local_mean, completed)
        completed_valid = completed_valid | new_valid
        propagation_distance = torch.where(
            new_valid, torch.full_like(propagation_distance, float(distance)), propagation_distance,
        )
        lidar_reliability = torch.where(
            new_valid, distance_confidence * (0.5 + 0.5 * stability), lidar_reliability,
        )
    propagated = completed_valid & ~static_valid
    fused_depth = torch.where(completed_valid, completed, stereo_depth)
    fusion_confidence = torch.where(
        lidar_reliability > 0, lidar_reliability, stereo_confidence,
    )
    return fused_depth, fusion_confidence, lidar_reliability, propagation_distance


def bounded_lidar_depth_correction(
    stereo_depth: torch.Tensor,
    stereo_confidence: torch.Tensor,
    lidar_depth: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy 15x15 LiDAR correction retained only for controlled A/B tests."""
    if lidar_depth is None:
        zeros = torch.zeros_like(stereo_depth)
        return stereo_depth, stereo_confidence, zeros, zeros
    lidar = F.interpolate(lidar_depth.float(), stereo_depth.shape[-2:], mode="nearest")
    lidar_valid = (lidar > 0).to(stereo_depth.dtype)
    local_support = F.avg_pool2d(lidar_valid, 15, 1, 7, count_include_pad=False)
    local_correction = F.avg_pool2d(
        (lidar - stereo_depth) * lidar_valid, 15, 1, 7, count_include_pad=False,
    ) / local_support.clamp(min=1e-6)
    correction_limit = stereo_depth * 0.25
    local_correction = torch.maximum(
        torch.minimum(local_correction, correction_limit), -correction_limit,
    )
    correction_weight = (local_support * 25.0).clamp(0.0, 1.0)
    depth = stereo_depth + correction_weight * local_correction
    confidence = torch.maximum(stereo_confidence, correction_weight * 0.8)
    return depth, confidence, correction_weight, torch.zeros_like(stereo_depth)


def homography_warp(
    source_features: torch.Tensor,
    source_K: torch.Tensor,
    source_T_world: torch.Tensor,
    target_K: torch.Tensor,
    target_T_world: torch.Tensor,
    depth: float,
    H: int,
    W: int,
    source_D: torch.Tensor | None = None,
    target_D: torch.Tensor | None = None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Warp source features to target view at given depth plane.
    
    Args:
        source_features: [B, C, Hs, Ws] source image features
        source_K: [B, 3, 3] source camera intrinsics
        source_T_world: [B, 4, 4] source camera extrinsics (world to camera)
        target_K: [B, 3, 3] target camera intrinsics
        target_T_world: [B, 4, 4] target camera extrinsics (world to camera)
        depth: scalar depth value for the plane
        H, W: target image size
    
    Returns:
        warped_features: [B, C, H, W] warped source features
    """
    B, C, Hs, Ws = source_features.shape
    device = source_features.device
    source_K = source_K.float()
    source_T_world = source_T_world.float()
    target_K = target_K.float()
    target_T_world = target_T_world.float()
    
    # Generate target pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    ones = torch.ones_like(x)
    target_pixels = torch.stack([x, y, ones], dim=-1)  # [H, W, 3]
    target_pixels = target_pixels.reshape(-1, 3).t()  # [3, H*W]
    
    # Backproject target pixels to 3D at given depth
    target_T_world_inv = torch.linalg.inv(target_T_world)
    
    rays = pixels_to_camera_rays(target_pixels, target_K, target_D)
    rays = rays * depth  # scale to depth
    
    # Convert to homogeneous
    rays_homo = torch.cat([rays, torch.ones(B, 1, H*W, device=device)], dim=1)  # [B, 4, H*W]
    
    # Transform to world space
    world_points = target_T_world_inv @ rays_homo  # [B, 4, H*W]
    
    # Transform to source camera space
    source_points = source_T_world @ world_points  # [B, 4, H*W]
    
    # Project to source image
    source_pixels_homo = project_camera_points(source_points[:, :3, :], source_K, source_D)
    homogeneous_scale = source_pixels_homo[:, 2:3, :]
    source_pixels = source_pixels_homo[:, :2, :] / homogeneous_scale.clamp(min=1e-4)
    
    # Normalize to [-1, 1] for grid_sample
    source_pixels[:, 0, :] = 2.0 * source_pixels[:, 0, :] / (Ws - 1) - 1.0
    source_pixels[:, 1, :] = 2.0 * source_pixels[:, 1, :] / (Hs - 1) - 1.0
    valid = (
        (source_points[:, 2:3, :] > 1e-4)
        & (source_pixels[:, 0:1, :] >= -1.0 - 1e-5)
        & (source_pixels[:, 0:1, :] <= 1.0 + 1e-5)
        & (source_pixels[:, 1:2, :] >= -1.0 - 1e-5)
        & (source_pixels[:, 1:2, :] <= 1.0 + 1e-5)
    ).reshape(B, 1, H, W)
    
    source_pixels = source_pixels.permute(0, 2, 1).reshape(B, H, W, 2)  # [B, H, W, 2]
    
    # Warp source features
    warped = F.grid_sample(
        source_features,
        source_pixels,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    if return_mask:
        return warped, valid.to(warped.dtype)
    return warped


def homography_warp_dense_depth(
    source_image: torch.Tensor,
    source_K: torch.Tensor,
    source_T_world: torch.Tensor,
    target_K: torch.Tensor,
    target_T_world: torch.Tensor,
    target_depth: torch.Tensor,
    source_depth: torch.Tensor | None = None,
    source_D: torch.Tensor | None = None,
    target_D: torch.Tensor | None = None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Warp full-resolution source RGB using a per-pixel target depth map."""
    batch, _, source_height, source_width = source_image.shape
    height, width = target_depth.shape[-2:]
    device = source_image.device
    source_K = source_K.float()
    source_T_world = source_T_world.float()
    target_K = target_K.float()
    target_T_world = target_T_world.float()
    target_depth = target_depth.float()

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
    rays = pixels_to_camera_rays(pixels, target_K, target_D)
    rays = rays * target_depth.reshape(batch, 1, -1)
    rays_homogeneous = torch.cat(
        [rays, torch.ones(batch, 1, height * width, device=device)],
        dim=1,
    )
    world_points = torch.linalg.inv(target_T_world) @ rays_homogeneous
    source_points = source_T_world @ world_points
    projected = project_camera_points(source_points[:, :3], source_K, source_D)
    homogeneous_scale = projected[:, 2:3]
    source_pixels = projected[:, :2] / homogeneous_scale.clamp(min=1e-4)
    source_pixels[:, 0] = 2.0 * source_pixels[:, 0] / max(source_width - 1, 1) - 1.0
    source_pixels[:, 1] = 2.0 * source_pixels[:, 1] / max(source_height - 1, 1) - 1.0
    valid = (
        (target_depth.reshape(batch, 1, -1) > 0)
        & (source_points[:, 2:3] > 1e-4)
        & (source_pixels[:, 0:1] >= -1.0 - 1e-5)
        & (source_pixels[:, 0:1] <= 1.0 + 1e-5)
        & (source_pixels[:, 1:2] >= -1.0 - 1e-5)
        & (source_pixels[:, 1:2] <= 1.0 + 1e-5)
    ).reshape(batch, 1, height, width)
    grid = source_pixels.permute(0, 2, 1).reshape(batch, height, width, 2)
    if source_depth is not None:
        sampled_source_depth = F.grid_sample(
            source_depth.float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        has_source_depth = sampled_source_depth > 0
        front_tolerance = torch.maximum(
            sampled_source_depth * 0.02,
            sampled_source_depth.new_full((), 0.15),
        )
        source_surface_visible = (
            source_points[:, 2:3].reshape(batch, 1, height, width)
            <= sampled_source_depth + front_tolerance
        )
        valid = valid & (~has_source_depth | source_surface_visible)
    warped = F.grid_sample(
        source_image.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    if return_mask:
        return warped, valid.to(warped.dtype)
    return warped


class CostVolumeBuilder(nn.Module):
    """Build cost volume from multi-view features."""
    
    def __init__(self, feature_channels: int = 32):
        super().__init__()
        self.feature_channels = feature_channels
    
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(
        self,
        source_features_list: list[torch.Tensor],
        source_calibs: list[dict],
        target_calib: dict,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_features_list: list of [B, C, Hs, Ws] source view features
            source_calibs: list of {K, T_world} dicts for each source view
            target_calib: {K, T_world} dict for target view
            depth_values: [D] depth hypothesis values
        
        Returns:
            cost_volume: [B, C, D, H, W]
        """
        B, C, H, W = source_features_list[0].shape
        D = len(depth_values)
        num_views = len(source_features_list)
        
        cost_volume = torch.zeros(B, C, D, H, W, device=source_features_list[0].device)
        
        for d_idx, depth in enumerate(depth_values):
            # Warp each source view to target at this depth
            feature_sum = torch.zeros(B, C, H, W, device=source_features_list[0].device)
            feature_sq_sum = torch.zeros_like(feature_sum)
            valid_count = torch.zeros(B, 1, H, W, device=feature_sum.device)
            for src_idx, source_features in enumerate(source_features_list):
                source_calib = source_calibs[src_idx]
                warped, valid = homography_warp(
                    source_features,
                    source_calib['K'],
                    source_calib['T_world'],
                    target_calib['K'],
                    target_calib['T_world'],
                    depth.item(),
                    H, W,
                    source_D=source_calib.get('D'),
                    target_D=target_calib.get('D'),
                    return_mask=True,
                )
                feature_sum += warped * valid
                feature_sq_sum += warped.square() * valid
                valid_count += valid
            mean = feature_sum / valid_count.clamp(min=1.0)
            variance = (feature_sq_sum / valid_count.clamp(min=1.0) - mean.square()).clamp(min=0.0)
            variance = variance + (valid_count < 2).to(variance.dtype)
            
            cost_volume[:, :, d_idx, :, :] = variance
        
        return cost_volume


class CostRegularization3D(nn.Module):
    """3D CNN to regularize cost volume."""
    
    def __init__(self, in_channels: int = 32):
        super().__init__()
        
        # U-Net style 3D network
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        
        self.deconv0 = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 3, padding=1),  # Output single channel cost
        )
    
    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_volume: [B, C, D, H, W]
        
        Returns:
            regularized_cost: [B, 1, D, H, W]
        """
        # Encoder
        conv0 = self.conv0(cost_volume)  # [B, 16, D, H, W]
        conv1 = self.conv1(conv0)         # [B, 32, D/2, H/2, W/2]
        conv2 = self.conv2(conv1)         # [B, 64, D/4, H/4, W/4]
        
        # Bottleneck
        conv3 = self.conv3(conv2)
        
        # Decoder with skip connections
        deconv2 = self.deconv2(conv3)     # [B, 32, D/2, H/2, W/2]
        deconv2 = torch.cat([deconv2, conv1], dim=1)  # [B, 64, D/2, H/2, W/2]
        
        deconv1 = self.deconv1(deconv2)   # [B, 16, D, H, W]
        deconv1 = torch.cat([deconv1, conv0], dim=1)  # [B, 32, D, H, W]
        
        cost_reg = self.deconv0(deconv1)  # [B, 1, D, H, W]
        
        return cost_reg


class PlaneSweepRenderer(nn.Module):
    """Complete plane-sweep stereo renderer."""
    
    def __init__(
        self,
        feature_channels: int = 32,
        num_depth_layers: int = 48,
        depth_min: float = 2.0,
        depth_max: float = 80.0,
        narrow_depth_layers: int | None = None,
        narrow_focal_ratio_threshold: float = 1.2,
        lidar_depth_mode: str = "primary",
    ):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.num_depth_layers = num_depth_layers
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.narrow_depth_layers = narrow_depth_layers
        self.narrow_focal_ratio_threshold = narrow_focal_ratio_threshold
        if lidar_depth_mode not in {"primary", "bounded", "stereo"}:
            raise ValueError(f"unsupported lidar_depth_mode: {lidar_depth_mode}")
        self.lidar_depth_mode = lidar_depth_mode
        self.log_baseline_temperature = nn.Parameter(torch.tensor(math.log(0.50)))
        self.log_baseline_temperature_narrow = nn.Parameter(torch.tensor(math.log(0.50)))
        
        # Feature extractor for images
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_channels, 3, padding=1),
        )
        
        # Cost volume builder
        self.cost_volume_builder = CostVolumeBuilder(feature_channels)
        
        # 3D cost regularization
        self.cost_regularization = CostRegularization3D(feature_channels)

    @property
    def baseline_temperature(self) -> torch.Tensor:
        return self.log_baseline_temperature.exp().clamp(0.05, 2.0)

    @property
    def baseline_temperature_narrow(self) -> torch.Tensor:
        return self.log_baseline_temperature_narrow.exp().clamp(0.05, 2.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        temperature_key = prefix + "log_baseline_temperature"
        if temperature_key not in state_dict:
            state_dict[temperature_key] = self.log_baseline_temperature.detach().clone()
        narrow_temperature_key = prefix + "log_baseline_temperature_narrow"
        if narrow_temperature_key not in state_dict:
            state_dict[narrow_temperature_key] = state_dict[temperature_key].detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(
        self,
        source_images: torch.Tensor,
        source_calibs: list[dict],
        target_calib: dict,
        lidar_depth_guide: torch.Tensor | None = None,
        lidar_dynamic_mask: torch.Tensor | None = None,
        source_lidar_depths: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            source_images: [B, num_views, 3, H, W] source RGB images
            source_calibs: list of calibration dicts
            target_calib: target camera calibration
            lidar_depth_guide: [B, 1, H, W] optional LiDAR depth to guide range
        
        Returns:
            dict with:
                - rgb: [B, 3, H, W] reconstructed dense RGB
                - depth: [B, 1, H, W] depth map
                - confidence: [B, 1, H, W] reconstruction confidence
                - depth_prob: [B, D, H, W] depth probability distribution
        """
        B, num_views, _, H, W = source_images.shape
        device = source_images.device
        source_occlusion_depths = None
        if source_lidar_depths is not None:
            source_occlusion_depths = reliable_lidar_occlusion_depth(
                source_lidar_depths.reshape(B * num_views, 1, H, W)
            ).reshape(B, num_views, 1, H, W)
        
        # Extract features from all views
        source_features_list = []
        for view_idx in range(num_views):
            view_img = source_images[:, view_idx, :, :, :]  # [B, 3, H, W]
            features = self.feature_extractor(view_img)  # [B, C, H, W]
            source_features_list.append(features)
        feature_H, feature_W = source_features_list[0].shape[-2:]
        intrinsic_scale = source_images.new_tensor([
            [feature_W / W, 0.0, 0.0],
            [0.0, feature_H / H, 0.0],
            [0.0, 0.0, 1.0],
        ])[None]
        feature_source_calibs = [
            {'K': intrinsic_scale @ calib['K'], 'T_world': calib['T_world']}
            for calib in source_calibs
        ]
        feature_target_calib = {
            'K': intrinsic_scale @ target_calib['K'],
            'T_world': target_calib['T_world'],
            'D': target_calib.get('D'),
        }
        for feature_calib, source_calib in zip(feature_source_calibs, source_calibs):
            feature_calib['D'] = source_calib.get('D')
        
        # Determine depth range
        if lidar_depth_guide is not None and lidar_depth_guide.sum() > 0:
            valid_depths = lidar_depth_guide[lidar_depth_guide > 0]
            depth_min = max(valid_depths.min().item() * 0.5, self.depth_min)
            depth_max = min(valid_depths.max().item() * 1.5, self.depth_max)
        else:
            depth_min = self.depth_min
            depth_max = self.depth_max
        
        # Sample depth layers
        target_focal_ratio = float(
            (target_calib['K'][:, 0, 0] / max(W, 1)).detach().mean()
        )
        depth_layer_count = self.num_depth_layers
        if (
            self.narrow_depth_layers is not None
            and target_focal_ratio >= self.narrow_focal_ratio_threshold
        ):
            depth_layer_count = self.narrow_depth_layers
        baseline_temperature = (
            self.baseline_temperature_narrow
            if target_focal_ratio >= self.narrow_focal_ratio_threshold
            else self.baseline_temperature
        )
        depth_values = inverse_depth_sampling(depth_min, depth_max, depth_layer_count)
        depth_values = depth_values.to(device)
        
        # Build cost volume
        cost_volume = self.cost_volume_builder(
            source_features_list,
            feature_source_calibs,
            feature_target_calib,
            depth_values,
        )
        
        # Regularize cost volume
        cost_reg = self.cost_regularization(cost_volume)  # [B, 1, D, H, W]
        cost_reg = cost_reg.squeeze(1)  # [B, D, H, W]
        
        # Soft depth probability
        depth_prob = F.softmax(-cost_reg, dim=1)  # [B, D, H, W], lower cost = higher prob
        
        # Expected depth
        depth_values_expanded = depth_values.view(1, -1, 1, 1).expand(B, -1, feature_H, feature_W)
        depth_map = (depth_prob * depth_values_expanded).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Confidence (entropy-based)
        entropy = -(depth_prob * (depth_prob + 1e-10).log()).sum(dim=1, keepdim=True)
        max_entropy = np.log(depth_layer_count)
        confidence = 1.0 - (entropy / max_entropy).clamp(0, 1)
        
        # Estimate depth at low resolution, then sample original source RGB once
        # per view at full resolution. This preserves source-image high frequencies.
        depth_map = F.interpolate(depth_map, (H, W), mode='bilinear', align_corners=False)
        stereo_confidence = F.interpolate(confidence, (H, W), mode='bilinear', align_corners=False)
        stereo_depth = depth_map
        if self.lidar_depth_mode == "primary":
            depth_map, depth_confidence, lidar_reliability, lidar_propagated = fuse_lidar_metric_depth(
                stereo_depth, stereo_confidence, lidar_depth_guide, lidar_dynamic_mask,
            )
        elif self.lidar_depth_mode == "bounded":
            depth_map, depth_confidence, lidar_reliability, lidar_propagated = bounded_lidar_depth_correction(
                stereo_depth, stereo_confidence, lidar_depth_guide,
            )
        else:
            depth_map, depth_confidence = stereo_depth, stereo_confidence
            lidar_reliability = torch.zeros_like(stereo_depth)
            lidar_propagated = torch.zeros_like(stereo_depth)
        warped_views = []
        valid_views = []
        for view_idx, source_image in enumerate(source_images.unbind(1)):
            source_depth = (
                source_occlusion_depths[:, view_idx]
                if source_occlusion_depths is not None else None
            )
            warped_rgb, valid = homography_warp_dense_depth(
                source_image,
                source_calibs[view_idx]['K'],
                source_calibs[view_idx]['T_world'],
                target_calib['K'],
                target_calib['T_world'],
                depth_map,
                source_depth=source_depth,
                source_D=source_calibs[view_idx].get('D'),
                target_D=target_calib.get('D'),
                return_mask=True,
            )
            warped_views.append(warped_rgb)
            valid_views.append(valid)
        warped_stack = torch.stack(warped_views, dim=1)
        valid_stack = torch.stack(valid_views, dim=1)
        target_rotation = target_calib['T_world'][:, :3, :3].float()
        target_forward = target_rotation.transpose(1, 2)[:, :, 2]
        target_center = torch.linalg.inv(target_calib['T_world'].float())[:, :3, 3]
        view_similarities = []
        view_baselines = []
        for source_calib in source_calibs:
            source_rotation = source_calib['T_world'][:, :3, :3].float()
            source_forward = source_rotation.transpose(1, 2)[:, :, 2]
            similarity = F.cosine_similarity(source_forward, target_forward, dim=1).clamp(min=0.0)
            view_similarities.append(similarity[:, None, None, None])
            source_center = torch.linalg.inv(source_calib['T_world'].float())[:, :3, 3]
            baseline = torch.linalg.vector_norm(source_center - target_center, dim=1)
            view_baselines.append(baseline[:, None, None, None])
        view_similarity = torch.stack(view_similarities, dim=1)
        view_baseline = torch.stack(view_baselines, dim=1)
        facing_mask = (view_similarity > 0.1).to(valid_stack.dtype)
        valid_stack = valid_stack * facing_mask
        valid_count = valid_stack.sum(dim=1).clamp(min=1.0)
        consensus = (warped_stack * valid_stack).sum(dim=1) / valid_count
        photometric_error = (warped_stack - consensus[:, None]).abs().mean(dim=2, keepdim=True)
        flattened_valid = valid_stack.reshape(B * num_views, 1, H, W)
        feather = F.avg_pool2d(flattened_valid, 31, stride=1, padding=15).reshape(B, num_views, 1, H, W)
        view_logits = (
            -photometric_error / 0.05
            + feather.clamp(min=1e-4).log()
            + 2.0 * view_similarity.clamp(min=1e-4).log()
            - view_baseline / baseline_temperature
        )
        view_logits = view_logits.masked_fill(valid_stack < 0.5, -1e4)
        view_weights = F.softmax(view_logits, dim=1) * valid_stack
        view_weights = F.avg_pool2d(
            view_weights.reshape(B * num_views, 1, H, W),
            15,
            stride=1,
            padding=7,
        ).reshape(B, num_views, 1, H, W) * valid_stack
        view_weights = view_weights / view_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        rgb_reconstruction = (warped_stack * view_weights).sum(dim=1).clamp(0, 1)
        rgb_validity = (valid_stack.sum(dim=1) > 0).to(rgb_reconstruction.dtype)
        view_dominance = view_weights.max(dim=1).values
        disagreement = (photometric_error * view_weights).sum(dim=1)
        raw_valid_count = valid_stack.sum(dim=1)
        multi_view_support = (raw_valid_count >= 2.0).to(rgb_reconstruction.dtype)
        single_view_support = (raw_valid_count == 1.0).to(rgb_reconstruction.dtype)
        photometric_agreement = torch.exp(-disagreement / 0.10)
        photometric_confidence = (
            multi_view_support * photometric_agreement
            + single_view_support * 0.35
        )
        confidence = depth_confidence * photometric_confidence
        seam_mask = (
            (1.0 - photometric_agreement)
            + (1.0 - multi_view_support)
        ).clamp(0.0, 1.0) * rgb_validity
        warped_feature_views = []
        valid_feature_views = []
        for view_idx, source_features in enumerate(source_features_list):
            warped_features, valid_features = homography_warp_dense_depth(
                source_features,
                feature_source_calibs[view_idx]['K'],
                feature_source_calibs[view_idx]['T_world'],
                feature_target_calib['K'],
                feature_target_calib['T_world'],
                F.interpolate(depth_map, (feature_H, feature_W), mode='bilinear', align_corners=False),
                source_depth=(
                    F.interpolate(
                        source_occlusion_depths[:, view_idx].float(),
                        (feature_H, feature_W),
                        mode="nearest",
                    )
                    if source_occlusion_depths is not None else None
                ),
                source_D=feature_source_calibs[view_idx].get('D'),
                target_D=feature_target_calib.get('D'),
                return_mask=True,
            )
            warped_feature_views.append(warped_features)
            valid_feature_views.append(valid_features)
        warped_feature_stack = torch.stack(warped_feature_views, dim=1)
        valid_feature_stack = torch.stack(valid_feature_views, dim=1) * facing_mask
        feature_weights = F.interpolate(
            view_weights.reshape(B * num_views, 1, H, W),
            (feature_H, feature_W),
            mode='bilinear',
            align_corners=False,
        ).reshape(B, num_views, 1, feature_H, feature_W)
        feature_weights = feature_weights * valid_feature_stack
        feature_weights = feature_weights / feature_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        rendered_features = (warped_feature_stack * feature_weights).sum(dim=1)
        rendered_features = F.interpolate(rendered_features, (H, W), mode='bilinear', align_corners=False)
        depth_prob_output = F.interpolate(depth_prob, (H, W), mode='bilinear', align_corners=False)
        
        return {
            'rgb': rgb_reconstruction,
            'depth': depth_map,
            'stereo_depth': stereo_depth,
            'confidence': confidence,
            'depth_confidence': depth_confidence,
            'stereo_confidence': stereo_confidence,
            'lidar_reliability': lidar_reliability,
            'lidar_propagated_mask': lidar_propagated,
            'lidar_exact_mask': (
                (lidar_reliability > 0) & (lidar_propagated == 0)
            ).to(depth_map.dtype),
            'structure_validity': (
                (lidar_reliability > 0).to(depth_map.dtype)
                if self.lidar_depth_mode == "primary"
                else rgb_validity.clamp(0, 1)
            ),
            'lidar_depth_mode': self.lidar_depth_mode,
            'photometric_confidence': photometric_confidence,
            'photometric_agreement': photometric_agreement,
            'view_dominance': view_dominance,
            'valid_source_count': raw_valid_count,
            'appearance_validity': (
                raw_valid_count >= 1.0
            ).to(rgb_reconstruction.dtype),
            'valid_mask': rgb_validity.clamp(0, 1),
            # Keep the per-source masks for calibration/coverage diagnostics.
            # The source dimension has the same order as source_calibs.
            'per_view_valid_mask': valid_stack.clamp(0, 1),
            'view_weights': view_weights,
            'view_similarity': view_similarity,
            'view_baseline': view_baseline,
            'baseline_temperature': baseline_temperature,
            'depth_prob': depth_prob_output,
            'depth_layer_count': depth_map.new_tensor(depth_layer_count),
            'target_focal_ratio': depth_map.new_tensor(target_focal_ratio),
            'seam_mask': seam_mask.clamp(0, 1),
            'features': rendered_features,
        }
