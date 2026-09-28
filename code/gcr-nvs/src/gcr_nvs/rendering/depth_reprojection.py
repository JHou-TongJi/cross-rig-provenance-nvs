"""Dense source-depth reprojection for small target-camera pose changes."""

from __future__ import annotations

import torch

from gcr_nvs.rendering.plane_sweep import pixels_to_camera_rays, project_camera_points


def align_inverse_depth_to_lidar(
    relative_inverse_depth: torch.Tensor,
    sparse_metric_depth: torch.Tensor,
    minimum_depth: float = 1.0,
    maximum_depth: float = 80.0,
) -> torch.Tensor:
    """Robustly align relative inverse depth to sparse metric LiDAR depth."""
    if relative_inverse_depth.ndim == 3:
        relative_inverse_depth = relative_inverse_depth[:, None]
    if sparse_metric_depth.ndim == 3:
        sparse_metric_depth = sparse_metric_depth[:, None]
    metric_batches = []
    for relative, sparse in zip(relative_inverse_depth, sparse_metric_depth):
        relative_prediction = relative[0].clamp(min=1e-4)
        valid = (
            torch.isfinite(relative_prediction)
            & torch.isfinite(sparse[0])
            & (sparse[0] >= minimum_depth)
            & (sparse[0] <= maximum_depth)
        )
        prediction_samples = relative_prediction[valid]
        inverse_depth_samples = sparse[0][valid].reciprocal()
        if prediction_samples.numel() < 16:
            scale = inverse_depth_samples.median() / prediction_samples.median().clamp(min=1e-4)
            shift = prediction_samples.new_zeros(())
        else:
            keep = torch.ones_like(prediction_samples, dtype=torch.bool)
            scale = prediction_samples.new_tensor(1.0)
            shift = prediction_samples.new_zeros(())
            for _ in range(3):
                design = torch.stack(
                    [prediction_samples[keep], torch.ones_like(prediction_samples[keep])],
                    dim=1,
                )
                solution = torch.linalg.lstsq(design, inverse_depth_samples[keep, None]).solution[:, 0]
                scale, shift = solution[0], solution[1]
                residual = inverse_depth_samples - (scale * prediction_samples + shift)
                median = residual.median()
                mad = (residual - median).abs().median().clamp(min=1e-4)
                keep = (residual - median).abs() <= 3.0 * mad
            if not torch.isfinite(scale) or scale <= 0:
                scale = (inverse_depth_samples / prediction_samples.clamp(min=1e-4)).median()
                shift = prediction_samples.new_zeros(())
        metric_inverse_depth = (scale * relative_prediction + shift).clamp(min=1.0 / maximum_depth)
        metric_batches.append(metric_inverse_depth.reciprocal().clamp(minimum_depth, maximum_depth))
    return torch.stack(metric_batches, dim=0)[:, None]


def align_metric_depth_to_lidar(
    predicted_metric_depth: torch.Tensor,
    sparse_metric_depth: torch.Tensor,
    minimum_depth: float = 1.0,
    maximum_depth: float = 80.0,
) -> torch.Tensor:
    """Calibrate metric depth scale and bias against sparse LiDAR samples."""
    if predicted_metric_depth.ndim == 3:
        predicted_metric_depth = predicted_metric_depth[:, None]
    if sparse_metric_depth.ndim == 3:
        sparse_metric_depth = sparse_metric_depth[:, None]
    aligned_batches = []
    for predicted, sparse in zip(predicted_metric_depth, sparse_metric_depth):
        prediction = predicted[0]
        valid = (
            torch.isfinite(prediction)
            & (prediction > 0)
            & torch.isfinite(sparse[0])
            & (sparse[0] >= minimum_depth)
            & (sparse[0] <= maximum_depth)
        )
        prediction_samples = prediction[valid]
        depth_samples = sparse[0][valid]
        ratios = depth_samples / prediction_samples.clamp(min=1e-4)
        scale = ratios.median()
        for _ in range(2):
            residual = ratios - scale
            mad = residual.abs().median().clamp(min=0.01)
            inliers = residual.abs() <= 3.0 * mad
            scale = ratios[inliers].median()
        aligned_batches.append((scale * prediction).clamp(minimum_depth, maximum_depth))
    return torch.stack(aligned_batches, dim=0)[:, None]


def forward_splat_rgb(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    source_intrinsic: torch.Tensor,
    source_external: torch.Tensor,
    target_intrinsic: torch.Tensor,
    target_external: torch.Tensor,
    source_distortion: torch.Tensor | None = None,
    target_distortion: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-project source RGB and depth into a target camera with a z-buffer."""
    batch, channels, height, width = source_rgb.shape
    device = source_rgb.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
    rays = pixels_to_camera_rays(
        pixels, source_intrinsic.float(), source_distortion,
    )
    source_points = rays * source_depth.float().reshape(batch, 1, -1)
    source_homogeneous = torch.cat(
        [source_points, torch.ones(batch, 1, height * width, device=device)],
        dim=1,
    )
    world_points = torch.linalg.inv(source_external.float()) @ source_homogeneous
    target_points = target_external.float() @ world_points
    projected = project_camera_points(
        target_points[:, :3], target_intrinsic.float(), target_distortion,
    )
    target_depth = target_points[:, 2]
    target_x = projected[:, 0] / projected[:, 2].clamp(min=1e-4)
    target_y = projected[:, 1] / projected[:, 2].clamp(min=1e-4)
    source_depth_valid = source_depth.reshape(batch, -1) > 0

    source_colors = source_rgb.float().reshape(batch, channels, -1).permute(0, 2, 1)
    batch_indices = torch.arange(batch, device=device)[:, None].expand(batch, height * width)
    candidates = []
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        pixel_x = torch.floor(target_x).long() + offset_x
        pixel_y = torch.floor(target_y).long() + offset_y
        weight_x = 1.0 - (target_x - pixel_x.float()).abs()
        weight_y = 1.0 - (target_y - pixel_y.float()).abs()
        weight = (weight_x * weight_y).clamp(min=0.0)
        valid = (
            source_depth_valid
            & torch.isfinite(target_depth)
            & (target_depth > 1e-4)
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
            & (weight > 0)
        )
        flat_index = batch_indices * (height * width) + pixel_y.clamp(0, height - 1) * width + pixel_x.clamp(0, width - 1)
        candidates.append((flat_index[valid], target_depth[valid], weight[valid], source_colors[valid]))

    indices = torch.cat([candidate[0] for candidate in candidates])
    depths = torch.cat([candidate[1] for candidate in candidates])
    weights = torch.cat([candidate[2] for candidate in candidates])
    colors = torch.cat([candidate[3] for candidate in candidates])
    pixel_count = batch * height * width
    z_buffer = source_rgb.new_full((pixel_count,), float("inf"))
    z_buffer.scatter_reduce_(0, indices, depths, reduce="amin", include_self=True)
    front = depths <= z_buffer[indices] + 0.02 * z_buffer[indices].clamp(min=1.0)
    indices = indices[front]
    weights = weights[front]
    colors = colors[front]
    color_sum = source_rgb.new_zeros(pixel_count, channels)
    weight_sum = source_rgb.new_zeros(pixel_count, 1)
    color_sum.scatter_add_(0, indices[:, None].expand(-1, channels), colors * weights[:, None])
    weight_sum.scatter_add_(0, indices[:, None], weights[:, None])
    rendered = (color_sum / weight_sum.clamp(min=1e-6)).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    validity = (weight_sum > 0).to(source_rgb.dtype).reshape(batch, 1, height, width)
    rendered_depth = z_buffer.reshape(batch, 1, height, width)
    rendered_depth = torch.where(validity > 0, rendered_depth, torch.zeros_like(rendered_depth))
    return rendered.clamp(0.0, 1.0), rendered_depth, validity
