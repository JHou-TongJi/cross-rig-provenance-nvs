"""Render deterministic LiDAR appearance points with gsplat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gcr_nvs.datasets.manifest import build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.rendering.lidar_gaussians import render_fixed_lidar_gaussians


def _load_target(path: Path, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0


def _psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    mse = float(np.mean((prediction[mask] - target[mask]) ** 2))
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("appearance", type=Path)
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frame-id", type=int, default=75)
    parser.add_argument("--target-camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--pixel-radius", type=float, default=0.85)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/geometry_diagnostics/lidar_gaussians"),
    )
    args = parser.parse_args()

    with np.load(args.appearance) as payload:
        points = payload["surface_points"].astype(np.float32)
        colors = payload["point_rgb"].astype(np.float32)
        validity = payload["point_appearance_validity"].astype(bool)
        source_count = payload["point_source_count"].astype(np.float32)
    sequence_dir = Path(args.sequence)
    record = next(
        item for item in build_sequence_manifest(sequence_dir, compute_quality=False)
        if item.frame_id == args.frame_id
    )
    calibration = load_calibrations(
        sequence_dir / record.camera_config,
        Path("camera_intric.yaml"),
    )[args.target_camera]
    device = torch.device(args.device)
    with torch.inference_mode():
        result = render_fixed_lidar_gaussians(
            torch.from_numpy(points).to(device),
            torch.from_numpy(colors).to(device),
            torch.from_numpy(validity).to(device),
            calibration,
            width=args.width,
            height=args.height,
            source_count=torch.from_numpy(source_count).to(device),
            pixel_radius=args.pixel_radius,
            alpha_threshold=args.alpha_threshold,
        )
    rgb = result.rgb.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
    depth = result.depth[0].cpu().numpy()
    alpha = result.alpha[0].cpu().numpy().clip(0.0, 1.0)
    valid = result.appearance_validity[0].cpu().numpy()
    target = _load_target(
        sequence_dir / record.cameras[args.target_camera],
        args.width,
        args.height,
    )
    high_alpha = alpha >= 0.5
    depth_normalized = np.zeros_like(depth)
    if valid.any():
        depth_normalized[valid] = np.clip(depth[valid] / 80.0, 0.0, 1.0)
    depth_rgb = np.stack([
        np.clip(1.5 - 2.0 * depth_normalized, 0.0, 1.0),
        np.clip(1.0 - np.abs(2.0 * depth_normalized - 1.0), 0.0, 1.0),
        np.clip(2.0 * depth_normalized - 0.5, 0.0, 1.0),
    ], axis=-1)
    depth_rgb[~valid] = 0.0
    alpha_rgb = np.repeat(alpha[..., None], 3, axis=2)
    panel = np.concatenate([target, rgb, alpha_rgb, depth_rgb], axis=1)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(panel * 255.0).astype(np.uint8)).resize(
        (panel.shape[1] * 3, panel.shape[0] * 3), Image.Resampling.NEAREST,
    ).save(args.output_prefix.with_suffix(".png"))
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        rgb=rgb,
        depth=depth,
        alpha=alpha,
        appearance_validity=valid,
    )
    report = {
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "target_camera": args.target_camera,
        "fixed_centers": True,
        "position_offsets": False,
        "gaussian_point_count": result.point_count,
        "appearance_coverage": float(valid.mean()),
        "high_alpha_coverage": float(high_alpha.mean()),
        "observed_psnr_db": _psnr(rgb, target, valid),
        "high_alpha_psnr_db": _psnr(rgb, target, high_alpha),
        "invalid_rgb_exactly_zero": bool(np.all(rgb[~valid] == 0.0)),
        "depth_finite": bool(np.isfinite(depth).all()),
        "alpha_finite": bool(np.isfinite(alpha).all()),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
