"""Strict-LOO RGB/DINO appearance lift on LiDAR-supported surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.appearance import project_surfaces_to_sources
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.models.lidar_appearance import LiDARSurfaceAppearanceLift
from gcr_nvs.rendering.fixed_splat import FixedSplatRenderer


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frame-id", type=int, default=75)
    parser.add_argument("--target-camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--source-width", type=int, default=224)
    parser.add_argument("--source-height", type=int, default=126)
    parser.add_argument("--rgb-source-width", type=int)
    parser.add_argument("--rgb-source-height", type=int)
    parser.add_argument("--output-width", type=int, default=128)
    parser.add_argument("--output-height", type=int, default=72)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hard-rgb-selection", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/geometry_diagnostics/lidar_appearance"),
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    sequence_dir = Path(args.sequence)
    record = next(
        item for item in build_sequence_manifest(sequence_dir, compute_quality=False)
        if item.frame_id == args.frame_id
    )
    calibrations = load_calibrations(
        sequence_dir / record.camera_config,
        Path("camera_intric.yaml"),
    )
    source_names = tuple(name for name in CAMERA_NAMES if name != args.target_camera)
    source_images_np = np.stack([
        _load_rgb(
            sequence_dir / record.cameras[name],
            args.source_width,
            args.source_height,
        ).transpose(2, 0, 1)
        for name in source_names
    ])
    rgb_source_width = args.rgb_source_width or args.source_width
    rgb_source_height = args.rgb_source_height or args.source_height
    rgb_source_images_np = np.stack([
        _load_rgb(
            sequence_dir / record.cameras[name],
            rgb_source_width,
            rgb_source_height,
        ).transpose(2, 0, 1)
        for name in source_names
    ])
    source_calibrations = [
        TargetCamera(calibrations[name]).resized(
            args.source_width, args.source_height,
        ).calibration
        for name in source_names
    ]
    rgb_source_calibrations = [
        TargetCamera(calibrations[name]).resized(
            rgb_source_width, rgb_source_height,
        ).calibration
        for name in source_names
    ]
    source_optical_axes = np.stack([
        calibration.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
        for calibration in source_calibrations
    ]).astype(np.float32)
    target_calibration = calibrations[args.target_camera]
    target_center = target_calibration.camera_center.astype(np.float32)
    target_axis = (
        target_calibration.external[:3, :3].T @ np.asarray([0.0, 0.0, 1.0])
    ).astype(np.float32)

    with np.load(args.geometry) as payload:
        surface_map = payload["surface_points"].astype(np.float32)
        structure_validity = payload["structure_validity"].astype(bool)
        geometry_features = payload["surface_features"].astype(np.float32)
        geometry_depth = payload["depth"].astype(np.float32)
        geometry_normal = payload["normal"].astype(np.float32)
        geometry_occupancy = payload["occupancy"].astype(np.float32)
        geometry_confidence = payload["confidence"].astype(np.float32)
        geometry_query_coverage = payload["query_coverage"].astype(np.float32)
        geometry_occupied_support = payload["occupied_support"].astype(np.float32)
    points = surface_map[structure_validity]
    point_geometry_features = geometry_features[structure_validity]
    source_uv, source_valid, _, source_centers = project_surfaces_to_sources(
        points, source_calibrations,
    )
    rgb_source_uv, rgb_source_valid, _, _ = project_surfaces_to_sources(
        points, rgb_source_calibrations,
    )
    source_valid &= rgb_source_valid
    device = torch.device(args.device)
    model = LiDARSurfaceAppearanceLift(
        feature_dim=64,
        top_k=args.top_k,
        use_dino=True,
        dino_pretrained=True,
        hard_rgb_selection=args.hard_rgb_selection,
    ).to(device).eval()
    with torch.inference_mode():
        appearance = model(
            torch.from_numpy(source_images_np[None]).to(device),
            torch.from_numpy(points[None]).to(device),
            torch.from_numpy(source_uv[None]).to(device),
            torch.from_numpy(source_valid[None]).to(device),
            torch.from_numpy(source_centers[None]).to(device),
            target_camera_center=torch.from_numpy(target_center[None]).to(device),
            source_optical_axes=torch.from_numpy(source_optical_axes[None]).to(device),
            target_optical_axis=torch.from_numpy(target_axis[None]).to(device),
            rgb_source_images=torch.from_numpy(rgb_source_images_np[None]).to(device),
            rgb_source_uv=torch.from_numpy(rgb_source_uv[None]).to(device),
        )

    colors = appearance.rgb[0].cpu().numpy()
    color_validity = appearance.appearance_validity[0].cpu().numpy()
    source_count = appearance.source_count[0].cpu().numpy().astype(np.float32)
    provenance = appearance.source_provenance[0].cpu().numpy().astype(np.int16)
    features = appearance.features[0].cpu().numpy()
    render = FixedSplatRenderer().render(
        points,
        colors,
        target_calibration,
        source_observation_count=source_count,
        source_provenance=provenance,
        color_validity=color_validity,
        output_size=(args.output_width, args.output_height),
    )
    target_rgb = _load_rgb(
        sequence_dir / record.cameras[args.target_camera],
        args.output_width,
        args.output_height,
    )
    target_point_rgb = target_rgb[structure_validity]
    observed = render.appearance_validity
    error = render.coarse_rgb.transpose(1, 2, 0)[observed] - target_rgb[observed]
    mse = float(np.mean(error**2)) if len(error) else None
    psnr = float(-10.0 * np.log10(max(mse, 1e-12))) if mse is not None else None

    rgb_image = np.rint(
        render.coarse_rgb.transpose(1, 2, 0).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    validity_image = np.repeat(
        (render.appearance_validity[..., None] * 255).astype(np.uint8), 3, axis=2,
    )
    count_image = np.zeros_like(rgb_image)
    normalized_count = np.clip(render.source_count[0] / max(len(source_names), 1), 0.0, 1.0)
    count_image[..., 1] = np.rint(normalized_count * 255.0).astype(np.uint8)
    target_image = np.rint(target_rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    panel = np.concatenate([target_image, rgb_image, validity_image, count_image], axis=1)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).resize(
        (panel.shape[1] * 3, panel.shape[0] * 3), Image.Resampling.NEAREST,
    ).save(args.output_prefix.with_suffix(".png"))
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        rgb=render.coarse_rgb,
        structure_validity=render.structure_validity,
        appearance_validity=render.appearance_validity,
        source_count=render.source_count,
        source_provenance=render.source_provenance,
        point_features=features,
        point_rgb=colors,
        point_appearance_validity=color_validity,
        surface_points=points,
        point_source_count=source_count,
        point_source_provenance=provenance,
        per_source_features=appearance.per_source_features[0].cpu().numpy(),
        per_source_rgb=appearance.per_source_rgb[0].cpu().numpy(),
        per_source_validity=appearance.per_source_validity[0].cpu().numpy(),
        per_source_scores=appearance.per_source_scores[0].cpu().numpy(),
        target_point_rgb=target_point_rgb,
        point_geometry_features=point_geometry_features,
        point_depth=geometry_depth[structure_validity],
        point_normal=geometry_normal[structure_validity],
        point_occupancy=geometry_occupancy[structure_validity],
        point_geometry_confidence=geometry_confidence[structure_validity],
        point_query_coverage=geometry_query_coverage[structure_validity],
        point_occupied_support=geometry_occupied_support[structure_validity],
    )
    report = {
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "target_camera": args.target_camera,
        "source_cameras": source_names,
        "strict_loo": True,
        "surface_count": int(len(points)),
        "point_appearance_coverage": float(color_validity.mean()),
        "target_structure_coverage": float(render.structure_validity.mean()),
        "target_appearance_coverage": float(render.appearance_validity.mean()),
        "observed_psnr_db": psnr,
        "mean_source_count": float(source_count[color_validity].mean()) if color_validity.any() else 0.0,
        "invalid_rgb_exactly_zero": bool(np.all(colors[~color_validity] == 0.0)),
        "invalid_features_exactly_zero": bool(np.all(features[~color_validity] == 0.0)),
        "features_finite": bool(np.isfinite(features).all()),
        "feature_norm_mean": float(np.linalg.norm(features[color_validity], axis=1).mean()) if color_validity.any() else 0.0,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
