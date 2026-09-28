"""Render and audit strict-LOO RGB from the rectified Unified 3D Field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.datasets.unified_3d_field_dataset import Unified3DFieldDataset
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.geometry.camera import TargetCamera, sparse_depth_from_points
from gcr_nvs.geometry.dynamic_mask import SegFormerDynamicMasker
from gcr_nvs.models.unified_3d_field import Unified3DField
from gcr_nvs.rendering.environment import (
    EnvironmentRenderResult,
    render_directional_source_fallback,
    render_infinite_sky,
)
from gcr_nvs.rendering.unified_field_raymarch import (
    UnifiedFieldRenderResult,
    raymarch_unified_field,
)


def _calibration(name: str, intrinsic, external, width: int, height: int):
    return CameraCalibration(
        name=name,
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        distortion=np.zeros(5, dtype=np.float64),
        external=np.asarray(external, dtype=np.float64),
        width=width,
        height=height,
    )


def _perturb_target(
    calibration: CameraCalibration,
    translation_xyz: tuple[float, float, float],
    rotation_rpy_deg: tuple[float, float, float],
) -> CameraCalibration:
    roll, pitch, yaw = np.deg2rad(rotation_rpy_deg)
    rx = np.asarray([
        [1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    ry = np.asarray([
        [np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    rz = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1],
    ])
    camera_to_vehicle = np.linalg.inv(calibration.external)
    camera_to_vehicle[:3, :3] = rz @ ry @ rx @ camera_to_vehicle[:3, :3]
    camera_to_vehicle[:3, 3] += np.asarray(translation_xyz, dtype=np.float64)
    return CameraCalibration(
        name=calibration.name,
        intrinsic=calibration.intrinsic.copy(),
        distortion=calibration.distortion.copy(),
        external=np.linalg.inv(camera_to_vehicle),
        width=calibration.width,
        height=calibration.height,
    )


def _psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    mse = np.square(prediction[mask] - target[mask]).mean()
    return float(-10.0 * np.log10(max(float(mse), 1e-8)))


def _densify_lidar_prior(depth: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return depth
    valid = depth > 1e-4
    inverse = torch.where(valid, depth.reciprocal(), torch.zeros_like(depth))
    pooled = torch.nn.functional.max_pool2d(
        inverse,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    filled = torch.where(pooled > 0, pooled.reciprocal(), torch.zeros_like(pooled))
    return torch.where(valid, depth, filled)


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    geometry_checkpoint_override: Path | None = None,
):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    geometry_checkpoint_path = (
        geometry_checkpoint_override
        if geometry_checkpoint_override is not None
        else Path(config["geometry_checkpoint"])
    )
    geometry_state = torch.load(
        geometry_checkpoint_path, map_location="cpu", weights_only=False,
    )
    model = Unified3DField(
        lidar_input_dim=7,
        geometry_channels=tuple(
            geometry_state.get("config", {}).get(
                "geometry_channels", (32, 64, 128, 256),
            )
        ),
        use_dino=bool(config.get("use_dino", False)),
        dino_model_name=config.get("dino_model_name", "dinov2_vitb14"),
        dino_pretrained=True,
        appearance_dim=int(config.get("appearance_dim", 64)),
        fusion_hidden_dim=int(config.get("fusion_hidden_dim", 96)),
        decoder_dim=int(config.get("decoder_dim", 48)),
        residual_hidden_dim=int(config.get("residual_hidden_dim", 96)),
        semantic_upsampling=config.get("semantic_upsampling", "bilinear"),
        anyup_checkpoint=Path(config.get("anyup_checkpoint", "outputs/checkpoints/anyup_multi_backbone.pth")),
        anyup_root=Path(config.get("anyup_root", "third_party/anyup")),
        anyup_q_chunk_size=int(config.get("anyup_q_chunk_size", 8192)),
        query_position_encoding=bool(
            geometry_state.get("config", {}).get("query_position_encoding", False)
        ),
        target_camera_conditioning=bool(config.get("target_camera_conditioning", False)),
    ).to(device).eval()
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    allowed_missing = (
        "target_camera_embedding.", "camera_rgb_residual.",
        "camera_rgb_gate.", "camera_color_affine.",
    )
    invalid_missing = [key for key in missing if not key.startswith(allowed_missing)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing={invalid_missing}, unexpected={unexpected}"
        )
    if geometry_checkpoint_override is not None:
        model.geometry.load_state_dict(geometry_state["model"], strict=True)
    return model, config, geometry_state.get("config", {}), missing


def _save_outputs(output_root: Path, result, target_rgb: np.ndarray):
    output_root.mkdir(parents=True, exist_ok=True)
    rgb = result.rgb.detach().cpu().numpy().clip(0.0, 1.0)
    validity = result.output_validity.detach().cpu().numpy()
    rgba = np.concatenate([rgb, validity[..., None].astype(np.float32)], axis=-1)
    Image.fromarray(np.rint(rgba * 255.0).astype(np.uint8), "RGBA").save(
        output_root / "render_rgba.png"
    )
    Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), "RGB").save(
        output_root / "reconstruction_rgb.png"
    )
    Image.fromarray(np.rint(target_rgb * 255.0).astype(np.uint8)).save(
        output_root / "target_rectified.png"
    )
    structure = result.structure_validity.detach().cpu().numpy()
    appearance = result.appearance_validity.detach().cpu().numpy()
    environment = result.environment_validity.detach().cpu().numpy()
    fallback = result.fallback_validity.detach().cpu().numpy()
    mask_rgb = np.zeros((*validity.shape, 3), dtype=np.uint8)
    mask_rgb[structure & appearance] = (40, 190, 80)
    mask_rgb[structure & ~appearance] = (230, 55, 50)
    mask_rgb[~structure] = (45, 100, 225)
    mask_rgb[environment] = (60, 205, 220)
    mask_rgb[fallback] = (235, 170, 35)
    Image.fromarray(mask_rgb).save(output_root / "validity_classes.png")
    rendered_for_panel = np.rint(rgb * 255.0).astype(np.uint8)
    rendered_for_panel[~validity] = (230, 0, 210)
    panel = np.concatenate([
        np.rint(target_rgb * 255.0).astype(np.uint8),
        rendered_for_panel,
        mask_rgb,
    ], axis=1)
    Image.fromarray(panel).save(output_root / "audit_panel.png")
    np.savez_compressed(
        output_root / "render_fields.npz",
        rgb=rgb,
        surface_rgb=result.surface_rgb.detach().cpu().numpy(),
        generated_rgb=result.generated_rgb.detach().cpu().numpy(),
        generation_alpha=result.generation_alpha.detach().cpu().numpy(),
        generation_logvar=result.generation_logvar.detach().cpu().numpy(),
        depth=result.depth.detach().cpu().numpy(),
        normal=result.normal.detach().cpu().numpy(),
        uncertainty=result.uncertainty.detach().cpu().numpy(),
        structure_validity=structure,
        appearance_validity=appearance,
        surface_validity=result.surface_validity.detach().cpu().numpy(),
        environment_validity=environment,
        fallback_validity=fallback,
        output_validity=validity,
        true_disocclusion=result.true_disocclusion.detach().cpu().numpy(),
        unknown_structure=result.unknown_structure.detach().cpu().numpy(),
        source_provenance=result.source_provenance.detach().cpu().numpy(),
        source_weight=result.source_weight.detach().cpu().numpy(),
    )


def _crop_environment(environment, x0, y0, width, height):
    if environment is None:
        return None
    return EnvironmentRenderResult(
        rgb=environment.rgb[y0:y0 + height, x0:x0 + width],
        validity=environment.validity[y0:y0 + height, x0:x0 + width],
        source_count=environment.source_count[y0:y0 + height, x0:x0 + width],
        source_provenance=environment.source_provenance[y0:y0 + height, x0:x0 + width],
    )


def _render_native_tiled(
    *,
    model,
    sample,
    dataset,
    target,
    environment,
    fallback_environment,
    target_depth_prior,
    width,
    height,
    tile_size,
    device,
    sample_count,
    refine_sample_count,
    occupancy_threshold,
    occupied_support_stride,
    rgb_weight_sharpness,
    ignore_source_lidar_validity,
):
    source_images = sample["source_images"].to(device)[None]
    source_rgb_images = sample["source_rgb_images"].to(device)[None]
    source_features = model.encode_sources(source_images)
    encoded_geometry = model.geometry.encode_sparse(
        sample["input_features"].to(device),
        sample["input_indices"].to(device),
        tuple(int(value) for value in sample["spatial_shape"].tolist()),
        batch_size=1,
    )
    tensor_names = (
        "rgb", "surface_rgb", "generated_rgb", "generation_alpha",
        "generation_logvar", "depth", "normal", "structure_confidence",
        "appearance_confidence", "uncertainty", "structure_validity",
        "appearance_validity", "surface_validity", "environment_validity",
        "fallback_validity",
        "output_validity", "unknown_structure", "true_disocclusion",
        "source_provenance", "source_weight",
    )
    stitched = {}
    last_geometry = None
    with torch.inference_mode():
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                tw = min(tile_size, width - x0)
                th = min(tile_size, height - y0)
                result = raymarch_unified_field(
                    model,
                    sample["input_features"].to(device),
                    sample["input_indices"].to(device),
                    tuple(int(value) for value in sample["spatial_shape"].tolist()),
                    target,
                    dataset.geometry_datasets[0].grid,
                    source_images,
                    sample["source_intrinsics"].to(device)[None],
                    sample["source_extrinsics"].to(device)[None],
                    sample["source_distortions"].to(device)[None],
                    width=tw,
                    height=th,
                    source_lidar_depth=sample["source_lidar_depth"].to(device)[None],
                    source_validity=(
                        None if ignore_source_lidar_validity
                        else sample["source_validity"].to(device)[None]
                    ),
                    source_time_offsets=sample["source_time_offsets"].to(device)[None],
                    source_alignment_confidence=sample["source_alignment_confidence"].to(device)[None],
                    focal_role=sample["focal_role"].to(device)[None],
                    source_camera_indices=sample["source_camera_indices"].to(device)[None],
                    target_camera_index=sample["target_camera_index"].to(device),
                    target_focal_role=sample["target_focal_role"].to(device)[None],
                    environment=_crop_environment(environment, x0, y0, tw, th),
                    fallback_environment=_crop_environment(
                        fallback_environment, x0, y0, tw, th,
                    ),
                    sample_count=sample_count,
                    geometry_refine_sample_count=refine_sample_count,
                    target_depth_prior=target_depth_prior[..., y0:y0 + th, x0:x0 + tw],
                    occupancy_threshold=occupancy_threshold,
                    occupied_support_stride=occupied_support_stride,
                    source_rgb_images=source_rgb_images,
                    source_rgb_intrinsics=sample["source_rgb_intrinsics"].to(device)[None],
                    source_rgb_distortions=sample["source_rgb_distortions"].to(device)[None],
                    source_features=source_features,
                    pixel_origin=(x0, y0),
                    encoded_geometry=encoded_geometry,
                )
                last_geometry = result.geometry
                for name in tensor_names:
                    value = getattr(result, name)
                    if name not in stitched:
                        stitched[name] = value.new_zeros((height, width, *value.shape[2:]))
                    stitched[name][y0:y0 + th, x0:x0 + tw] = value
    return UnifiedFieldRenderResult(
        **stitched,
        geometry=last_geometry,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--geometry-checkpoint-override", type=Path)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--teacher-root", type=Path, default=Path("outputs/geometry_teacher"))
    parser.add_argument("--split", type=Path, default=Path("configs/splits/pilot_15seq_v1.yaml"))
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="val")
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--camera", choices=CAMERA_NAMES, default="CAM_FRONT_NARROW")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    # These are the production defaults.  The old 64/17 setting was useful
    # for quick previews but skipped thin surfaces and made coverage appear
    # worse than it is.  Keep explicit CLI overrides for fast smoke tests.
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--refine-sample-count", type=int, default=65)
    parser.add_argument("--max-input-points", type=int)
    parser.add_argument("--max-positive-queries", type=int)
    parser.add_argument("--max-negative-queries", type=int)
    parser.add_argument("--source-rgb-width", type=int)
    parser.add_argument("--source-rgb-height", type=int)
    parser.add_argument("--with-sky", action="store_true")
    parser.add_argument("--rgb-weight-sharpness", type=float, default=1.0)
    parser.add_argument(
        "--rgb-ignore-lidar-validity", action="store_true",
        help="diagnostic: do not require a sparse LiDAR return for RGB visibility",
    )
    parser.add_argument(
        "--temporal-rgb", action="store_true",
        help="append calibrated neighboring-frame RGB views from teacher metadata",
    )
    parser.add_argument(
        "--temporal-rgb-max-views", type=int, default=7,
        help="maximum topology-approved RGB views loaded from temporal frames",
    )
    parser.add_argument("--temporal-rgb-max-frames", type=int, default=1)
    parser.add_argument(
        "--include-target-source", action=argparse.BooleanOptionalAction,
        default=None,
        help="include the current same-camera RGB; defaults on for shifted targets",
    )
    parser.add_argument("--lidar-prior-radius", type=int, default=0)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--occupied-support-stride", type=int, default=4)
    parser.add_argument("--translate-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--rotate-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--native-resolution", action="store_true")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, config, geometry_config, missing = _load_model(
        args.checkpoint, device, args.geometry_checkpoint_override,
    )
    # Match the geometry/appearance training contract unless the caller
    # intentionally requests a diagnostic override.  Using a larger or
    # smaller point budget at evaluation changes the sparse field itself.
    args.max_input_points = int(
        args.max_input_points
        if args.max_input_points is not None
        else config.get("max_input_points", geometry_config.get("max_input_points", 80_000))
    )
    args.max_positive_queries = int(
        args.max_positive_queries
        if args.max_positive_queries is not None
        else config.get("max_positive_queries", geometry_config.get("max_positive_queries", 40_000))
    )
    args.max_negative_queries = int(
        args.max_negative_queries
        if args.max_negative_queries is not None
        else config.get("max_negative_queries", geometry_config.get("max_negative_queries", 20_000))
    )
    if args.rgb_weight_sharpness < 1.0:
        raise ValueError("rgb-weight-sharpness must be at least one")
    model.voxel_fusion.rgb_weight_sharpness = args.rgb_weight_sharpness
    source_size = (
        int(config.get("source_width", 320)), int(config.get("source_height", 180)),
    )
    if (args.source_rgb_width is None) != (args.source_rgb_height is None):
        raise ValueError("source-rgb-width and source-rgb-height must be set together")
    configured_rgb_size = (
        (int(config["appearance_rgb_width"]), int(config["appearance_rgb_height"]))
        if config.get("appearance_rgb_width") is not None else source_size
    )
    appearance_rgb_size = (
        configured_rgb_size
        if args.source_rgb_width is None
        else (args.source_rgb_width, args.source_rgb_height)
    )
    if args.native_resolution:
        args.width, args.height = (
            (3840, 2160)
            if args.camera in {"CAM_FRONT_NARROW", "CAM_FRONT_WIDE"}
            else (1920, 1080)
        )
        if args.source_rgb_width is None:
            appearance_rgb_size = (args.width, args.height)
    shifted_target = any(abs(value) > 1e-9 for value in (
        *args.translate_xyz, *args.rotate_rpy_deg,
    ))
    include_target_source = (
        shifted_target
        if args.include_target_source is None
        else bool(args.include_target_source)
    )
    dataset = Unified3DFieldDataset(
        root=args.root,
        teacher_root=args.teacher_root,
        split_file=args.split,
        split_name=args.split_name,
        distortion_path=args.distortion,
        source_size=source_size,
        appearance_rgb_size=appearance_rgb_size,
        max_input_points=args.max_input_points,
        max_positive_queries=args.max_positive_queries,
        max_negative_queries=args.max_negative_queries,
        include_free_input=True,
        include_temporal_input=bool(
            geometry_config.get("include_temporal_input", False)
        ),
        temporal_input_fraction=float(
            geometry_config.get("temporal_input_fraction", 0.40)
        ),
        temporal_rgb=args.temporal_rgb,
        temporal_rgb_max_views=args.temporal_rgb_max_views,
        temporal_rgb_max_frames=args.temporal_rgb_max_frames,
        include_target_source=include_target_source,
        target_camera=args.camera,
    )
    sample = dataset[args.sample_index]
    source_names = sample["source_cameras"].split("|")
    source_images = sample["source_rgb_images"].permute(0, 2, 3, 1).numpy()
    source_calibrations = [
        _calibration(
            name,
            sample["source_rgb_intrinsics"][index].numpy(),
            sample["source_extrinsics"][index].numpy(),
            appearance_rgb_size[0], appearance_rgb_size[1],
        )
        for index, name in enumerate(source_names)
    ]
    target_calibration = _calibration(
        args.camera,
        (
            sample["target_intrinsic_high"].numpy()
            if args.native_resolution else sample["target_intrinsic"].numpy()
        ),
        sample["target_extrinsic"].numpy(),
        (
            appearance_rgb_size[0]
            if args.native_resolution else source_size[0]
        ),
        (
            appearance_rgb_size[1]
            if args.native_resolution else source_size[1]
        ),
    )
    target = TargetCamera(target_calibration).resized(args.width, args.height)
    target = TargetCamera(_perturb_target(
        target.calibration,
        tuple(args.translate_xyz),
        tuple(args.rotate_rpy_deg),
    ))
    environment = None
    if args.with_sky:
        segmenter = SegFormerDynamicMasker(device=args.device)
        source_sky_masks = [
            segmenter.predict_label_mask(
                np.rint(image.clip(0.0, 1.0) * 255.0).astype(np.uint8),
                {"sky"},
                output_size=appearance_rgb_size,
            )
            for image in source_images
        ]
        environment = render_infinite_sky(
            list(source_images),
            source_sky_masks,
            source_calibrations,
            target.calibration,
            args.width,
            args.height,
        )
        del segmenter
        torch.cuda.empty_cache()
    # Always prepare a source-RGB fallback.  It is composited only where the
    # 3D field and sky have no answer, and remains excluded from geometry metrics.
    fallback_environment = render_directional_source_fallback(
        list(source_images), source_calibrations, target.calibration,
        args.width, args.height,
    )
    with torch.inference_mode():
        target_depth_numpy, _, _ = sparse_depth_from_points(
            sample["geometry_input_xyz"].numpy(),
            target.calibration,
            (args.width, args.height),
        )
        target_depth_prior = torch.from_numpy(target_depth_numpy).to(device)[None, None]
        raw_target_depth_coverage = float(
            (target_depth_prior > 1e-4).float().mean()
        )
        target_depth_prior = _densify_lidar_prior(
            target_depth_prior, args.lidar_prior_radius,
        )
        if args.native_resolution:
            result = _render_native_tiled(
                model=model,
                sample=sample,
                dataset=dataset,
                target=target,
                environment=environment,
                fallback_environment=fallback_environment,
                target_depth_prior=target_depth_prior,
                width=args.width,
                height=args.height,
                tile_size=max(64, args.tile_size),
                device=device,
                sample_count=args.sample_count,
                refine_sample_count=args.refine_sample_count,
                occupancy_threshold=args.occupancy_threshold,
                occupied_support_stride=args.occupied_support_stride,
                rgb_weight_sharpness=args.rgb_weight_sharpness,
                ignore_source_lidar_validity=args.rgb_ignore_lidar_validity,
            )
        else:
            result = raymarch_unified_field(
                model,
                sample["input_features"].to(device),
                sample["input_indices"].to(device),
                tuple(int(value) for value in sample["spatial_shape"].tolist()),
                target,
                dataset.geometry_datasets[0].grid,
                sample["source_images"].to(device)[None],
                sample["source_intrinsics"].to(device)[None],
                sample["source_extrinsics"].to(device)[None],
                sample["source_distortions"].to(device)[None],
                width=args.width,
                height=args.height,
                source_lidar_depth=sample["source_lidar_depth"].to(device)[None],
                source_validity=(
                    None if args.rgb_ignore_lidar_validity
                    else sample["source_validity"].to(device)[None]
                ),
                source_time_offsets=sample["source_time_offsets"].to(device)[None],
                source_alignment_confidence=sample["source_alignment_confidence"].to(device)[None],
                focal_role=sample["focal_role"].to(device)[None],
                source_camera_indices=sample["source_camera_indices"].to(device)[None],
                target_camera_index=sample["target_camera_index"].to(device),
                target_focal_role=sample["target_focal_role"].to(device)[None],
                environment=environment,
                fallback_environment=fallback_environment,
                sample_count=args.sample_count,
                geometry_refine_sample_count=args.refine_sample_count,
                target_depth_prior=target_depth_prior,
                occupancy_threshold=args.occupancy_threshold,
                occupied_support_stride=args.occupied_support_stride,
                source_rgb_images=sample["source_rgb_images"].to(device)[None],
                source_rgb_intrinsics=sample["source_rgb_intrinsics"].to(device)[None],
                source_rgb_distortions=sample["source_rgb_distortions"].to(device)[None],
            )
    target_source = (
        sample["target_rgb_high"] if args.native_resolution else sample["target_rgb"]
    )
    target_rgb = np.asarray(
        Image.fromarray(
            np.rint(target_source.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        ).resize((args.width, args.height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    _save_outputs(args.output, result, target_rgb)
    rendered = result.rgb.detach().cpu().numpy()
    surface = result.surface_validity.detach().cpu().numpy()
    disocclusion = result.true_disocclusion.detach().cpu().numpy()
    fallback = result.fallback_validity.detach().cpu().numpy()
    env_valid = result.environment_validity.detach().cpu().numpy()
    output_valid = result.output_validity.detach().cpu().numpy()
    report = {
        "checkpoint": str(args.checkpoint),
        "geometry_checkpoint_override": (
            str(args.geometry_checkpoint_override)
            if args.geometry_checkpoint_override is not None else None
        ),
        "sequence_id": sample["sequence_id"],
        "frame_id": int(sample["frame_id"]),
        "target_camera": args.camera,
        "target_translation_xyz_m": list(args.translate_xyz),
        "target_rotation_rpy_deg": list(args.rotate_rpy_deg),
        "shifted_target_has_rgb_ground_truth": not shifted_target,
        "source_cameras": source_names,
        "source_time_offsets_s": [
            float(value) for value in sample["source_time_offsets"].tolist()
        ],
        "temporal_rgb_enabled": args.temporal_rgb,
        "temporal_rgb_max_frames": args.temporal_rgb_max_frames,
        "include_current_target_source": include_target_source,
        "appearance_topology_enforced": True,
        "resolution": [args.width, args.height],
        "native_tiled_render": args.native_resolution,
        "tile_size": args.tile_size if args.native_resolution else None,
        "source_feature_resolution": list(source_size),
        "source_rgb_resolution": list(appearance_rgb_size),
        "max_input_points": args.max_input_points,
        "max_positive_queries": args.max_positive_queries,
        "max_negative_queries": args.max_negative_queries,
        "rectified_pinhole": True,
        "rgb_weight_sharpness": args.rgb_weight_sharpness,
        "rgb_ignore_lidar_validity": args.rgb_ignore_lidar_validity,
        "legacy_checkpoint_missing_conditioning": list(missing),
        "structure_coverage": float(result.structure_validity.float().mean()),
        "surface_rgb_coverage": float(result.surface_validity.float().mean()),
        "environment_coverage": float(result.environment_validity.float().mean()),
        "fallback_coverage": float(result.fallback_validity.float().mean()),
        "output_coverage": float(result.output_validity.float().mean()),
        "true_disocclusion_ratio": float(result.true_disocclusion.float().mean()),
        "unknown_structure_ratio": float(result.unknown_structure.float().mean()),
        "target_geometry_depth_coverage": float(
            (target_depth_prior > 1e-4).float().mean()
        ),
        "raw_target_geometry_depth_coverage": raw_target_depth_coverage,
        "lidar_prior_radius": args.lidar_prior_radius,
        "occupancy_threshold": args.occupancy_threshold,
        "occupied_support_stride": args.occupied_support_stride,
        "surface_psnr_db": None if shifted_target else _psnr(rendered, target_rgb, surface),
        "fallback_psnr_db": None if shifted_target else _psnr(
            rendered, target_rgb, fallback,
        ),
        "disocclusion_psnr_db": None if shifted_target else _psnr(
            rendered, target_rgb, disocclusion,
        ),
        "environment_psnr_db": None if shifted_target else _psnr(rendered, target_rgb, env_valid),
        "valid_output_psnr_db": None if shifted_target else _psnr(
            rendered, target_rgb, surface | env_valid,
        ),
        "display_psnr_db": None if shifted_target else _psnr(
            rendered, target_rgb, output_valid,
        ),
        "all_output_psnr_db": None if shifted_target else _psnr(
            rendered, target_rgb, np.ones_like(output_valid, dtype=bool),
        ),
        "geometry_and_sky_validity_excludes_fallback": True,
        "audit_panel_layout": [
            "target_rectified", "reconstruction_with_invalid_magenta", "validity_classes",
        ],
        "render_rgba_transparent_where_invalid": True,
    }
    if device.type == "cuda":
        report["gpu_peak_allocated_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 3,
        )
        report["gpu_peak_reserved_gb"] = round(
            torch.cuda.max_memory_reserved(device) / 1024**3, 3,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
