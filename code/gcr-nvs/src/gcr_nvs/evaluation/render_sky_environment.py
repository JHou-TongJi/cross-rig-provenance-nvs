"""Composite an infinity-sky environment behind fixed LiDAR Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.camera import TargetCamera
from gcr_nvs.geometry.dynamic_mask import SegFormerDynamicMasker
from gcr_nvs.rendering.environment import render_infinite_sky


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
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
    parser.add_argument("gaussian", type=Path)
    parser.add_argument("--sequence", default="2026-05-22-15-09-17")
    parser.add_argument("--frame-id", type=int, default=75)
    parser.add_argument("--target-camera", default="CAM_FRONT_NARROW")
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/geometry_diagnostics/sky_environment"),
    )
    args = parser.parse_args()

    sequence_dir = Path(args.sequence)
    record = next(
        item for item in build_sequence_manifest(sequence_dir, compute_quality=False)
        if item.frame_id == args.frame_id
    )
    calibrations = load_calibrations(
        sequence_dir / record.camera_config,
        Path("camera_intric.yaml"),
    )
    target_calibration = calibrations[args.target_camera]
    width = args.width or target_calibration.width
    height = args.height or target_calibration.height
    if (args.width is None) != (args.height is None):
        raise ValueError("set both --width and --height, or neither for native resolution")
    source_names = tuple(name for name in CAMERA_NAMES if name != args.target_camera)
    source_images = [
        _load_rgb(
            sequence_dir / record.cameras[name],
            args.source_width,
            args.source_height,
        )
        for name in source_names
    ]
    source_calibrations = [
        TargetCamera(calibrations[name]).resized(
            args.source_width, args.source_height,
        ).calibration
        for name in source_names
    ]
    segmenter = SegFormerDynamicMasker(
        device=args.device,
        dilation_pixels=0,
    )
    source_sky_masks = [
        segmenter.predict_label_mask(
            np.rint(image * 255.0).astype(np.uint8),
            {"sky"},
            output_size=(args.source_width, args.source_height),
        )
        for image in source_images
    ]
    environment = render_infinite_sky(
        source_images,
        source_sky_masks,
        source_calibrations,
        target_calibration,
        width=width,
        height=height,
    )
    target = _load_rgb(
        sequence_dir / record.cameras[args.target_camera],
        width,
        height,
    )
    target_sky = segmenter.predict_label_mask(
        np.rint(target * 255.0).astype(np.uint8),
        {"sky"},
        output_size=(width, height),
    )
    with np.load(args.gaussian) as payload:
        gaussian_rgb = payload["rgb"].astype(np.float32)
        alpha = payload["alpha"].astype(np.float32)
        gaussian_valid = payload["appearance_validity"].astype(bool)
    composite = gaussian_rgb.copy()
    sky = environment.validity
    composite[sky] = (
        gaussian_rgb[sky] * alpha[sky, None]
        + environment.rgb[sky] * (1.0 - alpha[sky, None])
    )
    final_valid = gaussian_valid | sky
    intersection = sky & target_sky
    union = sky | target_sky
    precision = float(intersection.sum() / max(sky.sum(), 1))
    recall = float(intersection.sum() / max(target_sky.sum(), 1))
    target_image = np.rint(target * 255.0).astype(np.uint8)
    gaussian_image = np.rint(gaussian_rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    sky_image = np.rint(environment.rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    composite_image = np.rint(composite.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    mask_image = np.zeros_like(target_image)
    mask_image[..., 2] = sky.astype(np.uint8) * 255
    mask_image[..., 1] = target_sky.astype(np.uint8) * 255
    panel = np.concatenate([
        target_image,
        gaussian_image,
        sky_image,
        composite_image,
        mask_image,
    ], axis=1)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_image).save(args.output_prefix.with_suffix(".png"))
    comparison = Image.fromarray(panel)
    comparison.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    comparison.save(args.output_prefix.with_name(
        args.output_prefix.name + ".comparison.png"
    ))
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        rgb=composite,
        validity=final_valid,
        sky_rgb=environment.rgb,
        sky_validity=sky,
        sky_source_count=environment.source_count,
        target_sky_mask=target_sky,
        gaussian_rgb=gaussian_rgb,
        gaussian_alpha=alpha,
    )
    report = {
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "target_camera": args.target_camera,
        "infinity_background": True,
        "finite_sky_depth": False,
        "output_resolution": [width, height],
        "native_resolution": [target_calibration.width, target_calibration.height],
        "native_size_output": bool(
            width == target_calibration.width and height == target_calibration.height
        ),
        "gaussian_appearance_coverage": float(gaussian_valid.mean()),
        "sky_coverage": float(sky.mean()),
        "final_coverage": float(final_valid.mean()),
        "target_sky_ratio": float(target_sky.mean()),
        "sky_iou": float(intersection.sum() / max(union.sum(), 1)),
        "sky_precision": precision,
        "sky_recall": recall,
        "sky_psnr_db": _psnr(environment.rgb, target, intersection),
        "composite_valid_psnr_db": _psnr(composite, target, final_valid),
        "invalid_rgb_exactly_zero": bool(np.all(composite[~final_valid] == 0.0)),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
