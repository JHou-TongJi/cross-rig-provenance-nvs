"""Render every camera view and compare an unchanged and shifted target rig."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset
from gcr_nvs.evaluation.metrics import psnr, ssim_proxy
from gcr_nvs.models.foundation_v2 import FoundationV2SingleFrame
from gcr_nvs.models.refiner import LiteTemporalRefiner
from gcr_nvs.rendering.depth_reprojection import align_metric_depth_to_lidar, forward_splat_rgb


def _calibrations(batch: dict, device: str):
    source_intrinsics = batch["source_intrinsics"].to(device)
    source_extrinsics = batch["source_extrinsics"].to(device)
    source_distortions = batch["source_distortions"].to(device)
    source = [
        {
            "K": source_intrinsics[:, index],
            "D": source_distortions[:, index],
            "T_world": source_extrinsics[:, index],
        }
        for index in range(source_intrinsics.shape[1])
    ]
    target = {
        "K": batch["target_intrinsic"].to(device),
        "D": batch["target_distortion"].to(device),
        "T_world": batch["target_extrinsic"].to(device),
    }
    return source, target


def _image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype("uint8"), "RGB")


def _render(model, batch: dict, device: str):
    source_images = batch["source_images"].to(device)
    geometry = batch["geometry"].to(device)
    source_calibs, target_calib = _calibrations(batch, device)
    return model(
        source_images,
        source_calibs,
        target_calib,
        batch["ray_map"].to(device),
        geometry[:, 3:4],
        geometry,
    )


def _dataset(
    root: Path,
    sequence: str,
    camera: str,
    distortion: Path,
    rig: Path | None,
    include_target_source: bool = False,
):
    return LeaveOneOutTorchDataset(
        root,
        [sequence],
        distortion,
        width=512,
        height=288,
        max_points=2000,
        target_rig=rig,
        target_camera=camera,
        foundation_minimal_inputs=True,
        include_target_source=include_target_source,
        max_samples_per_sequence=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--distortion", type=Path, required=True)
    parser.add_argument("--shifted-rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FoundationV2SingleFrame(plane_sweep_depth_layers=32).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model, cache_dir="outputs/model_cache")
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        args.depth_model, cache_dir="outputs/model_cache",
    ).to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    overview = Image.new("RGB", (1536, 288 * len(CAMERA_NAMES)), "black")
    for row_index, camera in enumerate(CAMERA_NAMES):
        base_loader = DataLoader(
            _dataset(args.root, args.sequence, camera, args.distortion, None),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        shifted_loader = DataLoader(
            _dataset(
                args.root,
                args.sequence,
                camera,
                args.distortion,
                args.shifted_rig,
                include_target_source=True,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        base_batch = next(iter(base_loader))
        shifted_batch = next(iter(shifted_loader))
        with torch.no_grad():
            base_output = _render(model, base_batch, device)
            shifted_output = _render(model, shifted_batch, device)

            depth_inputs = depth_processor(images=_image(base_batch["target_rgb"]), return_tensors="pt")
            depth_inputs = {key: value.to(device) for key, value in depth_inputs.items()}
            predicted_metric_depth = depth_model(**depth_inputs).predicted_depth[:, None]
            predicted_metric_depth = F.interpolate(
                predicted_metric_depth, (288, 512), mode="bicubic", align_corners=False,
            )
            metric_source_depth = align_metric_depth_to_lidar(
                predicted_metric_depth,
                base_batch["geometry"][:, 3:4].to(device),
            )
            reprojected_rgb, _, reprojected_valid = forward_splat_rgb(
                base_batch["target_rgb"].to(device),
                metric_source_depth,
                base_batch["target_intrinsic"].to(device),
                base_batch["target_extrinsic"].to(device),
                shifted_batch["target_intrinsic"].to(device),
                shifted_batch["target_extrinsic"].to(device),
            )
            shifted_rgb = LiteTemporalRefiner._multiband_blend(
                reprojected_rgb,
                shifted_output["rgb"],
                reprojected_valid,
            ).clamp(0.0, 1.0)

        source_image = _image(base_batch["target_rgb"])
        base_image = _image(base_output["rgb"])
        shifted_image = _image(shifted_rgb)
        camera_dir = args.output / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        source_image.save(camera_dir / "source_rgb.png")
        base_image.save(camera_dir / "reconstruction_base.png")
        shifted_image.save(camera_dir / "reconstruction_up20cm.png")

        comparison = Image.new("RGB", (1536, 288), "black")
        comparison.paste(source_image, (0, 0))
        comparison.paste(base_image, (512, 0))
        comparison.paste(shifted_image, (1024, 0))
        comparison_draw = ImageDraw.Draw(comparison)
        for x, label in (
            (8, f"{camera} | SOURCE"),
            (520, "BASE RECONSTRUCTION"),
            (1032, "UP 0.20M RECONSTRUCTION"),
        ):
            comparison_draw.text((x, 8), label, fill="yellow", stroke_width=2, stroke_fill="black")
        comparison.save(camera_dir / "comparison.png")
        overview.paste(comparison, (0, row_index * 288))

        rows.append({
            "camera": camera,
            "base_source_cameras": [name for name in CAMERA_NAMES if name != camera],
            "shifted_source_cameras": list(CAMERA_NAMES),
            "shifted_includes_target_source": True,
            "target_resolution": [512, 288],
            "base_psnr_vs_source": psnr(base_output["rgb"], base_batch["target_rgb"].to(device)),
            "base_ssim_vs_source": ssim_proxy(base_output["rgb"], base_batch["target_rgb"].to(device)),
            "shifted_psnr_vs_source_reference": psnr(
                shifted_rgb, shifted_batch["target_rgb"].to(device),
            ),
            "shifted_ssim_vs_source_reference": ssim_proxy(
                shifted_rgb, shifted_batch["target_rgb"].to(device),
            ),
            "base_coverage": float(base_output["target_geometry"][:, 7:8].mean()),
            "shifted_coverage": float(shifted_output["target_geometry"][:, 7:8].mean()),
            "dense_reprojection_coverage": float(reprojected_valid.mean()),
        })

    overview.save(args.output / "all_views_overview.png")
    (args.output / "report.json").write_text(json.dumps({
        "sequence": args.sequence,
        "shift": {"vehicle_offset_m": [0.0, 0.0, 0.20]},
        "note": "shifted metrics use the original camera image as a reference, not ground truth for the shifted viewpoint",
        "views": rows,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "views": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
