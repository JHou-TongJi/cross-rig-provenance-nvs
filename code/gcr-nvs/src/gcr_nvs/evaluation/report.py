"""Evaluate GCR-NVS checkpoints with mask- and geometry-aware metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset
from gcr_nvs.evaluation.metrics import depth_errors, lpips_score, psnr, ssim_proxy, uncertainty_metrics
from gcr_nvs.models.foundation import GCRNVSFoundationSingleFrame
from gcr_nvs.models.refiner import SingleFrameRefiner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--distortion", type=Path, default=None)
    parser.add_argument("--target-rig", type=Path, default=None)
    parser.add_argument("--model", choices=("lite", "foundation"), default="lite")
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dynamic-mask-root", type=Path, default=None)
    parser.add_argument("--target-camera", default="CAM_FRONT_WIDE")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rgb-residual", type=float, default=0.25)
    parser.add_argument("--samples-per-sequence", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/evaluation.json"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeaveOneOutTorchDataset(args.root, args.sequences, args.distortion, args.width, args.height, max_samples=args.max_samples, target_rig=args.target_rig, dynamic_mask_root=args.dynamic_mask_root, target_camera=args.target_camera, include_foundation_inputs=args.model == "foundation", cache_root=args.cache_root, max_samples_per_sequence=args.samples_per_sequence)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.model == "foundation":
        model = GCRNVSFoundationSingleFrame(use_dino=not args.no_dino).to(device)
    else:
        config = state.get("config", {}) if isinstance(state, dict) else {}
        model = SingleFrameRefiner(input_channels=48, base_channels=int(config.get("base_channels", 48)), max_rgb_residual=args.max_rgb_residual).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            geometry = batch["geometry"].to(device)
            rays = batch["ray_map"].to(device)
            target = batch["target_rgb"].to(device)
            valid = batch["valid_mask"].to(device)
            if args.model == "foundation":
                output = model(
                    geometry,
                    rays,
                    batch["source_images"].to(device),
                    batch["anchor_xyz"].to(device),
                    batch["source_uv"].to(device),
                    batch["target_uv"].to(device),
                    batch["target_anchor_valid"].to(device),
                    batch["source_valid"].to(device),
                    batch["plane_sweep_uv"].to(device),
                    batch["plane_sweep_valid"].to(device),
                )
            else:
                output = model(geometry, rays, batch["rendered_feature"].to(device))
            observed = geometry[:, 7:8]
            hole = 1.0 - observed
            row = {
                "coarse_psnr": psnr(geometry[:, :3], target),
                "coarse_ssim": ssim_proxy(geometry[:, :3], target),
                "final_psnr": psnr(output["rgb"], target),
                "final_ssim": ssim_proxy(output["rgb"], target),
                "observed_ratio": float(observed.mean()),
                "inpaint_ratio": float((output["alpha_effective"] > 0.5).float().mean()),
                "observed_psnr": psnr(output["rgb"], target, observed),
                "hole_psnr": psnr(output["rgb"], target, hole),
                "lpips": lpips_score(output["rgb"], target),
                "depth_support": float((geometry[:, 3:4] > 0).float().mean()),
            }
            row.update(uncertainty_metrics(output["rgb"], target, output["log_uncertainty"], valid))
            values.append(row)
    keys = [key for key in values[0] if values and all(value.get(key) is not None for value in values)]
    report = {"model": args.model, "samples": len(values), **{key: sum(value[key] for value in values) / max(1, len(values)) for key in keys}, "per_sample": values}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
