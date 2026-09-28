"""Export standalone full RGB reconstruction artifacts, not only comparison panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset
from gcr_nvs.models.refiner import SingleFrameRefiner
from gcr_nvs.models.foundation import GCRNVSFoundationSingleFrame


def save_rgb(tensor: torch.Tensor, path: Path, crop_height: int | None = None):
    image = tensor.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    if crop_height is not None:
        image = image[:crop_height]
    Image.fromarray((image * 255).astype(np.uint8)).save(path, format="PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--distortion", type=Path, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--sample-index", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstructions"))
    parser.add_argument("--model", choices=("lite", "foundation"), default="lite")
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--target-rig", type=Path, default=None)
    parser.add_argument("--dynamic-mask-root", type=Path, default=None)
    parser.add_argument("--target-camera", default="CAM_FRONT_WIDE")
    parser.add_argument("--max-rgb-residual", type=float, default=0.25)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeaveOneOutTorchDataset(args.root, [args.sequence], args.distortion, args.width, args.height, max_samples=args.sample_index + 1, target_rig=args.target_rig, dynamic_mask_root=args.dynamic_mask_root, target_camera=args.target_camera)
    sample = dataset[args.sample_index]
    batch = {key: (value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]) for key, value in sample.items()}
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {}) if isinstance(state, dict) else {}
    if args.model == "foundation":
        model = GCRNVSFoundationSingleFrame(use_dino=not args.no_dino).to(device)
    else:
        model = SingleFrameRefiner(input_channels=48, base_channels=int(config.get("base_channels", 48)), max_rgb_residual=args.max_rgb_residual).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    with torch.no_grad():
        geometry = batch["geometry"].to(device)
        rays = batch["ray_map"].to(device)
        rendered_feature = batch["rendered_feature"].to(device)
        if args.model == "foundation":
            output = model(
                geometry, rays, batch["source_images"].to(device), batch["anchor_xyz"].to(device),
                batch["source_uv"].to(device), batch["target_uv"].to(device),
                batch["target_anchor_valid"].to(device), batch["source_valid"].to(device),
                batch["plane_sweep_uv"].to(device), batch["plane_sweep_valid"].to(device),
            )
        else:
            output = model(geometry, rays, rendered_feature)
    name = f"{args.sequence}_{int(batch['frame_id'][0]):06d}_{batch['target_camera'][0]}"
    output_dir = args.output / name
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_height = min(540, args.height)
    save_rgb(batch["target_rgb"][0], output_dir / "target_rgb.png", crop_height)
    save_rgb(geometry[0, :3], output_dir / "coarse_rgb.png", crop_height)
    save_rgb(output["coarse_rgb"][0], output_dir / "coarse_contract_rgb.png", crop_height)
    save_rgb(output["rgb"][0], output_dir / "final_rgb.png", crop_height)
    alpha = output["alpha_effective"][0, 0].detach().cpu().clamp(0, 1).numpy()[:crop_height]
    Image.fromarray((alpha * 255).astype(np.uint8)).save(output_dir / "alpha.png")
    np.save(output_dir / "target_depth.npy", geometry[0, 3].detach().cpu().numpy()[:crop_height])
    np.save(output_dir / "confidence.npy", geometry[0, 8].detach().cpu().numpy()[:crop_height])
    np.save(output_dir / "valid_mask.npy", (geometry[0, 7] > 0).detach().cpu().numpy()[:crop_height])
    np.save(output_dir / "inpaint_mask.npy", (output["alpha_effective"][0, 0] > 0.5).detach().cpu().numpy()[:crop_height])
    np.save(output_dir / "dynamic_mask.npy", geometry[0, 11].detach().cpu().numpy()[:crop_height])
    np.save(output_dir / "source_count.npy", geometry[0, 10].detach().cpu().numpy()[:crop_height])
    np.save(output_dir / "source_provenance.npy", batch["source_provenance"][0].detach().cpu().numpy()[:crop_height])
    print(output_dir)


if __name__ == "__main__":
    main()
