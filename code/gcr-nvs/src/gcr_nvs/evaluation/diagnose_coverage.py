"""Compare two plane-sweep checkpoints at pixel-level coverage boundaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset
from gcr_nvs.rendering.plane_sweep import PlaneSweepRenderer


def calibrations(batch, device):
    source = [
        {
            "K": batch["source_intrinsics"][:, index].to(device),
            "D": batch["source_distortions"][:, index].to(device),
            "T_world": batch["source_extrinsics"][:, index].to(device),
        }
        for index in range(batch["source_intrinsics"].shape[1])
    ]
    target = {
        "K": batch["target_intrinsic"].to(device),
        "D": batch["target_distortion"].to(device),
        "T_world": batch["target_extrinsic"].to(device),
    }
    return source, target


def radial_regions(height: int, width: int, device):
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    radius = torch.sqrt(x.square() + y.square()).clamp(max=1.0)[None, None]
    return radius, {
        "center": radius < 0.50,
        "middle": (radius >= 0.50) & (radius < 0.80),
        "edge": radius >= 0.80,
    }


def mask_rate(mask: torch.Tensor, region: torch.Tensor) -> float:
    return float((mask & region).sum() / region.sum().clamp(min=1))


def load_model(path: Path, depth_layers: int, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PlaneSweepRenderer(num_depth_layers=depth_layers).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return model.eval()


def save_difference(old_valid, new_valid, path: Path):
    old = old_valid[0, 0].detach().cpu() > 0.5
    new = new_valid[0, 0].detach().cpu() > 0.5
    image = torch.zeros((*old.shape, 3), dtype=torch.uint8)
    image[old & new] = torch.tensor([180, 180, 180], dtype=torch.uint8)
    image[old & ~new] = torch.tensor([255, 40, 40], dtype=torch.uint8)
    image[~old & new] = torch.tensor([40, 220, 80], dtype=torch.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.numpy()).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("old_checkpoint", type=Path)
    parser.add_argument("new_checkpoint", type=Path)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--distortion", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--depth-layers", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeaveOneOutTorchDataset(
        args.root, args.sequences, args.distortion, args.width, args.height,
        max_points=args.max_points, target_camera=None, foundation_minimal_inputs=True,
        balance_target_cameras=True, max_samples_per_sequence=args.samples_per_sequence,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    old_model = load_model(args.old_checkpoint, args.depth_layers, device)
    new_model = load_model(args.new_checkpoint, args.depth_layers, device)
    _, regions = radial_regions(args.height, args.width, device)
    rows = []
    saved = set()
    with torch.no_grad():
        for batch in loader:
            source, target = calibrations(batch, device)
            source_images = batch["source_images"].to(device)
            lidar_depth = batch["geometry"][:, 3:4].to(device)
            old = old_model(source_images, source, target, lidar_depth)
            new = new_model(source_images, source, target, lidar_depth)
            old_valid = old["valid_mask"] > 0.5
            new_valid = new["valid_mask"] > 0.5
            lost = old_valid & ~new_valid
            gained = ~old_valid & new_valid
            camera = batch["target_camera"][0]
            row = {
                "camera": camera,
                "old_coverage": float(old_valid.float().mean()),
                "new_coverage": float(new_valid.float().mean()),
                "lost_fraction": float(lost.float().mean()),
                "gained_fraction": float(gained.float().mean()),
                "lost_by_region": {name: mask_rate(lost, region) for name, region in regions.items()},
                "gained_by_region": {name: mask_rate(gained, region) for name, region in regions.items()},
                "old_confidence": float(old["confidence"].mean()),
                "new_confidence": float(new["confidence"].mean()),
                "old_depth_confidence": float(old["depth_confidence"].mean()),
                "new_depth_confidence": float(new["depth_confidence"].mean()),
                "old_photometric_confidence": float(old["photometric_confidence"].mean()),
                "new_photometric_confidence": float(new["photometric_confidence"].mean()),
                "mean_depth_delta_m": float((new["depth"] - old["depth"]).abs().mean()),
                "lost_depth_delta_m": float(
                    ((new["depth"] - old["depth"]).abs() * lost).sum()
                    / lost.sum().clamp(min=1)
                ),
            }
            rows.append(row)
            if args.images is not None and camera not in saved:
                save_difference(old_valid, new_valid, args.images / f"{camera}_coverage_delta.png")
                saved.add(camera)

    cameras = sorted({row["camera"] for row in rows})
    numeric = (
        "old_coverage", "new_coverage", "lost_fraction", "gained_fraction",
        "old_confidence", "new_confidence", "old_depth_confidence", "new_depth_confidence",
        "old_photometric_confidence", "new_photometric_confidence",
        "mean_depth_delta_m", "lost_depth_delta_m",
    )
    aggregate = {key: sum(row[key] for row in rows) / len(rows) for key in numeric}
    aggregate["lost_by_region"] = {
        name: sum(row["lost_by_region"][name] for row in rows) / len(rows) for name in regions
    }
    aggregate["gained_by_region"] = {
        name: sum(row["gained_by_region"][name] for row in rows) / len(rows) for name in regions
    }
    per_camera = defaultdict(dict)
    for camera in cameras:
        selected = [row for row in rows if row["camera"] == camera]
        per_camera[camera] = {key: sum(row[key] for row in selected) / len(selected) for key in numeric}
        per_camera[camera]["lost_by_region"] = {
            name: sum(row["lost_by_region"][name] for row in selected) / len(selected) for name in regions
        }
    report = {"samples": len(rows), "aggregate": aggregate, "per_camera": per_camera, "per_sample": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"samples": len(rows), "aggregate": aggregate, "per_camera": per_camera}, indent=2))


if __name__ == "__main__":
    main()
