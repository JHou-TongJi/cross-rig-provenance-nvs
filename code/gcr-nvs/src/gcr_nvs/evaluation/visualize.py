"""Save GT/coarse/final comparison panels for qualitative inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from gcr_nvs.datasets.torch_dataset import LeaveOneOutTorchDataset
from gcr_nvs.models.refiner import SingleFrameRefiner


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return Image.fromarray((array * 255).astype(np.uint8))


def make_panel(images: list[tuple[str, Image.Image]]) -> Image.Image:
    width, height = images[0][1].size
    panel = Image.new("RGB", (width * len(images), height + 28), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, image) in enumerate(images):
        panel.paste(image, (index * width, 28))
        draw.text((index * width + 8, 7), label, fill="black")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--distortion", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeaveOneOutTorchDataset(args.root, [args.sequence], args.distortion, args.width, args.height, max_samples=args.samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_config = state.get("config", {}) if isinstance(state, dict) else {}
    model = SingleFrameRefiner(input_channels=48, base_channels=int(saved_config.get("base_channels", 48))).to(device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            geometry = batch["geometry"].to(device)
            rays = batch["ray_map"].to(device)
            rendered_feature = batch["rendered_feature"].to(device)
            target = batch["target_rgb"].to(device)
            output = model(geometry, rays, rendered_feature)
            alpha = output["alpha_effective"][0, 0].clamp(0, 1).cpu().numpy()
            alpha_image = Image.fromarray((alpha * 255).astype(np.uint8)).convert("RGB")
            panel = make_panel([
                ("GT", to_image(target[0])),
                ("Coarse", to_image(geometry[0, :3])),
                ("Dense", to_image(context[0])),
                ("Final", to_image(output["rgb"][0])),
                ("Alpha", alpha_image),
            ])
            name = f"{index:04d}_{batch['target_camera'][0]}_{int(batch['frame_id'][0]):06d}.jpg"
            panel.save(args.output / name, quality=95)
            print(args.output / name)


if __name__ == "__main__":
    main()
