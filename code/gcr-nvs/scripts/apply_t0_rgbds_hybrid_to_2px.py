"""Apply the A+D2 RGB-D-S diffusion route to yesterday's 2px T0 holes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionInpaintPipeline, T2IAdapter

from gcr_nvs.datasets.t0_rgbds_diffusion import build_rgbds_conditions, resize_hole_mask
from gcr_nvs.dynamics.masks import DynamicMaskProvider
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor
from gcr_nvs.inference.rgbds_completion import load_adapter, prompt_embeddings, sample


POSES = ("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm", "target_rig")
CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def _font(size: int):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _seven_view(root: Path, pose: str, route: str) -> None:
    cell = (640, 360); title = 36
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, camera in enumerate(CAMERAS):
        x = index % 4 * cell[0]; y = index // 4 * (cell[1] + title)
        draw.text((x + 8, y + 6), camera, fill=(15, 20, 30), font=_font(18))
        image = Image.open(root / pose / camera / f"rgbds_{route}_4k.png").convert("RGB")
        canvas.paste(image.resize(cell, Image.Resampling.LANCZOS), (x, y + title))
    canvas.save(root / f"{pose}_rgbds_{route}_seven_view.jpg", quality=95, subsampling=0)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--prefilled-root", type=Path,
        help="Optional UniWorld/DA3 gate output root. Its filled pixels are locked and only the residual hole is generated.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-a", type=Path, required=True)
    parser.add_argument("--adapter-d2", type=Path)
    parser.add_argument(
        "--route", choices=("a", "hybrid"), default="hybrid",
        help="Use the validated general expert alone, or route D2 inside vehicle-risk regions.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--poses", nargs="+", choices=POSES, default=list(POSES))
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    size = (args.width, args.height)
    device = torch.device("cuda")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    for module in (pipe.vae, pipe.text_encoder, pipe.unet):
        module.requires_grad_(False).eval()
    prompt = prompt_embeddings(pipe, device).to(torch.float16)
    adapter_a = load_adapter(args.adapter_a, device)
    if args.route == "hybrid" and args.adapter_d2 is None:
        parser.error("--adapter-d2 is required when --route=hybrid")
    adapter_d2 = load_adapter(args.adapter_d2, device) if args.route == "hybrid" else None
    descriptor = FrozenDinoAnyUpDescriptor(
        model_name="dinov2_vitl14", output_channels=8, dino_input_size=(364, 644),
        anyup_checkpoint=project / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=project / "third_party/anyup", q_chunk_size=4096,
    ).to(device).eval()
    detector = DynamicMaskProvider(device="cuda", score_threshold=0.5)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for pose_index, pose in enumerate(args.poses):
        report[pose] = {}
        for camera_index, camera in enumerate(args.cameras):
            source = args.input / pose / camera
            output = args.output / pose / camera
            started = time.perf_counter()
            output.mkdir(parents=True, exist_ok=True)
            blue_hd = np.asarray(Image.open(source / "t0_surface_2px_repaired_blue.png").convert("RGB"))
            valid_hd = np.load(source / "validity_after_2px_surface_repair.npy").astype(bool)
            depth_hd = np.load(source / "target_surface_depth_2px_repaired.npy").astype(np.float32)
            base_hd = blue_hd.copy()
            gate_fill_hd = np.zeros_like(valid_hd, dtype=bool)
            if args.prefilled_root is not None:
                prefilled = args.prefilled_root / pose / camera / "t0_uniview_background_gate.png"
                gate_mask = args.prefilled_root / pose / camera / "uniview_gate_fill_mask.png"
                if not prefilled.exists() or not gate_mask.exists():
                    raise FileNotFoundError(f"missing prefilled gate artifacts for {pose}/{camera}: {prefilled}, {gate_mask}")
                base_hd = np.asarray(Image.open(prefilled).convert("RGB"))
                gate_fill_hd = np.asarray(Image.open(gate_mask).convert("L")) > 127
                if base_hd.shape != blue_hd.shape or gate_fill_hd.shape != valid_hd.shape:
                    raise ValueError("prefilled gate artifacts must match the T0 native resolution")
                gate_fill_hd &= ~valid_hd
            residual_hole_hd = (~valid_hd) & (~gate_fill_hd)
            rgb_low = cv2.resize(base_hd, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            hole_low = resize_hole_mask(residual_hole_hd, size)
            rgb_low[hole_low] = 0.0
            depth_low = cv2.resize(depth_hd, size, interpolation=cv2.INTER_NEAREST)
            dynamic_low, _, _ = detector(np.uint8(rgb_low * 255))
            image = torch.from_numpy(rgb_low.transpose(2, 0, 1).copy()).to(device)[None]
            with torch.inference_mode():
                semantic = descriptor(image, (args.height, args.width))[0].cpu().numpy()
            conditions = build_rgbds_conditions(
                rgb_low, depth_low, dynamic_low, semantic, hole_low,
            )
            batch = {
                "target_rgb": torch.from_numpy(rgb_low.transpose(2, 0, 1).copy())[None],
                "masked_rgb": torch.from_numpy(conditions["masked_rgb"].transpose(2, 0, 1).copy())[None],
                "hole_mask": torch.from_numpy(conditions["hole_mask"][None, None].copy()),
                "control": torch.from_numpy(conditions["control"][None].copy()),
                "forbid_foreground_extension": torch.from_numpy(
                    conditions["forbid_foreground_extension"][None, None].copy()
                ),
            }
            seed = args.seed + pose_index * 100 + camera_index
            masked, hole, _, final_a = sample(
                pipe, adapter_a, batch, prompt, "a", args.inference_steps, seed,
            )
            if args.route == "hybrid":
                _, _, _, final_d2 = sample(
                    pipe, adapter_d2, batch, prompt, "d", args.inference_steps, seed,
                )
                risk = batch["forbid_foreground_extension"].to(device)
                completed = final_a * (1.0 - risk) + final_d2 * risk
                contract = "A outside vehicle risk; D2 inside risk; 2px observed pixels immutable"
            else:
                completed = final_a
                contract = "validated A-only completion; 2px observed pixels immutable"
            generated_low = completed[0].float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            generated_hd = cv2.resize(
                generated_low, (blue_hd.shape[1], blue_hd.shape[0]), interpolation=cv2.INTER_LANCZOS4,
            )
            final_hd = base_hd.copy()
            final_hd[residual_hole_hd] = np.uint8(np.clip(generated_hd[residual_hole_hd], 0, 1) * 255)
            Image.fromarray(final_hd).save(output / f"rgbds_{args.route}_4k.png")
            Image.fromarray(np.uint8(generated_low * 255)).save(output / f"rgbds_{args.route}_low.png")
            Image.fromarray(np.uint8(conditions["forbid_foreground_extension"] * 255)).save(
                output / "vehicle_extension_risk.png"
            )
            report[pose][camera] = {
                "original_2px_coverage": float(valid_hd.mean()),
                "gate_filled_fraction_of_hole": float(gate_fill_hd.sum() / max((~valid_hd).sum(), 1)),
                "generated_fraction": float(residual_hole_hd.mean()),
                "vehicle_risk_fraction_of_hole_low": float(
                    conditions["forbid_foreground_extension"].sum() / max(hole_low.sum(), 1)
                ),
                "locked_pixels_modified": int(np.any(final_hd[valid_hd | gate_fill_hd] != base_hd[valid_hd | gate_fill_hd], axis=1).sum()),
                "route": args.route,
                "contract": contract + "; DA3/UniWorld gate-filled pixels locked" if args.prefilled_root else contract,
                "timing_s": {"rgbds_completion_s": time.perf_counter() - started, "device": str(device), "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None},
            }
            print(json.dumps({"pose": pose, "camera": camera, **report[pose][camera]}), flush=True)
        if tuple(args.cameras) == CAMERAS:
            _seven_view(args.output, pose, args.route)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
