"""Apply DA3/UniWorld-style background visibility gating to a T0 render.

Only pixels that are invalid in T0 and supported by the DA3-first
triple-reprojection diagnostic are copied. Existing valid T0 pixels are never
read from the gate image and therefore remain byte-identical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)
POSES = ("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm", "target_rig")


def _font(size: int):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _sheet(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    cell = (640, 360)
    title = 36
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title) * ((len(entries) + 3) // 4)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(entries):
        x = index % 4 * cell[0]
        y = index // 4 * (cell[1] + title)
        draw.text((x + 8, y + 5), label, fill=(15, 20, 30), font=_font(17))
        if image.dtype != np.uint8:
            image = np.uint8(np.clip(image, 0, 1) * 255)
        canvas.paste(Image.fromarray(image).convert("RGB").resize(cell, Image.Resampling.LANCZOS), (x, y + title))
    canvas.save(path, quality=95, subsampling=0)


def main() -> None:
    bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-root", type=Path, default=bundle / "runs/t0_target_surface_2px_repair_all_20260824")
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose", choices=POSES, default="mixed_20_50cm")
    parser.add_argument("--camera", choices=CAMERAS, default="CAM_BACK_LEFT")
    args = parser.parse_args()

    source = args.t0_root / args.pose / args.camera
    gate = args.gate_root / args.pose / args.camera
    output = args.output / args.pose / args.camera
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    t0_path = source / "t0_surface_2px_repaired_blue.png"
    valid_path = source / "validity_after_2px_surface_repair.npy"
    t0 = np.asarray(Image.open(t0_path).convert("RGB"))
    valid = np.load(valid_path).astype(bool)
    gate_rgb = np.asarray(Image.open(gate / "background_rgb_nearest.png").convert("RGB"))
    gate_mask = np.asarray(Image.open(gate / "triple_reprojection_visibility.png").convert("L")) > 127
    gate_mask = _resize_mask(gate_mask, valid.shape)
    if gate_rgb.shape[:2] != valid.shape:
        gate_rgb = cv2.resize(gate_rgb, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    fill = (~valid) & gate_mask
    final = t0.copy()
    final[fill] = gate_rgb[fill]
    Image.fromarray(final).save(output / "t0_uniview_background_gate.png")
    Image.fromarray(np.uint8(fill * 255)).save(output / "uniview_gate_fill_mask.png")
    _sheet(output / "t0_uniview_background_gate_panel.jpg", [
        ("T0 blue residual", t0),
        ("DA3 gate RGB", gate_rgb),
        ("triple visibility", gate_mask.astype(np.float32)),
        ("T0 + gated background", final),
    ])
    report = {
        "pose": args.pose,
        "camera": args.camera,
        "resolution": [int(valid.shape[1]), int(valid.shape[0])],
        "t0_valid_fraction": float(valid.mean()),
        "triple_visibility_fraction": float(gate_mask.mean()),
        "filled_fraction_of_all_pixels": float(fill.mean()),
        "filled_fraction_of_t0_hole": float(fill.sum() / max((~valid).sum(), 1)),
        "observed_pixels_modified": int(np.any(final[valid] != t0[valid], axis=1).sum()),
        "contract": "only T0-invalid pixels supported by DA3 triple visibility are copied; no diffusion; valid T0 pixels immutable",
        "source": str(t0_path),
        "gate": str(gate),
        "timing_s": {"gate_composite_s": time.perf_counter() - started},
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
