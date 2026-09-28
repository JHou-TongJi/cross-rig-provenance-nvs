"""Export yesterday's 2px T0 result without the blue validity overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


POSES = ("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm")
CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def _font(size: int):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _save_comparison(path: Path, source: Image.Image, blue: Image.Image, clean: Image.Image) -> None:
    cell = (960, 540)
    title_height = 44
    canvas = Image.new("RGB", (cell[0] * 3, cell[1] + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    entries = (
        ("去畸变源图", source),
        ("昨天 2px（蓝色标记无效区）", blue),
        ("昨天 2px（撤掉蓝色，源图兜底）", clean),
    )
    for index, (label, image) in enumerate(entries):
        x = index * cell[0]
        draw.text((x + 10, 7), label, fill=(15, 20, 30), font=_font(23))
        canvas.paste(image.resize(cell, Image.Resampling.LANCZOS), (x, title_height))
    canvas.save(path, quality=95, subsampling=0)


def _save_seven_view(path: Path, root: Path, pose: str) -> None:
    cell = (640, 360)
    title_height = 36
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, camera in enumerate(CAMERAS):
        x = index % 4 * cell[0]
        y = index // 4 * (cell[1] + title_height)
        draw.text((x + 8, y + 6), camera, fill=(15, 20, 30), font=_font(18))
        image = Image.open(root / pose / camera / "t0_surface_2px_no_blue_source_fallback.png").convert("RGB")
        canvas.paste(image.resize(cell, Image.Resampling.LANCZOS), (x, y + title_height))
    canvas.save(path, quality=95, subsampling=0)


def _save_route_comparison(path: Path, input_root: Path, direct_root: Path,
                           reference_root: Path, pose: str) -> None:
    cell = (640, 360)
    title_height = 42
    canvas = Image.new("RGB", (cell[0] * 3, (cell[1] + title_height) * len(CAMERAS)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, camera in enumerate(CAMERAS):
        images = (
            Image.open(input_root / pose / camera / "source_native_rectified.png").convert("RGB"),
            Image.open(direct_root / pose / camera / "t0_surface_2px_no_blue_source_fallback.png").convert("RGB"),
            Image.open(reference_root / pose / camera / "rgbds_a_4k.png").convert("RGB"),
        )
        labels = (f"{camera} / 去畸变源图", "2px T0 + 源图直接合成", "2px T0 + A-1000 扩散")
        y = row * (cell[1] + title_height)
        for column, (label, image) in enumerate(zip(labels, images)):
            x = column * cell[0]
            draw.text((x + 8, y + 7), label, fill=(15, 20, 30), font=_font(19))
            canvas.paste(image.resize(cell, Image.Resampling.LANCZOS), (x, y + title_height))
    canvas.save(path, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path,
        help="Optional A-only diffusion output root used to build source/direct/diffusion comparison sheets.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for pose in POSES:
        report[pose] = {}
        for camera in CAMERAS:
            source_root = args.input / pose / camera
            output_root = args.output / pose / camera
            output_root.mkdir(parents=True, exist_ok=True)
            source = np.asarray(Image.open(source_root / "source_native_rectified.png").convert("RGB"))
            blue = np.asarray(Image.open(source_root / "t0_surface_2px_repaired_blue.png").convert("RGB"))
            valid = np.load(source_root / "validity_after_2px_surface_repair.npy").astype(bool)
            if source.shape != blue.shape or source.shape[:2] != valid.shape:
                raise ValueError(f"shape mismatch for {pose}/{camera}")

            clean = source.copy()
            clean[valid] = blue[valid]
            rgba = np.concatenate([blue, np.uint8(valid[..., None]) * 255], axis=2)
            Image.fromarray(clean).save(output_root / "t0_surface_2px_no_blue_source_fallback.png")
            Image.fromarray(rgba, mode="RGBA").save(output_root / "t0_surface_2px_valid_rgba.png")
            _save_comparison(
                output_root / "source_vs_2px_blue_vs_no_blue.jpg",
                Image.fromarray(source), Image.fromarray(blue), Image.fromarray(clean),
            )
            report[pose][camera] = {
                "two_px_valid_fraction": float(valid.mean()),
                "source_fallback_fraction": float((~valid).mean()),
                "valid_t0_pixels_changed": int(np.any(clean[valid] != blue[valid], axis=1).sum()),
                "contract": "valid 2px T0 pixels are immutable; invalid pixels show rectified same-camera source RGB",
            }
        _save_seven_view(args.output / f"{pose}_2px_no_blue_seven_view.jpg", args.output, pose)
        if args.reference is not None:
            _save_route_comparison(
                args.output / f"{pose}_source_direct_vs_diffusion.jpg",
                args.input, args.output, args.reference, pose,
            )
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
