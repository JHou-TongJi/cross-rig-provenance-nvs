"""Build source-vs-blue-T0 comparison panels from the validated result cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)
POSES = {
    "mixed_5_10cm": (0.06, -0.08, 0.05),
    "mixed_10_20cm": (-0.14, 0.12, 0.18),
    "mixed_20_50cm": (0.32, -0.24, 0.42),
}


def font(size: int):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def panel(path: Path, source: Image.Image, blue: Image.Image, validity: np.ndarray, title: str):
    cell = (720, 405)
    title_h = 44
    canvas = Image.new("RGB", (cell[0] * 3, cell[1] + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    mask = Image.fromarray(np.uint8(validity) * 255).convert("RGB")
    for index, (label, image) in enumerate((("去畸变源图", source), ("T0 重建（蓝色=未观测）", blue), ("有效性 mask", mask))):
        x = index * cell[0]
        draw.text((x + 10, 8), f"{title} | {label}", fill=(15, 20, 30), font=font(22))
        canvas.paste(image.convert("RGB").resize(cell, Image.Resampling.LANCZOS), (x, title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95, subsampling=0)


def seven_view(path: Path, root: Path, pose: str):
    cell = (480, 270)
    title_h = 34
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title_h) * 7), "white")
    draw = ImageDraw.Draw(canvas)
    for row, camera in enumerate(CAMERAS):
        folder = root / pose / camera
        source = Image.open(folder / "source_native_rectified.png").convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        blue = Image.open(folder / "t0_surface_2px_repaired_blue.png").convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        y = row * (cell[1] + title_h)
        draw.text((6, y + 6), f"{camera} | 左：源图    右：T0 蓝色未观测区", fill=(15, 20, 30), font=font(18))
        canvas.paste(source, (0, y + title_h))
        canvas.paste(blue, (cell[0], y + title_h))
        # Keep two blank cells available for consistent 4-column paper layout.
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95, subsampling=0)


def overview(path: Path, root: Path):
    cell = (320, 180)
    title_h = 30
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title_h) * 6), "white")
    draw = ImageDraw.Draw(canvas)
    for row, pose in enumerate(POSES):
        for col, camera in enumerate(CAMERAS[:4]):
            folder = root / pose / camera
            image = Image.open(folder / "t0_surface_2px_repaired_blue.png").convert("RGB").resize(cell, Image.Resampling.LANCZOS)
            x = col * cell[0]
            y = row * 2 * (cell[1] + title_h)
            draw.text((x + 5, y + 4), f"{pose} / {camera} / T0", fill=(15, 20, 30), font=font(14))
            canvas.paste(image, (x, y + title_h))
    canvas.save(path, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed source RGB and validated blue T0 outputs")
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", default="2026-05-22-10-25-21")
    parser.add_argument("--frame", type=int, default=73)
    args = parser.parse_args()
    reports = {}
    for pose, translation in POSES.items():
        reports[pose] = {}
        for camera in CAMERAS:
            folder = args.reconstruction_root / pose / camera
            source_path = folder / "source_native_rectified.png"
            blue_path = folder / "t0_surface_2px_repaired_blue.png"
            valid_path = folder / "validity_after_2px_surface_repair.npy"
            for path in (source_path, blue_path, valid_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            source = Image.open(source_path).convert("RGB")
            blue = Image.open(blue_path).convert("RGB")
            valid = np.load(valid_path).astype(bool)
            panel(args.output / pose / camera / "source_vs_t0_blue_panel.jpg", source, blue, valid, f"{pose} | {camera}")
            reports[pose][camera] = {
                "source": str(source_path), "reconstruction": str(blue_path),
                "validity": str(valid_path), "resolution": [source.width, source.height],
                "translation_xyz_m": list(translation), "coverage": float(valid.mean()),
                "unobserved_fraction": float((~valid).mean()),
                "contract": "same rectified source frame; blue marks invalid/unobserved T0 pixels",
            }
        seven_view(args.output / f"{pose}_source_vs_t0_blue_seven_view.jpg", args.reconstruction_root, pose)
    overview(args.output / "all_poses_t0_blue_overview.jpg", args.reconstruction_root)
    manifest = {
        "sequence": args.sequence, "frame": args.frame,
        "reconstruction_root": str(args.reconstruction_root),
        "poses": {name: {"translation_xyz_m": list(value), "range": name.replace("mixed_", "")} for name, value in POSES.items()},
        "reports": reports,
        "visual_contract": "source_native_rectified.png and t0_surface_2px_repaired_blue.png are paired by identical pose/camera/frame",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "poses": list(POSES), "cameras": list(CAMERAS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
