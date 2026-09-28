"""Compare fixed source RGB against the validated A-ONLY final result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
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


def _font(size: int):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _diff(source: np.ndarray, result: np.ndarray, scale: float = 8.0):
    if source.shape != result.shape:
        result = cv2.resize(result, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    delta = cv2.absdiff(source, result)
    gray = np.max(delta, axis=2).astype(np.float32)
    amplified = np.clip(gray * scale, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(amplified, cv2.COLORMAP_TURBO)[..., ::-1]
    return delta, heat


def _panel(path: Path, source: np.ndarray, result: np.ndarray, heat: np.ndarray, title: str):
    cell = (640, 360)
    title_h = 42
    canvas = Image.new("RGB", (cell[0] * 4, cell[1] + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    entries = (("去畸变源图", source), ("A-ONLY 最终重建", result), ("绝对差异 x8", heat), ("差异叠加", cv2.addWeighted(result, 0.55, heat, 0.45, 0)))
    for col, (label, image) in enumerate(entries):
        x = col * cell[0]
        draw.text((x + 8, 8), f"{title} | {label}", fill=(15, 20, 30), font=_font(17))
        canvas.paste(Image.fromarray(image).convert("RGB").resize(cell, Image.Resampling.LANCZOS), (x, title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95, subsampling=0)


def _seven_view(path: Path, root: Path, pose: str):
    cell = (480, 270)
    title_h = 34
    canvas = Image.new("RGB", (cell[0] * 4, (cell[1] + title_h) * 7), "white")
    draw = ImageDraw.Draw(canvas)
    for row, camera in enumerate(CAMERAS):
        folder = root / pose / camera
        source = np.asarray(Image.open(folder / "source_native_rectified.png").convert("RGB"))
        result = np.asarray(Image.open(folder / "rgbds_a_4k.png").convert("RGB"))
        _, heat = _diff(source, result)
        y = row * (cell[1] + title_h)
        draw.text((6, y + 6), f"{camera} | 源图 | A-ONLY | DIFF", fill=(15, 20, 30), font=_font(17))
        canvas.paste(Image.fromarray(source).resize(cell, Image.Resampling.LANCZOS), (0, y + title_h))
        canvas.paste(Image.fromarray(result).resize(cell, Image.Resampling.LANCZOS), (cell[0], y + title_h))
        canvas.paste(Image.fromarray(heat).resize(cell, Image.Resampling.LANCZOS), (cell[0] * 2, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Source vs validated A-ONLY final reconstruction DIFF")
    parser.add_argument("--a-only-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="The corresponding 2px T0 root containing source_native_rectified.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", default="2026-05-22-10-25-21")
    parser.add_argument("--frame", type=int, default=73)
    parser.add_argument("--diff-scale", type=float, default=8.0)
    args = parser.parse_args()
    report = {"sequence": args.sequence, "frame": args.frame, "a_only_root": str(args.a_only_root), "source_root": str(args.source_root), "poses": {}}
    for pose, translation in POSES.items():
        report["poses"][pose] = {}
        for camera in CAMERAS:
            source_path = args.source_root / pose / camera / "source_native_rectified.png"
            result_path = args.a_only_root / pose / camera / "rgbds_a_4k.png"
            valid_path = args.source_root / pose / camera / "validity_after_2px_surface_repair.npy"
            for path in (source_path, result_path, valid_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            source = np.asarray(Image.open(source_path).convert("RGB"))
            result = np.asarray(Image.open(result_path).convert("RGB"))
            delta, heat = _diff(source, result, args.diff_scale)
            valid = np.load(valid_path).astype(bool)
            if valid.shape != source.shape[:2]:
                raise ValueError(f"validity shape mismatch for {pose}/{camera}: {valid.shape} vs {source.shape[:2]}")
            if result.shape != source.shape:
                result = cv2.resize(result, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                delta, heat = _diff(source, result, args.diff_scale)
            out = args.output / pose / camera
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(delta).save(out / "a_only_abs_diff.png")
            Image.fromarray(heat).save(out / "a_only_abs_diff_x8_heatmap.png")
            _panel(out / "source_vs_a_only_diff_panel.jpg", source, result, heat, f"{pose} | {camera}")
            abs_float = delta.astype(np.float32) / 255.0
            report["poses"][pose][camera] = {
                "source": str(source_path), "a_only": str(result_path), "validity": str(valid_path),
                "resolution": [int(source.shape[1]), int(source.shape[0])],
                "translation_xyz_m": list(translation), "coverage": float(valid.mean()),
                "mae_all": float(abs_float.mean()), "mae_observed": float(abs_float[valid].mean()) if valid.any() else None,
                "mae_unobserved": float(abs_float[~valid].mean()) if (~valid).any() else None,
                "diff_gt_8bit_fraction": float((np.max(delta, axis=2) > 8).mean()),
                "contract": "same sequence/frame/pose/camera; source and A-ONLY aligned to identical rectified raster",
            }
        # Build a dedicated wide paper sheet with source, A-ONLY and DIFF.
        cell = (480, 270); title_h = 34
        canvas = Image.new("RGB", (cell[0] * 3, (cell[1] + title_h) * 7), "white")
        draw = ImageDraw.Draw(canvas)
        for row, camera in enumerate(CAMERAS):
            src = np.asarray(Image.open(args.source_root / pose / camera / "source_native_rectified.png").convert("RGB"))
            res = np.asarray(Image.open(args.a_only_root / pose / camera / "rgbds_a_4k.png").convert("RGB"))
            _, heat = _diff(src, res, args.diff_scale)
            y = row * (cell[1] + title_h)
            draw.text((6, y + 6), f"{camera} | 源图 / A-ONLY / DIFF", fill=(15, 20, 30), font=_font(17))
            for col, image in enumerate((src, res, heat)):
                canvas.paste(Image.fromarray(image).resize(cell, Image.Resampling.LANCZOS), (col * cell[0], y + title_h))
        canvas.save(args.output / f"{pose}_source_a_only_diff_triplet_seven_view.jpg", quality=95, subsampling=0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "poses": list(POSES), "cameras": list(CAMERAS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
