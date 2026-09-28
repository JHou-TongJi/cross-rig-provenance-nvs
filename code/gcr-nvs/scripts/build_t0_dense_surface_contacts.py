"""Build seven-camera contacts for the T0 dense-surface inference stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STAGES = (
    ("source_native_rectified.png", "去畸变源 RGB", False),
    ("da3_depth_before_structure.png", "原始 DA3 稠密深度", True),
    ("depth_after_structure.png", "结构校准后源表面", True),
    ("target_surface_depth_before_rgb.png", "SE(3) 后目标表面（RGB 前）", False),
    ("validity.png", "目标表面有效性", False),
    ("t0_highres_blue.png", "源图取色后 T0 RGB", False),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    cameras = summary["target_cameras"]
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    title_font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    camera_font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    cell = (480, 270)
    top, row_label = 44, 170
    for pose in summary["reports"]:
        canvas = Image.new(
            "RGB", (row_label + cell[0] * len(STAGES), top + cell[1] * len(cameras)), "white",
        )
        draw = ImageDraw.Draw(canvas)
        for column, (_, label, _) in enumerate(STAGES):
            draw.text((row_label + column * cell[0] + 8, 8), label, fill=(15, 20, 30), font=title_font)
        for row, camera in enumerate(cameras):
            y = top + row * cell[1]
            draw.text((10, y + 116), camera, fill=(15, 20, 30), font=camera_font)
            for column, (filename, _, sidecar) in enumerate(STAGES):
                source = (
                    args.root / "source_sidecars" / camera / filename
                    if sidecar else args.root / pose / camera / filename
                )
                image = Image.open(source).convert("RGB").resize(cell, Image.Resampling.LANCZOS)
                canvas.paste(image, (row_label + column * cell[0], y))
        canvas.save(args.root / f"{pose}_dense_surface_all_cameras.jpg", quality=94, subsampling=0)


if __name__ == "__main__":
    main()
