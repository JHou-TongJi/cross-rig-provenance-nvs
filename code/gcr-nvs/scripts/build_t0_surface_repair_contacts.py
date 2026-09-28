"""Build original-vs-2px target-surface repair contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    cameras = summary["target_cameras"]
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    title_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    camera_font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    cell = (960, 540)
    top, label_width = 48, 190
    for pose in summary["reports"]:
        canvas = Image.new("RGB", (label_width + cell[0] * 2, top + cell[1] * len(cameras)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((label_width + 12, 10), "T0 原始 RGB（大洞蓝色）", fill=(15, 20, 30), font=title_font)
        draw.text((label_width + cell[0] + 12, 10), "目标表面 2px 修补后（大洞保留）", fill=(15, 20, 30), font=title_font)
        for row, camera in enumerate(cameras):
            y = top + row * cell[1]
            draw.text((8, y + 245), camera, fill=(15, 20, 30), font=camera_font)
            original = Image.open(args.root / pose / camera / "t0_highres_blue.png").convert("RGB")
            repaired = Image.open(args.root / pose / camera / "t0_surface_2px_repaired_blue.png").convert("RGB")
            canvas.paste(original.resize(cell, Image.Resampling.LANCZOS), (label_width, y))
            canvas.paste(repaired.resize(cell, Image.Resampling.LANCZOS), (label_width + cell[0], y))
        canvas.save(args.root / f"{pose}_surface_repair_comparison.jpg", quality=94, subsampling=0)


if __name__ == "__main__":
    main()
