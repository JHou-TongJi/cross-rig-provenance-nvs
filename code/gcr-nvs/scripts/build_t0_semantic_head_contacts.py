"""Build seven-view contacts for the trained semantic confidence head."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    cell = (960, 540)
    title = 40
    for pose in ("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"):
        canvas = Image.new("RGB", (cell[0] * 2, (cell[1] + title) * len(CAMERAS)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, camera in enumerate(CAMERAS):
            y = row * (cell[1] + title)
            draw.text((8, y + 8), camera + " / 2px 修补后", fill=(15, 20, 30), font=font)
            draw.text((cell[0] + 8, y + 8), camera + " / L/644 安全语义头", fill=(15, 20, 30), font=font)
            left = Image.open(args.root.parent / "t0_target_surface_2px_repair_all_20260824" / pose / camera / "t0_surface_2px_repaired_blue.png").convert("RGB")
            right = Image.open(args.root / pose / camera / "t0_dino_l_head_recovered_blue_residual.png").convert("RGB")
            canvas.paste(left.resize(cell, Image.Resampling.LANCZOS), (0, y + title))
            canvas.paste(right.resize(cell, Image.Resampling.LANCZOS), (cell[0], y + title))
        args.output.mkdir(parents=True, exist_ok=True)
        canvas.save(args.output / f"{pose}_semantic_head_comparison.jpg", quality=94, subsampling=0)


if __name__ == "__main__":
    main()
