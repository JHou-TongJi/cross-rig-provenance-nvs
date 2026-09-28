"""Build a direct B/448 versus L/644 semantic-repair comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("b_root", type=Path)
    parser.add_argument("l_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    b = args.b_root / "mixed_20_50cm/CAM_FRONT_NARROW"
    l = args.l_root / "mixed_20_50cm/CAM_FRONT_NARROW"
    entries = [
        ("语义找回后的 RGB", b / "t0_semantic_recovered_blue_residual.png", l / "t0_semantic_recovered_blue_residual.png"),
        ("安全找回 mask", b / "recoverable_mask.png", l / "recoverable_mask.png"),
        ("剩余大洞 mask", b / "residual_true_hole_mask.png", l / "residual_true_hole_mask.png"),
    ]
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(font_path), 25) if font_path.exists() else ImageFont.load_default()
    thumb = (960, 540)
    title = 44
    canvas = Image.new("RGB", (thumb[0] * 2, (thumb[1] + title) * len(entries)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (label, left, right) in enumerate(entries):
        y = row * (thumb[1] + title)
        draw.text((12, y + 7), label + " / B/448", fill=(15, 20, 30), font=font)
        draw.text((thumb[0] + 12, y + 7), label + " / L/644", fill=(15, 20, 30), font=font)
        canvas.paste(Image.open(left).convert("RGB").resize(thumb, Image.Resampling.LANCZOS), (0, y + title))
        canvas.paste(Image.open(right).convert("RGB").resize(thumb, Image.Resampling.LANCZOS), (thumb[0], y + title))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=94, subsampling=0)


if __name__ == "__main__":
    main()
