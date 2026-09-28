"""Build compact seven-camera reconstruction contact sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=270)
    parser.add_argument("--kind", choices=("reconstruction_rgb.png", "audit_panel.png"), default="reconstruction_rgb.png")
    args = parser.parse_args()
    for group in sorted(p.name for p in args.root.iterdir() if p.is_dir()):
        canvas = Image.new("RGB", (args.tile_width * 2, (args.tile_height + 34) * 4), "white")
        draw = ImageDraw.Draw(canvas)
        for index, camera in enumerate(CAMERAS):
            path = args.root / group / camera / args.kind
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((args.tile_width, args.tile_height), Image.Resampling.LANCZOS)
            x = (index % 2) * args.tile_width
            y = (index // 2) * (args.tile_height + 34)
            canvas.paste(image, (x, y + 24))
            draw.text((x + 8, y + 4), camera, fill="black")
        suffix = "reconstruction" if args.kind.startswith("reconstruction") else "audit"
        output = args.root / f"{group}_{suffix}_contact.jpg"
        canvas.save(output, quality=95)
        print(output)


if __name__ == "__main__":
    main()
