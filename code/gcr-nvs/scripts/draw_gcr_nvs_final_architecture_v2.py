"""Compact paper-style architecture figure for the final GCR-NVS route."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 4600, 2460
BG = "#F8FAFD"
INK = "#182235"
MUTED = "#66748A"
GRID = "#D2DAE6"
WHITE = "#FFFFFF"
BLUE = "#2C6CC2"
BLUE_L = "#E7F0FE"
GREEN = "#238A68"
GREEN_L = "#E3F5ED"
PURPLE = "#7657B4"
PURPLE_L = "#F0EAFA"
ORANGE = "#C66E1F"
ORANGE_L = "#FFF0E0"
RED = "#C74451"
RED_L = "#FBE8EB"
TEAL = "#168B99"


def fnt(size: int, bold: bool = False):
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def rounded(draw, box, fill, outline, width=4, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center(draw, box, text, size=22, color=INK, bold=False, spacing=4):
    font = fnt(size, bold)
    bb = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (box[0] + box[2] - tw) / 2
    y = (box[1] + box[3] - th) / 2 - bb[1]
    draw.multiline_text((x, y), text, fill=color, font=font, spacing=spacing, align="center")


def arrow(draw, start, end, color=INK, width=6, head=19, dashed=False):
    x0, y0 = start
    x1, y1 = end
    if dashed:
        length = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1.0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        cursor = 0.0
        while cursor < length - head:
            nxt = min(cursor + 24, length - head)
            draw.line((x0 + ux * cursor, y0 + uy * cursor, x0 + ux * nxt, y0 + uy * nxt), fill=color, width=width)
            cursor += 40
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    length = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1.0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    px, py = -uy, ux
    draw.polygon([(x1, y1), (x1 - head * ux + head * .42 * px, y1 - head * uy + head * .42 * py), (x1 - head * ux - head * .42 * px, y1 - head * uy - head * .42 * py)], fill=color)


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    ratio = size[0] / size[1]
    sr = image.width / image.height
    if sr > ratio:
        crop = int(image.height * ratio)
        left = (image.width - crop) // 2
        image = image.crop((left, 0, left + crop, image.height))
    else:
        crop = int(image.width / ratio)
        top = (image.height - crop) // 2
        image = image.crop((0, top, image.width, top + crop))
    return image.resize(size, Image.Resampling.LANCZOS)


def thumb(canvas, draw, path: Path, box, title, color):
    x0, y0, x1, y1 = box
    if path.exists():
        canvas.paste(fit(path, (x1 - x0, y1 - y0)), (x0, y0))
    else:
        draw.rectangle(box, fill="#E4E9F0")
    draw.rectangle(box, outline=color, width=4)
    draw.rectangle((x0, y0, x1, y0 + 34), fill=(0, 0, 0))
    draw.text((x0 + 10, y0 + 5), title, font=fnt(18, True), fill=WHITE)


def block(draw, box, title, subtitle, fill, outline, title_size=25):
    rounded(draw, box, fill, outline, 4, 16)
    x0, y0, x1, y1 = box
    center(draw, (x0 + 8, y0 + 8, x1 - 8, y0 + 55), title, title_size, outline, True)
    if subtitle:
        center(draw, (x0 + 8, y0 + 58, x1 - 8, y1 - 8), subtitle, 17, MUTED)


def net_blocks(draw, box, labels, outline, fill=WHITE):
    x0, y0, x1, y1 = box
    gap = 15
    bw = (x1 - x0 - gap * (len(labels) - 1)) // len(labels)
    for index, label in enumerate(labels):
        bx = x0 + index * (bw + gap)
        rounded(draw, (bx, y0, bx + bw, y1), fill, outline, 3, 10)
        center(draw, (bx + 4, y0 + 4, bx + bw - 4, y1 - 4), label, 17, outline, True)
        if index + 1 < len(labels):
            arrow(draw, (bx + bw + 3, (y0 + y1) // 2), (bx + bw + gap - 3, (y0 + y1) // 2), outline, 3, 11)


def stage(draw, x, number, label, color):
    draw.ellipse((x, 145, x + 42, 187), fill=color)
    center(draw, (x, 145, x + 42, 187), str(number), 19, WHITE, True)
    draw.text((x + 56, 150), label, fill=color, font=fnt(24, True))


def draw(args):
    pipeline = args.pipeline.resolve()
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 35), "GCR-NVS", font=fnt(52, True), fill=INK)
    draw.text((360, 53), "RGB-guided dense geometry + semantic novel-view synthesis", font=fnt(24), fill=MUTED)
    draw.line((70, 115, W - 70, 115), fill=GRID, width=3)

    stage(draw, 75, 1, "INPUT", TEAL)
    stage(draw, 900, 2, "DEPTH", BLUE)
    stage(draw, 1840, 3, "SEMANTIC", GREEN)
    stage(draw, 2860, 4, "3D VIEW", PURPLE)
    stage(draw, 3830, 5, "RGB OUTPUT", ORANGE)

    # Input and calibration lane
    thumb(canvas, draw, pipeline / "01_01_raw_rgb.png", (75, 245, 360, 430), "RGB", TEAL)
    thumb(canvas, draw, pipeline / "02_02_rectified_rgb.png", (410, 245, 695, 430), "RECTIFY", TEAL)
    block(draw, (75, 485, 695, 660), "7 x {RGB, K, D, T}", "topology mask + timestamps", TEAL, TEAL)
    arrow(draw, (360, 338), (410, 338), TEAL)
    arrow(draw, (552, 430), (552, 485), TEAL)

    # Depth lane
    block(draw, (900, 245, 1195, 430), "DA3 ViT-L", "DPT head", BLUE_L, BLUE)
    net_blocks(draw, (925, 480, 1170, 600), ["ViT-L", "DPT", "D_m"], BLUE)
    thumb(canvas, draw, pipeline / "04_04_da3_depth.png", (1225, 245, 1510, 430), "D_m", BLUE)
    arrow(draw, (695, 338), (900, 338), BLUE)
    arrow(draw, (1047, 430), (1047, 480), BLUE)
    arrow(draw, (1195, 338), (1225, 338), BLUE)
    block(draw, (900, 725, 1510, 895), "Metric alignment", "inv-depth affine + confidence", GREEN_L, GREEN)
    arrow(draw, (1047, 600), (1047, 725), BLUE)

    # LiDAR branch enters alignment and geometry
    block(draw, (75, 760, 360, 930), "LiDAR", "PCD + time", BLUE_L, BLUE)
    net_blocks(draw, (410, 795, 695, 895), ["voxel", "temporal", "mask"], BLUE)
    arrow(draw, (360, 845), (410, 845), BLUE)
    arrow(draw, (695, 845), (900, 810), BLUE)
    block(draw, (75, 975, 695, 1135), "LiDAR authority", "metric scale | local depth | visibility", BLUE_L, BLUE)
    arrow(draw, (500, 930), (500, 975), BLUE)
    arrow(draw, (695, 1055), (1510, 810), BLUE, dashed=True)
    thumb(canvas, draw, pipeline / "06_06_fused_depth.png", (1225, 925, 1510, 1110), "D_dense", GREEN)
    arrow(draw, (1205, 810), (1300, 925), GREEN)

    # Semantic lane
    block(draw, (1840, 245, 2140, 430), "DINOv2", "ViT-L/14", GREEN_L, GREEN)
    net_blocks(draw, (1865, 480, 2115, 600), ["tokens", "AnyUP", "FPN"], GREEN)
    thumb(canvas, draw, pipeline / "11_11_dinov2_plus_anyup.png", (2170, 245, 2455, 430), "F_64", GREEN)
    arrow(draw, (695, 338), (1840, 338), GREEN)
    arrow(draw, (1990, 430), (1990, 480), GREEN)
    arrow(draw, (2140, 338), (2170, 338), GREEN)
    block(draw, (1840, 725, 2455, 895), "Semantic features", "DINOv2 + AnyUP at H x W", GREEN_L, GREEN)
    arrow(draw, (1990, 600), (1990, 725), GREEN)

    # 3D lane
    block(draw, (2860, 245, 3210, 430), "Dense 3D", "D_dense + K^-1", PURPLE_L, PURPLE)
    net_blocks(draw, (2885, 480, 3185, 600), ["backproj", "SE(3)", "z-buffer"], PURPLE)
    thumb(canvas, draw, pipeline / "08_08_3d_surface___se(3).png", (3240, 245, 3525, 430), "P_t", PURPLE)
    arrow(draw, (1510, 810), (2860, 338), PURPLE)
    arrow(draw, (2455, 810), (2860, 338), GREEN)
    arrow(draw, (3035, 430), (3035, 480), PURPLE)
    arrow(draw, (3210, 338), (3240, 338), PURPLE)
    block(draw, (2860, 725, 3525, 895), "Visibility", "observed | uncertain | disocclusion", PURPLE_L, PURPLE)
    arrow(draw, (3035, 600), (3035, 725), PURPLE)

    # RGB output lane
    block(draw, (3830, 245, 4180, 430), "RGB sampler", "source RGB + provenance", ORANGE_L, ORANGE)
    thumb(canvas, draw, pipeline / "09_09_rgb_warp.png", (4210, 245, 4495, 430), "RGB_warp", ORANGE)
    arrow(draw, (3525, 338), (3830, 338), ORANGE)
    arrow(draw, (4180, 338), (4210, 338), ORANGE)
    block(draw, (3830, 725, 4495, 895), "Cross-view fusion", "RGB + F_64 + depth gate", GREEN_L, GREEN)
    arrow(draw, (3525, 810), (3830, 810), GREEN)
    block(draw, (3830, 1010, 4495, 1180), "Restormer", "residual / completion head", ORANGE_L, ORANGE)
    thumb(canvas, draw, pipeline / "12_12_completion_output.png", (4210, 1235, 4495, 1420), "RGB_t", ORANGE)
    arrow(draw, (4160, 895), (4160, 1010), ORANGE)
    arrow(draw, (4160, 1180), (4350, 1235), ORANGE)
    block(draw, (3830, 1455, 4495, 1625), "Route", "observed: small residual\ndisocclusion: generated", ORANGE_L, ORANGE)
    arrow(draw, (4160, 1420), (4160, 1455), ORANGE)

    # compact lower explanation band
    draw.line((75, 1740, W - 75, 1740), fill=GRID, width=3)
    block(draw, (75, 1790, 1510, 2240), "DATA / GEOMETRY", "LiDAR -> metric scale, local depth, visibility\nDA3 -> continuous dense depth candidate\nD_dense -> 3D surface -> SE(3) -> z-buffer\nLiDAR never overwrites RGB", BLUE_L, BLUE, 24)
    block(draw, (1565, 1790, 3000, 2240), "APPEARANCE / SEMANTICS", "RGB -> color and high-frequency texture\nDINOv2 -> scene semantics\nAnyUP -> high-resolution feature localization\nview fusion uses topology + depth consistency", GREEN_L, GREEN, 24)
    block(draw, (3055, 1790, 4525, 2240), "TRAIN / EVAL", "strict LOO: hidden target with RGB GT\nshifted: 10cm SE(3) source anchor\nobserved / uncertain / disocclusion losses\nmetrics: depth holdout + per-view RGB + coverage", RED_L, RED, 24)
    draw.text((75, 2325), "Current figure: DA3 dense-depth pilot, AnyUP semantic audit, same-camera priority, and +Y 10cm target perturbation.", font=fnt(20), fill=MUTED)
    draw.text((75, 2365), "Architecture contract: LiDAR constrains geometry; RGB supplies appearance; DINOv2 + AnyUP supplies semantic localization; completion only handles missing observations.", font=fnt(21, True), fill=INK)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args)


if __name__ == "__main__":
    main()

