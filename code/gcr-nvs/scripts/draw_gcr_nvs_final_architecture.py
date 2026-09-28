"""Draw the final GCR-NVS architecture with real intermediate thumbnails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 4600, 2550
BG = "#F7F9FC"
INK = "#172033"
MUTED = "#637087"
LINE = "#9AA8BA"
BLUE = "#2F6FC1"
BLUE_LIGHT = "#E5F0FE"
GREEN = "#238A68"
GREEN_LIGHT = "#E2F5ED"
PURPLE = "#7657B4"
PURPLE_LIGHT = "#F0EAFB"
ORANGE = "#C56C1E"
ORANGE_LIGHT = "#FFF0DF"
RED = "#C94350"
RED_LIGHT = "#FCE8EA"
TEAL = "#168E9A"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False):
    paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded(draw, box, fill, outline=LINE, width=4, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw, box, text, size=24, fill=INK, bold=False, spacing=5):
    f = font(size, bold)
    bb = draw.multiline_textbbox((0, 0), text, font=f, spacing=spacing, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (box[0] + box[2] - tw) / 2
    y = (box[1] + box[3] - th) / 2 - bb[1]
    draw.multiline_text((x, y), text, font=f, fill=fill, spacing=spacing, align="center")


def arrow(draw, start, end, color=INK, width=7, head=22, dashed=False):
    if dashed:
        x0, y0 = start
        x1, y1 = end
        length = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        cursor = 0.0
        while cursor < length - head:
            nxt = min(cursor + 24, length - head)
            draw.line((x0 + ux * cursor, y0 + uy * cursor, x0 + ux * nxt, y0 + uy * nxt), fill=color, width=width)
            cursor += 40
    else:
        draw.line((*start, *end), fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    length = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    px, py = -uy, ux
    draw.polygon([(x1, y1), (x1 - head * ux + head * 0.42 * px, y1 - head * uy + head * 0.42 * py), (x1 - head * ux - head * 0.42 * px, y1 - head * uy - head * 0.42 * py)], fill=color)


def image_fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
    ratio = size[0] / size[1]
    source_ratio = image.width / image.height
    if source_ratio > ratio:
        crop = int(image.height * ratio)
        left = (image.width - crop) // 2
        image = image.crop((left, 0, left + crop, image.height))
    else:
        crop = int(image.width / ratio)
        top = (image.height - crop) // 2
        image = image.crop((0, top, image.width, top + crop))
    return image.resize(size, Image.Resampling.LANCZOS)


def thumb(canvas, draw, path: Path, box, title, subtitle, border=LINE):
    x0, y0, x1, y1 = box
    if path.exists():
        image = image_fit(path, (x1 - x0, y1 - y0))
        canvas.paste(image, (x0, y0))
    else:
        draw.rectangle(box, fill="#DFE4EB")
    draw.rectangle(box, outline=border, width=4)
    draw.rectangle((x0, y0, x1, y0 + 38), fill=(0, 0, 0))
    draw.text((x0 + 12, y0 + 7), title, font=font(20, True), fill=WHITE)
    draw.text((x0 + 12, y1 - 30), subtitle, font=font(16), fill=WHITE)


def module(draw, box, title, body, fill, outline, title_size=27):
    rounded(draw, box, fill, outline, 5, 24)
    x0, y0, x1, y1 = box
    text_center(draw, (x0 + 16, y0 + 18, x1 - 16, y0 + 70), title, title_size, outline, True)
    text_center(draw, (x0 + 20, y0 + 75, x1 - 20, y1 - 16), body, 20, MUTED, False, 6)


def mini_network(draw, box, labels, color):
    x0, y0, x1, y1 = box
    gap = 18
    bw = (x1 - x0 - gap * (len(labels) - 1)) // len(labels)
    for i, label in enumerate(labels):
        bx = x0 + i * (bw + gap)
        rounded(draw, (bx, y0, bx + bw, y1), WHITE, color, 4, 12)
        text_center(draw, (bx, y0 + 8, bx + bw, y1 - 8), label, 17, color, True)
        if i < len(labels) - 1:
            arrow(draw, (bx + bw + 4, (y0 + y1) // 2), (bx + bw + gap - 4, (y0 + y1) // 2), color, 4, 13)


def draw(args):
    pipeline = args.pipeline.resolve()
    semantic = args.semantic.resolve()
    stage = {
        "raw": pipeline / "01_01_raw_rgb.png",
        "rectified": pipeline / "02_02_rectified_rgb.png",
        "da3": pipeline / "04_04_da3_depth.png",
        "fused": pipeline / "06_06_fused_depth.png",
        "surface": pipeline / "08_08_3d_surface___se(3).png",
        "warp": pipeline / "09_09_rgb_warp.png",
        "semantic": pipeline / "11_11_dinov2_plus_anyup.png",
        "completion": pipeline / "12_12_completion_output.png",
    }
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((70, 42), "GCR-NVS", font=font(54, True), fill=INK)
    draw.text((360, 58), "GCR-NVS-T0-UniWorld-A | final inference route", font=font(25), fill=MUTED)
    draw.line((70, 125, W - 70, 125), fill="#D2D9E2", width=3)

    # Stage labels
    stages = [(80, "1  INPUT / CALIBRATION", TEAL), (930, "2  DENSE STRUCTURE", BLUE), (1980, "3  SEMANTIC / GATE", GREEN), (3100, "4  3D REPROJECTION", PURPLE), (3970, "5  CONTROLLED RGB-D-S", ORANGE)]
    for x, label, color in stages:
        draw.ellipse((x, 155, x + 46, 201), fill=color)
        draw.text((x + 13, 163), label.split()[0], font=font(21, True), fill=WHITE)
        draw.text((x + 60, 160), " ".join(label.split()[1:]), font=font(25, True), fill=color)

    # Input thumbnails and contracts
    thumb(canvas, draw, stage["raw"], (75, 260, 425, 470), "RAW RGB", "H x W x 3, Brown distorted")
    thumb(canvas, draw, stage["rectified"], (485, 260, 835, 470), "RECTIFIED RGB", "H x W x 3, D=0, new K")
    module(draw, (75, 525, 835, 700), "7-view topology + calibration", "CAM_FRONT_* and CAM_BACK_* are separated by physical topology\nK, D, T, timestamps, dynamic masks", TEAL, TEAL)
    arrow(draw, (425, 365), (485, 365), TEAL, 7)
    arrow(draw, (660, 470), (660, 525), TEAL, 6)

    # LiDAR branch
    module(draw, (75, 790, 430, 1010), "LiDAR input", "sparse points + time\nmetric scale / visibility\nNOT RGB override", BLUE_LIGHT, BLUE)
    mini_network(draw, (470, 820, 835, 980), ["voxelize", "temporal\nfilter", "sparse\ngeometry"], BLUE)
    arrow(draw, (430, 900), (470, 900), BLUE, 7)

    # DA3 branch
    module(draw, (930, 260, 1320, 470), "DA3Metric-Large", "ViT-L depth teacher\ninput: rectified RGB + new K\noutput: dense D_m, confidence", BLUE_LIGHT, BLUE)
    mini_network(draw, (950, 515, 1300, 665), ["ViT-L", "DPT", "D_m"], BLUE)
    arrow(draw, (835, 365), (930, 365), BLUE, 7)
    arrow(draw, (1125, 470), (1125, 515), BLUE, 6)
    thumb(canvas, draw, stage["da3"], (1345, 260, 1665, 470), "DA3 DEPTH", "dense candidate")
    arrow(draw, (1320, 365), (1345, 365), BLUE, 6)

    # Alignment branch
    module(draw, (930, 790, 1665, 1010), "T0StructureConstraintNet", "LiDAR-anchored correction of the DA3 dense surface\nmetric depth, confidence, visibility and dynamic masks\nLiDAR constrains geometry; it never paints RGB", GREEN_LIGHT, GREEN)
    arrow(draw, (1125, 665), (1125, 790), BLUE, 7)
    arrow(draw, (835, 900), (930, 900), BLUE, 7)
    thumb(canvas, draw, stage["fused"], (1345, 1040, 1665, 1250), "FUSED DEPTH", "continuous surface candidate")
    arrow(draw, (1295, 1010), (1450, 1040), GREEN, 6)

    # semantic branch
    module(draw, (1980, 260, 2345, 470), "DINOv2", "ViT-L/14 tokens\nscene semantics\nview-consistent features", GREEN_LIGHT, GREEN)
    mini_network(draw, (2000, 515, 2325, 665), ["tokens", "AnyUP", "FPN/NAF"], GREEN)
    arrow(draw, (835, 365), (1980, 365), GREEN, 7)
    arrow(draw, (2160, 470), (2160, 515), GREEN, 6)
    thumb(canvas, draw, stage["semantic"], (2380, 260, 2700, 470), "SEMANTIC MAP", "DINOv2 + AnyUP, HxWx64")
    # Replace semantic thumbnail with actual projected semantic visualization if supplied.
    semantic_preview = args.semantic_preview.resolve()
    if semantic_preview.exists():
        thumb(canvas, draw, semantic_preview, (2380, 260, 2700, 470), "SEMANTIC MAP", "DINOv2 + AnyUP, HxWx64", GREEN)
    arrow(draw, (2345, 365), (2380, 365), GREEN, 6)

    # 3D branch
    module(draw, (1980, 790, 2700, 1010), "Dense surface construction", "D_dense + K^-1 -> source 3D points\nworld / vehicle coordinates\nconfidence and dynamic masks", PURPLE_LIGHT, PURPLE)
    arrow(draw, (1665, 1145), (1980, 900), GREEN, 7)
    arrow(draw, (835, 900), (1980, 900), BLUE, 5, dashed=True)
    module(draw, (1980, 1100, 2700, 1320), "Target camera transform", "target K, T + small SE(3) perturbation\n10cm translation / 2 degree rotation\nno target-view monocular depth", PURPLE_LIGHT, PURPLE)
    arrow(draw, (2340, 1010), (2340, 1100), PURPLE, 7)
    module(draw, (1980, 1410, 2700, 1630), "z-buffer + visibility", "nearest surface wins\nsource provenance\nobserved / uncertain / disocclusion masks", PURPLE_LIGHT, PURPLE)
    arrow(draw, (2340, 1320), (2340, 1410), PURPLE, 7)
    thumb(canvas, draw, stage["surface"], (2740, 1100, 3060, 1310), "TARGET SURFACE", "SE(3) + z-buffer")
    # The pipeline panel stores the stage, but save a reliable fallback from the audit output if needed.
    if args.surface_preview.exists():
        thumb(canvas, draw, args.surface_preview, (2740, 1100, 3060, 1310), "TARGET SURFACE", "SE(3) + z-buffer", PURPLE)
    arrow(draw, (2700, 1520), (2740, 1200), PURPLE, 6)

    # appearance fusion and output
    module(draw, (3100, 790, 3460, 1010), "Real RGB sampler", "source RGB at T0 target surface\nobserved pixels are locked\nprovenance is recorded per pixel", ORANGE_LIGHT, ORANGE)
    module(draw, (3100, 1100, 3460, 1320), "UniWorld / DA3 gate", "source -> target -> source triple reprojection\nbackground candidate only when visibility and depth agree\ngate-filled pixels are also locked", GREEN_LIGHT, GREEN)
    arrow(draw, (2700, 900), (3100, 900), ORANGE, 7)
    arrow(draw, (2700, 1200), (3100, 1200), GREEN, 7)
    thumb(canvas, draw, stage["warp"], (3530, 790, 3850, 1000), "RGB WARP", "observed appearance")
    arrow(draw, (3460, 900), (3530, 900), ORANGE, 6)
    module(draw, (3100, 1410, 3850, 1630), "A-only RGB-D-S completion", "frozen diffusion backbone + trained adapter\nonly residual true hole is writable\nobserved and gate regions cannot be regenerated", ORANGE_LIGHT, ORANGE)
    arrow(draw, (3690, 1000), (3690, 1410), ORANGE, 7)
    thumb(canvas, draw, stage["completion"], (3970, 1410, 4290, 1620), "FINAL RGB", "same target resolution")
    arrow(draw, (3850, 1520), (3970, 1520), ORANGE, 7)

    # Lower training / correctness band
    draw.line((75, 1770, 4525, 1770), fill="#CAD2DE", width=4)
    text_center(draw, (75, 1810, 1450, 1880), "TRAINING SUPERVISION", 25, BLUE, True)
    text_center(draw, (1570, 1810, 3000, 1880), "REGION-SPLIT OBJECTIVES", 25, GREEN, True)
    text_center(draw, (3150, 1810, 4525, 1880), "CORRECTNESS GATES", 25, RED, True)
    module(draw, (75, 1900, 1450, 2250), "strict LOO + shifted 10cm", "strict LOO: hidden target camera, real RGB GT\nshifted: same-camera RGB anchor + SE(3) perturbation\nnearest temporal frame only", BLUE_LIGHT, BLUE)
    module(draw, (1570, 1900, 3000, 2250), "observed / uncertain / disocclusion", "observed: RGB fidelity + source preservation\nuncertain: bounded residual + uncertainty\ndisocclusion: perceptual / pseudo-GT completion", GREEN_LIGHT, GREEN)
    module(draw, (3150, 1900, 4525, 2250), "metrics before model selection", "LiDAR holdout depth by camera and range\nper-camera RGB/SSIM/LPIPS and coverage\nno gray fill, no invalid RGB override, no topology leakage", RED_LIGHT, RED)
    draw.text((75, 2370), "Final contract: LiDAR/T0 constrains metric geometry; source RGB supplies appearance; DINOv2 + AnyUP supplies semantic localization; gate recovers only supported background; A-only writes residual holes.", font=font(23, True), fill=INK)
    draw.text((75, 2415), "Wan2.1/VACE is a future 4D backend reservation, not part of the current final inference result.", font=font(19), fill=MUTED)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True, help="directory with pipeline intermediates")
    parser.add_argument("--semantic", type=Path, required=True, help="semantic audit directory")
    parser.add_argument("--semantic-preview", type=Path, required=True, help="RGB preview of projected semantic features")
    parser.add_argument("--surface-preview", type=Path, required=True, help="RGB preview of target surface depth")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args)


if __name__ == "__main__":
    main()
