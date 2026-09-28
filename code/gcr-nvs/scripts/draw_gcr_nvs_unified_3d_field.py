"""Draw the Unified 3D Field architecture as a paper-style raster figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.pcd import read_pcd


W, H = 3000, 1760
BG = "#F7F9FC"
INK = "#182033"
MUTED = "#657084"
LINE = "#AAB5C6"
BLUE = "#2468B4"
BLUE_LIGHT = "#E5F0FB"
GREEN = "#24835C"
GREEN_LIGHT = "#E5F5ED"
PURPLE = "#7555A7"
PURPLE_LIGHT = "#F0EAF8"
ORANGE = "#C57025"
ORANGE_LIGHT = "#FCEEDD"
RED = "#C7464D"
WHITE = "#FFFFFF"
NAVY = "#253850"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=LINE, width=3, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw: ImageDraw.ImageDraw, box, text: str, size: int, color=INK, bold=False):
    fnt = font(size, bold)
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=5, align="center")
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = (box[0] + box[2] - tw) / 2
    y = (box[1] + box[3] - th) / 2 - bounds[1]
    draw.multiline_text((x, y), text, font=fnt, fill=color, spacing=5, align="center")


def arrow(draw: ImageDraw.ImageDraw, start, end, color=INK, width=6, head=18):
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    a = (end[0] - head * math.cos(angle - 0.55), end[1] - head * math.sin(angle - 0.55))
    b = (end[0] - head * math.cos(angle + 0.55), end[1] - head * math.sin(angle + 0.55))
    draw.polygon([end, a, b], fill=color)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
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
    return image.resize(size, RESAMPLE)


def camera_stack(canvas: Image.Image, draw: ImageDraw.ImageDraw, paths: list[Path], box):
    x0, y0, x1, y1 = box
    thumb_w, thumb_h = 252, 142
    offsets = [(0, 0), (24, 20), (48, 40), (72, 60), (96, 80), (120, 100), (144, 120)]
    for path, (dx, dy) in reversed(list(zip(paths, offsets))):
        image = fit_image(path, (thumb_w, thumb_h))
        x, y = x0 + dx, y0 + dy
        draw.rounded_rectangle((x - 4, y - 4, x + thumb_w + 4, y + thumb_h + 4), 9, WHITE, GREEN, 3)
        canvas.paste(image, (x, y))
    centered(draw, (x0, y1 - 44, x1, y1), "7-view RGB  I1 ... I7", 25, GREEN, True)


def lidar_bev(points: np.ndarray, size=(390, 255)) -> Image.Image:
    result = Image.new("RGB", size, "#071625")
    draw = ImageDraw.Draw(result)
    xyz = points[:, :3]
    valid = np.isfinite(xyz).all(1) & (np.abs(xyz[:, 0]) < 65) & (np.abs(xyz[:, 1]) < 40)
    xyz = xyz[valid]
    if len(xyz) > 70000:
        xyz = xyz[np.linspace(0, len(xyz) - 1, 70000).astype(np.int64)]
    px = ((xyz[:, 0] + 65) / 130 * (size[0] - 1)).astype(np.int32)
    py = ((40 - xyz[:, 1]) / 80 * (size[1] - 1)).astype(np.int32)
    distance = np.linalg.norm(xyz[:, :2], axis=1)
    t = np.clip(distance / 70, 0, 1)
    colors = np.stack([50 + 205 * t, 220 - 100 * t, 245 - 185 * t], axis=1).astype(np.uint8)
    for x, y, color in zip(px, py, colors):
        draw.point((int(x), int(y)), fill=tuple(map(int, color)))
    cx, cy = size[0] // 2, size[1] // 2
    draw.rectangle((cx - 14, cy - 8, cx + 14, cy + 8), fill=WHITE, outline="#42C6E5", width=2)
    return result


def feature_pyramid(draw: ImageDraw.ImageDraw, origin, color, light, labels):
    x, y = origin
    widths = [230, 190, 150, 110]
    heights = [112, 96, 80, 64]
    for index, (w, h, label) in enumerate(zip(widths, heights, labels)):
        bx = x + index * 54
        by = y + index * 23
        rounded(draw, (bx, by, bx + w, by + h), light, color, 3, 11)
        centered(draw, (bx, by, bx + w, by + h), label, 22, color, True)


def unet(draw: ImageDraw.ImageDraw, origin):
    x, y = origin
    sizes = [(96, 150), (82, 125), (70, 102), (58, 82)]
    enc = []
    for i, (bw, bh) in enumerate(sizes):
        bx, by = x + i * 112, y + i * 56
        box = (bx, by, bx + bw, by + bh)
        rounded(draw, box, BLUE_LIGHT, BLUE, 3, 9)
        centered(draw, box, f"{32 * 2**i}\n³D", 23, BLUE, True)
        enc.append(box)
        if i:
            arrow(draw, (enc[i - 1][2], enc[i - 1][1] + 76), (box[0], box[1] + 58), BLUE, 4, 13)
    for i, source in enumerate(reversed(enc[:-1])):
        bx = x + 455 + i * 100
        by = y + 112 - i * 51
        bw, bh = sizes[2 - i]
        box = (bx, by, bx + bw, by + bh)
        rounded(draw, box, BLUE_LIGHT, BLUE, 3, 9)
        centered(draw, box, f"{128 // 2**i}\n³D", 23, BLUE, True)
        if i == 0:
            arrow(draw, (enc[-1][2], enc[-1][1] + 42), (box[0], box[1] + 50), BLUE, 4, 13)
        else:
            previous = (x + 455 + (i - 1) * 100, y + 112 - (i - 1) * 51)
            arrow(draw, (previous[0] + sizes[3 - i][0], previous[1] + 45), (box[0], box[1] + 50), BLUE, 4, 13)
        draw.line((source[2], source[1] + 18, box[0], box[1] + 18), fill="#79A9D8", width=3)


def voxel_field(draw: ImageDraw.ImageDraw, box, seed=2):
    x0, y0, x1, y1 = box
    rounded(draw, box, PURPLE_LIGHT, PURPLE, 4, 15)
    rng = np.random.default_rng(seed)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2 + 10
    for _ in range(260):
        gx, gy, gz = rng.integers(-7, 8), rng.integers(-6, 7), rng.integers(-2, 5)
        if rng.random() > (0.40 + 0.04 * max(0, 4 - abs(gx))):
            continue
        px = center_x + (gx - gy) * 10
        py = center_y + (gx + gy) * 5 - gz * 10
        color = BLUE if rng.random() < 0.38 else GREEN
        draw.rectangle((px - 3, py - 3, px + 3, py + 3), fill=color)
    centered(draw, (x0, y0 + 12, x1, y0 + 55), "Persistent Unified 3D Field", 26, PURPLE, True)
    centered(draw, (x0, y1 - 65, x1, y1 - 12), "occupancy · SDF · normal · confidence · appearance", 19, MUTED)


def stage_header(draw: ImageDraw.ImageDraw, x, text, color):
    draw.ellipse((x, 123, x + 42, 165), fill=color)
    centered(draw, (x, 123, x + 42, 165), str(stage_header.counter), 22, WHITE, True)
    draw.text((x + 54, 124), text, font=font(26, True), fill=color)
    stage_header.counter += 1


stage_header.counter = 1


def draw(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    sequence = root / args.sequence
    config_path = sequence / "Key_frames/camera_config" / f"{args.frame_id:06d}.json"
    payload = json.loads(config_path.read_text())
    rgb_paths = [sequence / "Key_frames" / name / payload["sensors"][name] for name in CAMERA_NAMES]
    lidar_path = sequence / "Key_frames/LIDAR_CONCAT" / payload["sensors"]["LIDAR_CONCAT"]
    points = read_pcd(lidar_path)

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((65, 37), "GCR-NVS Unified RGB–LiDAR 3D Field", font=font(50, True), fill=INK)
    d.text((68, 97), "LiDAR learns metric hidden structure; RGB learns semantics and high-frequency appearance", font=font(25), fill=MUTED)

    headers = [(65, "Multi-modal input", GREEN), (530, "Metric geometry", BLUE), (1370, "RGB appearance", GREEN), (1990, "3D fusion", PURPLE), (2510, "Novel-view render", ORANGE)]
    for x, label, color in headers:
        stage_header(d, x, label, color)

    camera_stack(canvas, d, rgb_paths, (70, 205, 485, 540))
    bev = lidar_bev(points)
    rounded(d, (70, 610, 480, 930), WHITE, BLUE, 4, 14)
    canvas.paste(bev, (80, 635))
    centered(d, (80, 890, 470, 925), f"Temporal LiDAR · {len(points):,} raw points", 19, BLUE, True)
    rounded(d, (100, 985, 450, 1090), "#EEF2F7", LINE, 3, 12)
    centered(d, (100, 985, 450, 1090), "48k current + 32k aligned static\n8k measured free-space", 22, NAVY, True)

    arrow(d, (480, 770), (535, 770), BLUE)
    rounded(d, (535, 655, 790, 880), BLUE_LIGHT, BLUE, 4, 14)
    centered(d, (535, 655, 790, 880), "occupied / free / unknown\n\n0.08m near\n0.16m mid\n0.32m far", 20, BLUE, True)
    arrow(d, (790, 765), (830, 765), BLUE)
    unet(d, (830, 625))
    rounded(d, (650, 1020, 1265, 1140), "#D9E9F8", BLUE, 4, 14)
    centered(d, (650, 1020, 1265, 1140), "Sparse 3D U-Net → occupancy / SDF / normal / visibility / confidence", 23, BLUE, True)
    d.line((665, 1140, 665, 1188, 1260, 1188, 1260, 1140), fill=BLUE, width=4)
    centered(d, (650, 1188, 1275, 1270), "LiDAR measured occupied/free voxels are immutable", 24, RED, True)

    arrow(d, (485, 380), (555, 380), GREEN)
    rounded(d, (555, 275, 905, 485), GREEN_LIGHT, GREEN, 4, 14)
    centered(d, (555, 275, 905, 485), "RGB rectification\n\ncamera_intric.yaml\nraw K,D -> K_rect,D=0\nall later stages use rectified RGB", 22, GREEN, True)
    arrow(d, (905, 380), (1375, 380), GREEN)
    feature_pyramid(d, (1380, 250), GREEN, GREEN_LIGHT, ["CNN 1×", "NAF 1/2", "DINO 1/4", "DINO 1/8"])
    rounded(d, (1420, 550, 1825, 700), GREEN_LIGHT, GREEN, 4, 14)
    centered(d, (1420, 550, 1825, 700), "DINOv2 tokens -> bilinear / AnyUp\n+ CNN/NAF detail features", 23, GREEN, True)
    arrow(d, (1620, 700), (1620, 780), GREEN)
    rounded(d, (1390, 780, 1880, 1030), GREEN_LIGHT, GREEN, 4, 14)
    centered(d, (1390, 780, 1880, 1030), "Rectified 3D projection\n\nlow-res semantic query\n+ high-res RGB sampling\n\nz-buffer + LiDAR consistency", 24, GREEN, True)
    rounded(d, (1415, 1085, 1855, 1190), "#DDF2E7", GREEN, 3, 12)
    centered(d, (1415, 1085, 1855, 1190), "per-view feature + visibility + focal-role token", 22, GREEN, True)

    arrow(d, (1265, 885), (2010, 885), BLUE)
    arrow(d, (1880, 900), (2010, 900), GREEN)
    rounded(d, (2010, 690, 2350, 1050), PURPLE_LIGHT, PURPLE, 4, 16)
    centered(d, (2010, 690, 2350, 1050), "Gated Cross-View\nVoxel Transformer\n\nLiDAR authority gate\nvisibility attention\ndepth-consistency gate\nwide/narrow focal token", 25, PURPLE, True)
    arrow(d, (2350, 875), (2400, 875), PURPLE)
    voxel_field(d, (2400, 630, 2810, 1110))

    arrow(d, (2605, 1110), (2605, 1250), ORANGE)
    rounded(d, (2245, 1250, 2735, 1450), ORANGE_LIGHT, ORANGE, 4, 15)
    centered(d, (2245, 1250, 2735, 1450), "Target rays  K_rect, D=0, Tt\nLiDAR-locked ray marching\ninfinity sky + uncertainty", 24, ORANGE, True)
    arrow(d, (2735, 1350), (2820, 1350), ORANGE)
    output = fit_image(rgb_paths[CAMERA_NAMES.index(args.target_camera)], (430, 242))
    rounded(d, (2810, 1195, 2970, 1535), WHITE, ORANGE, 4, 15)
    canvas.paste(output.resize((140, 242), RESAMPLE), (2820, 1250))
    centered(d, (2815, 1205, 2965, 1245), "native-resolution RGB", 19, ORANGE, True)
    centered(d, (2815, 1492, 2965, 1528), "RGB + depth + U", 19, ORANGE, True)

    d.line((60, 1320, 2160, 1320), fill=LINE, width=3)
    rounded(d, (65, 1370, 665, 1670), WHITE, "#8E9AAD", 3, 14)
    centered(d, (65, 1370, 665, 1420), "DATASET", 24, NAVY, True)
    centered(d, (90, 1430, 640, 1650), "13 train sequences / 1 val / 1 test\nsequence-level isolation\n7 RGB + LiDAR + K/D/T + timestamps\ndynamic mask + measurement provenance", 23, MUTED)
    arrow(d, (665, 1520), (735, 1520), NAVY)
    rounded(d, (735, 1370, 1370, 1670), WHITE, BLUE, 3, 14)
    centered(d, (735, 1370, 1370, 1420), "STAGED TRAINING", 24, BLUE, True)
    centered(d, (760, 1430, 1345, 1650), "① geometry-only on all sequences\n② freeze geometry, learn appearance\n③ joint fine-tune (geometry LR × 0.05)\n④ shifted poses with external RGB supervision", 23, MUTED)
    arrow(d, (1370, 1520), (1440, 1520), NAVY)
    rounded(d, (1440, 1370, 2110, 1670), WHITE, RED, 3, 14)
    centered(d, (1440, 1370, 2110, 1420), "CORRECTNESS GATES", 24, RED, True)
    centered(d, (1470, 1430, 2080, 1650), "LiDAR holdout depth P50/P90\nnear / mid / far and every camera\nobserved / uncertain / disocclusion metrics\ncombined x/y/z/yaw/pitch stress tests", 23, MUTED)

    rounded(d, (2170, 1535, 2970, 1670), "#FFF1F1", RED, 3, 13)
    centered(d, (2190, 1535, 2970, 1670), "No target-view monocular depth\nNo gray fill · No 2D Voronoi/Gaussian expansion", 21, RED, True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    preview_path = args.output.with_name(args.output.stem + "_preview.png")
    canvas.resize((1500, 880), RESAMPLE).save(preview_path, optimize=True)
    print(json.dumps({"output": str(args.output), "preview": str(preview_path), "shape": [H, W], "sequence": args.sequence, "frame_id": args.frame_id}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-26-11-44-28")
    parser.add_argument("--frame-id", type=int, default=11)
    parser.add_argument("--target-camera", choices=CAMERA_NAMES, default="CAM_FRONT_NARROW")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_figures/gcr_nvs_unified_3d_field.png"))
    draw(parser.parse_args())


if __name__ == "__main__":
    main()
