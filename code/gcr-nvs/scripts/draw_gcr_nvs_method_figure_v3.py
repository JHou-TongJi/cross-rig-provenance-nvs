"""Draw a visual, paper-style GCR-NVS method figure with PIL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from draw_gcr_nvs_paper_architecture import (
    AMBER,
    BLUE,
    BLUE_LIGHT,
    CYAN,
    GREEN,
    GREEN_LIGHT,
    INK,
    LINE,
    MUTED,
    PURPLE,
    RED,
    RESAMPLE_LANCZOS,
    WHITE,
    arrow,
    depth_color,
    fit_image,
    font,
    lidar_bev,
    rounded_box,
    sparse_depth_preview,
    text_bbox,
)
from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.pcd import read_pcd


CANVAS = (4800, 1900)
BG = "#FFFFFF"
LIGHT = "#F5F7FA"
TEAL = "#8FCDBB"
TEAL_DARK = "#3D9D82"
VIOLET = "#8580C3"
PINK = "#E68CAD"
ORANGE = "#E6A04B"


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    color: str = INK,
    bold: bool = False,
) -> None:
    text_font = font(size, bold=bold)
    bbox = text_bbox(draw, (0, 0), text, text_font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) // 2, xy[1]), text, font=text_font, fill=color)


def feature_stack(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    count: int,
    fill: str,
    outline: str,
    label: str = "",
    step: tuple[int, int] = (14, -12),
) -> tuple[int, int, int, int]:
    for index in range(count - 1, -1, -1):
        dx = index * step[0]
        dy = index * step[1]
        draw.rectangle((x + dx, y + dy, x + dx + w, y + dy + h), fill=fill, outline=outline, width=3)
    if label:
        centered_text(draw, (x + w // 2 + (count - 1) * step[0] // 2, y + h + 14), label, 24, MUTED)
    return x, y + (count - 1) * step[1], x + w + (count - 1) * step[0], y + h


def image_stack(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    paths: list[Path],
    x: int,
    y: int,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    visible = paths[:4]
    for index in range(len(visible) - 1, -1, -1):
        dx = index * 22
        dy = -index * 17
        image = fit_image(visible[index], size)
        draw.rectangle((x + dx - 4, y + dy - 4, x + dx + size[0] + 4, y + dy + size[1] + 4), fill=WHITE, outline="#687386", width=3)
        canvas.paste(image, (x + dx, y + dy))
    return x, y - (len(visible) - 1) * 17, x + size[0] + (len(visible) - 1) * 22, y + size[1]


def vertical_module(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    label: str,
    fill: str = LIGHT,
    outline: str = "#9AA2AC",
) -> None:
    rounded_box(draw, xy, fill, outline, radius=20, width=4)
    text_font = font(29, bold=True)
    bbox = text_bbox(draw, (0, 0), label, text_font)
    text = Image.new("RGBA", (bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text)
    text_draw.text((10, 10 - bbox[1]), label, font=text_font, fill=INK)
    rotated = text.rotate(90, expand=True, resample=Image.BICUBIC)
    cx = (xy[0] + xy[2]) // 2
    cy = (xy[1] + xy[3]) // 2
    canvas.paste(rotated, (cx - rotated.width // 2, cy - rotated.height // 2), rotated)


def transformer_block(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    box = (x, y, x + 690, y + 450)
    rounded_box(draw, box, "#FCFCFC", "#8E9399", radius=38, width=4)
    centered_text(draw, (x + 345, y + 24), "DINOv2 ViT-B/14", 34, INK, True)
    colors = ["#EFF1F3", "#EFF1F3", "#DCE8B8", "#EFF1F3", "#EFF1F3", "#DCE8B8"]
    cursor = x + 55
    for index, color in enumerate(colors):
        rounded_box(draw, (cursor, y + 105, cursor + 72, y + 350), color, "#A6AAAE", radius=12, width=3)
        if index < len(colors) - 1:
            arrow(draw, (cursor + 72, y + 228), (cursor + 93, y + 228), LINE, width=3, head=10)
        cursor += 101
    centered_text(draw, (x + 205, y + 375), "2 / 5 / 8 / 11", 23, MUTED)
    return box


def naf_branch(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    centered_text(draw, (x + 270, y - 42), "NAF detail CNN", 30, GREEN, True)
    widths = [96, 118, 142, 170]
    heights = [210, 185, 155, 125]
    cursor = x
    boxes = []
    for index, (width, height) in enumerate(zip(widths, heights)):
        top = y + (210 - height) // 2
        box = (cursor, top, cursor + width, top + height)
        rounded_box(draw, box, GREEN_LIGHT, GREEN, radius=10, width=3)
        for stripe in range(3):
            sx = cursor + 16 + stripe * 18
            draw.line((sx, top + 16, sx, top + height - 16), fill="#83C9AD", width=6)
        boxes.append(box)
        if index:
            arrow(draw, (boxes[index - 1][2], y + 105), (box[0], y + 105), GREEN, width=4, head=13)
        cursor += width + 34
    return boxes[0][0], y, boxes[-1][2], y + 210


def sparse_unet(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    centered_text(draw, (x + 420, y - 54), "SparseConv U-Net", 34, BLUE, True)
    channels = [32, 64, 128, 256]
    boxes = []
    for index, channel in enumerate(channels):
        bx = x + index * 145
        by = y + index * 95
        size = 120 - index * 12
        box = (bx, by, bx + size, by + size)
        rounded_box(draw, box, BLUE_LIGHT, BLUE, radius=10, width=3)
        centered_text(draw, ((box[0] + box[2]) // 2, box[1] + size // 2 - 18), str(channel), 25, BLUE, True)
        boxes.append(box)
        if index:
            arrow(draw, (boxes[index - 1][2], boxes[index - 1][1] + (boxes[index - 1][3] - boxes[index - 1][1]) // 2), (box[0], box[1] + size // 2), BLUE, width=4, head=13)
    decoder = []
    for index, channel in enumerate([128, 64, 32]):
        bx = x + 600 + index * 145
        by = y + 190 - index * 95
        size = 96 + index * 12
        box = (bx, by, bx + size, by + size)
        rounded_box(draw, box, "#DDEAFF", BLUE, radius=10, width=3)
        centered_text(draw, ((box[0] + box[2]) // 2, box[1] + size // 2 - 18), str(channel), 25, BLUE, True)
        decoder.append(box)
        if index:
            arrow(draw, (decoder[index - 1][2], decoder[index - 1][1] + (decoder[index - 1][3] - decoder[index - 1][1]) // 2), (box[0], box[1] + size // 2), BLUE, width=4, head=13)
    arrow(draw, (boxes[-1][2], boxes[-1][1] + 45), (decoder[0][0], decoder[0][1] + 48), BLUE, width=4, head=13)
    for source, target in zip(boxes[2::-1], decoder):
        sx = (source[0] + source[2]) // 2
        tx = (target[0] + target[2]) // 2
        top = y - 24
        draw.line((sx, source[1], sx, top, tx, top, tx, target[1]), fill="#75A8E9", width=3)
    return x, y - 24, decoder[-1][2], y + 400


def attention_block(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    box = (x, y, x + 480, y + 610)
    rounded_box(draw, box, "#FBFAFE", VIOLET, radius=34, width=4)
    centered_text(draw, (x + 240, y + 24), "RGB Appearance Attention", 29, PURPLE, True)
    token_x = [x + 95, x + 190, x + 285, x + 380]
    token_y = [y + 145, y + 255, y + 365, y + 475]
    colors = [TEAL, ORANGE, VIOLET, PINK]
    for index, (tx, ty, color) in enumerate(zip(token_x, token_y, colors)):
        draw.rectangle((tx - 30, ty - 30, tx + 30, ty + 30), fill=color, outline="#555A65", width=2)
        if index:
            draw.line((token_x[index - 1], token_y[index - 1], tx, ty), fill="#A5A3BE", width=4)
    for i in range(4):
        for j in range(i + 2, 4):
            draw.arc((min(token_x[i], token_x[j]) - 30, min(token_y[i], token_y[j]) - 55, max(token_x[i], token_x[j]) + 30, max(token_y[i], token_y[j]) + 55), 195, 330, fill="#B8A7D5", width=3)
    centered_text(draw, (x + 240, y + 545), "Top-K = 3   geometry query", 23, MUTED)
    return box


def gaussian_cloud(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    title_color: str,
    seed: int = 17,
) -> tuple[int, int, int, int]:
    rng = np.random.RandomState(seed)
    box = (x, y, x + 480, y + 560)
    centered_text(draw, (x + 240, y - 45), title, 32, title_color, True)
    draw.polygon([(x + 60, y + 450), (x + 245, y + 535), (x + 430, y + 430), (x + 245, y + 355)], fill="#F2F0F8", outline="#B5A9CC")
    palette = ["#4CB5AE", "#6C72BE", "#D98AA6", "#E3A452", "#6EAF6C"]
    for _ in range(175):
        px = int(rng.normal(x + 245, 105))
        py = int(rng.normal(y + 300, 105))
        if not (x + 45 <= px <= x + 435 and y + 90 <= py <= y + 470):
            continue
        rw = int(rng.randint(8, 27))
        rh = int(rng.randint(4, 15))
        color = palette[int(rng.randint(0, len(palette)))]
        draw.ellipse((px - rw, py - rh, px + rw, py + rh), fill=color, outline="#FFFFFF")
    draw.line((x + 245, y + 90, x + 245, y + 500), fill="#9F9BAA", width=2)
    draw.line((x + 80, y + 430, x + 420, y + 430), fill="#9F9BAA", width=2)
    return box


def renderer_icon(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    centered_text(draw, (x + 165, y - 50), "gsplat", 34, AMBER, True)
    draw.polygon([(x + 20, y + 80), (x + 130, y + 25), (x + 130, y + 275), (x + 20, y + 220)], fill="#FFF1D9", outline=AMBER)
    draw.polygon([(x + 190, y + 35), (x + 310, y + 95), (x + 310, y + 235), (x + 190, y + 295)], fill="#FDE4B9", outline=AMBER)
    for offset in [0, 30, 60, 90, 120]:
        draw.line((x + 130, y + 55 + offset, x + 190, y + 65 + offset), fill=AMBER, width=3)
    return x + 20, y + 25, x + 310, y + 295


def restormer(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    centered_text(draw, (x + 335, y - 50), "RGB-only Target Decoder", 31, PURPLE, True)
    boxes = []
    for index, channel in enumerate([64, 64, 64, 64]):
        bx = x + index * 88
        by = y + index * 72
        height = 124 - index * 12
        box = (bx, by, bx + 66, by + height)
        rounded_box(draw, box, "#EEE8F8", PURPLE, radius=8, width=3)
        centered_text(draw, (bx + 33, by + height // 2 - 14), str(channel), 20, PURPLE, True)
        boxes.append(box)
        if index:
            arrow(draw, (boxes[index - 1][2], boxes[index - 1][1] + 60), (box[0], box[1] + 48), PURPLE, width=3, head=11)
    dec = []
    for index, channel in enumerate([64, 64, 3]):
        bx = x + 370 + index * 88
        by = y + 144 - index * 72
        height = 100 + index * 12
        box = (bx, by, bx + 66, by + height)
        rounded_box(draw, box, "#E7DDF5", PURPLE, radius=8, width=3)
        centered_text(draw, (bx + 33, by + height // 2 - 14), str(channel), 20, PURPLE, True)
        dec.append(box)
        if index:
            arrow(draw, (dec[index - 1][2], dec[index - 1][1] + 48), (box[0], box[1] + 60), PURPLE, width=3, head=11)
    arrow(draw, (boxes[-1][2], boxes[-1][1] + 42), (dec[0][0], dec[0][1] + 45), PURPLE, width=3, head=11)
    for source, target in zip(boxes[2::-1], dec):
        draw.line(((source[0] + source[2]) // 2, source[1], (source[0] + source[2]) // 2, y - 22, (target[0] + target[2]) // 2, y - 22, (target[0] + target[2]) // 2, target[1]), fill="#AE9CCE", width=2)
    return x, y - 22, dec[-1][2], y + 360


def temporal_teacher(draw: ImageDraw.ImageDraw, x: int, y: int) -> tuple[int, int, int, int]:
    stages = [
        ("t", CYAN),
        ("SE(3)", BLUE),
        ("M", GREEN),
        ("S", PURPLE),
        ("Z*", AMBER),
    ]
    boxes = []
    for index, (label, color) in enumerate(stages):
        bx = x + index * 135
        box = (bx, y, bx + 92, y + 92)
        rounded_box(draw, box, "#FAFBFC", color, radius=16, width=3)
        centered_text(draw, (bx + 46, y + 27), label, 25, color, True)
        boxes.append(box)
        if index:
            arrow(draw, (boxes[index - 1][2], y + 46), (box[0], y + 46), LINE, width=3, head=11)
    centered_text(draw, (x + 316, y + 112), "temporal LiDAR teacher", 25, MUTED)
    return boxes[0][0], y, boxes[-1][2], y + 145


def loss_circle(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, color: str) -> None:
    draw.ellipse((x - 35, y - 35, x + 35, y + 35), fill=WHITE, outline=color, width=4)
    centered_text(draw, (x, y - 18), label, 24, color, True)


def checkpoint(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, label: str) -> None:
    draw.ellipse((x, y, x + 92, y + 26), fill=WHITE, outline=color, width=3)
    draw.rectangle((x, y + 13, x + 92, y + 90), fill=WHITE, outline=color, width=3)
    draw.ellipse((x, y + 76, x + 92, y + 102), fill="#F8F8FA", outline=color, width=3)
    centered_text(draw, (x + 46, y + 112), label, 22, color, True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-26-11-44-28")
    parser.add_argument("--frame-id", type=int, default=11)
    parser.add_argument("--target-camera", choices=CAMERA_NAMES, default="CAM_BACK_LEFT")
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--reconstruction", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_figures/gcr_nvs_method_figure_v3.png"))
    args = parser.parse_args()

    root = args.root.resolve()
    sequence_dir = root / args.sequence
    config_path = sequence_dir / "Key_frames/camera_config" / f"{args.frame_id:06d}.json"
    payload = json.loads(config_path.read_text())
    calibrations = load_calibrations(config_path, args.distortion)
    lidar_path = sequence_dir / "Key_frames/LIDAR_CONCAT" / payload["sensors"]["LIDAR_CONCAT"]
    points = read_pcd(lidar_path)

    source_cameras = [camera for camera in CAMERA_NAMES if camera != args.target_camera]
    source_paths = [sequence_dir / "Key_frames" / camera / payload["sensors"][camera] for camera in source_cameras]
    target_path = sequence_dir / "Key_frames" / args.target_camera / payload["sensors"][args.target_camera]

    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 35), "GCR-NVS", font=font(52, bold=True), fill=INK)
    draw.text((330, 54), "LiDAR hidden structure  +  RGB appearance  +  3D feature rendering", font=font(27), fill=MUTED)

    rgb_box = image_stack(canvas, draw, source_paths, 65, 250, (275, 155))
    centered_text(draw, ((rgb_box[0] + rgb_box[2]) // 2, rgb_box[3] + 25), "I_s x 6", 27, MUTED, True)

    bev = lidar_bev(points, (310, 230))
    for index in range(2, -1, -1):
        px = 66 + index * 18
        py = 1030 - index * 14
        draw.rectangle((px - 4, py - 4, px + 314, py + 234), fill=WHITE, outline=CYAN, width=3)
        canvas.paste(bev, (px, py))
    centered_text(draw, (235, 1285), "P_t", 27, CYAN, True)

    patch = (440, 205, 570, 510)
    vertical_module(canvas, draw, patch, "Patch Embed")
    arrow(draw, (rgb_box[2] + 20, 330), (patch[0], 330), INK, width=5, head=16)
    tokens = feature_stack(draw, 625, 235, 50, 50, 7, TEAL, TEAL_DARK, "tokens", (0, 62))
    arrow(draw, (patch[2], 330), (625, 330), INK, width=5, head=16)

    dino = transformer_block(draw, 755, 150)
    arrow(draw, (tokens[2] + 15, 330), (dino[0], 330), INK, width=5, head=16)
    dino_tokens = feature_stack(draw, 1495, 220, 52, 52, 7, TEAL, TEAL_DARK, "F_D", (0, 62))
    arrow(draw, (dino[2], 330), (1495, 330), INK, width=5, head=16)

    naf = naf_branch(draw, 755, 680)
    arrow(draw, (rgb_box[2] + 10, 390), (700, 785), GREEN, width=5, head=16)
    dpt = feature_stack(draw, 1610, 255, 150, 210, 4, "#9ED8C7", TEAL_DARK, "DPT/FPN", (18, -16))
    arrow(draw, (dino_tokens[2] + 18, 330), (1610, 330), GREEN, width=5, head=16)
    naf_maps = feature_stack(draw, 1610, 690, 150, 150, 4, "#B9E3D5", GREEN, "detail", (18, -16))
    arrow(draw, (naf[2], 785), (1610, 765), GREEN, width=5, head=16)

    voxel = feature_stack(draw, 445, 1045, 92, 92, 5, "#A6CDF6", BLUE, "voxels", (15, -12))
    arrow(draw, (395, 1145), (voxel[0], 1100), BLUE, width=5, head=16)
    unet = sparse_unet(draw, 655, 1000)
    arrow(draw, (voxel[2] + 20, 1100), (unet[0], 1100), BLUE, width=5, head=16)
    geom_maps = feature_stack(draw, 1640, 1060, 155, 210, 5, "#AFCFF3", BLUE, "Z O N V U", (18, -16))
    arrow(draw, (unet[2], 1110), (1640, 1135), BLUE, width=5, head=16)

    teacher = temporal_teacher(draw, 700, 1580)
    draw.line((235, 1285, 235, 1625, teacher[0], 1625), fill=CYAN, width=4)
    arrow(draw, (teacher[2], 1625), (1700, 1320), BLUE, width=4, head=14, dashed=True)
    loss_circle(draw, 1510, 1510, "L_G", BLUE)
    draw.line((teacher[2] - 15, 1625, 1470, 1535), fill=BLUE, width=3)
    draw.line((1545, 1495, 1690, 1320), fill=BLUE, width=3)
    checkpoint(draw, 1660, 1580, BLUE, "theta_G")
    arrow(draw, (1540, 1518), (1690, 1580), BLUE, width=3, head=11, dashed=True)

    attention = attention_block(draw, 1980, 350)
    arrow(draw, (dpt[2] + 25, 340), (attention[0], 470), GREEN, width=5, head=16)
    arrow(draw, (naf_maps[2] + 25, 765), (attention[0], 650), GREEN, width=5, head=16)
    arrow(draw, (geom_maps[2] + 10, 1110), (attention[0] + 90, attention[3]), BLUE, width=4, head=14, dashed=True)

    surfel = gaussian_cloud(draw, 2050, 1050, "LiDAR surfels", BLUE, seed=9)
    arrow(draw, (geom_maps[2] + 25, 1150), (surfel[0], 1250), BLUE, width=5, head=16)
    draw.line((2220, attention[3], 2220, surfel[1] + 20), fill=GREEN, width=5)
    arrow(draw, (2220, surfel[1] + 20), (2290, surfel[1] + 75), GREEN, width=5, head=16)

    gaussian = gaussian_cloud(draw, 2675, 560, "3D Feature Field", PURPLE, seed=31)
    arrow(draw, (attention[2], 650), (gaussian[0], 760), PURPLE, width=5, head=16)
    arrow(draw, (surfel[2], 1270), (gaussian[0], 980), BLUE, width=5, head=16)

    render = renderer_icon(draw, 3260, 700)
    arrow(draw, (gaussian[2], 820), (render[0], 835), AMBER, width=5, head=16)
    rendered_maps = feature_stack(draw, 3595, 690, 145, 225, 5, "#E7C4D7", PINK, "F32 Z N V A", (18, -15))
    arrow(draw, (render[2], 835), (3595, 835), AMBER, width=5, head=16)

    target_camera_x, target_camera_y = 3195, 1300
    draw.polygon([(target_camera_x, target_camera_y), (target_camera_x + 85, target_camera_y - 45), (target_camera_x + 85, target_camera_y + 45)], fill="#F1EFF8", outline=PURPLE)
    draw.line((target_camera_x + 85, target_camera_y - 45, target_camera_x + 210, target_camera_y - 110), fill=PURPLE, width=3)
    draw.line((target_camera_x + 85, target_camera_y + 45, target_camera_x + 210, target_camera_y + 110), fill=PURPLE, width=3)
    draw.line((target_camera_x + 210, target_camera_y - 110, target_camera_x + 210, target_camera_y + 110), fill=PURPLE, width=3)
    centered_text(draw, (target_camera_x + 105, target_camera_y + 135), "K_t D_t T_t", 26, PURPLE, True)
    arrow(draw, (target_camera_x + 210, target_camera_y), (render[0] + 60, render[3]), PURPLE, width=4, head=14)

    restore = restormer(draw, 3835, 665)
    arrow(draw, (rendered_maps[2] + 20, 835), (restore[0], 835), PURPLE, width=5, head=16)

    if args.reconstruction is None:
        target = fit_image(target_path, (250, 141))
    else:
        with np.load(args.reconstruction) as reconstruction:
            target = Image.fromarray(np.rint(
                reconstruction["rgb"].clip(0.0, 1.0) * 255.0
            ).astype(np.uint8)).resize((250, 141), RESAMPLE_LANCZOS)
    depth = sparse_depth_preview(points, calibrations[args.target_camera], (250, 141))
    output_x = 4510
    for index, image in enumerate([target, depth]):
        oy = 590 + index * 235
        draw.rectangle((output_x - 4, oy - 4, output_x + 254, oy + 145), fill=WHITE, outline=RED if index == 0 else BLUE, width=4)
        canvas.paste(image, (output_x, oy))
        centered_text(draw, (output_x + 125, oy + 158), "I_t" if index == 0 else "Z_t / V_t", 25, RED if index == 0 else BLUE, True)
    arrow(draw, (restore[2], 835), (output_x - 12, 835), RED, width=5, head=16)
    draw.line((rendered_maps[2], 940, 3780, 1100, 4465, 1100, 4465, 966), fill=BLUE, width=3)
    arrow(draw, (4465, 966), (output_x - 12, 966), BLUE, width=3, head=11)

    environment = (3835, 1120, 4165, 1250)
    rounded_box(draw, environment, "#E8F4F7", CYAN, radius=20, width=4)
    centered_text(draw, (4000, 1152), "Infinity Environment", 27, CYAN, True)
    arrow(draw, (4165, 1185), (4490, 730), CYAN, width=4, head=13)

    loss_circle(draw, 4280, 1390, "L_R", RED)
    draw.line((4635, 1230, 4635, 1390, 4315, 1390), fill=RED, width=3)
    draw.line((4245, 1390, 4120, 1020), fill=RED, width=3)
    checkpoint(draw, 4380, 1510, RED, "theta_A,R")
    arrow(draw, (4310, 1415), (4410, 1510), RED, width=3, head=11, dashed=True)

    draw.line((50, 1775, 4750, 1775), fill="#D4D7DC", width=2)
    centered_text(draw, (610, 1800), "training only", 23, MUTED)
    draw.line((485, 1840, 735, 1840), fill=LINE, width=3)
    for px in range(485, 735, 24):
        draw.line((px, 1840, min(px + 12, 735), 1840), fill=BLUE, width=4)
    centered_text(draw, (2190, 1800), "geometry", 23, BLUE, True)
    centered_text(draw, (2790, 1800), "appearance", 23, GREEN, True)
    centered_text(draw, (3460, 1800), "render", 23, AMBER, True)
    centered_text(draw, (4180, 1800), "restore", 23, PURPLE, True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    preview = args.output.with_name(args.output.stem + "_preview.png")
    canvas.resize((2400, 950), RESAMPLE_LANCZOS).save(preview, optimize=True)
    print(json.dumps({"output": str(args.output), "preview": str(preview), "size": list(CANVAS)}, indent=2))


if __name__ == "__main__":
    main()
