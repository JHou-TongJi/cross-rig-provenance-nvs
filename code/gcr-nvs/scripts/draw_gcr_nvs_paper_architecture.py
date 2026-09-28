"""Draw the proposed LiDAR-structure/RGB-appearance GCR-NVS paper figure with PIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.calibration import load_calibrations, project_world_points
from gcr_nvs.geometry.pcd import read_pcd


CANVAS = (4800, 3000)
BACKGROUND = "#F7F9FC"
INK = "#152238"
MUTED = "#64748B"
LINE = "#A7B4C7"
WHITE = "#FFFFFF"
BLUE = "#246BCE"
BLUE_LIGHT = "#E9F2FF"
CYAN = "#008EA8"
CYAN_LIGHT = "#E4F8FB"
GREEN = "#16835A"
GREEN_LIGHT = "#E8F7F0"
AMBER = "#B96800"
AMBER_LIGHT = "#FFF3DE"
PURPLE = "#7551B2"
PURPLE_LIGHT = "#F1EBFA"
RED = "#C63C43"
RED_LIGHT = "#FDECEE"
GRAY_LIGHT = "#EEF2F7"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_MEDIUM = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def font(size: int, bold: bool = False, medium: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_MEDIUM if medium else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    native = getattr(draw, "textbbox", None)
    if native is not None:
        return native(xy, text, font=text_font)
    width, height = draw.textsize(text, font=text_font)
    return xy[0], xy[1], xy[0] + width, xy[1] + height


def _rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    """Draw a rounded rectangle on Pillow builds without rounded_rectangle()."""
    native = getattr(draw, "rounded_rectangle", None)
    if native is not None:
        native(xy, radius=radius, fill=fill, outline=outline, width=width)
        return

    left, top, right, bottom = xy
    radius = max(0, min(radius, (right - left) // 2, (bottom - top) // 2))

    def draw_fill(bounds: tuple[int, int, int, int], inset: int, color: str) -> None:
        x0, y0, x1, y1 = bounds
        r = max(0, radius - inset)
        draw.rectangle((x0 + r, y0, x1 - r, y1), fill=color)
        draw.rectangle((x0, y0 + r, x1, y1 - r), fill=color)
        if r == 0:
            return
        diameter = 2 * r
        draw.pieslice((x0, y0, x0 + diameter, y0 + diameter), 180, 270, fill=color)
        draw.pieslice((x1 - diameter, y0, x1, y0 + diameter), 270, 360, fill=color)
        draw.pieslice((x1 - diameter, y1 - diameter, x1, y1), 0, 90, fill=color)
        draw.pieslice((x0, y1 - diameter, x0 + diameter, y1), 90, 180, fill=color)

    if outline and width > 0:
        draw_fill((left, top, right, bottom), 0, outline)
        inset = min(width, (right - left) // 2, (bottom - top) // 2)
        draw_fill((left + inset, top + inset, right - inset, bottom - inset), inset, fill)
    else:
        draw_fill((left, top, right, bottom), 0, fill)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str = LINE,
    width: int = 3,
    radius: int = 22,
) -> None:
    _rounded_rectangle(draw, xy, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and text_bbox(draw, (0, 0), candidate, text_font)[2] > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current)
    return "\n".join(lines)


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    outline: str,
    title_color: str = INK,
    body_color: str = MUTED,
    title_size: int = 41,
    body_size: int = 31,
    title_height: int = 68,
) -> None:
    rounded_box(draw, xy, fill, outline)
    left, top, right, _ = xy
    draw.text((left + 28, top + 22), title, font=font(title_size, bold=True), fill=title_color)
    if body:
        wrapped = wrap_text(draw, body, font(body_size), right - left - 56)
        draw.multiline_text(
            (left + 28, top + title_height),
            wrapped,
            font=font(body_size),
            fill=body_color,
            spacing=11,
        )


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str,
    color: str,
    outline: str | None = None,
    size: int = 27,
) -> None:
    _rounded_rectangle(
        draw,
        xy,
        radius=(xy[3] - xy[1]) // 2,
        fill=fill,
        outline=outline,
        width=2,
    )
    bbox = text_bbox(draw, (0, 0), text, font(size, medium=True))
    x = xy[0] + (xy[2] - xy[0] - (bbox[2] - bbox[0])) // 2
    y = xy[1] + (xy[3] - xy[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, font=font(size, medium=True), fill=color)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = INK,
    width: int = 8,
    head: int = 22,
    dashed: bool = False,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if dashed:
        distance = max(abs(x1 - x0), abs(y1 - y0))
        segments = max(1, distance // 34)
        for index in range(segments):
            if index % 2:
                continue
            t0 = index / segments
            t1 = min(1.0, (index + 1) / segments)
            draw.line(
                (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                 x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
                fill=color,
                width=width,
            )
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = np.arctan2(y1 - y0, x1 - x0)
    left = (
        x1 - head * np.cos(angle - np.pi / 6),
        y1 - head * np.sin(angle - np.pi / 6),
    )
    right = (
        x1 - head * np.cos(angle + np.pi / 6),
        y1 - head * np.sin(angle + np.pi / 6),
    )
    draw.polygon([(x1, y1), left, right], fill=color)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    target_ratio = size[0] / size[1]
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        crop_width = int(image.height * target_ratio)
        left = (image.width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, image.height))
    else:
        crop_height = int(image.width / target_ratio)
        top = (image.height - crop_height) // 2
        image = image.crop((0, top, image.width, top + crop_height))
    return image.resize(size, RESAMPLE_LANCZOS)


def depth_color(depth: np.ndarray) -> np.ndarray:
    normalized = np.clip((depth - 2.0) / 58.0, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=1) * 255).astype(np.uint8)


def lidar_bev(points: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, "#071827")
    draw = ImageDraw.Draw(image)
    xyz = points[:, :3]
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= -55.0) & (xyz[:, 0] <= 55.0)
        & (xyz[:, 1] >= -35.0) & (xyz[:, 1] <= 35.0)
        & (xyz[:, 2] >= -4.0) & (xyz[:, 2] <= 8.0)
    )
    selected = xyz[valid]
    if len(selected) > 55_000:
        selected = selected[np.linspace(0, len(selected) - 1, 55_000).astype(np.int64)]
    x = ((selected[:, 0] + 55.0) / 110.0 * (size[0] - 1)).astype(np.int32)
    y = ((35.0 - selected[:, 1]) / 70.0 * (size[1] - 1)).astype(np.int32)
    colors = depth_color(np.linalg.norm(selected[:, :2], axis=1))
    for px, py, color in zip(x, y, colors):
        draw.point((int(px), int(py)), fill=tuple(int(value) for value in color))
    center_x = int(55.0 / 110.0 * size[0])
    center_y = int(35.0 / 70.0 * size[1])
    _rounded_rectangle(
        draw,
        (center_x - 20, center_y - 12, center_x + 20, center_y + 12),
        radius=5,
        fill="#FFFFFF",
        outline="#35B7D4",
        width=3,
    )
    return image


def sparse_depth_preview(
    points: np.ndarray,
    calibration,
    size: tuple[int, int],
) -> Image.Image:
    image = Image.new("RGB", size, "#071827")
    draw = ImageDraw.Draw(image)
    pixels, valid = project_world_points(points[:, :3], calibration)
    camera_points = (
        calibration.external
        @ np.c_[points[:, :3], np.ones(len(points))].T
    ).T[:, :3]
    indices = np.flatnonzero(valid & (camera_points[:, 2] > 0))
    if len(indices) > 45_000:
        indices = indices[np.linspace(0, len(indices) - 1, 45_000).astype(np.int64)]
    u = pixels[indices, 0] / calibration.width * size[0]
    v = pixels[indices, 1] / calibration.height * size[1]
    colors = depth_color(camera_points[indices, 2])
    order = np.argsort(camera_points[indices, 2])[::-1]
    for index in order:
        px, py = int(u[index]), int(v[index])
        if 0 <= px < size[0] and 0 <= py < size[1]:
            color = tuple(int(value) for value in colors[index])
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
    return image


def paste_framed(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    xy: tuple[int, int],
    caption: str,
    border: str,
    caption_fill: str,
    caption_color: str,
) -> None:
    x, y = xy
    _rounded_rectangle(
        draw,
        (x - 4, y - 4, x + image.width + 4, y + image.height + 48),
        radius=12,
        fill=WHITE,
        outline=border,
        width=4,
    )
    canvas.paste(image, (x, y))
    draw.rectangle((x, y + image.height, x + image.width, y + image.height + 44), fill=caption_fill)
    draw.text((x + 12, y + image.height + 6), caption, font=font(24, medium=True), fill=caption_color)


def draw_figure(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    sequence_dir = root / args.sequence
    config_path = sequence_dir / "Key_frames/camera_config" / f"{args.frame_id:06d}.json"
    payload = json.loads(config_path.read_text())
    calibrations = load_calibrations(config_path, args.distortion)
    lidar_path = sequence_dir / "Key_frames/LIDAR_CONCAT" / payload["sensors"]["LIDAR_CONCAT"]
    points = read_pcd(lidar_path)

    canvas = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((70, 48), "GCR-NVS：LiDAR 结构主导、RGB 外观承载的新视角重建架构", font=font(70, bold=True), fill=INK)
    draw.text(
        (74, 132),
        "Proposed architecture and end-to-end data flow  |  真实样本："
        f"{args.sequence} / frame {args.frame_id:06d}",
        font=font(31),
        fill=MUTED,
    )
    pill(draw, (3970, 55, 4695, 125), "目标：正确重建，而非灰色均值补洞", RED_LIGHT, RED, RED, 29)

    # Dataset sample panel.
    rounded_box(draw, (55, 205, 1110, 1945), WHITE, "#CAD4E1", radius=24)
    draw.text((90, 235), "A. 数据集样本与加载契约", font=font(46, bold=True), fill=INK)
    draw.text(
        (90, 295),
        "15 sequences / 1,199 frames / 8,393 images  |  每帧 7 RGB + LiDAR + K/T/D",
        font=font(27),
        fill=MUTED,
    )
    thumb_size = (300, 169)
    positions = [(90, 355), (420, 355), (750, 355), (90, 585), (420, 585), (750, 585), (255, 815)]
    for camera, position in zip(CAMERA_NAMES, positions):
        image_path = sequence_dir / "Key_frames" / camera / payload["sensors"][camera]
        image = fit_image(image_path, thumb_size)
        is_target = camera == args.target_camera
        paste_framed(
            canvas,
            draw,
            image,
            position,
            f"{camera}  {'TARGET/GT' if is_target else 'SOURCE'}",
            RED if is_target else GREEN,
            RED_LIGHT if is_target else GREEN_LIGHT,
            RED if is_target else GREEN,
        )
    pill(draw, (90, 1050, 590, 1110), "split：13 train / 1 val / 1 test", RED_LIGHT, RED, RED)
    pill(draw, (615, 1050, 1025, 1110), "训练严格 LOO；推理给定 K/T", PURPLE_LIGHT, PURPLE, PURPLE)

    bev = lidar_bev(points, (610, 405))
    depth = sparse_depth_preview(points, calibrations[args.target_camera], (300, 405))
    paste_framed(canvas, draw, bev, (90, 1160), f"LiDAR BEV  |  {len(points):,} points", CYAN, CYAN_LIGHT, CYAN)
    paste_framed(canvas, draw, depth, (750, 1160), f"{args.target_camera} sparse Z", BLUE, BLUE_LIGHT, BLUE)
    draw.text((90, 1640), "加载与检查", font=font(32, bold=True), fill=INK)
    checks = [
        "✓ 相机图像尺寸与解码",
        "✓ timestamp / frame pairing",
        "✓ Brown 畸变 D 与 K/T 契约",
        "✓ PCD x,y,z,intensity,ring,time",
        "✓ train / val / test 按序列隔离",
    ]
    for index, item in enumerate(checks):
        draw.text((105 + (index % 2) * 490, 1690 + (index // 2) * 58), item, font=font(26), fill=INK)

    # Loader and preprocessing.
    label(
        draw,
        (1180, 245, 1585, 730),
        "B. Dataset Loader",
        "manifest 读取\n同步帧校验\n相机 resize\nLiDAR 原始点保留\n目标相机/目标位姿采样\n\n输出：\nRGB[V,3,H,W]\nPCD[N,6]\nK,D,T 与 validity",
        GRAY_LIGHT,
        "#8797AD",
        title_size=39,
        body_size=29,
    )
    label(
        draw,
        (1180, 790, 1585, 1285),
        "几何预处理",
        "LiDAR 时序/deskew 检查\nvehicle frame 统一\n距离自适应 voxel\n动态点 mask\n精确 z-buffer\n\n禁止：\n无 RGB 点填 0.5 灰色",
        CYAN_LIGHT,
        CYAN,
        title_size=39,
        body_size=29,
    )
    label(
        draw,
        (1180, 1345, 1585, 1845),
        "外观预处理",
        "RGB 归一化\n相机曝光/颜色校准\nDINO 语义特征\n高频 CNN 细节特征\n\n仅提供场景内容；\n不输出公制深度",
        GREEN_LIGHT,
        GREEN,
        title_size=39,
        body_size=29,
    )
    arrow(draw, (1110, 660), (1180, 490), LINE)
    arrow(draw, (1110, 1370), (1180, 1020), CYAN)
    arrow(draw, (1110, 890), (1180, 1570), GREEN)

    # LiDAR structure lane.
    draw.text((1660, 235), "C1. 结构分支：只由 LiDAR 决定公制几何", font=font(43, bold=True), fill=BLUE)
    label(
        draw,
        (1650, 315, 2050, 650),
        "Sparse 3D Encoder",
        "SparseConv / PointTransformer\n输入仅为 x,y,z,intensity,ring,time\n学习局部表面与占据关系",
        BLUE_LIGHT,
        BLUE,
        title_size=36,
        body_size=25,
    )
    label(
        draw,
        (2150, 315, 2580, 650),
        "LiDAR Depth Completion",
        "预测 target-view dense Z\n显式前/后表面概率\n边界与动态区域置信度\n用 LiDAR holdout 监督",
        BLUE_LIGHT,
        BLUE,
        title_size=35,
        body_size=27,
    )
    label(
        draw,
        (2680, 315, 3100, 650),
        "Metric Geometry",
        "dense depth Z_t\noccupancy O_t\nvisibility V_t\nnormal N_t\nuncertainty U_z",
        BLUE_LIGHT,
        BLUE,
        title_size=36,
        body_size=28,
    )
    arrow(draw, (2050, 480), (2150, 480), BLUE)
    arrow(draw, (2580, 480), (2680, 480), BLUE)
    arrow(draw, (1585, 1020), (1650, 480), BLUE)
    pill(draw, (1800, 690, 2920, 756), "硬约束：LiDAR exact > LiDAR completion；不回退为 RGB plane-sweep 深度", BLUE, WHITE, BLUE, 28)

    # RGB appearance lane.
    draw.text((1660, 830), "C2. 外观分支：RGB 学场景语义、材质与细节", font=font(43, bold=True), fill=GREEN)
    label(
        draw,
        (1650, 910, 2050, 1245),
        "Multi-view RGB Encoder",
        "DINOv2 semantic pyramid\nNAF/CNN 高频纹理\n每路相机独立编码\n保留 view id 与曝光信息",
        GREEN_LIGHT,
        GREEN,
        title_size=36,
        body_size=27,
    )
    label(
        draw,
        (2150, 910, 2580, 1245),
        "Geometry-guided Lift",
        "按 LiDAR 三维位置采样 RGB feature\nBrown-aware projection\nz-buffer 可见性检查\n只写入真实观测特征",
        GREEN_LIGHT,
        GREEN,
        title_size=36,
        body_size=27,
    )
    label(
        draw,
        (2680, 910, 3100, 1245),
        "3D Appearance Field",
        "80k-120k surfels / Gaussians\n每点存 feature + color + validity\n多视图 attention 融合\n不使用伪灰色颜色",
        GREEN_LIGHT,
        GREEN,
        title_size=36,
        body_size=27,
    )
    arrow(draw, (1585, 1570), (1650, 1075), GREEN)
    arrow(draw, (2050, 1075), (2150, 1075), GREEN)
    arrow(draw, (2580, 1075), (2680, 1075), GREEN)

    # Scene fusion and renderer.
    label(
        draw,
        (3195, 345, 3655, 820),
        "D. 3D Scene Fusion",
        "结构 token：Z/O/V/N\n外观 token：RGB features\n\n分离 validity：\nstructure_validity\nappearance_validity\ntrue_disocclusion\n\n融合发生在 3D，而不是\n目标图像上的稀疏散点插值",
        PURPLE_LIGHT,
        PURPLE,
        title_size=40,
        body_size=28,
    )
    arrow(draw, (3100, 500), (3195, 540), BLUE)
    arrow(draw, (3100, 1075), (3195, 700), GREEN)
    label(
        draw,
        (3195, 900, 3655, 1315),
        "Target-view Renderer",
        "输入目标 K_t, D_t, T_t\nsoft z-buffer / Gaussian splat\n输出 observed RGB、depth、coverage\n几何边界由 LiDAR visibility 决定",
        AMBER_LIGHT,
        AMBER,
        title_size=40,
        body_size=29,
    )
    arrow(draw, (3425, 820), (3425, 900), PURPLE)
    pill(draw, (3225, 1355, 3625, 1420), "目标相机参数 K/D/T", PURPLE, WHITE, PURPLE, 28)
    arrow(draw, (3425, 1420), (3425, 1315), PURPLE)

    # Output and completion.
    label(
        draw,
        (3760, 260, 4685, 690),
        "E. Correctness-aware Routing",
        "① observed + reliable structure：直接保留真实重投影\n② structure valid / appearance missing：调用 3D appearance completion\n③ structure missing：标记 uncertainty，不把灰色当观测\n④ true disocclusion：仅由有外部真值的生成分支处理",
        WHITE,
        "#8B6BB8",
        title_size=43,
        body_size=31,
    )
    arrow(draw, (3655, 1100), (3920, 690), AMBER)
    target_path = sequence_dir / "Key_frames" / args.target_camera / payload["sensors"][args.target_camera]
    target_preview = fit_image(target_path, (510, 287))
    paste_framed(canvas, draw, target_preview, (3790, 810), "Desired target RGB / training GT", RED, RED_LIGHT, RED)
    depth_large = sparse_depth_preview(points, calibrations[args.target_camera], (330, 287))
    paste_framed(canvas, draw, depth_large, (4340, 810), "LiDAR Z + validity", BLUE, BLUE_LIGHT, BLUE)
    label(
        draw,
        (3760, 1190, 4685, 1775),
        "最终输出",
        "RGB reconstruction I_t\nmetric depth Z_t\nstructure / appearance coverage\nuncertainty U_t\nsource provenance\n\n核心约束：\n已观测区域追求几何正确与像素保真；\n未观测区域必须有 temporal / 3DGS / real rig 外部监督，\n否则只输出不确定性，不能伪装成正确重建。",
        WHITE,
        "#5D6C82",
        title_size=42,
        body_size=30,
    )

    # Bottom training and inference band.
    draw.line((55, 2010, 4745, 2010), fill="#CCD5E2", width=4)
    draw.text((65, 2040), "F. 训练流程、监督信号与产物", font=font(48, bold=True), fill=INK)
    label(
        draw,
        (65, 2120, 920, 2840),
        "训练样本构造",
        "同一时刻 7 路相机 + LiDAR\n随机选择 target camera\n其余相机作为 RGB source\n\n目标位姿：\n• strict LOO：真实目标 RGB\n• perturbed pose：3DGS / temporal pseudo GT\n\n按 sequence 切分，禁止相邻帧泄漏",
        GRAY_LIGHT,
        "#8797AD",
        title_size=40,
        body_size=29,
    )
    label(
        draw,
        (1020, 2120, 1965, 2840),
        "训练损失",
        "LiDAR structure\n• sparse depth holdout\n• occupancy / visibility\n• edge-aware depth\n• temporal geometry consistency\n\nRGB appearance\n• observed Charbonnier + SSIM\n• perceptual loss on supervised holes\n• feature consistency\n• uncertainty calibration\n\n三类区域分别统计，禁止混合 PSNR 掩盖退化",
        BLUE_LIGHT,
        BLUE,
        title_size=40,
        body_size=27,
    )
    arrow(draw, (920, 2480), (1020, 2480), INK)
    label(
        draw,
        (2065, 2120, 2950, 2840),
        "训练产物",
        "checkpoint/\n• lidar_structure.pt\n• rgb_appearance.pt\n• scene_renderer.pt\n• completion.pt\n\nmetrics/\n• per-camera PSNR / SSIM\n• depth holdout error\n• observed / missing / disocclusion\n• coverage 与 uncertainty\n\nvisualizations/\n• RGB / depth / masks / provenance",
        GREEN_LIGHT,
        GREEN,
        title_size=40,
        body_size=27,
    )
    arrow(draw, (1965, 2480), (2065, 2480), INK)
    label(
        draw,
        (3050, 2120, 3970, 2840),
        "推理过程",
        "输入：单帧 7 RGB + LiDAR\n      + 任意目标 K/D/T\n\n1. LiDAR-only dense geometry\n2. RGB feature lift 到 3D\n3. target-view rendering\n4. correctness-aware routing\n5. 输出 RGB/depth/validity/U\n\n不使用目标 RGB；不依赖训练期 3DGS",
        PURPLE_LIGHT,
        PURPLE,
        title_size=40,
        body_size=29,
    )
    arrow(draw, (2950, 2480), (3050, 2480), INK)
    label(
        draw,
        (4070, 2120, 4735, 2840),
        "必须通过的门槛",
        "✓ 每个相机不系统性退化\n✓ LiDAR holdout 深度不崩溃\n✓ 黑洞/灰块单独量化\n✓ shifted 只做压力测试\n✓ checkpoint 记录数据/标定哈希\n✓ 可视化必须与 mask 对齐\n\n失败实验立即清理，不进入默认路径",
        RED_LIGHT,
        RED,
        title_size=39,
        body_size=28,
    )

    # Forbidden shortcuts callout.
    _rounded_rectangle(
        draw,
        (1650, 1510, 3655, 1870),
        radius=22,
        fill=RED_LIGHT,
        outline=RED,
        width=4,
    )
    draw.text((1690, 1540), "禁止的错误捷径", font=font(39, bold=True), fill=RED)
    forbidden = (
        "✗ LiDAR 不支持时直接让 RGB plane-sweep 决定主深度    "
        "✗ 无 RGB 观测点写入常数 0.5\n"
        "✗ 把 structure_validity 与 appearance_validity 合成一个 hole    "
        "✗ 只训练几百步就让二维 completion head 猜整块场景"
    )
    draw.multiline_text((1690, 1605), forbidden, font=font(29, medium=True), fill=RED, spacing=18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    preview = canvas.resize((2400, 1500), RESAMPLE_LANCZOS)
    preview_path = args.output.with_name(args.output.stem + "_preview.png")
    preview.save(preview_path, optimize=True)
    print(json.dumps({
        "output": str(args.output),
        "preview": str(preview_path),
        "canvas": list(CANVAS),
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "target_camera": args.target_camera,
        "lidar_points": int(len(points)),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-26-11-44-28")
    parser.add_argument("--frame-id", type=int, default=11)
    parser.add_argument("--target-camera", choices=CAMERA_NAMES, default="CAM_BACK_LEFT")
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_figures/gcr_nvs_lidar_rgb_architecture.png"),
    )
    draw_figure(parser.parse_args())


if __name__ == "__main__":
    main()
