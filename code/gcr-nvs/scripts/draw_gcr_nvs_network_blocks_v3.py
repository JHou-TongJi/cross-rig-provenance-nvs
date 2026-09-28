"""Detailed neural-network block diagram for the final GCR-NVS architecture."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 5000, 2850
BG = "#F7F9FC"
INK = "#172033"
MUTED = "#66738A"
GRID = "#D5DDE8"
WHITE = "#FFFFFF"
BLUE = "#2E70C9"
BLUE_L = "#E6F0FF"
GREEN = "#238A68"
GREEN_L = "#E0F4EB"
PURPLE = "#7758B7"
PURPLE_L = "#F0E9FB"
ORANGE = "#C76D1D"
ORANGE_L = "#FFF0DE"
RED = "#C74450"
RED_L = "#FBE8EB"
TEAL = "#168B99"
SHADOW = "#C8D0DB"


def font(size: int, bold: bool = False):
    paths = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded(draw, box, fill, outline, width=4, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center(draw, box, text, size=22, color=INK, bold=False, spacing=4):
    f = font(size, bold)
    bb = draw.multiline_textbbox((0, 0), text, font=f, spacing=spacing, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text(((box[0] + box[2] - tw) / 2, (box[1] + box[3] - th) / 2 - bb[1]), text, fill=color, font=f, spacing=spacing, align="center")


def arrow(draw, start, end, color=INK, width=6, head=18, dashed=False):
    x0, y0 = start
    x1, y1 = end
    length = max(math.hypot(x1 - x0, y1 - y0), 1.0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    if dashed:
        cursor = 0.0
        while cursor < length - head:
            nxt = min(cursor + 24, length - head)
            draw.line((x0 + ux * cursor, y0 + uy * cursor, x0 + ux * nxt, y0 + uy * nxt), fill=color, width=width)
            cursor += 40
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    px, py = -uy, ux
    draw.polygon([(x1, y1), (x1 - head * ux + head * .42 * px, y1 - head * uy + head * .42 * py), (x1 - head * ux - head * .42 * px, y1 - head * uy - head * .42 * py)], fill=color)


def prism(draw, x, y, w, h, dx, dy, front, side, top, label="", label_color=INK, label_size=18):
    """Draw a paper-style 3D cuboid with a front, top and side face."""
    front_poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top_poly = [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)]
    side_poly = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    draw.polygon(top_poly, fill=top, outline=side)
    draw.polygon(side_poly, fill=side, outline=side)
    draw.polygon(front_poly, fill=front, outline=side)
    draw.line((*front_poly[0], *front_poly[1]), fill=side, width=2)
    if label:
        center(draw, (x + 4, y + 4, x + w - 4, y + h - 4), label, label_size, label_color, True)
    return (x, y + dy, x + w + dx, y + h)


def thumb(canvas, draw, path: Path, box, label, color):
    x0, y0, x1, y1 = box
    if path.exists():
        with Image.open(path) as source:
            image = source.convert("RGB")
        ratio = (x1 - x0) / (y1 - y0)
        sr = image.width / image.height
        if sr > ratio:
            crop = int(image.height * ratio)
            left = (image.width - crop) // 2
            image = image.crop((left, 0, left + crop, image.height))
        else:
            crop = int(image.width / ratio)
            top = (image.height - crop) // 2
            image = image.crop((0, top, image.width, top + crop))
        canvas.paste(image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), (x0, y0))
    else:
        draw.rectangle(box, fill="#E1E7EF")
    draw.rectangle(box, outline=color, width=4)
    draw.rectangle((x0, y0, x1, y0 + 32), fill=(0, 0, 0))
    draw.text((x0 + 9, y0 + 4), label, font=font(17, True), fill=WHITE)


def block(draw, box, title, subtitle, fill, outline, title_size=24):
    rounded(draw, box, fill, outline, 4, 16)
    x0, y0, x1, y1 = box
    center(draw, (x0 + 8, y0 + 8, x1 - 8, y0 + 58), title, title_size, outline, True)
    center(draw, (x0 + 8, y0 + 60, x1 - 8, y1 - 8), subtitle, 17, MUTED)


def transformer_stack(draw, x, y, count, width, height, color, labels):
    for i in range(count):
        bx = x + i * 35
        by = y - i * 10
        prism(draw, bx, by, width, height, 16, -10, color[0], color[1], color[2], labels[i] if i < len(labels) else "Block", WHITE, 16)
    return x, y - (count - 1) * 10, x + width + (count - 1) * 35 + 16, y + height


def conv_stack(draw, x, y, labels, color, width=120, height=86, gap=28):
    for i, label in enumerate(labels):
        bx = x + i * (width + gap)
        prism(draw, bx, y, width, height, 14, -10, color[0], color[1], color[2], label, WHITE, 16)
        if i < len(labels) - 1:
            arrow(draw, (bx + width + 7, y + height // 2), (bx + width + gap - 7, y + height // 2), color[0], 4, 12)


def section_title(draw, x, y, number, title, color):
    draw.ellipse((x, y, x + 42, y + 42), fill=color)
    center(draw, (x, y, x + 42, y + 42), str(number), 19, WHITE, True)
    draw.text((x + 56, y + 6), title, font=font(25, True), fill=color)


def draw(args):
    pipeline = args.pipeline.resolve()
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 34), "GCR-NVS", font=font(54, True), fill=INK)
    draw.text((365, 53), "RGB 引导的稠密几何与语义新视角重建", font=font(25), fill=MUTED)
    draw.line((70, 115, W - 70, 115), fill=GRID, width=3)

    # Four main lanes
    section_title(draw, 75, 150, 1, "输入与公制深度", TEAL)
    section_title(draw, 1260, 150, 2, "神经特征金字塔", BLUE)
    section_title(draw, 2550, 150, 3, "三维场与视角变换", PURPLE)
    section_title(draw, 3700, 150, 4, "目标视图合成", ORANGE)

    # Lane 1: inputs and depth teacher
    thumb(canvas, draw, pipeline / "01_01_raw_rgb.png", (75, 245, 330, 410), "原始 RGB", TEAL)
    thumb(canvas, draw, pipeline / "02_02_rectified_rgb.png", (370, 245, 625, 410), "去畸变 RGB", TEAL)
    arrow(draw, (330, 327), (370, 327), TEAL)
    input_contract = (75, 475, 625, 620)
    rounded(draw, input_contract, TEAL, TEAL, 4, 16)
    center(draw, (90, 490, 610, 545), "K、D、T  |  相机拓扑  |  时间戳", 22, WHITE, True)
    center(draw, (90, 545, 610, 605), "七路相机标定契约", 18, WHITE)
    arrow(draw, (500, 410), (500, 475), TEAL)

    thumb(canvas, draw, pipeline / "04_04_da3_depth.png", (75, 710, 330, 875), "单目深度 D_m", BLUE)
    block(draw, (370, 690, 625, 895), "DA3Metric-Large", "ViT-L/14 + DPT", BLUE_L, BLUE, 22)
    transformer_stack(draw, 405, 730, 5, 34, 105, (BLUE, "#8DB3E9", "#C7DBF8"), ["Patch", "Block", "Block", "Block", "DPT"])
    arrow(draw, (625, 792), (75, 792), BLUE, dashed=True)
    arrow(draw, (330, 792), (370, 792), BLUE)

    block(draw, (75, 960, 330, 1145), "真实 LiDAR", "点坐标 + 时间字段", BLUE_L, BLUE, 22)
    conv_stack(draw, 370, 990, ["体素化", "时序", "动态掩码"], (BLUE, "#8DB3E9", "#C7DBF8"), width=70, height=110, gap=22)
    arrow(draw, (330, 1050), (370, 1050), BLUE)
    block(draw, (75, 1210, 625, 1375), "公制深度对齐", "逆深度仿射 + LiDAR 留出验证 + 置信度", GREEN_L, GREEN, 22)
    arrow(draw, (500, 895), (500, 1210), GREEN)
    arrow(draw, (625, 1050), (625, 1290), BLUE)
    thumb(canvas, draw, pipeline / "06_06_fused_depth.png", (370, 1410, 625, 1575), "融合深度 D_dense", GREEN)
    arrow(draw, (500, 1375), (500, 1410), GREEN)

    # Lane 2: DINO and AnyUP
    block(draw, (1260, 245, 1540, 420), "DINOv2", "ViT-L/14 语义编码", GREEN_L, GREEN, 24)
    transformer_stack(draw, 1292, 475, 6, 36, 118, (GREEN, "#8AC5AC", "#C4E6D6"), ["Patch", "Block", "Block", "Block", "Block", "Block"])
    thumb(canvas, draw, pipeline / "11_11_dinov2_plus_anyup.png", (1580, 245, 1855, 420), "语义特征 F_64", GREEN)
    arrow(draw, (625, 327), (1260, 327), GREEN)
    arrow(draw, (1400, 420), (1400, 475), GREEN)
    arrow(draw, (1540, 327), (1580, 327), GREEN)

    block(draw, (1260, 690, 1855, 875), "AnyUP 特征上采样", "像素查询引导的语义恢复", GREEN_L, GREEN, 22)
    conv_stack(draw, 1300, 745, ["查询 Q", "键值 K/V", "像素查询", "FPN"], (GREEN, "#8AC5AC", "#C4E6D6"), width=105, height=74, gap=24)
    arrow(draw, (1400, 610), (1400, 690), GREEN)
    block(draw, (1260, 960, 1855, 1145), "细节 CNN / NAF", "RGB 边缘与高频纹理分支", GREEN_L, GREEN, 22)
    conv_stack(draw, 1300, 1010, ["Stem", "NAF", "NAF", "NAF"], (GREEN, "#8AC5AC", "#C4E6D6"), width=105, height=74, gap=24)
    arrow(draw, (1400, 875), (1400, 960), GREEN)
    block(draw, (1260, 1230, 1855, 1415), "语义 / 细节融合", "F_64 @ H × W", GREEN_L, GREEN, 22)
    arrow(draw, (1550, 1145), (1550, 1230), GREEN)

    # Lane 3: geometry / 3D view
    block(draw, (2550, 245, 2850, 420), "SparseConv U-Net", "LiDAR 几何编码器", BLUE_L, BLUE, 22)
    conv_stack(draw, 2585, 475, ["32", "64", "128", "256"], (BLUE, "#8DB3E9", "#C7DBF8"), width=62, height=110, gap=24)
    arrow(draw, (625, 1290), (2550, 327), BLUE)
    block(draw, (2550, 690, 2850, 875), "稠密三维表面", "D_dense → X_3D", PURPLE_L, PURPLE, 22)
    net_blocks = ["K⁻¹", "反投影", "X_3D"]
    conv_stack(draw, 2588, 745, net_blocks, (PURPLE, "#B4A0DC", "#DDD3F1"), width=70, height=74, gap=22)
    arrow(draw, (1855, 1320), (2550, 780), GREEN)
    arrow(draw, (2700, 420), (2700, 690), PURPLE)
    block(draw, (2550, 960, 2850, 1145), "目标相机位姿", "K_t、T_t、SE(3)", PURPLE_L, PURPLE, 22)
    conv_stack(draw, 2590, 1015, ["旋转 R", "平移 t", "目标射线"], (PURPLE, "#B4A0DC", "#DDD3F1"), width=70, height=70, gap=22)
    arrow(draw, (2700, 875), (2700, 960), PURPLE)
    block(draw, (2550, 1230, 2850, 1415), "Z-buffer 可见性", "深度排序与来源掩码", PURPLE_L, PURPLE, 22)
    conv_stack(draw, 2590, 1285, ["深度", "近表面", "掩码"], (PURPLE, "#B4A0DC", "#DDD3F1"), width=70, height=70, gap=22)
    arrow(draw, (2700, 1145), (2700, 1230), PURPLE)
    thumb(canvas, draw, pipeline / "08_08_3d_surface___se(3).png", (2940, 960, 3215, 1125), "目标表面 P_t", PURPLE)
    arrow(draw, (2850, 1320), (2940, 1040), PURPLE)

    # Cross-view appearance and decoder
    block(draw, (3310, 245, 3610, 420), "RGB 颜色采样", "grid_sample @ X_3D", ORANGE_L, ORANGE, 22)
    thumb(canvas, draw, pipeline / "09_09_rgb_warp.png", (3650, 245, 3925, 420), "RGB 重投影", ORANGE)
    arrow(draw, (3215, 1040), (3310, 327), ORANGE)
    arrow(draw, (3610, 327), (3650, 327), ORANGE)
    block(draw, (3310, 690, 3925, 875), "跨视图多头注意力", "Q=几何  K/V=RGB+F_64", PURPLE_L, PURPLE, 22)
    transformer_stack(draw, 3360, 745, 4, 48, 82, (PURPLE, "#B4A0DC", "#DDD3F1"), ["Q", "K", "V", "MHA"])
    arrow(draw, (3215, 1040), (3310, 780), PURPLE)
    arrow(draw, (1855, 1320), (3310, 810), GREEN, dashed=True)
    block(draw, (3310, 960, 3925, 1145), "深度 / 视图门控", "置信度 + 来源 + 焦距角色", GREEN_L, GREEN, 22)
    conv_stack(draw, 3350, 1015, ["置信度", "深度", "视图", "门控"], (GREEN, "#8AC5AC", "#C4E6D6"), width=105, height=72, gap=16)
    arrow(draw, (3610, 875), (3610, 960), GREEN)
    block(draw, (3310, 1230, 3925, 1415), "Restormer", "编码器 → 瓶颈 → 解码器", ORANGE_L, ORANGE, 22)
    # U-shaped Restormer blocks
    conv_stack(draw, 3350, 1285, ["48", "96", "192", "96", "48"], (ORANGE, "#E2B27B", "#F5DEC0"), width=76, height=72, gap=14)
    arrow(draw, (3610, 1145), (3610, 1230), ORANGE)
    thumb(canvas, draw, pipeline / "12_12_completion_output.png", (3650, 1485, 3925, 1650), "最终 RGB_t", ORANGE)
    arrow(draw, (3925, 1320), (4050, 1485), ORANGE)
    block(draw, (3310, 1740, 3925, 1905), "双路输出", "可观测区：受限残差\n新暴露区：生成补全", ORANGE_L, ORANGE, 22)
    arrow(draw, (3610, 1415), (3610, 1740), ORANGE)

    # Explanation band
    draw.line((75, 2050, W - 75, 2050), fill=GRID, width=3)
    block(draw, (75, 2100, 1550, 2470), "几何分工", "LiDAR：公制尺度 / 局部结构 / 可见性\nDA3：连续稠密深度候选\nD_dense → X_3D → SE(3) → Z-buffer\nLiDAR 不直接覆盖 RGB", BLUE_L, BLUE, 25)
    block(draw, (1680, 2100, 3150, 2470), "外观与语义分工", "RGB：颜色与纹理\nDINOv2：场景语义\nAnyUP：高分辨率语义定位\nMHA：拓扑与深度一致的视图融合", GREEN_L, GREEN, 25)
    block(draw, (3280, 2100, 4925, 2470), "训练与验收", "严格 LOO + 10cm SE(3) 小扰动\n可观测 / 不确定 / 新暴露区域损失\n逐相机深度留出 + RGB 指标\n生成器只补缺失观测区域", RED_L, RED, 25)
    draw.text((75, 2555), "上方展示网络结构块；下方统一解释数据分工、语义分工与验收协议。", font=font(21), fill=MUTED)
    draw.text((75, 2600), "当前试验：7 路 DA3 缓存、AnyUP 语义审计、同名相机优先、目标位姿上移 10cm。", font=font(21, True), fill=INK)

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
