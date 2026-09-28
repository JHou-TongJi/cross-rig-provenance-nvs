"""Polished paper-style architecture figure for GCR-NVS.

This version keeps the real intermediate thumbnails from the pilot pipeline,
but gives the neural-network blocks a stronger visual hierarchy and a tighter
four-column layout suitable for slides or a paper appendix.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from draw_gcr_nvs_network_blocks_v3 import (
    BLUE,
    BLUE_L,
    GREEN,
    GREEN_L,
    INK,
    MUTED,
    ORANGE,
    ORANGE_L,
    PURPLE,
    PURPLE_L,
    RED,
    RED_L,
    SHADOW,
    TEAL,
    WHITE,
    arrow,
    block,
    center,
    conv_stack,
    font,
    prism,
    rounded,
    thumb,
    transformer_stack,
)


W, H = 6200, 3450
BG = "#F4F7FB"
GRID = "#D9E1EC"
PANEL = "#FFFFFF"
INK_DARK = "#0F1B2D"


def line(draw, points, color, width=8, dashed=False):
    if dashed:
        for a, b in zip(points[:-1], points[1:]):
            x0, y0 = a
            x1, y1 = b
            length = max(math.hypot(x1 - x0, y1 - y0), 1.0)
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            pos = 0.0
            while pos < length:
                nxt = min(pos + 34, length)
                draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * nxt, y0 + uy * nxt), fill=color, width=width)
                pos += 58
    else:
        draw.line(points, fill=color, width=width, joint="curve")


def pill(draw, box, text, fill, outline, size=20):
    rounded(draw, box, fill, outline, 3, 24)
    center(draw, box, text, size, outline, True)


def lane(draw, box, num, title, accent, pale):
    rounded(draw, box, PANEL, GRID, 3, 30)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 92), radius=30, fill=pale, outline=None)
    draw.rectangle((x0, y0 + 62, x1, y0 + 92), fill=pale)
    draw.ellipse((x0 + 26, y0 + 23, x0 + 76, y0 + 73), fill=accent)
    center(draw, (x0 + 26, y0 + 23, x0 + 76, y0 + 73), str(num), 23, WHITE, True)
    draw.text((x0 + 95, y0 + 25), title, font=font(30, True), fill=accent)


def shadow_panel(canvas, draw, box, fill=PANEL, outline=GRID, radius=24):
    x0, y0, x1, y1 = box
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x0 + 10, y0 + 13, x1 + 10, y1 + 13), radius=radius, fill=(20, 37, 60, 30))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    canvas.alpha_composite(shadow)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)


def label(draw, x, y, text, size=20, color=MUTED, bold=False):
    draw.text((x, y), text, font=font(size, bold), fill=color)


def draw_v5(args):
    pipeline = args.pipeline.resolve()
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((90, 42), "GCR-NVS", font=font(64, True), fill=INK_DARK)
    draw.text((430, 63), "RGB 引导的稠密几何与语义新视角重建", font=font(31), fill=MUTED)
    pill(draw, (4800, 48, 6080, 100), "论文版 · 单帧前馈推理", "#E9EEF6", "#58708F", 19)
    draw.line((90, 135, W - 90, 135), fill=GRID, width=4)

    # Four tightly packed lanes.
    boxes = [(75, 175, 1510, 2520), (1570, 175, 3010, 2520), (3070, 175, 4510, 2520), (4570, 175, 6125, 2520)]
    lane(draw, boxes[0], 1, "输入与公制深度", TEAL, "#E4F5F6")
    lane(draw, boxes[1], 2, "特征金字塔", GREEN, "#E6F4ED")
    lane(draw, boxes[2], 3, "三维场与 SE(3)", PURPLE, "#F0EAFB")
    lane(draw, boxes[3], 4, "目标视图合成", ORANGE, "#FFF0E0")

    # Lane 1: source images, DA3, LiDAR and metric fusion.
    thumb(canvas, draw, pipeline / "01_01_raw_rgb.png", (115, 300, 445, 505), "原始 RGB", TEAL)
    thumb(canvas, draw, pipeline / "02_02_rectified_rgb.png", (490, 300, 820, 505), "去畸变 RGB", TEAL)
    arrow(draw, (445, 402), (490, 402), TEAL, 7, 20)
    pill(draw, (895, 330, 1460, 405), "K · D · T", "#D6F0F2", TEAL, 24)
    label(draw, 905, 420, "相机拓扑 · 时间戳 · 视角角色", 18, MUTED)
    arrow(draw, (820, 402), (895, 368), TEAL, 7, 20)

    thumb(canvas, draw, pipeline / "04_04_da3_depth.png", (115, 615, 445, 820), "DA3 单目深度", BLUE)
    shadow_panel(canvas, draw, (490, 585, 1460, 855), BLUE_L, "#78A9E6")
    label(draw, 530, 610, "DA3Metric-Large", 27, BLUE, True)
    label(draw, 530, 649, "ViT-L / DPT · 连续深度候选", 18, MUTED)
    transformer_stack(draw, 550, 705, 6, 54, 105, (BLUE, "#8DB3E9", "#C7DBF8"), ["Patch", "Block", "Block", "Block", "Block", "DPT"])
    arrow(draw, (275, 505), (275, 615), BLUE, 7, 20)
    arrow(draw, (820, 505), (820, 585), BLUE, 7, 20)

    shadow_panel(canvas, draw, (115, 965, 1460, 1225), BLUE_L, "#78A9E6")
    label(draw, 155, 990, "真实 LiDAR 时序点云", 27, BLUE, True)
    label(draw, 155, 1031, "点坐标 · 时间字段 · 动态掩码", 18, MUTED)
    conv_stack(draw, 590, 1025, ["体素", "时序", "去动态", "Sparse"], (BLUE, "#8DB3E9", "#C7DBF8"), width=118, height=92, gap=22)
    arrow(draw, (820, 855), (820, 965), BLUE, 7, 20)

    shadow_panel(canvas, draw, (115, 1355, 1460, 1615), GREEN_L, "#65B795")
    label(draw, 155, 1380, "公制深度对齐", 27, GREEN, True)
    label(draw, 155, 1422, "逆深度仿射 · LiDAR 留出验证 · 置信度门控", 18, MUTED)
    conv_stack(draw, 585, 1415, ["尺度", "对齐", "融合", "D_dense"], (GREEN, "#8AC5AC", "#C4E6D6"), width=120, height=92, gap=20)
    arrow(draw, (820, 1225), (820, 1355), GREEN, 7, 20)
    thumb(canvas, draw, pipeline / "06_06_fused_depth.png", (960, 1665, 1415, 1890), "融合深度 D_dense", GREEN)
    arrow(draw, (820, 1615), (1185, 1665), GREEN, 7, 20)

    # Lane 2: DINO, AnyUP, high-frequency RGB detail.
    thumb(canvas, draw, pipeline / "11_11_dinov2_plus_anyup.png", (1615, 300, 2035, 505), "DINOv2 + AnyUP", GREEN)
    shadow_panel(canvas, draw, (2080, 280, 2965, 750), GREEN_L, "#65B795")
    label(draw, 2120, 305, "DINOv2 ViT-L/14", 27, GREEN, True)
    label(draw, 2120, 346, "Patch Embedding → Transformer", 18, MUTED)
    transformer_stack(draw, 2140, 410, 7, 62, 118, (GREEN, "#8AC5AC", "#C4E6D6"), ["Patch", "Block", "Block", "Block", "Block", "Block", "Norm"])
    arrow(draw, (820, 402), (1615, 402), GREEN, 8, 22)

    shadow_panel(canvas, draw, (1615, 850, 2965, 1130), GREEN_L, "#65B795")
    label(draw, 1655, 875, "AnyUP 语义特征上采样", 27, GREEN, True)
    conv_stack(draw, 1660, 940, ["Q 查询", "K/V", "像素查询", "FPN", "F64"], (GREEN, "#8AC5AC", "#C4E6D6"), width=135, height=88, gap=17)
    arrow(draw, (2520, 750), (2220, 850), GREEN, 7, 20)

    shadow_panel(canvas, draw, (1615, 1220, 2965, 1500), GREEN_L, "#65B795")
    label(draw, 1655, 1245, "RGB 细节分支", 27, GREEN, True)
    label(draw, 1655, 1286, "高频纹理 · 边缘 · 局部颜色", 18, MUTED)
    conv_stack(draw, 1660, 1350, ["Stem", "NAF", "NAF", "NAF", "Fuse"], (GREEN, "#8AC5AC", "#C4E6D6"), width=135, height=88, gap=17)
    arrow(draw, (2290, 1130), (2290, 1220), GREEN, 7, 20)
    pill(draw, (1810, 1615, 2770, 1690), "语义 F64 + RGB 细节 → 3D 外观特征", "#DDF1E6", GREEN, 22)
    arrow(draw, (2290, 1500), (2290, 1615), GREEN, 7, 20)

    # Lane 3: 3D geometry and target ray casting.
    shadow_panel(canvas, draw, (3115, 280, 4465, 735), BLUE_L, "#78A9E6")
    label(draw, 3155, 305, "SparseConv U-Net", 27, BLUE, True)
    label(draw, 3155, 347, "LiDAR 几何编码 · 多尺度体素特征", 18, MUTED)
    conv_stack(draw, 3175, 420, ["32", "64", "128", "256"], (BLUE, "#8DB3E9", "#C7DBF8"), width=155, height=110, gap=26)
    arrow(draw, (1460, 1485), (3115, 470), BLUE, 9, 26)

    shadow_panel(canvas, draw, (3115, 850, 4465, 1130), PURPLE_L, "#9A7CD0")
    label(draw, 3155, 875, "稠密三维表面", 27, PURPLE, True)
    conv_stack(draw, 3170, 940, ["K⁻¹", "反投影", "X_3D", "表面"], (PURPLE, "#B4A0DC", "#DDD3F1"), width=145, height=88, gap=20)
    arrow(draw, (3800, 735), (3800, 850), PURPLE, 7, 20)
    arrow(draw, (2965, 1650), (3115, 1000), GREEN, 8, 22)

    shadow_panel(canvas, draw, (3115, 1220, 4465, 1500), PURPLE_L, "#9A7CD0")
    label(draw, 3155, 1245, "目标位姿与可见性", 27, PURPLE, True)
    conv_stack(draw, 3170, 1310, ["R", "t", "射线", "Z-buffer", "Mask"], (PURPLE, "#B4A0DC", "#DDD3F1"), width=138, height=88, gap=14)
    arrow(draw, (3800, 1130), (3800, 1220), PURPLE, 7, 20)
    thumb(canvas, draw, pipeline / "08_08_3d_surface___se(3).png", (3540, 1665, 4050, 1915), "目标表面 P_t", PURPLE)
    arrow(draw, (3800, 1500), (3800, 1665), PURPLE, 7, 20)

    # Lane 4: appearance projection, attention and restricted completion.
    shadow_panel(canvas, draw, (4615, 280, 6080, 735), ORANGE_L, "#DEA15F")
    label(draw, 4655, 305, "RGB 外观采样", 27, ORANGE, True)
    label(draw, 4655, 347, "X_3D → 源图颜色 / 语义特征", 18, MUTED)
    conv_stack(draw, 4675, 420, ["grid", "RGB", "F64", "融合"], (ORANGE, "#E2B27B", "#F5DEC0"), width=155, height=110, gap=26)
    arrow(draw, (4050, 1790), (4675, 470), ORANGE, 8, 22)
    thumb(canvas, draw, pipeline / "09_09_rgb_warp.png", (5515, 300, 6035, 550), "RGB 重投影", ORANGE)
    arrow(draw, (5280, 475), (5515, 425), ORANGE, 7, 20)

    shadow_panel(canvas, draw, (4615, 850, 6080, 1130), PURPLE_L, "#9A7CD0")
    label(draw, 4655, 875, "跨视图注意力", 27, PURPLE, True)
    conv_stack(draw, 4675, 940, ["Q 几何", "K RGB", "V F64", "MHA", "Gate"], (PURPLE, "#B4A0DC", "#DDD3F1"), width=145, height=88, gap=14)
    arrow(draw, (4465, 1010), (4615, 1010), PURPLE, 8, 22)
    arrow(draw, (2965, 1650), (4615, 1010), GREEN, 7, 20, dashed=True)

    shadow_panel(canvas, draw, (4615, 1220, 6080, 1500), ORANGE_L, "#DEA15F")
    label(draw, 4655, 1245, "Restormer 细化与补全", 27, ORANGE, True)
    conv_stack(draw, 4675, 1310, ["48", "96", "192", "96", "48"], (ORANGE, "#E2B27B", "#F5DEC0"), width=145, height=88, gap=14)
    arrow(draw, (5350, 1130), (5350, 1220), ORANGE, 7, 20)
    thumb(canvas, draw, pipeline / "12_12_completion_output.png", (5515, 1665, 6035, 1915), "最终 RGB_t", ORANGE)
    arrow(draw, (5350, 1500), (5775, 1665), ORANGE, 7, 20)
    pill(draw, (4680, 1990, 6015, 2075), "观测区：小残差  ·  新暴露区：生成补全", "#FFF1DF", ORANGE, 22)
    arrow(draw, (5775, 1915), (5350, 1990), ORANGE, 7, 20)

    # Strong cross-column backbone.
    line(draw, [(1460, 2100), (1570, 2100), (3010, 2100), (3070, 2100), (4510, 2100), (4570, 2100), (6125, 2100)], "#B6C3D4", 13)
    for x, c in [(1515, TEAL), (3040, GREEN), (4540, PURPLE)]:
        draw.ellipse((x - 18, 2082, x + 18, 2118), fill=c)
    label(draw, 90, 2150, "统一张量主干：去畸变 RGB + D_dense + F64 + 目标位姿", 25, INK_DARK, True)
    label(draw, 90, 2198, "颜色来自 RGB；结构来自 LiDAR 对齐后的稠密深度；生成器只负责真实缺失观测区域。", 21, MUTED)

    # Bottom explanation cards, deliberately concise and readable.
    cards = [(90, 2290, 2000, 3180, BLUE_L, BLUE, "几何约束"), (2095, 2290, 4005, 3180, GREEN_L, GREEN, "外观与语义"), (4100, 2290, 6110, 3180, RED_L, RED, "训练与验收")]
    for x0, y0, x1, y1, fill, accent, title in cards:
        shadow_panel(canvas, draw, (x0, y0, x1, y1), fill, accent)
        draw.text((x0 + 42, y0 + 32), title, font=font(29, True), fill=accent)
    label(draw, 135, 2385, "LiDAR：公制尺度、局部结构、遮挡与可见性", 23, INK_DARK)
    label(draw, 135, 2435, "DA3：稠密深度候选，不直接当作真值", 23, INK_DARK)
    label(draw, 135, 2485, "D_dense → 反投影 → SE(3) → Z-buffer", 23, INK_DARK)
    label(draw, 135, 2535, "LiDAR 不覆盖 RGB，不直接决定颜色", 23, BLUE, True)

    label(draw, 2140, 2385, "RGB：颜色、纹理与高频细节主来源", 23, INK_DARK)
    label(draw, 2140, 2435, "DINOv2：场景语义；AnyUP：高分辨率定位", 23, INK_DARK)
    label(draw, 2140, 2485, "MHA：按拓扑、深度和置信度融合视图", 23, INK_DARK)
    label(draw, 2140, 2535, "同名相机优先，邻近拓扑相机只填洞", 23, GREEN, True)

    label(draw, 4145, 2385, "严格 LOO + ±10cm SE(3) 小扰动", 23, INK_DARK)
    label(draw, 4145, 2435, "可观测 / 不确定 / 新暴露区域分开验收", 23, INK_DARK)
    label(draw, 4145, 2485, "逐相机深度留出 + RGB / SSIM / 覆盖率", 23, INK_DARK)
    label(draw, 4145, 2535, "目标：清晰、连续、无大块空洞与撕裂", 23, RED, True)

    label(draw, 90, 3290, "GCR-NVS · 当前试验：7 路 DA3 缓存、AnyUP 语义审计、同名相机优先、目标位姿上移 10cm", 21, MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(args.output, quality=96)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw_v5(args)


if __name__ == "__main__":
    main()
