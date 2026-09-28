"""Clean input-output architecture figure for the unified GCR-NVS model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

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
    TEAL,
    WHITE,
    arrow,
    center,
    conv_stack,
    font,
    prism,
    thumb,
    transformer_stack,
)


W, H = 6600, 3300
BG = "#F5F8FC"
PAPER = "#FFFFFF"
INK_D = "#172338"
GRID = "#D5DFEA"
BODY = "#F8FAFD"
TEAL_D = "#248B94"


def panel(draw, box, fill, outline=GRID, width=3, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def tx(draw, xy, text, size=20, color=INK_D, bold=False):
    draw.text(xy, text, font=font(size, bold), fill=color)


def tag(draw, box, text, fill, outline, size=18):
    panel(draw, box, fill, outline, 2, 18)
    center(draw, box, text, size, outline, True)


def stage(draw, box, n, title, subtitle, accent, pale):
    x0, y0, x1, y1 = box
    panel(draw, box, PAPER, accent, 4, 24)
    draw.rounded_rectangle((x0, y0, x1, y0 + 74), radius=24, fill=pale)
    draw.rectangle((x0, y0 + 48, x1, y0 + 74), fill=pale)
    draw.ellipse((x0 + 22, y0 + 17, x0 + 65, y0 + 60), fill=accent)
    center(draw, (x0 + 22, y0 + 17, x0 + 65, y0 + 60), str(n), 20, WHITE, True)
    tx(draw, (x0 + 82, y0 + 19), title, 24, accent, True)
    tx(draw, (x0 + 26, y0 + 90), subtitle, 17, MUTED)


def output_card(draw, box, title, subtitle, accent, pale):
    panel(draw, box, pale, accent, 3, 18)
    x0, y0, x1, y1 = box
    tx(draw, (x0 + 18, y0 + 16), title, 20, accent, True)
    tx(draw, (x0 + 18, y0 + 51), subtitle, 16, MUTED)


def draw(args):
    pipeline = args.pipeline.resolve()
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Header and legend.
    tx(draw, (80, 42), "GCR-NVS", 65, INK_D, True)
    tx(draw, (420, 65), "端到端 RGB–深度–LiDAR 三维新视角重建模型", 31, MUTED)
    tag(draw, (5060, 45, 6490, 103), "输入 → 统一模型 → 输出", "#E7EFF8", "#45627F", 21)
    draw.line((80, 140, W - 80, 140), fill=GRID, width=4)

    # Left: clearly separated input contract.
    input_box = (70, 220, 1130, 2250)
    panel(draw, input_box, "#EDF5FB", "#99B8D2", 4, 28)
    tx(draw, (105, 255), "输入", 31, TEAL_D, True)
    tx(draw, (105, 310), "七路源图像 + 时序雷达 + 目标相机参数", 19, MUTED)
    tag(draw, (105, 365, 1090, 425), "I_rgb : 7 × 3 × H × W", "#DDF3F3", TEAL_D, 21)
    thumb(canvas, draw, pipeline / "02_02_rectified_rgb.png", (105, 465, 495, 690), "去畸变 RGB", TEAL_D)
    thumb(canvas, draw, pipeline / "11_11_dinov2_plus_anyup.png", (520, 465, 910, 690), "语义 / 细节", GREEN)
    tag(draw, (105, 735, 1090, 795), "P_lidar : {x, y, z, t}  →  时序体素", "#E3ECFA", BLUE, 19)
    panel(draw, (105, 835, 1090, 1070), BLUE_L, "#83A8D3", 3, 18)
    tx(draw, (135, 865), "LiDAR 结构输入", 23, BLUE, True)
    tx(draw, (135, 907), "当前帧 + 最近时序帧", 18, MUTED)
    conv_stack(draw, 145, 955, ["voxel", "time", "mask"], (BLUE, "#8DB3E9", "#C7DBF8"), 150, 70, 18)
    tag(draw, (105, 1140, 1090, 1200), "K, D, T  |  相机拓扑  |  时间戳", "#E4F5F6", TEAL_D, 19)
    panel(draw, (105, 1245, 1090, 1485), PURPLE_L, "#A08BD0", 3, 18)
    tx(draw, (135, 1275), "目标视角条件", 23, PURPLE, True)
    tx(draw, (135, 1318), "K_t, T_t, Δξ = (Δx, Δy, Δz, Δr)", 18, MUTED)
    conv_stack(draw, 145, 1370, ["K_t", "R", "t", "ray"], (PURPLE, "#B4A0DC", "#DDD3F1"), 145, 70, 16)
    tag(draw, (105, 1580, 1090, 1640), "预处理：去畸变 / 坐标统一 / 动态掩码", "#F1F5F9", "#60758B", 18)
    panel(draw, (105, 1730, 1090, 2180), PAPER, "#B8C7D7", 3, 18)
    tx(draw, (135, 1760), "输入张量契约", 22, INK_D, True)
    tx(draw, (135, 1810), "RGB → 颜色与语义", 19, GREEN)
    tx(draw, (135, 1860), "LiDAR → 公制结构与可见性", 19, BLUE)
    tx(draw, (135, 1910), "DA3 → 稠密深度候选", 19, PURPLE)
    tx(draw, (135, 1960), "目标参数 → SE(3) 视角条件", 19, ORANGE)
    tag(draw, (105, 2140, 1090, 2200), "所有输入先对齐到同一目标坐标系", "#DCE8F3", "#45627F", 19)

    # Main body: one enclosing end-to-end model.
    body = (1190, 220, 5750, 2250)
    panel(draw, body, BODY, "#526A83", 5, 30)
    tx(draw, (1240, 255), "统一端到端模型  GCR-NVS", 34, INK_D, True)
    tx(draw, (1240, 305), "多模态编码 → 3D 表面 → 目标视角渲染 → 受限生成解码", 19, MUTED)
    tag(draw, (4380, 252, 5670, 310), "RGB 主导外观 · LiDAR 约束结构", "#E8F2FC", "#426687", 18)

    # Main backbone stages.
    s1 = (1250, 390, 2020, 1290)
    s2 = (2110, 390, 2880, 1290)
    s3 = (2970, 390, 3750, 1290)
    s4 = (3840, 390, 4620, 1290)
    s5 = (4710, 390, 5690, 1290)
    stage(draw, s1, 1, "多模态编码", "RGB / 深度 / LiDAR", TEAL_D, "#DDF3F3")
    transformer_stack(draw, 1285, 535, 5, 47, 100, (GREEN, "#65B795", "#B8E4CD"), ["DINO", "Block", "Block", "AnyUP", "F64"])
    conv_stack(draw, 1285, 800, ["DA3", "D_dense", "Sparse"], (BLUE, "#5D90CC", "#B9D2EF"), 135, 75, 15)
    tag(draw, (1280, 1040, 1990, 1095), "F_rgb  F_sem  F_geo", "#E6F4ED", GREEN, 18)
    tag(draw, (1280, 1130, 1990, 1185), "D_dense + confidence", "#E3ECFA", BLUE, 18)

    stage(draw, s2, 2, "三维融合", "metric surface field", PURPLE, "#F0E9FB")
    conv_stack(draw, 2145, 540, ["F_sem", "F_geo", "F_rgb", "Fuse"], (PURPLE, "#9A7CD0", "#DDD3F1"), 145, 78, 15)
    prism(draw, 2180, 770, 115, 170, 38, -24, "#6D52AB", "#4D3C82", "#B8A7E6", "D", WHITE, 18)
    prism(draw, 2345, 742, 115, 198, 38, -24, "#6D52AB", "#4D3C82", "#B8A7E6", "X", WHITE, 18)
    prism(draw, 2510, 714, 115, 226, 38, -24, "#6D52AB", "#4D3C82", "#B8A7E6", "Surf", WHITE, 17)
    tag(draw, (2140, 1015, 2850, 1070), "K⁻¹ · 反投影 · 表面特征", "#EEE7FB", PURPLE, 18)
    tag(draw, (2140, 1110, 2850, 1165), "X_3D / depth / normals", "#EEE7FB", PURPLE, 18)

    stage(draw, s3, 3, "目标视角", "SE(3) + visibility", BLUE, "#E5EFFC")
    conv_stack(draw, 3005, 540, ["K_t", "R", "t", "ray"], (PURPLE, "#9A7CD0", "#DDD3F1"), 145, 78, 13)
    conv_stack(draw, 3005, 750, ["Z-buffer", "near", "mask"], (PURPLE, "#9A7CD0", "#DDD3F1"), 190, 78, 15)
    tag(draw, (3000, 1015, 3720, 1070), "P_t / visibility / source id", "#EEE7FB", PURPLE, 18)
    tag(draw, (3000, 1110, 3720, 1165), "目标表面与可见性先验", "#E7EEF8", "#486681", 18)

    stage(draw, s4, 4, "跨视图融合", "geometry-guided attention", ORANGE, "#FFF0DE")
    transformer_stack(draw, 3875, 540, 4, 56, 108, (ORANGE, "#C66B1C", "#F1B26B"), ["Q_geo", "K_rgb", "V_sem", "MHA"])
    conv_stack(draw, 3880, 805, ["view", "depth", "gate"], (GREEN, "#65B795", "#B8E4CD"), 145, 78, 15)
    tag(draw, (3875, 1015, 4590, 1070), "RGB warp + F64 + P_t", "#FFF0DE", ORANGE, 18)
    tag(draw, (3875, 1110, 4590, 1165), "拓扑先验 / 同名相机优先", "#E6F4ED", GREEN, 18)

    stage(draw, s5, 5, "受限解码", "Restormer + completion", ORANGE, "#FFF0DE")
    prism(draw, 4750, 540, 110, 122, 32, -21, "#C66B1C", "#8B4816", "#F1B26B", "48", WHITE, 18)
    prism(draw, 4900, 570, 110, 160, 32, -21, "#C66B1C", "#8B4816", "#F1B26B", "96", WHITE, 18)
    prism(draw, 5050, 600, 110, 198, 32, -21, "#C66B1C", "#8B4816", "#F1B26B", "192", WHITE, 18)
    transformer_stack(draw, 5220, 845, 3, 64, 94, (ORANGE, "#C66B1C", "#F1B26B"), ["MDTA", "GDFN", "Norm"])
    conv_stack(draw, 4750, 1015, ["Obs", "Unc", "Disocc", "RGB_t"], (ORANGE, "#C66B1C", "#F1B26B"), 135, 72, 11)
    tag(draw, (4745, 1130, 5655, 1185), "观测区小残差 · 新暴露区补全", "#FFF0DE", ORANGE, 18)

    # Main stage-to-stage backbone arrows.
    for a, b, c in [((2020, 820), (2110, 820), TEAL_D), ((2880, 820), (2970, 820), PURPLE), ((3750, 820), (3840, 820), BLUE), ((4620, 820), (4710, 820), ORANGE)]:
        arrow(draw, a, b, c, 11, 26)

    # Intermediate artifacts shelf: explicit side products of the same model.
    tx(draw, (1250, 1370), "模型内部副产物", 24, INK_D, True)
    output_card(draw, (1250, 1425, 2020, 1745), "F_sem / F64", "DINOv2 + AnyUP 高分辨率语义", GREEN_L, GREEN)
    thumb(canvas, draw, pipeline / "11_11_dinov2_plus_anyup.png", (1280, 1495, 1545, 1700), "语义特征", GREEN)
    tag(draw, (1580, 1505, 1985, 1570), "语义边界", "#E6F4ED", GREEN, 17)
    tag(draw, (1580, 1600, 1985, 1665), "像素查询", "#E6F4ED", GREEN, 17)

    output_card(draw, (2110, 1425, 2880, 1745), "D_dense / F_geo", "LiDAR 对齐后的稠密结构", BLUE_L, BLUE)
    thumb(canvas, draw, pipeline / "06_06_fused_depth.png", (2140, 1495, 2405, 1700), "融合深度", BLUE)
    tag(draw, (2440, 1505, 2840, 1570), "公制尺度", "#E3ECFA", BLUE, 17)
    tag(draw, (2440, 1600, 2840, 1665), "置信度", "#E3ECFA", BLUE, 17)

    output_card(draw, (2970, 1425, 3750, 1745), "X_3D / P_t", "三维表面与目标表面", PURPLE_L, PURPLE)
    thumb(canvas, draw, pipeline / "08_08_3d_surface___se(3).png", (3000, 1495, 3265, 1700), "目标表面", PURPLE)
    tag(draw, (3300, 1505, 3710, 1570), "SE(3)", "#EEE7FB", PURPLE, 17)
    tag(draw, (3300, 1600, 3710, 1665), "Z-buffer", "#EEE7FB", PURPLE, 17)

    output_card(draw, (3840, 1425, 4620, 1745), "RGB warp / Gate", "颜色采样与视图门控", ORANGE_L, ORANGE)
    thumb(canvas, draw, pipeline / "09_09_rgb_warp.png", (3870, 1495, 4135, 1700), "RGB 重投影", ORANGE)
    tag(draw, (4170, 1505, 4580, 1570), "视图权重", "#FFF0DE", ORANGE, 17)
    tag(draw, (4170, 1600, 4580, 1665), "遮挡 mask", "#FFF0DE", ORANGE, 17)

    output_card(draw, (4710, 1425, 5690, 1745), "残差 / RGB_t", "Restormer 解码结果", ORANGE_L, ORANGE)
    thumb(canvas, draw, pipeline / "12_12_completion_output.png", (4740, 1495, 5005, 1700), "最终输出", ORANGE)
    tag(draw, (5040, 1505, 5650, 1570), "observed residual", "#FFF0DE", ORANGE, 17)
    tag(draw, (5040, 1600, 5650, 1665), "disocclusion completion", "#FFF0DE", ORANGE, 17)
    for x in [1635, 2495, 3355, 4215, 5200]:
        arrow(draw, (x, 1290), (x, 1425), "#8CA1B8", 6, 17)

    # Training heads inside the same model body.
    tx(draw, (1250, 1855), "端到端训练头（训练阶段）", 24, INK_D, True)
    tag(draw, (1250, 1915, 2120, 1985), "光度 / Charbonnier / SSIM", "#E9F0F7", "#45627F", 18)
    tag(draw, (2160, 1915, 3030, 1985), "深度 / LiDAR holdout", "#E3ECFA", BLUE, 18)
    tag(draw, (3070, 1915, 3940, 1985), "语义 / 边缘一致性", "#E6F4ED", GREEN, 18)
    tag(draw, (3980, 1915, 4850, 1985), "可见性 / 覆盖率", "#EEE7FB", PURPLE, 18)
    tag(draw, (4890, 1915, 5690, 1985), "disocclusion 感知损失", "#FFF0DE", ORANGE, 18)
    arrow(draw, (5200, 1745), (5200, 1915), ORANGE, 7, 20)

    # Outer input/output arrows make the I/O contract unmistakable.
    arrow(draw, (1130, 720), (1250, 720), TEAL_D, 12, 30)
    arrow(draw, (1130, 1000), (1280, 980), BLUE, 10, 28)
    arrow(draw, (1130, 1410), (3000, 640), PURPLE, 8, 25, True)
    arrow(draw, (1130, 1600), (3850, 640), ORANGE, 8, 25, True)

    # Right: one clearly defined final output.
    out = (5810, 220, 6530, 2250)
    panel(draw, out, "#FFF7EC", "#D6964C", 4, 28)
    tx(draw, (5850, 255), "输出", 31, ORANGE, True)
    tx(draw, (5850, 310), "目标视角 RGB_t", 20, MUTED)
    thumb(canvas, draw, pipeline / "12_12_completion_output.png", (5850, 405, 6490, 790), "最终重建图", ORANGE)
    tag(draw, (5850, 850, 6490, 915), "H × W 原分辨率", "#FFF0DE", ORANGE, 20)
    tag(draw, (5850, 1000, 6490, 1065), "无大块空洞", "#E6F4ED", GREEN, 20)
    tag(draw, (5850, 1110, 6490, 1175), "结构连续", "#E3ECFA", BLUE, 20)
    tag(draw, (5850, 1220, 6490, 1285), "颜色来自 RGB", "#DDF3F3", TEAL_D, 20)
    tag(draw, (5850, 1370, 6490, 1435), "观测区：受限残差", "#FFF0DE", ORANGE, 19)
    tag(draw, (5850, 1470, 6490, 1535), "新暴露区：生成补全", "#FFF0DE", ORANGE, 19)
    arrow(draw, (5690, 820), (5850, 590), ORANGE, 12, 30)

    # Bottom caption, not mixed into the model.
    draw.line((80, 2400, W - 80, 2400), fill=GRID, width=4)
    tx(draw, (80, 2450), "图注", 27, INK_D, True)
    tx(draw, (80, 2505), "完整端到端模型：输入经过多模态编码、三维场融合、目标视角变换、跨视图注意力和 Restormer 解码，输出同分辨率目标 RGB。", 21, MUTED)
    tx(draw, (80, 2555), "副产物是同一模型内部的中间张量：F_sem、D_dense、F_geo、X_3D、P_t、可见性 mask、RGB warp 和残差。", 21, MUTED)
    tx(draw, (80, 2605), "核心分工：RGB 提供颜色与语义，LiDAR/DA3 提供对齐后的结构，生成器只补真实没有源观测的新暴露区域。", 21, MUTED)
    tx(draw, (80, 3150), "GCR-NVS · 统一多模态三维新视角重建网络 · 论文图版式 v7", 20, MUTED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=96)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args)


if __name__ == "__main__":
    main()
