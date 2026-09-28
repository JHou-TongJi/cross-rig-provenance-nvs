"""Unified model-style (not process-flow) architecture diagram for GCR-NVS."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from draw_gcr_nvs_network_blocks_v3 import (
    BLUE,
    GREEN,
    ORANGE,
    PURPLE,
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


W, H = 6400, 3600
BG = "#F3F6FA"
NAVY = "#10243B"
NAVY_2 = "#193A59"
LINE = "#9BB0C8"
TEXT = "#10243B"
MUTED = "#61758B"
WHITE_2 = "#EAF2FA"
CYAN = "#5FD0D3"
LILAC = "#BDA6F2"
AMBER = "#F1B26B"


def panel(draw, box, fill, outline, width=3, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def tag(draw, box, text, fill, color=WHITE, size=20):
    draw.rounded_rectangle(box, radius=18, fill=fill)
    center(draw, box, text, size, color, True)


def txt(draw, xy, text, size=20, color=TEXT, bold=False):
    draw.text(xy, text, font=font(size, bold), fill=color)


def arrow2(draw, a, b, color, width=8, dashed=False):
    arrow(draw, a, b, color, width, 22, dashed)


def model_stage(draw, box, title, subtitle, accent):
    x0, y0, x1, y1 = box
    panel(draw, box, NAVY_2, (46, 85, 120), 3, 26)
    draw.rectangle((x0, y0, x1, y0 + 62), fill=accent)
    txt(draw, (x0 + 22, y0 + 15), title, 25, NAVY, True)
    txt(draw, (x0 + 22, y0 + 75), subtitle, 17, WHITE_2)


def draw(args):
    pipeline = args.pipeline.resolve()
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Header: the whole diagram is explicitly a single unified model.
    txt(draw, (90, 44), "GCR-NVS", 66, TEXT, True)
    txt(draw, (430, 67), "统一多模态三维新视角重建网络", 32, MUTED)
    tag(draw, (5020, 52, 6280, 108), "Unified RGB–Depth–LiDAR Network", "#DCE8F5", "#385A7B", 19)
    draw.line((90, 145, W - 90, 145), fill="#D1DCE8", width=4)

    # Input tensors on the left.
    txt(draw, (90, 205), "输入张量", 28, TEXT, True)
    tag(draw, (90, 255, 1120, 305), "I_rgb ∈ R^{7×3×H×W}   ·   K,D,T   ·   时间戳", "#DDF3F3", "#237F82", 20)
    thumb(canvas, draw, pipeline / "02_02_rectified_rgb.png", (90, 345, 480, 590), "去畸变 RGB", TEAL)
    thumb(canvas, draw, pipeline / "04_04_da3_depth.png", (520, 345, 910, 590), "DA3 深度候选", BLUE)
    panel(draw, (90, 640, 910, 855), "#E8F0FA", "#7DA6D4", 3, 20)
    txt(draw, (122, 670), "P_lidar = {x,y,z,t}", 25, BLUE, True)
    txt(draw, (122, 718), "时序点云 · 体素坐标 · 动态掩码", 18, MUTED)
    conv_stack(draw, 125, 765, ["voxel", "time", "mask"], (BLUE, "#8DB3E9", "#C7DBF8"), 115, 60, 16)
    tag(draw, (90, 930, 910, 985), "目标相机：K_t,T_t, Δξ ∈ SE(3)", "#EEE7FB", "#6B4AA3", 20)

    # One contiguous model body.
    body = (1180, 205, 5670, 2460)
    panel(draw, body, NAVY, "#284766", 5, 34)
    txt(draw, (1240, 238), "GCR-NVS 统一模型主体", 34, WHITE, True)
    txt(draw, (1240, 284), "RGB 外观主导 · LiDAR 结构约束 · 深度对齐 · 受限生成补全", 18, WHITE_2)

    # Stage 1: encoders, all inside the same body.
    model_stage(draw, (1240, 360, 2310, 1470), "① 多模态编码器", "共享目标帧坐标系", CYAN)
    txt(draw, (1275, 465), "外观编码  E_rgb", 21, WHITE_2, True)
    transformer_stack(draw, 1290, 520, 6, 52, 110, (GREEN, "#65B795", "#B8E4CD"), ["Patch", "Block", "Block", "Block", "Block", "Norm"])
    tag(draw, (1290, 680, 2240, 735), "DINOv2 ViT-L/14 → F_sem", "#235C50", "#B8E4CD", 18)
    txt(draw, (1275, 800), "深度编码  E_depth", 21, WHITE_2, True)
    transformer_stack(draw, 1290, 850, 4, 52, 92, (BLUE, "#4B83C9", "#A9C8EC"), ["DA3", "DPT", "Align", "D_dense"])
    txt(draw, (1275, 1100), "结构编码  E_lidar", 21, WHITE_2, True)
    conv_stack(draw, 1290, 1150, ["Sparse", "32", "64", "128"], (BLUE, "#4B83C9", "#A9C8EC"), 145, 88, 18)
    tag(draw, (1290, 1300, 2240, 1355), "稀疏体素 + 时序补全 → F_geo", "#244C73", "#A9C8EC", 18)

    # Stage 2: metric 3D fusion / surface representation.
    model_stage(draw, (2390, 360, 3470, 1470), "② 三维场融合", "从深度到可变形表面", LILAC)
    txt(draw, (2425, 465), "特征对齐与融合", 21, WHITE_2, True)
    conv_stack(draw, 2440, 525, ["F_sem", "F_geo", "F_rgb", "Fuse"], (PURPLE, "#805FC0", "#C9BAEC"), 165, 94, 16)
    txt(draw, (2425, 735), "稠密表面模块", 21, WHITE_2, True)
    prism(draw, 2470, 805, 135, 220, 42, -26, "#6E53AF", "#4E3D87", "#B6A4E4", "D_dense", WHITE, 17)
    prism(draw, 2680, 770, 135, 255, 42, -26, "#6E53AF", "#4E3D87", "#B6A4E4", "X_3D", WHITE, 17)
    prism(draw, 2890, 735, 135, 290, 42, -26, "#6E53AF", "#4E3D87", "#B6A4E4", "Surf", WHITE, 17)
    arrow2(draw, (2605, 570), (2605, 790), LILAC, 7)
    tag(draw, (2430, 1120, 3400, 1175), "K⁻¹ · 反投影 · SparseConv U-Net", "#3A2D67", "#D7C9F4", 18)
    txt(draw, (2425, 1240), "尺度与置信度", 21, WHITE_2, True)
    conv_stack(draw, 2440, 1290, ["LiDAR holdout", "confidence", "D_dense"], (TEAL, "#309EA1", "#9DE2E4"), 210, 88, 18)

    # Stage 3: target camera transform and cross-view aggregation.
    model_stage(draw, (3550, 360, 4625, 1470), "③ 视角变换与注意力", "结构先变换，颜色后采样", AMBER)
    txt(draw, (3585, 465), "目标射线与 SE(3)", 21, WHITE_2, True)
    conv_stack(draw, 3600, 525, ["K_t", "R", "t", "ray"], (PURPLE, "#805FC0", "#C9BAEC"), 160, 88, 16)
    txt(draw, (3585, 735), "可见性与表面选择", 21, WHITE_2, True)
    conv_stack(draw, 3600, 795, ["Z-buffer", "near", "mask"], (PURPLE, "#805FC0", "#C9BAEC"), 180, 88, 16)
    txt(draw, (3585, 1000), "跨视图多头注意力", 21, WHITE_2, True)
    transformer_stack(draw, 3610, 1065, 4, 62, 115, (ORANGE, "#C67A2D", "#F0C287"), ["Q_geo", "K_rgb", "V_sem", "MHA"])
    tag(draw, (3595, 1300, 4580, 1355), "拓扑先验 · 深度门控 · 同名相机优先", "#5E431F", "#F4C98D", 18)

    # Stage 4: unified decoder with skip connections.
    model_stage(draw, (4705, 360, 5630, 2250), "④ 受限解码器", "Restormer + completion head", ORANGE)
    txt(draw, (4740, 465), "多尺度编码", 21, WHITE_2, True)
    prism(draw, 4760, 540, 115, 130, 32, -22, "#C66B1C", "#8B4816", "#F1B26B", "48", WHITE, 20)
    prism(draw, 4925, 575, 115, 175, 32, -22, "#C66B1C", "#8B4816", "#F1B26B", "96", WHITE, 20)
    prism(draw, 5090, 610, 115, 220, 32, -22, "#C66B1C", "#8B4816", "#F1B26B", "192", WHITE, 20)
    txt(draw, (4740, 900), "Restormer 瓶颈", 21, WHITE_2, True)
    transformer_stack(draw, 4760, 970, 4, 70, 120, (ORANGE, "#C66B1C", "#F1B26B"), ["MDTA", "GDFN", "MDTA", "GDFN"])
    txt(draw, (4740, 1250), "受限残差 + 生成补全", 21, WHITE_2, True)
    conv_stack(draw, 4760, 1320, ["Obs", "Unc", "Disocc", "RGB_t"], (ORANGE, "#C66B1C", "#F1B26B"), 145, 100, 14)
    tag(draw, (4750, 1580, 5585, 1635), "观测区小改 · 空洞区生成", "#5E431F", "#F4C98D", 19)
    # Decoder skip path, visibly inside the model body.
    arrow2(draw, (3610, 1170), (4760, 1370), AMBER, 8, True)
    arrow2(draw, (2240, 700), (4760, 1380), CYAN, 7, True)
    arrow2(draw, (3400, 1360), (4760, 1390), LILAC, 7, True)

    # Inter-stage tensor arrows, all inside the single model panel.
    arrow2(draw, (2310, 910), (2390, 910), CYAN, 10)
    arrow2(draw, (3470, 910), (3550, 910), LILAC, 10)
    arrow2(draw, (4625, 910), (4705, 910), AMBER, 10)

    # Latent workspace: this is the part that turns the four stages into one
    # network rather than four independent process boxes.
    txt(draw, (1275, 1665), "统一潜空间 / Token Mixer", 23, WHITE_2, True)
    prism(draw, 1320, 1745, 180, 120, 38, -24, "#2E7E82", "#205C60", "#79D8D7", "F_rgb", WHITE, 18)
    prism(draw, 1575, 1718, 180, 147, 38, -24, "#3E70B5", "#2A4E82", "#9DC1EC", "F_sem", WHITE, 18)
    prism(draw, 1830, 1690, 180, 174, 38, -24, "#6E53AF", "#4E3D87", "#B6A4E4", "F_geo", WHITE, 18)
    prism(draw, 2085, 1662, 180, 202, 38, -24, "#6E53AF", "#4E3D87", "#B6A4E4", "D_dense", WHITE, 18)
    arrow2(draw, (1500, 1805), (1575, 1790), CYAN, 6)
    arrow2(draw, (1755, 1790), (1830, 1765), LILAC, 6)
    arrow2(draw, (2010, 1765), (2085, 1740), LILAC, 6)
    transformer_stack(draw, 2410, 1690, 5, 76, 160, (ORANGE, "#C66B1C", "#F1B26B"), ["Cross", "Token", "MHA", "Gate", "Norm"])
    arrow2(draw, (2265, 1740), (2410, 1760), LILAC, 8)
    conv_stack(draw, 2930, 1775, ["view", "surface", "render"], (ORANGE, "#C66B1C", "#F1B26B"), 155, 100, 16)
    arrow2(draw, (2780, 1770), (2930, 1825), AMBER, 8)
    tag(draw, (1320, 2055, 3650, 2120), "几何 token 与外观 token 在同一个 latent field 中交互", "#24496B", "#D6E8F8", 19)
    arrow2(draw, (3220, 1875), (4760, 1440), AMBER, 8, True)
    arrow2(draw, (2600, 1850), (4760, 1390), AMBER, 8, True)
    tag(draw, (1310, 2265, 5520, 2330), "统一主干：F_rgb + F_sem + F_geo + D_dense + (K_t,T_t)  →  RGB_t", "#24496B", "#D6E8F8", 23)

    # Output panel, outside the model body.
    txt(draw, (5770, 205), "输出", 28, TEXT, True)
    thumb(canvas, draw, pipeline / "12_12_completion_output.png", (5770, 345, 6300, 660), "目标视图 RGB_t", ORANGE)
    tag(draw, (5770, 735, 6300, 790), "H × W 原分辨率", "#FFF0DE", "#B86D20", 20)
    arrow2(draw, (5630, 1390), (5770, 500), ORANGE, 10)

    # Bottom explanation, intentionally separated from the architecture itself.
    draw.line((90, 2640, W - 90, 2640), fill="#D1DCE8", width=4)
    txt(draw, (90, 2690), "图注", 28, TEXT, True)
    txt(draw, (90, 2750), "RGB 负责颜色、纹理与语义；LiDAR 只提供公制结构、遮挡和可见性约束；DA3 提供连续深度候选并通过 LiDAR 对齐。", 22, MUTED)
    txt(draw, (90, 2800), "模型主体内部完成三维表面建模、SE(3) 视角变换、跨视图注意力和 Restormer 解码；生成分支只处理真实新暴露区域。", 22, MUTED)
    txt(draw, (90, 2850), "验收：严格 LOO、±10cm 小扰动、逐相机覆盖率 / PSNR / SSIM，并单独检查可观测区与 disocclusion 区。", 22, MUTED)
    txt(draw, (90, 3350), "GCR-NVS · 统一多模态三维新视角重建网络 · 当前试验：7 路 DA3、AnyUP、同名相机优先", 20, MUTED)

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
