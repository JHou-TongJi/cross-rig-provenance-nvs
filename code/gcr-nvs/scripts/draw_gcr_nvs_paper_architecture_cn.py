"""Paper-style GCR-NVS architecture figure.

This intentionally follows the visual grammar of the dual-branch feature-field
figure: free-form network blocks, tensor stacks, 3-D feature fields, sparse
geometry and small real raster outputs.  PIL owns the complete composition.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from draw_gcr_nvs_network_blocks_v3 import (
    BLUE, BLUE_L, GREEN, GREEN_L, INK, MUTED, ORANGE, ORANGE_L,
    PURPLE, PURPLE_L, RED, RED_L, TEAL, WHITE,
    arrow, center, conv_stack, font, prism, thumb, transformer_stack,
)


W, H = 6200, 2850
BG = "#FFFFFF"
GRID = "#D4DCE7"
TEXT = "#172033"


def line(draw, points, color, width=7, dashed=False):
    if not dashed:
        draw.line(points, fill=color, width=width, joint="curve")
        return
    for a, b in zip(points[:-1], points[1:]):
        x0, y0 = a
        x1, y1 = b
        length = max(math.hypot(x1 - x0, y1 - y0), 1.0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pos = 0.0
        while pos < length:
            nxt = min(pos + 28, length)
            draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * nxt, y0 + uy * nxt), fill=color, width=width)
            pos += 50


def tag(draw, box, text, fill, color=WHITE, size=19):
    draw.rounded_rectangle(box, radius=18, fill=fill)
    center(draw, box, text, size, color, True)


def txt(draw, xy, value, size=20, color=TEXT, bold=False):
    draw.text(xy, value, font=font(size, bold), fill=color)


def cloud(draw, cx, cy, scale, color_set):
    """Small layered 3-D feature-field glyph made from ellipses and planes."""
    for j in range(3):
        draw.line((cx - 150 * scale, cy + 85 * scale + j * 18 * scale,
                   cx + 155 * scale, cy + 85 * scale + j * 18 * scale), fill="#D7DDE8", width=2)
    points = [(0, 0), (-70, 30), (68, 42), (-30, 78), (85, 92), (-95, 108),
              (12, 135), (120, 12), (-125, 58), (40, 170), (-48, 180), (150, 125)]
    for i, (dx, dy) in enumerate(points):
        r = 16 * scale if i % 3 else 22 * scale
        color = color_set[i % len(color_set)]
        draw.ellipse((cx + dx * scale - r, cy + dy * scale - r,
                       cx + dx * scale + r, cy + dy * scale + r), fill=color, outline=WHITE, width=2)


def draw(args):
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    root = Path("/media/2T_HD/GCR-NVS_DATA")
    source = root / "outputs/t0_video_reconstruction_test_frame73/mixed_10_20cm/CAM_FRONT_RIGHT"
    final_root = root / "outputs/evaluations/t0_uniview_rgbds_full_a_20260825/mixed_10_20cm/CAM_FRONT_RIGHT"
    diff_root = Path(args.diff_root or root / "outputs/source_vs_aonly_diff_20260826")
    paths = {
        "source": source / "source_native_rectified.png",
        "da3": root / "outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/depth_after_structure.png",
        "semantic": root / "outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/dino_anyup_semantic_pca.png",
        "t0": root / "t0_baseline_isolated_20260823/runs/t0_target_surface_2px_repair_all_20260824/mixed_10_20cm/CAM_FRONT_RIGHT/t0_highres_blue.png",
        "aonly": final_root / "rgbds_a_4k.png",
        "diff": diff_root / "mixed_10_20cm/CAM_FRONT_RIGHT/a_only_abs_diff.png",
        "final": root / "outputs/l4_simulated_frame73_submission_v4/distorted_final/target_rig/CAM_FRONT_RIGHT/final_distorted.png",
    }

    txt(draw, (85, 40), "GCR-NVS", 62, TEXT, True)
    txt(draw, (430, 64), "LiDAR 真实结构约束 + RGB 语义外观的新视角重建", 30, MUTED)
    tag(draw, (4760, 52, 6090, 108), "论文架构图 · 单帧前馈", "#E8EEF6", "#4B6583", 18)
    draw.line((85, 138, W - 85, 138), fill=GRID, width=3)

    # Left input stack.
    txt(draw, (80, 205), "输入", 27, TEXT, True)
    # Stacked RGB images.
    for i in range(4):
        x, y = 80 + i * 12, 310 - i * 12
        if paths["source"].exists():
            with Image.open(paths["source"]) as im:
                im = im.convert("RGB")
            im.thumbnail((300, 185), Image.Resampling.LANCZOS)
            canvas.paste(im, (x, y))
        draw.rectangle((x, y, x + 300, y + 185), outline="#7990A8", width=3)
    tag(draw, (105, 520, 380, 573), "I_rgb × 7", "#E6F5F5", TEAL, 20)
    # Calibration and LiDAR tensors.
    tag(draw, (460, 300, 840, 360), "K · D · T", "#E6F5F5", TEAL, 22)
    txt(draw, (470, 380), "相机拓扑 / 时间戳", 18, MUTED)
    draw.rectangle((80, 690, 840, 910), outline="#6E9FD7", width=3, fill=BLUE_L)
    txt(draw, (115, 720), "P_lidar = {x, y, z, t}", 25, BLUE, True)
    txt(draw, (115, 767), "当前帧 + 最近时序帧 · 动态 mask", 18, MUTED)
    conv_stack(draw, 120, 815, ["voxel", "注册", "时序", "mask"], (BLUE, "#6C9CDF", "#B8D1F0"), 115, 62, 15)
    tag(draw, (80, 1010, 840, 1070), "目标相机：Kₜ · Dₜ · Tₜ · Δξ ∈ SE(3)", "#F0EAFB", PURPLE, 18)

    # DINO upper branch.
    txt(draw, (1060, 205), "语义外观分支", 27, GREEN, True)
    if paths["semantic"].exists():
        thumb(canvas, draw, paths["semantic"], (1030, 320, 1360, 510), "DINOv2 + AnyUP", GREEN)
    tag(draw, (1410, 300, 1890, 358), "patch tokens", "#E5F4EC", GREEN, 19)
    transformer_stack(draw, 1450, 410, 6, 58, 115, (GREEN, "#61B48E", "#BEE5D1"), ["Patch", "Block", "Block", "Block", "Block", "Norm"])
    txt(draw, (1430, 600), "DINOv2 ViT-L/14", 23, GREEN, True)
    txt(draw, (1430, 638), "AnyUP → F_sem (H × W)", 18, MUTED)
    conv_stack(draw, 1060, 750, ["Q", "K/V", "像素查询", "FPN", "F64"], (GREEN, "#61B48E", "#BEE5D1"), 132, 82, 16)
    txt(draw, (1080, 860), "语义边界与高分辨率对应特征", 18, MUTED)

    # DA3 + T0 lower branch.
    txt(draw, (1060, 1110), "稠密几何分支", 27, BLUE, True)
    if paths["da3"].exists():
        thumb(canvas, draw, paths["da3"], (1030, 1230, 1360, 1420), "DA3 稠密深度", BLUE)
    transformer_stack(draw, 1420, 1210, 4, 65, 115, (BLUE, "#6C9CDF", "#B8D1F0"), ["ViT-L", "DPT", "Align", "D_m"])
    txt(draw, (1420, 1410), "RGB 对齐的连续深度候选", 18, MUTED)
    draw.rectangle((1030, 1550, 1890, 1760), outline="#6E9FD7", width=3, fill=BLUE_L)
    txt(draw, (1060, 1582), "T0StructureConstraintNet", 24, PURPLE, True)
    conv_stack(draw, 1080, 1640, ["LiDAR", "Fuse", "Transformer", "D_surface"], (PURPLE, "#A78DD4", "#DDD3F1"), 150, 75, 18)
    txt(draw, (1060, 1740), "LiDAR 只校准结构，不直接覆盖 RGB", 18, MUTED)

    # Center 3-D feature field and transform.
    txt(draw, (2180, 205), "统一三维特征场", 29, PURPLE, True)
    cloud(draw, 2460, 360, 1.0, ["#6A9BCE", "#7BC2A0", "#D5A35A", "#A37BC5", "#E887A7"])
    txt(draw, (2280, 620), "D_surface + F_sem + RGB", 18, MUTED)
    tag(draw, (2200, 700, 2730, 758), "稠密表面 / 语义特征 / provenance", "#F0EAFB", PURPLE, 17)
    # 3-D transformation blocks.
    conv_stack(draw, 2190, 855, ["K⁻¹", "X₃ᴅ", "SE(3)", "Kₜ", "z-buffer"], (PURPLE, "#A78DD4", "#DDD3F1"), 110, 76, 13)
    cloud(draw, 2470, 1130, 0.86, ["#6A9BCE", "#7BC2A0", "#D5A35A", "#A37BC5"])
    txt(draw, (2260, 1375), "目标视角稠密表面 Pₜ", 19, PURPLE, True)
    tag(draw, (2180, 1460, 2760, 1518), "observed · recoverable · true hole", "#F0EAFB", PURPLE, 17)

    # Right appearance / output side.
    txt(draw, (3070, 205), "外观重建分支", 29, ORANGE, True)
    conv_stack(draw, 3080, 330, ["grid", "RGB", "F64", "融合"], (ORANGE, "#D7964D", "#F4D2A7"), 140, 88, 18)
    txt(draw, (3090, 470), "provenance-guided RGB sampler", 19, ORANGE, True)
    if paths["t0"].exists():
        thumb(canvas, draw, paths["t0"], (3500, 285, 4000, 535), "T0 蓝色审计", RED)
    if paths["aonly"].exists():
        thumb(canvas, draw, paths["aonly"], (4200, 285, 4700, 535), "A-ONLY 最终 RGB", ORANGE)
    arrow(draw, (4000, 410), (4200, 410), ORANGE, 7, 20)
    # Gate and restricted decoder blocks.
    tag(draw, (3070, 650, 3560, 710), "UniWorld / DA3 gate", "#E5F4EC", GREEN, 18)
    conv_stack(draw, 3090, 760, ["Q_geo", "K_rgb", "MHA", "Gate"], (GREEN, "#61B48E", "#BEE5D1"), 125, 78, 15)
    tag(draw, (3650, 650, 4130, 710), "A-ONLY RGB-D-S", "#FFF0DF", ORANGE, 18)
    conv_stack(draw, 3670, 760, ["Obs", "Gate", "Hole", "RGB_t"], (ORANGE, "#D7964D", "#F4D2A7"), 125, 78, 15)
    txt(draw, (3090, 900), "observed / gate 像素 immutable", 18, RED, True)
    txt(draw, (3670, 900), "只生成 residual true hole", 18, ORANGE, True)
    # Final output / diff.
    if paths["final"].exists():
        thumb(canvas, draw, paths["final"], (4900, 285, 5450, 535), "最终 raw K/D 回畸变", ORANGE)
    if paths["diff"].exists():
        thumb(canvas, draw, paths["diff"], (5550, 285, 6100, 535), "a_only_abs_diff", RED)
    arrow(draw, (4700, 410), (4900, 410), ORANGE, 7, 20)
    arrow(draw, (5450, 410), (5550, 410), RED, 5, 17, dashed=True)
    txt(draw, (5590, 565), "审计旁路", 17, RED, True)

    # Main arrows and training-only path.
    arrow(draw, (840, 450), (1030, 450), TEAL, 8, 22)
    arrow(draw, (840, 860), (1030, 1320), BLUE, 8, 22)
    arrow(draw, (1890, 470), (2220, 470), GREEN, 8, 22)
    arrow(draw, (1890, 1680), (2210, 1050), PURPLE, 8, 22)
    arrow(draw, (2740, 1050), (3080, 370), ORANGE, 8, 22)
    line(draw, [(840, 1040), (920, 1040), (920, 1850), (1890, 1850), (2210, 1170)], BLUE, 5, True)
    tag(draw, (965, 1790, 1770, 1845), "训练期：LiDAR 结构 teacher / holdout", "#E8F0FA", BLUE, 17)

    # Bottom legend and equations.
    draw.line((80, 1940, W - 80, 1940), fill=GRID, width=3)
    txt(draw, (90, 1980), "模型契约", 25, TEXT, True)
    tag(draw, (260, 1972, 780, 2028), "geometry", BLUE_L, BLUE, 18)
    tag(draw, (820, 1972, 1340, 2028), "appearance", GREEN_L, GREEN, 18)
    tag(draw, (1380, 1972, 1900, 2028), "render", ORANGE_L, ORANGE, 18)
    tag(draw, (1940, 1972, 2460, 2028), "restore", PURPLE_L, PURPLE, 18)
    txt(draw, (90, 2080), "X = Z K⁻¹ [u,v,1]ᵀ     Xₜ = Tₜ Xₛ     pₜ ~ Kₜ Xₜ     Zₜ(u,v) = minᵢ Zᵢ(u,v)", 22, TEXT, True)
    txt(draw, (90, 2135), "LiDAR：真实公制结构与遮挡约束；DA3：RGB 对齐稠密候选；DINOv2 + AnyUP：语义和边界；RGB：最终颜色。", 20, MUTED)
    txt(draw, (90, 2185), "颜色不由稀疏 LiDAR 直接产生，observed 像素不被生成器重绘；A-ONLY 只处理未观测 residual hole。", 20, MUTED)

    # Bottom cards are explanations, not another flowchart.
    cards = [(90, 2270, 1900, 2750, BLUE_L, BLUE, "结构"), (1980, 2270, 3830, 2750, GREEN_L, GREEN, "语义与外观"), (3910, 2270, 6110, 2750, RED_L, RED, "验收与未来")]
    for x0, y0, x1, y1, fill, accent, title in cards:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=fill, outline=accent, width=3)
        txt(draw, (x0 + 28, y0 + 25), title, 24, accent, True)
    txt(draw, (130, 2350), "当前 LiDAR + 最近帧注册补偿", 19, TEXT)
    txt(draw, (130, 2395), "T0StructureConstraintNet 校准 DA3", 19, TEXT)
    txt(draw, (130, 2440), "稠密表面 → SE(3) → z-buffer", 19, TEXT)
    txt(draw, (130, 2485), "LiDAR 不直接覆盖 RGB", 19, BLUE, True)
    txt(draw, (2020, 2350), "DINOv2-L / AnyUP：语义与边界", 19, TEXT)
    txt(draw, (2020, 2395), "provenance RGB sampler：真实颜色", 19, TEXT)
    txt(draw, (2020, 2440), "拓扑 + 深度门控 + 同名相机优先", 19, TEXT)
    txt(draw, (2020, 2485), "A-ONLY：只生成 true hole", 19, GREEN, True)
    txt(draw, (3950, 2350), "完成率 98.24%｜锁定像素 0 px", 19, TEXT)
    txt(draw, (3950, 2395), "主 DIFF：a_only_abs_diff.png", 19, RED, True)
    txt(draw, (3950, 2440), "几何误差 / 语义保留 / 时序异常：按真值补测", 19, TEXT)
    txt(draw, (3950, 2485), "TODO：Wan2.1/VACE 仅接连续帧 residual hole", 19, RED, True)
    txt(draw, (90, 2810), "GCR-NVS · 当前定档：T0 + UniWorld/DA3 Gate + A-ONLY · 真实副产物来自 2026-05-22-10-25-21 / frame 73 / mixed_10_20cm", 18, MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=96)
    print(f"saved {args.output} ({W}x{H})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_figures/gcr_nvs_final_architecture_cn_20260826.png"))
    parser.add_argument("--diff-root", type=Path, default=None)
    args = parser.parse_args()
    draw(args)


if __name__ == "__main__":
    main()
