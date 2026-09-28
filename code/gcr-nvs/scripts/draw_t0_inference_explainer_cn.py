"""Draw a Chinese, paper-style explanation of one T0+ inference trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H = 5200, 3600
BG = "#F5F8FC"
INK = "#172338"
MUTED = "#536579"
GRID = "#CBD7E4"
BLUE = "#3977B8"
BLUE_L = "#E4F0FC"
GREEN = "#2E9271"
GREEN_L = "#E3F4EC"
PURPLE = "#7655AC"
PURPLE_L = "#F0E9FB"
ORANGE = "#C56D20"
ORANGE_L = "#FFF0DE"
TEAL = "#238B96"
TEAL_L = "#DDF3F3"
WHITE = "#FFFFFF"

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def ft(size: int, bold: bool = False):
    path = FONT_BOLD if bold else FONT
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def text(draw, xy, value, size=24, fill=INK, bold=False, anchor=None):
    draw.text(xy, value, font=ft(size, bold), fill=fill, anchor=anchor)


def rounded(draw, box, fill=WHITE, outline=GRID, width=3, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, a, b, color=INK, width=8):
    draw.line((*a, *b), fill=color, width=width)
    x0, y0 = a; x1, y1 = b
    length = max(((x1-x0)**2 + (y1-y0)**2) ** 0.5, 1)
    ux, uy = (x1-x0)/length, (y1-y0)/length
    px, py = -uy, ux
    tip = (x1, y1)
    left = (x1 - ux*25 + px*13, y1 - uy*25 + py*13)
    right = (x1 - ux*25 - px*13, y1 - uy*25 - py*13)
    draw.polygon([tip, left, right], fill=color)


def fit(path: Path, size=(300, 170)):
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#E9EEF4")
    canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return canvas


def rgb_heat(path: Path, size=(300, 170)):
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3:
        array = array[..., 0]
    array = np.asarray(array, np.float32)
    unique = np.unique(array[np.isfinite(array)])
    if len(unique) <= 2 and np.all(np.isin(unique, [0.0, 1.0])):
        image = Image.fromarray(np.repeat((array > 0.5).astype(np.uint8)[..., None], 3, axis=2) * 255)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "#E9EEF4")
        canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
        return canvas
    finite = np.isfinite(array)
    lo, hi = np.percentile(array[finite], [2, 98]) if finite.any() else (0, 1)
    scaled = np.clip((array-lo)/max(hi-lo, 1e-6), 0, 1)
    scaled[~finite] = 0
    image = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(np.uint8(scaled*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#E9EEF4")
    canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return canvas


def stage(draw, box, number, title, lines, accent, pale):
    rounded(draw, box, WHITE, accent, 4, 24)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y0+80), radius=24, fill=pale)
    draw.rectangle((x0, y0+52, x1, y0+80), fill=pale)
    draw.ellipse((x0+24, y0+20, x0+68, y0+64), fill=accent)
    text(draw, ((x0+46), y0+42), str(number), 21, WHITE, True, "mm")
    text(draw, (x0+85, y0+22), title, 28, accent, True)
    y = y0 + 112
    for line in lines:
        text(draw, (x0+28, y), line, 21, MUTED)
        y += 42


def card(canvas, draw, box, image_path, title, description, accent, heat=False):
    x0, y0, x1, y1 = box
    rounded(draw, box, WHITE, GRID, 2, 16)
    if image_path.exists():
        image = rgb_heat(image_path.with_suffix(".npy"), (330, 180)) if heat else fit(image_path, (330, 180))
        canvas.paste(image, (x0+20, y0+20))
    text(draw, (x0+375, y0+22), title, 22, accent, True)
    # Chinese descriptions are intentionally short so the figure remains readable.
    text(draw, (x0+375, y0+70), description, 18, MUTED)


def main():
    bundle = Path(__file__).resolve().parents[1]
    trace = bundle / "runs/t0_inference_trace_v2"
    output = bundle / "artifacts/paper_figures/t0_inference_explainer_cn.png"
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Header.
    text(draw, (90, 55), "GCR-NVS T0+ 单张图推理全链路", 58, INK, True)
    text(draw, (92, 135), "LiDAR 学公制结构，DA3 提供稠密候选，DINOv2 + AnyUP 保留语义，源 RGB 完成颜色采样", 25, MUTED)
    text(draw, (W-90, 72), "样本：CAM_FRONT_NARROW · 混合 10–20 cm", 21, TEAL, True, "ra")
    draw.line((90, 195, W-90, 195), fill=GRID, width=4)

    # Main route: 6 stages.
    text(draw, (90, 235), "一、完整技术路线：一张输入图像如何变成目标视角 RGB", 31, INK, True)
    boxes = [(90, 320, 820, 800), (930, 320, 1660, 800), (1770, 320, 2500, 800), (2610, 320, 3340, 800), (3450, 320, 4180, 800), (4290, 320, 5110, 800)]
    stage(draw, boxes[0], 1, "输入数据", ["七路去畸变 RGB", "当前 / 时序 LiDAR 点云", "目标相机 K、T、位移参数"], TEAL, TEAL_L)
    stage(draw, boxes[1], 2, "统一预处理", ["RGB、深度、点云统一 rectified 坐标", "动态掩码与时间对齐", "输出：同一像素网格"], BLUE, BLUE_L)
    stage(draw, boxes[2], 3, "多模态特征", ["DA3：稠密 RGB 对齐深度", "DINOv2 + AnyUP：语义 / 边界", "LiDAR：真实公制结构条件"], GREEN, GREEN_L)
    stage(draw, boxes[3], 4, "结构约束网络", ["CNN/U-Net 局部边缘", "Transformer 长距离连续表面", "输出 range 残差、gate、置信度"], PURPLE, PURPLE_L)
    stage(draw, boxes[4], 5, "目标视角渲染", ["连续表面反投影到 3D", "输入目标 SE(3)", "z-buffer 近表面与可见性"], ORANGE, ORANGE_L)
    stage(draw, boxes[5], 6, "RGB 输出", ["只从源 RGB 采样颜色", "T0 / T0+structure 对比", "输出同分辨率 RGB、mask、来源"], TEAL, TEAL_L)
    for i in range(5):
        arrow(draw, (boxes[i][2]+15, 560), (boxes[i+1][0]-15, 560), [TEAL, BLUE, GREEN, PURPLE, ORANGE][i], 9)

    # Responsibility strip.
    rounded(draw, (90, 865, 5110, 1080), WHITE, GRID, 3, 20)
    text(draw, (125, 900), "模块分工（避免概念混淆）", 25, INK, True)
    text(draw, (125, 960), "RGB", 23, TEAL, True); text(draw, (255, 960), "颜色、纹理、外观语义", 21, MUTED)
    text(draw, (1200, 960), "DA3", 23, GREEN, True); text(draw, (1325, 960), "稠密但不作为真实公制真值", 21, MUTED)
    text(draw, (2600, 960), "LiDAR", 23, BLUE, True); text(draw, (2745, 960), "真实尺度、表面结构、可见性约束", 21, MUTED)
    text(draw, (4100, 960), "T0 渲染", 23, ORANGE, True); text(draw, (4265, 960), "SE(3) + z-buffer + 源 RGB", 21, MUTED)

    # Actual tensor trace.
    text(draw, (90, 1160), "二、这张图实际产生的推理副产物（同一模型内部，不是额外模型）", 31, INK, True)
    text(draw, (90, 1210), "左到右是数据流；每个框下方同时保存了 PNG 可视化和 NPY 原始张量。", 21, MUTED)
    entries = [
        ("01_raw_rgb.png", "① 原始 RGB", "相机直接采集的输入", TEAL, False),
        ("02_rectified_rgb.png", "② 去畸变 RGB", "后续统一像素坐标", TEAL, False),
        ("03_da3_depth_z.png", "③ DA3 稠密深度", "RGB 对齐的候选 Z", GREEN, True),
        ("04_da3_confidence.png", "④ DA3 置信度", "深度可信程度", GREEN, True),
        ("05_current_lidar_range.png", "⑤ 当前 LiDAR", "真实公制稀疏结构", BLUE, True),
        ("07_temporal_lidar_range.png", "⑥ 时序 LiDAR", "邻帧补足结构支持", BLUE, True),
        ("09_lidar_support_distance_px.png", "⑦ 支持距离", "离真实点越远越保守", BLUE, True),
        ("10_rgb_edges.png", "⑧ RGB 边缘", "阻止深度跨物体传播", TEAL, False),
        ("11_dinov2_anyup_pca.png", "⑨ 语义特征", "DINOv2 + AnyUP 边界语义", GREEN, False),
        ("12_corrected_range_m.png", "⑩ 校正后 range", "结构网络的连续距离", PURPLE, True),
        ("13_residual_log_range.png", "⑪ 结构残差", "相对 DA3 的有界改动", PURPLE, True),
        ("14_correction_gate.png", "⑫ 校正 gate", "LiDAR 支持下才放大", PURPLE, True),
        ("15_geometry_confidence.png", "⑬ 几何置信度", "校正结果的可信度", PURPLE, True),
        ("16_boundary_probability.png", "⑭ 边界概率", "表面边界位置", PURPLE, True),
        ("17_corrected_depth_z.png", "⑮ 校正后 Z", "送入投影器的深度", ORANGE, True),
        ("18_t0_original_rgb.png", "⑯ 原始 T0", "未使用结构网络", ORANGE, False),
        ("19_t0_structure_rgb.png", "⑰ T0+结构", "使用校正后的表面", ORANGE, False),
        ("20_t0_original_validity.png", "⑱ T0 有效性", "原始可见像素", TEAL, True),
        ("21_t0_structure_validity.png", "⑲ T0+有效性", "结构网络后的覆盖", TEAL, True),
        ("22_t0_structure_provenance.png", "⑳ 来源 provenance", "每个像素来自哪个源相机", ORANGE, True),
    ]
    left, top = 90, 1270
    card_w, card_h = 1235, 270
    for index, (filename, title, desc, accent, heat) in enumerate(entries):
        col, row = index % 4, index // 4
        x = left + col * 1270
        y = top + row * 290
        card(canvas, draw, (x, y, x+card_w, y+card_h), trace / filename, title, desc, accent, heat)

    # Bottom interpretation.
    y0 = top + 5 * 290 + 30
    rounded(draw, (90, y0, 5110, y0+250), WHITE, GRID, 3, 20)
    text(draw, (125, y0+28), "三、如何读这张图", 25, INK, True)
    text(draw, (125, y0+82), "① 看 05/06/07：真实 LiDAR 只提供稀疏结构，不能直接生成 RGB。", 21, BLUE)
    text(draw, (125, y0+125), "② 看 11–15：结构网络只校正 DA3 表面，不输出颜色；gate 让远离 LiDAR 的区域接近 DA3 identity。", 21, PURPLE)
    text(draw, (125, y0+168), "③ 看 18/19：T0 与 T0+ 的 RGB 都来自源图 z-buffer 采样，差异来自结构投影，而不是颜色生成。", 21, ORANGE)
    text(draw, (125, y0+211), "④ 看 20/21/22：有效性与来源单独记录，避免把黑洞或错误覆盖误认为真实重建。", 21, TEAL)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=96)
    print(output)


if __name__ == "__main__":
    main()
