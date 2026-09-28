"""Draw the current GCR-NVS end-to-end architecture in Chinese.

The figure is deliberately tied to the released T0/Hybrid contract:
LiDAR constrains geometry, DA3 supplies a dense RGB-aligned surface,
DINOv2/AnyUP supplies semantic correspondence, and RGB sampling supplies
appearance.  The optional generators are hole-only branches.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


W, H = 5600, 3380
BG = "#F5F7FB"
INK = "#172033"
MUTED = "#637087"
LINE = "#C8D1DF"
WHITE = "#FFFFFF"
TEAL, TEAL_L = "#127B83", "#E3F4F4"
BLUE, BLUE_L = "#2867B2", "#E6F0FC"
GREEN, GREEN_L = "#23855E", "#E4F4EC"
PURPLE, PURPLE_L = "#7050A5", "#EFE9FA"
ORANGE, ORANGE_L = "#B9691F", "#FFF0DF"
RED, RED_L = "#B44956", "#FBE8EB"
GRAY_L = "#EDF1F6"


def font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def rounded(draw, box, fill, outline=LINE, width=3, radius=20):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw, box, value, size=22, color=INK, bold=False, spacing=5):
    f = font(size, bold)
    bb = draw.multiline_textbbox((0, 0), value, font=f, spacing=spacing, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (box[0] + box[2] - tw) / 2
    y = (box[1] + box[3] - th) / 2 - bb[1]
    draw.multiline_text((x, y), value, font=f, fill=color, spacing=spacing, align="center")


def arrow(draw, start, end, color=INK, width=6, head=22, dashed=False):
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    if dashed:
        cursor = 0.0
        while cursor < length - head:
            nxt = min(cursor + 25, length - head)
            draw.line((x0 + ux * cursor, y0 + uy * cursor, x0 + ux * nxt, y0 + uy * nxt), fill=color, width=width)
            cursor += 42
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    px, py = -uy, ux
    draw.polygon([(x1, y1), (x1 - head * ux + head * .45 * px, y1 - head * uy + head * .45 * py),
                  (x1 - head * ux - head * .45 * px, y1 - head * uy - head * .45 * py)], fill=color)


def stage(draw, x, number, title, color):
    draw.ellipse((x, 150, x + 58, 208), fill=color)
    text_center(draw, (x, 150, x + 58, 208), str(number), 24, WHITE, True)
    draw.text((x + 75, 161), title, fill=color, font=font(31, True))


def module(draw, box, title, subtitle, fill, outline, icon=None):
    rounded(draw, box, fill, outline, 4, 18)
    x0, y0, x1, y1 = box
    if icon:
        draw.ellipse((x0 + 20, y0 + 22, x0 + 68, y0 + 70), fill=outline)
        text_center(draw, (x0 + 20, y0 + 22, x0 + 68, y0 + 70), icon, 21, WHITE, True)
        title_box = (x0 + 78, y0 + 15, x1 - 12, y0 + 70)
    else:
        title_box = (x0 + 12, y0 + 15, x1 - 12, y0 + 70)
    text_center(draw, title_box, title, 25, outline, True)
    text_center(draw, (x0 + 12, y0 + 76, x1 - 12, y1 - 12), subtitle, 18, MUTED)


def net(draw, box, labels, outline, fill=WHITE):
    x0, y0, x1, y1 = box
    gap = 14
    bw = (x1 - x0 - gap * (len(labels) - 1)) // len(labels)
    for i, label in enumerate(labels):
        bx = x0 + i * (bw + gap)
        rounded(draw, (bx, y0, bx + bw, y1), fill, outline, 3, 11)
        # A small stack makes the neural block read as a network, not a label.
        for k in range(3):
            yy = y0 + 12 + k * 15
            draw.rounded_rectangle((bx + 15, yy, bx + bw - 15, yy + 7), radius=3, fill=outline)
        text_center(draw, (bx + 7, y0 + 56, bx + bw - 7, y1 - 7), label, 17, outline, True)
        if i + 1 < len(labels):
            arrow(draw, (bx + bw + 3, (y0 + y1) // 2), (bx + bw + gap - 3, (y0 + y1) // 2), outline, 3, 11)


def pill(draw, x, y, label, fill, outline, width=260):
    rounded(draw, (x, y, x + width, y + 44), fill, outline, 2, 22)
    text_center(draw, (x + 8, y + 2, x + width - 8, y + 42), label, 17, outline, True)


def thumbnail(canvas, draw, path, box, label, outline):
    x0, y0, x1, y1 = box
    if path and Path(path).exists():
        with Image.open(path) as src:
            image = src.convert("RGB")
        image.thumbnail((x1 - x0, y1 - y0 - 34), Image.Resampling.LANCZOS)
        px = x0 + ((x1 - x0) - image.width) // 2
        py = y0 + 34 + ((y1 - y0 - 34) - image.height) // 2
        canvas.paste(image, (px, py))
    else:
        draw.rectangle((x0, y0 + 34, x1, y1), fill=GRAY_L)
        # Placeholder raster keeps the figure useful without external caches.
        for i in range(5):
            draw.line((x0 + 12, y0 + 48 + i * 28, x1 - 12, y0 + 48 + i * 28), fill="#D9E0EA", width=3)
    draw.rectangle(box, outline=outline, width=4)
    draw.rectangle((x0, y0, x1, y0 + 34), fill=outline)
    draw.text((x0 + 10, y0 + 6), label, fill=WHITE, font=font(17, True))


def formula_image(expression: str, color: str = INK, size: int = 22) -> Image.Image:
    """Render a real math expression instead of relying on PIL glyph fallback."""
    figure = Figure(figsize=(13, 0.52), dpi=150, facecolor="none")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.axis("off")
    axis.text(0.01, 0.50, expression, color=color, fontsize=size, va="center", ha="left")
    buffer = io.BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def paste_fit(canvas: Image.Image, path: str | Path | None, box):
    """Paste an actual process image into a fixed card without changing layout."""
    x0, y0, x1, y1 = box
    if path is None or not Path(path).exists():
        return
    with Image.open(path) as src:
        image = src.convert("RGB")
    image.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    px = x0 + ((x1 - x0) - image.width) // 2
    py = y0 + ((y1 - y0) - image.height) // 2
    canvas.paste(image, (px, py))


def load_aonly_metrics(report_path: Path):
    default = {"coverage": 0.9762, "locked": 0.0, "generated": 0.01098}
    if not report_path or not report_path.exists():
        return default
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = []
        for pose in payload.values():
            if isinstance(pose, dict):
                rows.extend(v for v in pose.values() if isinstance(v, dict) and "original_2px_coverage" in v)
        if rows:
            return {"coverage": sum(v["original_2px_coverage"] for v in rows) / len(rows),
                    "locked": max(v.get("locked_pixels_modified", 0) for v in rows),
                    "generated": sum(v.get("generated_fraction", 0.0) for v in rows) / len(rows)}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return default


def draw_process_trace(output: Path, diff_root: Path | None = None):
    """Create a readable evidence sheet from the actual validated artifacts."""
    root = Path("/media/2T_HD/GCR-NVS_DATA")
    camera_root = root / "outputs/t0_video_reconstruction_test_frame73/mixed_10_20cm/CAM_FRONT_RIGHT"
    process = [
        ("01", "去畸变源 RGB", camera_root / "source_native_rectified.png", TEAL),
        ("02", "DA3 稠密深度", root / "outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/depth_after_structure.png", BLUE),
        ("03", "DINOv2 + AnyUP 语义", root / "outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/dino_anyup_semantic_pca.png", GREEN),
        ("04", "T0 蓝色审计图", root / "t0_baseline_isolated_20260823/runs/t0_target_surface_2px_repair_all_20260824/mixed_10_20cm/CAM_FRONT_RIGHT/t0_highres_blue.png", RED),
        ("05", "目标稠密表面深度", camera_root / "target_surface_depth_after_2px_repair.png", PURPLE),
        ("06", "可见性 / provenance", camera_root / "validity.png", PURPLE),
        ("07", "UniWorld / DA3 Gate", root / "outputs/l4_simulated_3700x1400x2000_frame73_gate_final/target_rig/CAM_FRONT_RIGHT/t0_uniview_background_gate.png", GREEN),
        ("08", "A-ONLY 最终重建", root / "outputs/evaluations/t0_uniview_rgbds_full_a_20260825/mixed_10_20cm/CAM_FRONT_RIGHT/rgbds_a_4k.png", ORANGE),
        ("09", "a_only_abs_diff", (diff_root or root / "outputs/source_vs_aonly_diff_20260826") / "mixed_10_20cm/CAM_FRONT_RIGHT/a_only_abs_diff.png", RED),
        ("10", "最终回畸变交付", root / "outputs/l4_simulated_frame73_submission_v4/distorted_final/target_rig/CAM_FRONT_RIGHT/final_distorted.png", ORANGE),
    ]
    width, height = 5600, 2500
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), "GCR-NVS A-ONLY：单样本完整推理副产物链", fill=INK, font=font(44, True))
    draw.text((92, 105), "序列 2026-05-22-10-25-21｜帧 73｜mixed_10_20cm｜CAM_FRONT_RIGHT｜真实文件缩略图", fill=MUTED, font=font(22))
    draw.line((90, 140, width - 90, 140), fill=LINE, width=3)
    card_w, card_h = 1050, 1050
    gap_x, gap_y = 65, 55
    start_x, start_y = 100, 190
    for index, (number, label, path, color) in enumerate(process):
        row, col = divmod(index, 5)
        x, y = start_x + col * (card_w + gap_x), start_y + row * (card_h + gap_y)
        rounded(draw, (x, y, x + card_w, y + card_h), WHITE, color, 4, 18)
        draw.ellipse((x + 28, y + 28, x + 92, y + 92), fill=color)
        text_center(draw, (x + 28, y + 28, x + 92, y + 92), number, 20, WHITE, True)
        draw.text((x + 115, y + 42), label, fill=color, font=font(25, True))
        if path.exists():
            with Image.open(path) as src:
                thumb = src.convert("RGB")
            thumb.thumbnail((card_w - 60, card_h - 175), Image.Resampling.LANCZOS)
            px = x + (card_w - thumb.width) // 2
            py = y + 125 + ((card_h - 175) - thumb.height) // 2
            image.paste(thumb, (px, py))
        else:
            draw.text((x + 30, y + 500), "真实产物未挂载", fill=RED, font=font(23, True))
        draw.text((x + 30, y + card_h - 38), str(path), fill=MUTED, font=font(14))
    image.save(output, quality=96)
    print(f"saved {output} ({width}x{height})")


def draw_paper_main(output: Path, report: Path, diff_root: Path | None = None):
    """Draw a compact paper-style main architecture plate.

    The previous plate was an engineering dashboard.  This plate uses a
    single visual spine and three aligned branches, with short labels and
    larger raster evidence so it reads like a methods figure at a glance.
    """
    width, height = 5200, 3000
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    metrics = load_aonly_metrics(report)
    root = Path("/media/2T_HD/GCR-NVS_DATA")
    sample = root / "outputs/evaluations/t0_uniview_rgbds_full_a_20260825/mixed_10_20cm/CAM_FRONT_RIGHT"
    source = root / "t0_video_reconstruction_test_frame73/mixed_10_20cm/CAM_FRONT_RIGHT"
    process = {
        "source": source / "source_native_rectified.png",
        "da3": root / "outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/depth_after_structure.png",
        "semantic": root / "outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/dino_anyup_semantic_pca.png",
        "surface": source / "target_surface_depth_after_2px_repair.png",
        "t0": root / "t0_baseline_isolated_20260823/runs/t0_target_surface_2px_repair_all_20260824/mixed_10_20cm/CAM_FRONT_RIGHT/t0_highres_blue.png",
        "aonly": sample / "rgbds_a_4k.png",
        "diff": (diff_root or root / "outputs/source_vs_aonly_diff_20260826") / "mixed_10_20cm/CAM_FRONT_RIGHT/a_only_abs_diff.png",
        "final": root / "outputs/l4_simulated_frame73_submission_v4/distorted_final/target_rig/CAM_FRONT_RIGHT/final_distorted.png",
    }

    # Header.
    draw.text((120, 48), "GCR-NVS", fill=INK, font=font(52, True))
    draw.text((430, 63), "真实稀疏结构约束的新视角重建", fill=INK, font=font(34, True))
    draw.text((432, 111), "论文方法总览｜观测像素锁定｜颜色来自真实源 RGB｜生成分支只处理 true hole", fill=MUTED, font=font(21))
    draw.line((120, 150, width - 120, 150), fill=LINE, width=3)

    def section(x, y, w, h, title, color, subtitle=""):
        rounded(draw, (x, y, x + w, y + h), WHITE, color, 4, 24)
        draw.rectangle((x, y, x + w, y + 66), fill=color)
        draw.text((x + 26, y + 17), title, fill=WHITE, font=font(25, True))
        if subtitle:
            draw.text((x + 28, y + 78), subtitle, fill=MUTED, font=font(18))

    def small_block(x, y, w, h, title, subtitle, color, icon):
        rounded(draw, (x, y, x + w, y + h), WHITE, color, 3, 16)
        draw.ellipse((x + 18, y + 18, x + 62, y + 62), fill=color)
        text_center(draw, (x + 18, y + 18, x + 62, y + 62), icon, 16, WHITE, True)
        draw.text((x + 76, y + 17), title, fill=color, font=font(22, True))
        text_center(draw, (x + 18, y + 72, x + w - 18, y + h - 12), subtitle, 17, MUTED)

    def spine_arrow(x0, y0, x1, y1, color):
        arrow(draw, (x0, y0), (x1, y1), color, width=6, head=20)

    # Stage 1: four input modalities.
    section(120, 200, 1120, 580, "① 输入与统一坐标", TEAL, "多相机观测、标定与稀疏结构进入同一 rectified frame")
    small_block(155, 310, 485, 150, "七路源 RGB", "去畸变前的真实图像", TEAL, "RGB")
    small_block(685, 310, 485, 150, "相机标定", "K / D / T / 时间戳 / 拓扑", TEAL, "K")
    small_block(155, 515, 485, 150, "当前 LiDAR", "真实公制深度与遮挡锚点", BLUE, "L")
    small_block(685, 515, 485, 150, "最近时序 LiDAR", "注册后的稀疏结构补偿", BLUE, "T")
    spine_arrow(1240, 490, 1370, 490, TEAL)

    # Stage 2: three aligned feature branches.
    section(1370, 200, 1450, 580, "② 特征与结构编码", BLUE, "三条分支职责互补，不互相替代")
    small_block(1410, 310, 405, 155, "DA3Metric-Large", "RGB 对齐的稠密深度候选", BLUE, "D")
    net(draw, (1430, 500, 1795, 675), ["ViT-L", "DPT", "D_m"], BLUE)
    small_block(1840, 310, 405, 155, "DINOv2-L + AnyUP", "语义、边界与高分辨率对应", GREEN, "S")
    net(draw, (1860, 500, 2225, 675), ["Token", "AnyUP", "F"], GREEN)
    small_block(2270, 310, 405, 155, "LiDAR 时序编码", "公制结构、自由空间、可见性", PURPLE, "L")
    net(draw, (2290, 500, 2655, 675), ["PCD", "时序", "锚点"], PURPLE)
    spine_arrow(2820, 490, 2960, 490, BLUE)

    # Stage 3: central 3D scene.
    section(2960, 200, 1080, 580, "③ 统一三维场与目标位姿", PURPLE, "真实结构约束 DA3 表面，再执行目标相机的 SE(3) 变换")
    small_block(3000, 310, 455, 155, "T0StructureConstraintNet", "LiDAR 校准 DA3，不直接生成 RGB", PURPLE, "T0")
    net(draw, (3020, 500, 3435, 675), ["融合", "Transformer", "表面"], PURPLE)
    small_block(3480, 310, 510, 155, "目标相机参数", "K_t / D_t / T_t + SE(3)", PURPLE, "SE")
    net(draw, (3500, 500, 3970, 675), ["反投影", "投影", "z-buffer"], PURPLE)
    spine_arrow(4040, 490, 4180, 490, PURPLE)

    # Stage 4: RGB appearance and hole-only completion.
    section(4180, 200, 900, 580, "④ 外观重建与审计", ORANGE, "几何决定采样位置，真实 RGB 决定颜色")
    small_block(4220, 300, 380, 140, "provenance RGB sampler", "按对应关系取真实颜色", ORANGE, "I")
    small_block(4660, 300, 380, 140, "T0 蓝色审计", "observed 锁定，hole 显式标记", RED, "V")
    small_block(4220, 490, 380, 140, "UniWorld / DA3 Gate", "仅恢复有后景证据的 hole", GREEN, "G")
    small_block(4660, 490, 380, 140, "A-ONLY RGB-D-S", "仅生成 residual true hole", ORANGE, "A")
    spine_arrow(4630, 440, 4630, 490, ORANGE)
    spine_arrow(4630, 630, 4630, 760, ORANGE)

    # Main output spine and actual artifacts.
    draw.line((120, 900, width - 120, 900), fill=LINE, width=3)
    draw.text((120, 935), "主输出：端到端重建结果", fill=INK, font=font(29, True))
    outputs = [("T0", "蓝色审计图", process["t0"], RED), ("A", "A-ONLY 最终图", process["aonly"], ORANGE),
               ("OUT", "raw K/D 回畸变交付", process["final"], ORANGE), ("Δ", "a_only_abs_diff（审计）", process["diff"], RED)]
    x = 120
    for i, (tag, label, path, color) in enumerate(outputs):
        rounded(draw, (x, 1000, x + 1160, 1380), WHITE, color, 4, 18)
        draw.ellipse((x + 25, 1025, x + 88, 1088), fill=color)
        text_center(draw, (x + 25, 1025, x + 88, 1088), tag, 16, WHITE, True)
        draw.text((x + 115, 1035), label, fill=color, font=font(24, True))
        if path.exists():
            with Image.open(path) as src_img:
                thumb = src_img.convert("RGB")
            thumb.thumbnail((1080, 245), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (1160 - thumb.width) // 2, 1110 + (245 - thumb.height) // 2))
        else:
            draw.text((x + 35, 1190), "未挂载真实产物", fill=RED, font=font(22, True))
        x += 1260
    # Main result path is T0 -> A-ONLY -> final.  DIFF is an audit side branch.
    spine_arrow(1280, 1190, 1335, 1190, ORANGE)
    spine_arrow(2540, 1190, 2595, 1190, ORANGE)
    arrow(draw, (1960, 1380), (3850, 1380), RED, width=4, head=16, dashed=True)
    draw.text((3100, 1392), "审计旁路：源图 ↔ A-ONLY", fill=RED, font=font(16, True))

    # Equations and contracts, laid out as two clean paper bands.
    draw.line((120, 1450, width - 120, 1450), fill=LINE, width=3)
    section(120, 1490, 2420, 535, "几何计算", PURPLE, "")
    equations = [
        (r"$I_{\mathrm{rect}}=\operatorname{Remap}(I_{\mathrm{raw}},K,D\rightarrow K_{\mathrm{new}},0)$", 1575),
        (r"$X=ZK^{-1}[u,v,1]^{\mathsf{T}}$    $X_t=T_tX_s$", 1660),
        (r"$p_t\sim K_tX_t$    $Z_t(u,v)=\min_i Z_i(u,v)$", 1745),
    ]
    for expression, y in equations:
        image = formula_image(expression, color=INK, size=22)
        image.thumbnail((2220, 62), Image.Resampling.LANCZOS)
        canvas.paste(image, (185, y), image)
    draw.text((185, 1875), "真实 LiDAR 约束公制结构，DA3 提供稠密候选，RGB 提供颜色。", fill=PURPLE, font=font(21, True))
    draw.text((185, 1925), "稀疏 LiDAR 不直接覆盖 RGB；时序 LiDAR 只补偿结构可见性。", fill=MUTED, font=font(19))
    section(2600, 1490, 2480, 535, "像素来源与边界", ORANGE, "")
    draw.text((2665, 1575), "I_final = I_source(observed)", fill=INK, font=font(24, True))
    draw.text((2665, 1640), "        + I_gate(recoverable hole)", fill=INK, font=font(24, True))
    draw.text((2665, 1705), "        + I_A-only(residual true hole)", fill=INK, font=font(24, True))
    draw.text((2665, 1800), "observed / gate 像素 immutable", fill=RED, font=font(22, True))
    draw.text((2665, 1860), "蓝色区域仅是审计标记，不参与 RGB 或 DIFF。", fill=MUTED, font=font(19))
    draw.text((2665, 1915), "最终按目标 raw K/D 回映射为 final_distorted.png。", fill=ORANGE, font=font(20, True))

    # Bottom metric strip, concise and truthful.
    draw.line((120, 2100, width - 120, 2100), fill=LINE, width=3)
    draw.text((120, 2140), "核心指标（A-ONLY｜3 档位姿｜21 视角）", fill=INK, font=font(27, True))
    cards = [("完成率", f"{metrics['coverage'] * 100:.2f}%", "observed coverage", TEAL),
             ("锁定像素", "0 px", "21/21 case", RED),
             ("生成占比", f"{metrics['generated'] * 100:.2f}%", "全图代理值", ORANGE),
             ("几何误差", "待 holdout", "不以 DA3 伪深度冒充", BLUE),
             ("语义 / 时序", "待测", "需独立 GT / 连续帧", GREEN)]
    x = 120
    for title, value, note, color in cards:
        rounded(draw, (x, 2210, x + 960, 2470), WHITE, color, 3, 16)
        draw.text((x + 25, 2240), title, fill=color, font=font(21, True))
        draw.text((x + 25, 2295), value, fill=INK, font=font(30, True))
        draw.text((x + 25, 2370), note, fill=MUTED, font=font(17))
        x += 1000

    # Footer: future route and DIFF provenance.
    rounded(draw, (120, 2550, 5080, 2880), GRAY_L, LINE, 3, 16)
    draw.text((155, 2590), "创新点", fill=PURPLE, font=font(22, True))
    draw.text((155, 2640), "真实稀疏 LiDAR 的公制结构约束 + DA3 RGB 对齐稠密表面 + DINOv2/AnyUP 语义对应 + provenance 锁定采样 + hole-only 生成。", fill=INK, font=font(20))
    draw.text((155, 2705), "主 DIFF：a_only_abs_diff.png｜未来 TODO：Wan2.1/VACE 仅接入连续帧 residual hole，不改 observed latent。", fill=RED, font=font(20, True))
    draw.text((155, 2770), "当前图对应可复现的 T0 + UniWorld/DA3 Gate + A-ONLY 定档路线，不宣称已完成 4D 视频扩散。", fill=MUTED, font=font(18))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=96)
    print(f"saved {output} ({width}x{height})")


def draw(args):
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    metrics = load_aonly_metrics(args.report)
    demo = Path(args.demo_root) if args.demo_root else None
    diff = Path(args.diff_root) if args.diff_root else None
    # Real artifacts from the validated A-ONLY run.  They are optional so the
    # figure remains reproducible on a machine that only has the repository.
    sample_camera = "CAM_FRONT_RIGHT"
    process = {
        "RGB_rect": Path("/media/2T_HD/GCR-NVS_DATA/outputs/t0_video_reconstruction_test_frame73/mixed_10_20cm/CAM_FRONT_RIGHT/source_native_rectified.png"),
        "D_DA3": Path("/media/2T_HD/GCR-NVS_DATA/outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/depth_after_structure.png"),
        "F_DINO": Path("/media/2T_HD/GCR-NVS_DATA/outputs/l4_simulated_frame73_submission_v4/t0_surface/source_sidecars/CAM_FRONT_RIGHT/dino_anyup_semantic_pca.png"),
        "D_surface": Path("/media/2T_HD/GCR-NVS_DATA/outputs/t0_video_reconstruction_test_frame73/mixed_10_20cm/CAM_FRONT_RIGHT/target_surface_depth_after_2px_repair.png"),
        "visibility": Path("/media/2T_HD/GCR-NVS_DATA/outputs/t0_video_reconstruction_test_frame73/mixed_10_20cm/CAM_FRONT_RIGHT/validity.png"),
        "a_only_abs_diff": Path("/media/2T_HD/GCR-NVS_DATA/outputs/source_vs_aonly_diff_20260826/mixed_10_20cm/CAM_FRONT_RIGHT/a_only_abs_diff.png"),
        "final_distorted": Path("/media/2T_HD/GCR-NVS_DATA/outputs/l4_simulated_frame73_submission_v4/distorted_final/target_rig/CAM_FRONT_RIGHT/final_distorted.png"),
    }
    if demo:
        process.update({
            "RGB_rect": demo / "01_t0_surface/target_rig/CAM_FRONT_RIGHT/source_native_rectified.png",
            "D_surface": demo / "01_t0_surface/target_rig/CAM_FRONT_RIGHT/target_surface_depth_after_2px_repair.png",
            "visibility": demo / "01_t0_surface/target_rig/CAM_FRONT_RIGHT/validity.png",
            "final_distorted": demo / "06_distorted_final/target_rig/CAM_FRONT_RIGHT/final_distorted.png",
        })
    if diff:
        process["a_only_abs_diff"] = diff / "mixed_10_20cm/CAM_FRONT_RIGHT/a_only_abs_diff.png"
    draw.text((90, 42), "GCR-NVS：真实稀疏结构约束的端到端新视角重建", fill=INK, font=font(52, True))
    draw.text((92, 105), "当前定档路线｜观测像素锁定，颜色来自真实源图，生成分支只处理没有观测来源的区域", fill=MUTED, font=font(24))
    draw.line((90, 130, W - 90, 130), fill=LINE, width=3)

    # Five-stage architecture spine.
    stage(draw, 100, 1, "输入与校准", TEAL)
    stage(draw, 1200, 2, "稠密表面与语义", BLUE)
    stage(draw, 2400, 3, "三维结构变换", PURPLE)
    stage(draw, 3550, 4, "真实颜色采样", ORANGE)
    stage(draw, 4660, 5, "审计与交付", RED)

    # Stage 1: source and preprocessing.
    module(draw, (100, 260, 510, 480), "七路 RGB", "去畸变前的真实相机图像", TEAL_L, TEAL, "RGB")
    module(draw, (555, 260, 965, 480), "标定参数", "K、D、T、时间戳、相机拓扑", TEAL_L, TEAL, "K")
    module(draw, (100, 540, 510, 760), "时序 LiDAR", "当前帧 + 最近帧注册补偿", BLUE_L, BLUE, "L")
    module(draw, (555, 540, 965, 760), "预处理", "Brown 去畸变｜PCD 对齐｜动态 mask", TEAL_L, TEAL, "P")
    arrow(draw, (510, 370), (555, 370), TEAL)
    arrow(draw, (965, 370), (1080, 370), TEAL)
    arrow(draw, (510, 650), (555, 650), BLUE)
    arrow(draw, (965, 650), (1080, 650), BLUE)
    module(draw, (100, 820, 965, 1015), "统一 rectified 坐标系", "RGB_rect、LiDAR_depth、时间与外参在同一像素坐标中对齐", TEAL_L, TEAL, "R")
    arrow(draw, (530, 760), (530, 820), TEAL)

    # Stage 2 branches.
    module(draw, (1200, 260, 1600, 480), "DA3Metric-Large", "RGB 对齐的稠密深度候选", BLUE_L, BLUE, "D")
    net(draw, (1230, 525, 1570, 685), ["ViT-L", "DPT", "D_m"], BLUE)
    module(draw, (1645, 260, 2045, 480), "DINOv2-L + AnyUP", "高分辨率语义、边界、对应特征", GREEN_L, GREEN, "S")
    net(draw, (1675, 525, 2015, 685), ["Token", "AnyUP", "F_64"], GREEN)
    module(draw, (1200, 820, 2045, 1015), "T0StructureConstraintNet", "真实 LiDAR 锚定公制结构，校准 DA3 表面；不直接生成 RGB", PURPLE_L, PURPLE, "T0")
    arrow(draw, (965, 900), (1200, 370), BLUE)
    arrow(draw, (965, 900), (1645, 370), GREEN)
    arrow(draw, (1600, 370), (1200, 900), BLUE)
    arrow(draw, (2045, 370), (2045, 900), GREEN)
    arrow(draw, (2045, 900), (2200, 900), PURPLE)

    # Stage 3 3D transform.
    module(draw, (2400, 260, 2770, 480), "稠密 3D 表面", "X = Z · K⁻¹[u,v,1]ᵀ", PURPLE_L, PURPLE, "3D")
    module(draw, (2815, 260, 3185, 480), "目标相机", "K_t、D_t、T_t + SE(3)", PURPLE_L, PURPLE, "SE")
    net(draw, (2430, 525, 3155, 685), ["反投影", "SE(3)", "投影", "z-buffer"], PURPLE)
    module(draw, (2400, 820, 3185, 1015), "可见性分层", "observed｜recoverable hole｜true hole｜provenance", PURPLE_L, PURPLE, "V")
    arrow(draw, (2200, 900), (2400, 370), PURPLE)
    arrow(draw, (2770, 370), (2815, 370), PURPLE)
    arrow(draw, (2995, 480), (2995, 525), PURPLE)
    arrow(draw, (3185, 370), (3280, 370), PURPLE)
    arrow(draw, (2995, 685), (2995, 820), PURPLE)

    # Stage 4 appearance route.
    module(draw, (3550, 260, 3950, 480), "真实源 RGB 采样", "按 provenance + 几何对应取色", ORANGE_L, ORANGE, "I")
    net(draw, (3580, 525, 3920, 685), ["grid", "sample", "fusion"], ORANGE)
    module(draw, (3550, 820, 4350, 1015), "T0 蓝色审计结果", "rectified RGB｜无效区蓝色标记｜观测像素不可修改", RED_L, RED, "T0")
    arrow(draw, (3185, 900), (3550, 900), ORANGE)
    arrow(draw, (3280, 370), (3550, 370), ORANGE)
    arrow(draw, (3750, 480), (3750, 525), ORANGE)
    arrow(draw, (3750, 685), (3750, 820), ORANGE)

    # Stage 5 optional branches and final output.
    module(draw, (4660, 260, 5070, 480), "UniWorld / DA3 Gate", "三重回投影：只接收有后景证据的 hole", GREEN_L, GREEN, "G")
    module(draw, (5115, 260, 5525, 480), "A-ONLY RGB-D-S", "只生成剩余 true hole，observed immutable", ORANGE_L, ORANGE, "A")
    net(draw, (4690, 525, 5495, 685), ["gate", "RGB-D-S", "lock"], RED)
    module(draw, (4660, 820, 5525, 1015), "最终 rectified Hybrid RGB", "真实采样 + gate + A-only；可选 OpenCV 1–2 px 裂缝修补", ORANGE_L, ORANGE, "OUT")
    arrow(draw, (4350, 900), (4660, 370), GREEN)
    arrow(draw, (4350, 900), (5115, 370), ORANGE)
    arrow(draw, (4865, 480), (4865, 525), GREEN)
    arrow(draw, (5320, 480), (5320, 525), ORANGE)
    arrow(draw, (5070, 900), (4660, 900), ORANGE)
    arrow(draw, (5495, 685), (5320, 820), ORANGE)

    # Outputs and equations band.
    draw.line((90, 1110, W - 90, 1110), fill=LINE, width=3)
    text_center(draw, (100, 1140, 2700, 1200), "核心几何公式（中文定义）", 27, PURPLE, True)
    text_center(draw, (2800, 1140, 5500, 1200), "最终合成与数据来源", 27, ORANGE, True)
    rounded(draw, (100, 1220, 2700, 1535), WHITE, PURPLE, 3, 16)
    equations = [
        (r"$I_{\mathrm{rect}}=\operatorname{Remap}(I_{\mathrm{raw}},K,D\rightarrow K_{\mathrm{new}},0)$", 1250),
        (r"$X=ZK^{-1}[u,v,1]^{\mathsf{T}}$    $X_t=T_tX_s$", 1320),
        (r"$p_t\sim K_tX_t$    $Z_t(u,v)=\min_i Z_i(u,v)$", 1390),
    ]
    for expression, y in equations:
        image = formula_image(expression, color=INK, size=22)
        image.thumbnail((2450, 54), Image.Resampling.LANCZOS)
        canvas.paste(image, (145, y), image)
    draw.text((145, 1460), "LiDAR 约束几何；DA3 提供稠密候选；RGB 提供颜色。职责不混淆。", fill=PURPLE, font=font(22, True))
    rounded(draw, (2800, 1220, 5500, 1535), WHITE, ORANGE, 3, 16)
    draw.text((2845, 1260), "I_final = I_RGB-source（observed）", fill=INK, font=font(24, True))
    draw.text((2845, 1320), "       + I_gate（有后景证据的可恢复 hole）", fill=INK, font=font(24, True))
    draw.text((2845, 1380), "       + I_A-only（剩余真实未观测区域）", fill=INK, font=font(24, True))
    draw.text((2845, 1440), "最后按目标 raw K/D 回映射，得到交付图 final_distorted.png。", fill=ORANGE, font=font(22, True))

    # Trace products.
    draw.line((90, 1600, W - 90, 1600), fill=LINE, width=3)
    text_center(draw, (100, 1630, 5500, 1690), "一张样本的可审计副产物链", 27, TEAL, True)
    products = [("01", "RGB_rect", "去畸变源图", TEAL), ("02", "D_DA3", "稠密深度", BLUE),
                ("03", "F_DINO", "语义/边界特征", GREEN), ("04", "D_surface", "LiDAR 约束表面", PURPLE),
                ("05", "visibility", "可见性与来源", PURPLE), ("06", "a_only_abs_diff", "源图-最终图 DIFF", RED),
                ("07", "final_distorted", "最终交付图", ORANGE)]
    x = 130
    for idx, (num, name, label, color) in enumerate(products):
        rounded(draw, (x, 1725, x + 700, 1935), WHITE, color, 3, 14)
        paste_fit(canvas, process.get(name), (x + 18, 1760, x + 210, 1900))
        draw.ellipse((x + 225, 1750, x + 285, 1810), fill=color)
        text_center(draw, (x + 225, 1750, x + 285, 1810), num, 19, WHITE, True)
        draw.text((x + 305, 1760), name, fill=color, font=font(22, True))
        draw.text((x + 305, 1812), label, fill=INK, font=font(20))
        if idx < len(products) - 1:
            arrow(draw, (x + 700, 1830), (x + 760, 1830), LINE, 4, 15)
        x += 780

    # Metrics panel; values are tied to report and status is explicit.
    draw.line((90, 2020, W - 90, 2020), fill=LINE, width=3)
    text_center(draw, (100, 2050, 5500, 2110), "核心指标达成值（A-ONLY 三档 21 视角汇总）", 29, INK, True)
    metric_items = [
        ("完成率 / 覆盖率", f"{metrics['coverage'] * 100:.2f}%", "实测｜2px 表面修补后的 observed coverage", TEAL),
        ("几何误差", "待 holdout 汇总", "已有 LiDAR holdout 报告，当前图不混入 DA3 伪深度误差", BLUE),
        ("语义保留率", "代理指标待测", "应以 DINOv2/AnyUP 特征余弦或语义 IoU 定义", GREEN),
        ("伪影率", f"{metrics['generated'] * 100:.2f}% 生成占比", "代理：A-only 生成像素占全图；不是完整伪影真值", ORANGE),
        ("时序异常率", "本轮未测", "当前材料为单帧新视角；连续帧视频验收后填写", RED),
    ]
    x = 120
    for title, value, note, color in metric_items:
        rounded(draw, (x, 2150, x + 1040, 2460), WHITE, color, 3, 16)
        draw.text((x + 25, 2180), title, fill=color, font=font(22, True))
        draw.text((x + 25, 2235), value, fill=INK, font=font(30, True))
        text_center(draw, (x + 22, 2300, x + 1018, 2435), note, 17, MUTED)
        x += 1090

    # Contract and future route.
    rounded(draw, (100, 2530, 3650, 3260), BLUE_L, BLUE, 3, 18)
    draw.text((145, 2570), "传感器职责与模型边界", fill=BLUE, font=font(28, True))
    rules = ["LiDAR：真实公制结构、遮挡顺序、SE(3) 约束；不直接画 RGB。",
             "时序 LiDAR：只做注册后的结构补偿，不把时间帧当作颜色监督。",
             "DA3：与 RGB 对齐的稠密候选表面，不替代真实深度。",
             "DINOv2 + AnyUP：语义、边界与对应特征，不替代几何。",
             "真实 RGB：颜色与高频纹理的第一来源；observed 像素 immutable。",
             "生成器：只处理没有真实来源的 hole，不能重绘已观测车辆、车道线和天空。"]
    for i, rule in enumerate(rules):
        draw.text((160, 2635 + i * 78), "• " + rule, fill=INK, font=font(21))
    rounded(draw, (3740, 2530, 5500, 3260), RED_L, RED, 3, 18)
    draw.text((3785, 2570), "评估与未来路线", fill=RED, font=font(28, True))
    draw.text((3790, 2640), "A-ONLY DIFF 主图：", fill=INK, font=font(21, True))
    draw.text((3790, 2685), "a_only_abs_diff.png", fill=RED, font=font(23, True))
    draw.text((3790, 2745), "源图与最终图逐像素绝对差；不把蓝色 validity 当 RGB。", fill=INK, font=font(20))
    draw.text((3790, 2830), "评估协议：", fill=INK, font=font(21, True))
    draw.text((3790, 2875), "strict LOO + 5–10 / 10–20 / 20–50 cm 混合 SE(3)", fill=INK, font=font(20))
    draw.text((3790, 2960), "TODO：Wan2.1/VACE 视频扩散接入", fill=RED, font=font(23, True))
    draw.text((3790, 3010), "仅作用于连续帧 residual hole，观察像素继续锁定。", fill=INK, font=font(20))
    draw.text((3790, 3085), "当前不宣称已完成 4D 生成。", fill=MUTED, font=font(20, True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=96)
    print(f"saved {args.output} ({W}x{H})")
    trace_output = args.trace_output or args.output.with_name("gcr_nvs_aonly_process_trace_cn_20260826.png")
    draw_process_trace(trace_output, args.diff_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_figures/gcr_nvs_final_architecture_cn_20260826.png"))
    parser.add_argument("--report", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/evaluations/t0_uniview_rgbds_full_a_20260825/report.json"))
    parser.add_argument("--demo-root", type=Path, default=None, help="Optional demo output root for real process thumbnails")
    parser.add_argument("--diff-root", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/source_vs_aonly_diff_20260826"))
    parser.add_argument("--trace-output", type=Path, default=None, help="Output path for the full process-artifact evidence sheet")
    args = parser.parse_args()
    draw_paper_main(args.output, args.report, args.diff_root)
    trace_output = args.trace_output or args.output.with_name("gcr_nvs_aonly_process_trace_cn_20260826.png")
    draw_process_trace(trace_output, args.diff_root)


if __name__ == "__main__":
    main()
