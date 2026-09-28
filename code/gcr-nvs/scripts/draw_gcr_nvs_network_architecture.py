"""Draw the block-level GCR-NVS network architecture using PIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from draw_gcr_nvs_paper_architecture import (
    AMBER,
    AMBER_LIGHT,
    BACKGROUND,
    BLUE,
    BLUE_LIGHT,
    CYAN,
    CYAN_LIGHT,
    GREEN,
    GREEN_LIGHT,
    INK,
    LINE,
    MUTED,
    PURPLE,
    PURPLE_LIGHT,
    RED,
    RED_LIGHT,
    RESAMPLE_LANCZOS,
    WHITE,
    arrow,
    depth_color,
    fit_image,
    font,
    label,
    lidar_bev,
    paste_framed,
    pill,
    rounded_box,
    sparse_depth_preview,
    text_bbox,
    wrap_text,
)
from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.pcd import read_pcd


CANVAS = (5200, 3300)


def compact_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    fill: str,
    outline: str,
    title_size: int = 31,
    body_size: int = 24,
) -> None:
    rounded_box(draw, xy, fill, outline, radius=16, width=3)
    left, top, right, _ = xy
    draw.text((left + 18, top + 14), title, font=font(title_size, bold=True), fill=INK)
    if subtitle:
        wrapped = wrap_text(draw, subtitle, font(body_size), right - left - 36)
        draw.multiline_text(
            (left + 18, top + 58), wrapped, font=font(body_size), fill=MUTED, spacing=7,
        )


def tensor_tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: str,
) -> None:
    bbox = text_bbox(draw, (0, 0), text, font(22, medium=True))
    width = bbox[2] - bbox[0] + 24
    height = 42
    pill(draw, (xy[0], xy[1], xy[0] + width, xy[1] + height), text, WHITE, color, color, 22)


def draw_sparse_unet(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    draw.text((x, y - 58), "SparseConv U-Net Geometry Student", font=font(35, bold=True), fill=BLUE)
    encoder = [
        ("Stem", "SubMConv3d\n6→32", 180),
        ("E1", "ResBlock×2\n32", 180),
        ("E2", "SpConv s2\n64", 180),
        ("E3", "SpConv s2\n128", 180),
        ("E4", "SpConv s2\n256", 180),
    ]
    boxes = []
    cursor = x
    for title, body, width in encoder:
        box = (cursor, y, cursor + width, y + 165)
        compact_block(draw, box, title, body, BLUE_LIGHT, BLUE, 29, 23)
        boxes.append(box)
        cursor += width + 46
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2], y + 82), (right[0], y + 82), BLUE, width=6, head=18)
    decoder_y = y + 270
    decoder = [
        ("D4", "SparseDeconv\n256→128"),
        ("D3", "Skip + Res\n128"),
        ("D2", "Skip + Res\n64"),
        ("D1", "Skip + Res\n32"),
    ]
    decoder_boxes = []
    cursor = boxes[-1][0]
    for title, body in decoder:
        box = (cursor, decoder_y, cursor + 190, decoder_y + 155)
        compact_block(draw, box, title, body, "#DDEBFF", BLUE, 28, 22)
        decoder_boxes.append(box)
        cursor -= 236
    for left, right in zip(decoder_boxes, decoder_boxes[1:]):
        arrow(draw, (left[0], decoder_y + 78), (right[2], decoder_y + 78), BLUE, width=6, head=18)
    arrow(draw, (boxes[-1][0] + 90, boxes[-1][3]), (decoder_boxes[0][0] + 95, decoder_y), BLUE, width=6)
    skip_pairs = [(boxes[3], decoder_boxes[1]), (boxes[2], decoder_boxes[2]), (boxes[1], decoder_boxes[3])]
    for source, target in skip_pairs:
        sx = (source[0] + source[2]) // 2
        tx = (target[0] + target[2]) // 2
        draw.line((sx, source[3], sx, decoder_y - 28, tx, decoder_y - 28, tx, decoder_y), fill="#79A9EB", width=4)
        draw.ellipse((tx - 7, decoder_y - 7, tx + 7, decoder_y + 7), fill=BLUE)
    tensor_tag(draw, (x, y + 190), "adaptive voxels: 0.08 / 0.16 / 0.32 m", BLUE)


def draw_rgb_encoder(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    draw.text((x, y - 58), "Multi-view RGB Appearance Encoder", font=font(35, bold=True), fill=GREEN)
    dino = (x, y, x + 370, y + 205)
    dpt = (x + 455, y, x + 790, y + 205)
    naf = (x, y + 275, x + 370, y + 480)
    detail = (x + 455, y + 275, x + 790, y + 480)
    fuse = (x + 900, y + 110, x + 1260, y + 385)
    compact_block(draw, dino, "DINOv2 ViT-B/14", "Frozen, 86M\n840×476 → 60×34\nblocks {2,5,8,11}\n768-D tokens", GREEN_LIGHT, GREEN, 31, 23)
    compact_block(draw, dpt, "DPT / FPN Adapter", "1×1 lateral conv\nupsample + fusion\n→ 128-D @ 1/4", GREEN_LIGHT, GREEN, 30, 23)
    compact_block(draw, naf, "NAF Detail Stem", "Conv3×3 + GN + SiLU\nNAFBlock×4\nfull / 1/2 / 1/4", "#E2F6ED", GREEN, 31, 23)
    compact_block(draw, detail, "Detail FPN", "edge / lane / pole\n64-D high-frequency\nfeature pyramid", "#E2F6ED", GREEN, 30, 23)
    compact_block(draw, fuse, "Semantic + Detail Fuse", "Concat 128+64\nConv3×3 + NAFBlock\nper-view 96-D feature\n[B,V,96,H/4,W/4]", GREEN_LIGHT, GREEN, 31, 24)
    arrow(draw, (dino[2], y + 102), (dpt[0], y + 102), GREEN, width=6)
    arrow(draw, (naf[2], y + 377), (detail[0], y + 377), GREEN, width=6)
    arrow(draw, (dpt[2], y + 102), (fuse[0], y + 185), GREEN, width=6)
    arrow(draw, (detail[2], y + 377), (fuse[0], y + 310), GREEN, width=6)


def draw_restormer(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    draw.text((x, y - 62), "Restormer-Small", font=font(34, bold=True), fill=PURPLE)
    levels = [
        ("L1", "48ch\n4 blocks", 0),
        ("L2", "96ch\n6 blocks", 95),
        ("L3", "192ch\n6 blocks", 190),
        ("Latent", "384ch\n8 blocks", 285),
    ]
    enc_boxes = []
    for index, (title, body, offset) in enumerate(levels):
        box = (x + index * 98, y + offset, x + index * 98 + 84, y + offset + 135)
        compact_block(draw, box, title, body, PURPLE_LIGHT, PURPLE, 22, 17)
        enc_boxes.append(box)
        if index:
            arrow(draw, (enc_boxes[index - 1][2], enc_boxes[index - 1][1] + 70), (box[0], box[1] + 70), PURPLE, width=5, head=15)
    dec_boxes = []
    for index in range(3):
        box = (
            x + 405 + index * 98,
            y + 190 - index * 95,
            x + 489 + index * 98,
            y + 325 - index * 95,
        )
        compact_block(draw, box, f"D{3-index}", f"up+skip\n{192 // (2**index)}ch", "#E9E0F7", PURPLE, 22, 17)
        dec_boxes.append(box)
    arrow(draw, (enc_boxes[-1][2], enc_boxes[-1][1] + 72), (dec_boxes[0][0], dec_boxes[0][1] + 72), PURPLE, width=5)
    for left, right in zip(dec_boxes, dec_boxes[1:]):
        arrow(draw, (left[2], left[1] + 72), (right[0], right[1] + 72), PURPLE, width=5)
    for source, target in zip(enc_boxes[2::-1], dec_boxes):
        sy = source[1]
        ty = target[1]
        draw.line((source[0] + 65, sy, source[0] + 65, y - 25, target[0] + 65, y - 25, target[0] + 65, ty), fill="#A98BD1", width=3)
    head = (x + 710, y + 28, x + 910, y + 365)
    compact_block(draw, head, "Heads", "observed residual\ncompletion RGB\nalpha gate\nlog uncertainty", RED_LIGHT, RED, 25, 19)
    arrow(draw, (dec_boxes[-1][2], dec_boxes[-1][1] + 72), (head[0], head[1] + 155), PURPLE, width=6)


def draw_teacher_strip(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    points,
) -> None:
    rounded_box(draw, (55, 185, 5145, 745), WHITE, "#C8D3E0", radius=24)
    draw.text((85, 210), "A. Training-only Temporal LiDAR Teacher", font=font(43, bold=True), fill=INK)
    draw.text((85, 265), "修复时间顺序后，用多帧 LiDAR 生成单帧几何学生的高置信监督；教师不进入正式推理。", font=font(28), fill=MUTED)
    bev = lidar_bev(points, (430, 300))
    paste_framed(canvas, draw, bev, (90, 335), "P_t current LiDAR", CYAN, CYAN_LIGHT, CYAN)
    blocks = [
        ((600, 355, 1010, 635), "Timestamp Sort", "按 lidar_timestamp 排序\nΔt 门控 / 异常窗口切断", "#EEF2F7", "#7A8CA5"),
        ((1080, 355, 1490, 635), "Deskew", "PCD time 语义确认后启用\nscan 内运动补偿", CYAN_LIGHT, CYAN),
        ((1560, 355, 1970, 635), "KISS-ICP", "T(t←k) + RMSE/inlier\n低质量邻帧直接禁用", BLUE_LIGHT, BLUE),
        ((2040, 355, 2450, 635), "SegFormer-B2", "车辆/行人动态 mask\n邻帧动态点不融合", GREEN_LIGHT, GREEN),
        ((2520, 355, 2970, 635), "Temporal Surfel Fusion", "P(t±2) → current frame\nvoxel median + front surface", PURPLE_LIGHT, PURPLE),
        ((3040, 355, 3480, 635), "Teacher Heads", "depth / occupancy / normal\nconfidence + source mask", AMBER_LIGHT, AMBER),
        ((3550, 355, 4050, 635), "Student Distillation", "masked inverse-depth L1\noccupancy BCE / normal cosine", BLUE_LIGHT, BLUE),
        ((4120, 355, 5070, 635), "Teacher Output Contract", "exact_current：硬真值\ntemporal_filled：高置信伪 GT\nunknown：不监督\n禁止 RGB 深度进入 teacher", RED_LIGHT, RED),
    ]
    for xy, title, body, fill, outline in blocks:
        compact_block(draw, xy, title, body, fill, outline, 29, 23)
    arrow(draw, (520, 490), (600, 490), CYAN, width=6)
    for left, right in zip(blocks, blocks[1:]):
        arrow(draw, (left[0][2], 490), (right[0][0], 490), LINE, width=5, head=16)


def draw_input_panel(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    sequence_dir: Path,
    payload: dict,
    points,
) -> None:
    rounded_box(draw, (55, 840, 725, 2380), WHITE, "#C8D3E0", radius=24)
    draw.text((85, 870), "B. Single-frame Input", font=font(42, bold=True), fill=INK)
    draw.text((85, 925), "正式推理只读取当前帧", font=font(27), fill=MUTED)
    thumb = (185, 104)
    positions = [(85 + (i % 3) * 205, 985 + (i // 3) * 155) for i in range(7)]
    for camera, position in zip(CAMERA_NAMES, positions):
        path = sequence_dir / "Key_frames" / camera / payload["sensors"][camera]
        paste_framed(canvas, draw, fit_image(path, thumb), position, camera.replace("CAM_", ""), GREEN, GREEN_LIGHT, GREEN)
    bev = lidar_bev(points, (580, 390))
    paste_framed(canvas, draw, bev, (85, 1510), f"LiDAR P_t  [{len(points):,}, 6]", CYAN, CYAN_LIGHT, CYAN)
    label(
        draw,
        (85, 1980, 695, 2325),
        "Input tensors",
        "RGB [B,7,3,H,W]\nLiDAR [B,N,6]\nsource K/D/T × 7\ntarget K_t/D_t/T_t\ndynamic mask + validity",
        "#F3F6FA",
        "#8A9AB0",
        title_size=32,
        body_size=25,
    )


def draw_main_network(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    sequence_dir: Path,
    payload: dict,
    calibrations,
    points,
    target_camera: str,
) -> None:
    draw.text((780, 820), "C. Single-frame Student and Neural Rendering Network", font=font(45, bold=True), fill=INK)
    draw_sparse_unet(draw, (800, 940))
    geom_heads = (800, 1545, 1350, 1830)
    compact_block(draw, geom_heads, "Geometry Heads", "occupancy O\ndepth / SDF Z\nnormal N\nvisibility V\nconfidence U_z", BLUE_LIGHT, BLUE, 32, 25)
    arrow(draw, (800, 1200), (725, 1420), BLUE, width=7)
    arrow(draw, (1185, 1288), (1075, 1545), BLUE, width=7)
    pill(draw, (820, 1465, 2290, 1525), "Hard replace exact current LiDAR points after the student head", BLUE, WHITE, BLUE, 26)

    draw_rgb_encoder(draw, (800, 1900))
    arrow(draw, (800, 2160), (725, 1300), GREEN, width=7)

    fusion = (3050, 1020, 3670, 2070)
    label(
        draw,
        fusion,
        "D. Geometry-guided 3D Appearance Field",
        "Surface Extraction\n80k-120k LiDAR/student surfels\n\nBrown-aware Projection\n3D center → 7 source views\n\nTop-K View Selection\nK=3, visibility + angle + baseline\n\nCross-View MHA\n4 heads, 96-D → 64-D token\n\nGaussian Parameter Heads\nfixed xyz / scale / rotation\nopacity / SH color / latent 32-D\nsource provenance + validity",
        PURPLE_LIGHT,
        PURPLE,
        title_size=36,
        body_size=27,
    )
    arrow(draw, (1350, 1690), (3050, 1390), BLUE, width=7)
    arrow(draw, (2060, 2190), (3050, 1870), GREEN, width=7)

    renderer = (3770, 1080, 4250, 1995)
    label(
        draw,
        renderer,
        "E. Differentiable Renderer",
        "Brown Projection Wrapper\n(K_t,D_t,T_t)\n\ngsplat CUDA rasterizer\nsoft z-buffer\nfront-surface compositing\n\nRendered tensors\nRGB 3\nfeature 32\ndepth 1 / normal 3\nopacity 1\nstructure validity 1\nappearance validity 1\nuncertainty 1",
        AMBER_LIGHT,
        AMBER,
        title_size=36,
        body_size=27,
    )
    arrow(draw, (3670, 1520), (3770, 1520), AMBER, width=7)
    tensor_tag(draw, (3830, 2020), "target camera K_t / D_t / T_t", PURPLE)

    draw_restormer(draw, (4260, 1130))
    arrow(draw, (4250, 1520), (4260, 1450), PURPLE, width=7)
    target_path = sequence_dir / "Key_frames" / target_camera / payload["sensors"][target_camera]
    target = fit_image(target_path, (350, 197))
    depth = sparse_depth_preview(points, calibrations[target_camera], (260, 197))
    paste_framed(canvas, draw, target, (4410, 2035), "Output RGB / GT reference", RED, RED_LIGHT, RED)
    paste_framed(canvas, draw, depth, (4790, 2035), "metric depth / validity", BLUE, BLUE_LIGHT, BLUE)


def draw_training_band(draw: ImageDraw.ImageDraw) -> None:
    draw.line((55, 2450, 5145, 2450), fill="#C9D3E0", width=4)
    draw.text((65, 2480), "F. Stage-wise Training, Losses, Tests and Artifacts", font=font(44, bold=True), fill=INK)
    blocks = [
        ((65, 2560, 1020, 3215), "Phase 0 · Data Contract", "真实 timestamp 排序\ntime 字段验证与 deskew\nKISS-ICP / gsplat / spconv smoke\n删除 color=0.5 sentinel\n拆分 structure/appearance/unknown", "#EEF2F7", "#7D8EA6"),
        ((1080, 2560, 2040, 3215), "Phase 1-2 · Geometry", "Temporal teacher：无网络\nSparse U-Net student：单序列过拟合\n\nLoss\nexact hard constraint\nteacher inverse-depth L1\noccupancy BCE\nnormal cosine\nconfidence NLL", BLUE_LIGHT, BLUE),
        ((2100, 2560, 3060, 3215), "Phase 3 · Appearance", "Frozen DINOv2-B\nDPT/FPN + NAF trainable\nCross-View MHA + Gaussian heads\n\nLoss\nobserved Charbonnier / SSIM\nDINO feature consistency\nprovenance / validity", GREEN_LIGHT, GREEN),
        ((3120, 2560, 4080, 3215), "Phase 4 · Rendering", "gsplat + Restormer-Small\n先 observed residual，后 completion\n\nLoss\nLPIPS only supervised holes\nuncertainty calibration\nprotected exact reconstruction\nper-camera non-regression", PURPLE_LIGHT, PURPLE),
        ((4140, 2560, 5135, 3215), "Required Outputs", "checkpoints：geometry / appearance / renderer\nmetrics：LiDAR holdout + RGB regions\nvisuals：RGB / depth / validity / provenance\n\nTests\ntimestamp monotonicity\ndeskew synthetic motion\nICP SE(3) recovery\nz-buffer front surface\nno RGB→depth gradient path", RED_LIGHT, RED),
    ]
    for xy, title, body, fill, outline in blocks:
        label(draw, xy, title, body, fill, outline, title_size=34, body_size=26)
    for left, right in zip(blocks, blocks[1:]):
        arrow(draw, (left[0][2], 2880), (right[0][0], 2880), INK, width=6)


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
        default=Path("outputs/paper_figures/gcr_nvs_network_architecture_v2.png"),
    )
    args = parser.parse_args()

    sequence_dir = args.root.resolve() / args.sequence
    config_path = sequence_dir / "Key_frames/camera_config" / f"{args.frame_id:06d}.json"
    payload = json.loads(config_path.read_text())
    lidar_path = sequence_dir / "Key_frames/LIDAR_CONCAT" / payload["sensors"]["LIDAR_CONCAT"]
    points = read_pcd(lidar_path)
    calibrations = load_calibrations(config_path, args.distortion)

    canvas = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 38), "GCR-NVS v2：Temporal-LiDAR Teacher + Sparse Geometry Student + DINOv2 Gaussian Renderer", font=font(64, bold=True), fill=INK)
    draw.text(
        (70, 116),
        "Block-level method figure  |  单帧正式推理  |  LiDAR 决定公制结构，RGB 决定三维外观",
        font=font(30),
        fill=MUTED,
    )
    pill(draw, (4300, 55, 5100, 125), "No RGB-to-depth fallback", RED_LIGHT, RED, RED, 29)

    draw_teacher_strip(draw, canvas, points)
    draw_input_panel(draw, canvas, sequence_dir, payload, points)
    draw_main_network(draw, canvas, sequence_dir, payload, calibrations, points, args.target_camera)
    draw_training_band(draw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    preview_path = args.output.with_name(args.output.stem + "_preview.png")
    canvas.resize((2600, 1650), RESAMPLE_LANCZOS).save(preview_path, optimize=True)
    print(json.dumps({
        "output": str(args.output),
        "preview": str(preview_path),
        "canvas": list(CANVAS),
        "architecture": {
            "geometry_teacher": "KISS-ICP temporal surfel fusion",
            "geometry_student": "spconv Sparse U-Net 32/64/128/256",
            "rgb_backbone": "frozen DINOv2 ViT-B/14 + NAF detail CNN",
            "view_fusion": "Top-K=3 Cross-View MHA, 4 heads",
            "renderer": "Brown wrapper + gsplat",
            "decoder": "Restormer-Small 48/96/192/384",
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
