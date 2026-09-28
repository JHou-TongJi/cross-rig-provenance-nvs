"""Render LiDAR overlays on explicitly rectified images for all seven cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.geometry.calibration import load_calibrations, project_world_points, rectify_image
from gcr_nvs.geometry.pcd import read_pcd


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def color_depth(depth: np.ndarray) -> np.ndarray:
    normalized = np.clip((depth - 2.0) / 70.0, 0.0, 1.0)
    return np.stack([
        np.clip(1.6 * normalized, 0, 1),
        np.clip(1.8 - np.abs(normalized * 3.0 - 1.2), 0, 1),
        np.clip(1.4 * (1.0 - normalized), 0, 1),
    ], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", default="2026-05-26-11-44-28")
    parser.add_argument("--frame-id", type=int, default=11)
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration_reports/rectified_lidar_alignment.png"))
    args = parser.parse_args()

    root = args.root.resolve()
    sequence = root / args.sequence
    config_path = sequence / "Key_frames/camera_config" / f"{args.frame_id:06d}.json"
    payload = json.loads(config_path.read_text())
    calibrations = load_calibrations(config_path, args.distortion)
    lidar_path = sequence / "Key_frames/LIDAR_CONCAT" / payload["sensors"]["LIDAR_CONCAT"]
    points = read_pcd(lidar_path)
    tile_w, tile_h = args.width, args.height + 52
    canvas = Image.new("RGB", (tile_w * 4, tile_h * 2 + 70), "#F5F7FA")
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype(str(FONT), 20)
    bold = ImageFont.truetype(str(FONT_BOLD), 23)
    title = ImageFont.truetype(str(FONT_BOLD), 31)
    draw.text((22, 16), "Rectified RGB + LiDAR projection (K_rect, D=0)", font=title, fill="#172033")
    rows = []
    for camera_index, name in enumerate(CAMERA_NAMES):
        calibration = calibrations[name]
        image_path = sequence / "Key_frames" / name / payload["sensors"][name]
        with Image.open(image_path) as source:
            raw = np.asarray(source.convert("RGB"))
        rectified, rectified_calibration = rectify_image(
            raw, calibration, (args.width, args.height), args.alpha,
        )
        uv, valid = project_world_points(points[:, :3], rectified_calibration)
        camera_xyz = (
            rectified_calibration.external
            @ np.c_[points[:, :3], np.ones(len(points))].T
        ).T[:, :3]
        indices = np.flatnonzero(valid)
        if len(indices) > 80_000:
            indices = indices[np.linspace(0, len(indices) - 1, 80_000).astype(np.int64)]
        order = indices[np.argsort(camera_xyz[indices, 2])[::-1]]
        overlay = rectified.copy()
        colors = (color_depth(camera_xyz[order, 2]) * 255).astype(np.uint8)
        for point_index, color in zip(order, colors):
            x, y = np.rint(uv[point_index]).astype(np.int32)
            cv2.circle(overlay, (int(x), int(y)), 1, tuple(map(int, color)), -1, lineType=cv2.LINE_AA)
        tile = Image.fromarray(overlay)
        x0 = (camera_index % 4) * tile_w
        y0 = 70 + (camera_index // 4) * tile_h
        canvas.paste(tile, (x0, y0))
        draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline="#7D8A9D", width=2)
        draw.text((x0 + 12, y0 + args.height + 9), name, font=bold, fill="#182033")
        draw.text(
            (x0 + 325, y0 + args.height + 12),
            f"visible {int(valid.sum()):,}  D={calibration.distortion[0]:+.3f},{calibration.distortion[1]:+.3f}",
            font=regular,
            fill="#526078",
        )
        rows.append({
            "camera": name,
            "visible_points": int(valid.sum()),
            "raw_size": [calibration.width, calibration.height],
            "rectified_size": [rectified_calibration.width, rectified_calibration.height],
            "raw_distortion": calibration.distortion.tolist(),
            "rectified_distortion": rectified_calibration.distortion.tolist(),
            "rectified_intrinsic": rectified_calibration.intrinsic.tolist(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({
        "sequence": args.sequence,
        "frame_id": args.frame_id,
        "mode": "rectified_pinhole",
        "distortion_config": str(args.distortion),
        "cameras": rows,
    }, indent=2))
    print(json.dumps({"output": str(args.output), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
