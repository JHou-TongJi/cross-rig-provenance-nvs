"""Audit frame/camera/image/calibration alignment for one Unified3DField sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, rectify_image, project_world_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--target-camera", required=True, choices=CAMERA_NAMES)
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sequence_dir = args.root / args.sequence
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    record = next(row for row in records if row.frame_id == args.frame_id)
    calibrations = load_calibrations(
        sequence_dir / record.camera_config, args.distortion,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = []
    report = {
        "sequence_id": args.sequence,
        "frame_id": args.frame_id,
        "camera_config": record.camera_config,
        "lidar": record.lidar,
        "cameras": {},
    }
    # Use the exact same rectification path as the training dataset, then
    # annotate every tile so a calibration-order error is immediately visible.
    for name in CAMERA_NAMES:
        path = sequence_dir / record.cameras[name]
        raw = np.asarray(Image.open(path).convert("RGB"))
        rectified, calibration = rectify_image(raw, calibrations[name], (480, 270))
        image = Image.fromarray(rectified)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 220, 26), fill=(0, 0, 0))
        draw.text((6, 6), name, fill=(255, 255, 0))
        tiles.append(image)
        report["cameras"][name] = {
            "path": str(path),
            "raw_size": [int(raw.shape[1]), int(raw.shape[0])],
            "rectified_size": [calibration.width, calibration.height],
            "intrinsic": calibration.intrinsic.tolist(),
            "external": calibration.external.tolist(),
            "camera_center": calibration.camera_center.tolist(),
        }
    sheet = Image.new("RGB", (960, 1080), (20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 480, (index // 2) * 270))
    sheet.save(args.output / "rectified_camera_contact_sheet.jpg", quality=95)

    # Check the measured current-frame points against every camera using the
    # same calibration objects.  This is independent of the neural renderer.
    pcd_path = sequence_dir / record.lidar
    try:
        from gcr_nvs.geometry.pcd import read_pcd
        points = read_pcd(pcd_path, fields=("x", "y", "z"))
        for name in CAMERA_NAMES:
            _, valid = project_world_points(points[:, :3], calibrations[name])
            report["cameras"][name]["current_lidar_projection_rate"] = float(valid.mean())
    except Exception as exc:  # diagnostics should retain image audit if PCD fields vary
        report["lidar_projection_error"] = f"{type(exc).__name__}: {exc}"

    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
