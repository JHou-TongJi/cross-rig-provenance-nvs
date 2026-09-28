"""Report exact and temporal LiDAR image coverage for every camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations, project_world_points
from gcr_nvs.geometry.temporal_lidar import EXACT_CURRENT


def _pixel_mask(points, calibration, width: int, height: int) -> np.ndarray:
    uv, valid = project_world_points(points[:, :3], calibration)
    scale = np.asarray([
        width / calibration.width,
        height / calibration.height,
    ], dtype=np.float64)
    pixels = np.floor(uv[valid] * scale).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    mask = np.zeros((height, width), dtype=bool)
    mask[pixels[:, 1], pixels[:, 0]] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--distortion", type=Path, default=Path("camera_intric.yaml"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = build_sequence_manifest(
        args.root / args.sequence, compute_quality=False,
    )
    record = next(row for row in records if row.frame_id == args.frame_id)
    calibrations = load_calibrations(
        args.root / args.sequence / record.camera_config,
        args.distortion,
    )
    with np.load(args.artifact) as payload:
        points = payload["points"]
        source_kind = payload["source_kind"]
    exact_points = points[source_kind == EXACT_CURRENT]
    temporal_points = points[source_kind != EXACT_CURRENT]

    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for camera in CAMERA_NAMES:
        exact = _pixel_mask(
            exact_points, calibrations[camera], args.width, args.height,
        )
        temporal = _pixel_mask(
            temporal_points, calibrations[camera], args.width, args.height,
        )
        added = temporal & ~exact
        fused = exact | temporal
        image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        image[exact] = (230, 70, 55)
        image[added] = (45, 205, 220)
        Image.fromarray(image).save(args.output / f"{camera}.png")
        rows[camera] = {
            "exact_coverage": float(exact.mean()),
            "temporal_coverage": float(temporal.mean()),
            "temporal_added_coverage": float(added.mean()),
            "fused_coverage": float(fused.mean()),
        }

    report = {
        "artifact": str(args.artifact),
        "sequence_id": args.sequence,
        "frame_id": args.frame_id,
        "resolution": [args.width, args.height],
        "exact_points": int(len(exact_points)),
        "temporal_points": int(len(temporal_points)),
        "per_camera": rows,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
