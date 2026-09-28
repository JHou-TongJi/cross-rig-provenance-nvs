"""Map a rectified target result back to the target camera's raw K/D grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.geometry.calibration import load_calibrations
from gcr_nvs.geometry.rig import load_target_rig, resolve_target_calibration


def _raw_to_rectified_map(raw, rectified):
    yy, xx = np.indices((raw.height, raw.width), dtype=np.float32)
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    rectified_points = cv2.undistortPoints(
        points.reshape(-1, 1, 2), raw.intrinsic, raw.distortion,
        R=np.eye(3), P=rectified.intrinsic,
    ).reshape(raw.height, raw.width, 2)
    return rectified_points[..., 0].astype(np.float32), rectified_points[..., 1].astype(np.float32)


def _map(array, map_x, map_y, interpolation):
    return cv2.remap(np.asarray(array), map_x, map_y, interpolation=interpolation,
                     borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distort rectified T0 output into target raw camera coordinates")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--target-rig", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Root containing target_rig/<camera>/<image-name>")
    parser.add_argument("--image-name", default="t0_highres_blue.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distortion", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    sequence_dir = args.source_root / args.sequence
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    record = next(row for row in records if row.frame_id == args.frame)
    source_calibrations = load_calibrations(sequence_dir / record.camera_config, args.distortion)
    rig = load_target_rig(args.target_rig)
    reports = {}
    for spec in rig.cameras:
        rectified = resolve_target_calibration(spec, source_calibrations)
        raw = resolve_target_calibration(spec, source_calibrations)
        # resolve_target_calibration returns the raw K/D contract. The T0
        # renderer rectifies it with the same K/D and stores the result raster.
        if not np.any(raw.distortion):
            rectified = raw
        else:
            from gcr_nvs.geometry.calibration import rectified_calibration
            rectified = rectified_calibration(raw, output_size=(spec.width, spec.height), alpha=0.0)
        source = args.input_root / "target_rig" / spec.name / args.image_name
        if not source.exists():
            raise FileNotFoundError(source)
        image = np.asarray(Image.open(source).convert("RGB"))
        if image.shape[:2] != (rectified.height, rectified.width):
            image = cv2.resize(image, (rectified.width, rectified.height), interpolation=cv2.INTER_LANCZOS4)
        map_x, map_y = _raw_to_rectified_map(raw, rectified)
        distorted = _map(image, map_x, map_y, cv2.INTER_LANCZOS4)
        output_dir = args.output / "target_rig" / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(distorted).save(output_dir / "final_distorted.png", quality=95, subsampling=0)
        reports[spec.name] = {
            "input": str(source), "output": str(output_dir / "final_distorted.png"),
            "resolution": [spec.width, spec.height], "raw_distortion": raw.distortion.tolist(),
            "contract": "rectified result -> target raw K/D coordinates; border replicate only for inverse-map outside pixels",
        }
    summary = {
        "sequence": args.sequence, "frame": args.frame,
        "target_rig": str(args.target_rig), "input_image_name": args.image_name,
        "reports": reports, "timing_s": {"distortion_mapping_total_s": time.perf_counter() - started},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "distortion_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
