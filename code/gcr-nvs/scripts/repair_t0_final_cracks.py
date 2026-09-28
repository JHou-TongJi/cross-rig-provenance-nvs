"""Repair only short raster cracks in a completed T0 Hybrid result.

Observed T0 pixels and UniWorld/DA3 gate pixels are immutable.  The mask is
restricted to residual generated pixels within a small distance of a locked
pixel, so this cannot turn a large disocclusion into an invented observation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--t0-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=2.0)
    args = parser.parse_args()
    started = time.perf_counter()
    reports = {}
    for camera in CAMERAS:
        src = args.final_root / "target_rig" / camera / "rgbds_a_4k.png"
        t0 = args.t0_root / "target_rig" / camera
        gate = args.gate_root / "target_rig" / camera
        image = np.asarray(Image.open(src).convert("RGB"))
        observed = np.load(t0 / "validity_after_2px_surface_repair.npy").astype(bool)
        gate_fill = np.asarray(Image.open(gate / "uniview_gate_fill_mask.png").convert("L")) > 127
        locked = observed | gate_fill
        residual = ~locked
        # Only thin residual components touching locked pixels are eligible.
        distance = cv2.distanceTransform(residual.astype(np.uint8), cv2.DIST_L2, 5)
        crack = residual & (distance <= float(args.radius))
        # A small morphological opening removes isolated one-pixel noise from
        # the repair mask without expanding it into true disocclusion regions.
        crack_u8 = cv2.morphologyEx(crack.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        crack = crack_u8.astype(bool) & residual
        repaired = cv2.inpaint(image, np.uint8(crack), 2.0, cv2.INPAINT_TELEA)
        repaired[locked] = image[locked]
        out = args.output / "target_rig" / camera
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(repaired).save(out / "rgbds_a_4k_repaired.png", quality=95, subsampling=0)
        Image.fromarray(np.uint8(crack * 255)).save(out / "opencv_crack_repair_mask.png")
        report = {
            "camera": camera,
            "resolution": [int(image.shape[1]), int(image.shape[0])],
            "radius_px": args.radius,
            "locked_fraction": float(locked.mean()),
            "crack_repaired_fraction": float(crack.mean()),
            "locked_pixels_modified": int(np.any(repaired[locked] != image[locked], axis=1).sum()),
            "method": "OpenCV TELEA inpaint on residual generated pixels within radius; observed/gate pixels immutable",
        }
        (out / "opencv_crack_repair_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[camera] = report
    summary = {"reports": reports, "timing_s": {"opencv_crack_repair_total_s": time.perf_counter() - started}, "contract": "hole-only 2px crack cleanup"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "opencv_crack_repair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
