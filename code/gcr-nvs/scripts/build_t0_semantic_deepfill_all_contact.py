"""Build seven-view contacts and verify the complete latest-T0 batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)
POSES = ("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm")


def contact(root: Path, pose: str) -> None:
    width, height, title = 960, 540, 38
    canvas = Image.new("RGB", (width * 3, (height + title) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    for index, camera in enumerate(CAMERAS):
        image = Image.open(root / pose / camera / "t0_semantic_deepfill_v2_pilot.png").convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        x = index % 3 * width
        y = index // 3 * (height + title)
        draw.text((x + 10, y + 6), camera, fill="black", font=font)
        canvas.paste(image, (x, y + title))
    canvas.save(root / f"{pose}_semantic_deepfill_v2_seven_view.png", compress_level=2)


def main() -> None:
    bundle = Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823")
    audit_root = bundle / "runs/t0_highres_source_audit_edge_all"
    root = bundle / "runs/t0_semantic_deepfill_v2_latest_t0_all"
    rows = []
    for pose in POSES:
        contact(root, pose)
        for camera in CAMERAS:
            audit = audit_root / pose / camera
            output = root / pose / camera
            original = np.asarray(Image.open(audit / "t0_highres_blue.png").convert("RGB"))
            validity = np.load(audit / "validity.npy").astype(bool)
            final = np.asarray(Image.open(output / "t0_semantic_deepfill_v2_pilot.png").convert("RGB"))
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            difference = np.abs(final.astype(np.int16) - original.astype(np.int16))
            rows.append({
                "pose": pose, "camera": camera,
                "coverage": float(validity.mean()),
                "original_hole_fraction": float((~validity).mean()),
                "generated_fraction": report["generated_fraction"],
                "observed_max_abs_diff": int(difference[validity].max(initial=0)),
                "all_original_pixels_max_abs_diff": int(difference.max(initial=0)),
                "resolution": list(final.shape[1::-1]),
            })
    if any(row["observed_max_abs_diff"] != 0 for row in rows):
        raise RuntimeError("at least one latest-T0 observed pixel was modified")
    summary = {
        "input": str(audit_root),
        "output": str(root),
        "cases": len(rows),
        "expected_cases": 21,
        "all_observed_pixels_unchanged": True,
        "rows": rows,
    }
    (root / "batch_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "cases": len(rows),
        "all_observed_pixels_unchanged": True,
        "mean_generated_fraction": float(np.mean([row["generated_fraction"] for row in rows])),
        "outputs": str(root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
