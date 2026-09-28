"""Compare source RGB, the validated T0 cache, and the new T0 cache.

Every group uses exactly one sequence/frame and one target pose for all seven
cameras.  The old and new renderers therefore differ only in their depth
cache, never in the sampled data or pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gcr_nvs.datasets.manifest import CAMERA_NAMES
from gcr_nvs.evaluation.render_dense_depth_pilot import render_target


RANGES = {
    "5_10cm": (0.05, 0.10),
    "10_30cm": (0.10, 0.30),
    "30_60cm": (0.30, 0.60),
}


def _complete_candidates(old_root: Path, new_root: Path) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for sequence_root in sorted(p for p in new_root.iterdir() if p.is_dir()):
        old_sequence_root = old_root / sequence_root.name
        for frame_root in sorted(p for p in sequence_root.iterdir() if p.is_dir() and p.name.isdigit()):
            old_frame_root = old_sequence_root / frame_root.name
            if all(
                (frame_root / camera / "report.json").exists()
                and (old_frame_root / camera / "report.json").exists()
                for camera in CAMERA_NAMES
            ):
                candidates.append((sequence_root.name, int(frame_root.name)))
    return candidates


def _pose(rng: np.random.Generator, bounds: tuple[float, float]) -> tuple[float, float, float]:
    low, high = bounds
    values = rng.uniform(low, high, size=3)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=3)
    return tuple(float(np.round(value * sign, 4)) for value, sign in zip(values, signs))


def _contact(group_root: Path, group: str, pose: tuple[float, float, float]) -> Path:
    canvas = Image.new("RGB", (1440, 7 * 300), "white")
    draw = ImageDraw.Draw(canvas)
    cameras = list(CAMERA_NAMES)
    for index, camera in enumerate(cameras):
        row = index * 300
        names = ("source_rgb_rectified.png", "reconstruction_rgb.png", "target_rectified.png")
        roots = (group_root / "source" / camera, group_root / "yesterday_t0" / camera, group_root / "today_t0" / camera)
        for column, (root, name) in enumerate(zip(roots, names)):
            image = Image.open(root / name).convert("RGB").resize((480, 270), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * 480, row + 25))
        draw.text((5, row + 4), f"{camera}   source | yesterday T0 | today T0", fill="black")
    output = group_root / f"{group}_source_yesterday_today_contact.jpg"
    canvas.save(output, quality=95)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/dense_route_dataset_v1"))
    parser.add_argument("--old-cache", type=Path, required=True)
    parser.add_argument("--new-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-range", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--frame-split", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    candidates = _complete_candidates(args.old_cache, args.new_cache)
    if len(candidates) < len(RANGES) * args.groups_per_range:
        raise RuntimeError(f"only {len(candidates)} complete old/new candidates, need {len(RANGES) * args.groups_per_range}")
    selected_indices = rng.choice(len(candidates), size=len(RANGES) * args.groups_per_range, replace=False)
    selected = [candidates[int(index)] for index in selected_indices]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "groups_per_range": args.groups_per_range, "candidate_count": len(candidates), "groups": []}
    cursor = 0
    for range_name, bounds in RANGES.items():
        for group_index in range(args.groups_per_range):
            sequence, frame_id = selected[cursor]
            cursor += 1
            pose = _pose(rng, bounds)
            group = f"{range_name}_group{group_index + 1:02d}"
            group_root = args.output / group
            rows = []
            for camera in CAMERA_NAMES:
                old_output = group_root / "yesterday_t0" / camera
                new_output = group_root / "today_t0" / camera
                old_report = render_target(
                    args.root, args.old_cache, sequence, frame_id, camera, pose,
                    Path("camera_intric.yaml"), old_output, splat_radius=0, color_invalid=False,
                )
                new_report = render_target(
                    args.root, args.new_cache, sequence, frame_id, camera, pose,
                    Path("camera_intric.yaml"), new_output, splat_radius=0, color_invalid=False,
                )
                rows.append({"camera": camera, "yesterday": old_report, "today": new_report})
                # The source image is identical for both renderers; keep one
                # copy next to the group so the contact sheet has a clear lane.
                source_output = group_root / "source" / camera
                source_output.mkdir(parents=True, exist_ok=True)
                source_path = old_output / "source_rgb_rectified.png"
                (source_output / "source_rgb_rectified.png").write_bytes(source_path.read_bytes())
                (source_output / "target_rectified.png").write_bytes((old_output / "target_rectified.png").read_bytes())
            contact = _contact(group_root, group, pose)
            manifest["groups"].append({
                "group": group,
                "range_m": list(bounds),
                "sequence": sequence,
                "frame_id": frame_id,
                "translation_xyz_m": list(pose),
                "contact_sheet": str(contact),
                "cameras": rows,
            })
            print(json.dumps({"group": group, "sequence": sequence, "frame_id": frame_id, "translation_xyz_m": pose, "contact": str(contact)}), flush=True)

    (args.output / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
