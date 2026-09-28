"""Build native-resolution T0 reconstruction frames for a real sequence.

The existing high-resolution T0 auditor is intentionally kept unchanged as
the reference implementation. This driver invokes it one target camera/frame
at a time to keep peak memory bounded, then retains only the reconstructed RGB
frame needed by the video renderer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest


POSES = ("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm")


def _native_size(sequence_root: Path, record, camera: str) -> tuple[int, int]:
    with Image.open(sequence_root / record.cameras[camera]) as image:
        return int(image.width), int(image.height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distortion", type=Path, required=True)
    parser.add_argument("--anyup-checkpoint", type=Path, required=True)
    parser.add_argument("--anyup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-list", type=int, nargs="+", default=None,
                        help="Explicit frame IDs with complete DA3/teacher caches.")
    parser.add_argument("--camera", action="append", choices=CAMERA_NAMES, default=None)
    parser.add_argument("--surface-repair-radius", type=float, default=0.0)
    parser.add_argument("--compute-max-width", type=int, default=1920)
    parser.add_argument("--compute-max-height", type=int, default=1080)
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--keep-temporary", action="store_true")
    args = parser.parse_args()
    if args.output_width <= 0 or args.output_height <= 0:
        parser.error("output dimensions must be positive")

    sequence_dir = args.sequence_root / args.sequence
    records = build_sequence_manifest(sequence_dir, compute_quality=False)
    selected = records
    if args.frame_start is not None:
        selected = [record for record in selected if record.frame_id >= args.frame_start]
    if args.frame_end is not None:
        selected = [record for record in selected if record.frame_id <= args.frame_end]
    if args.frame_list is not None:
        wanted = set(args.frame_list)
        selected = [record for record in selected if record.frame_id in wanted]
    if not selected:
        raise ValueError("no frames selected")
    cameras = args.camera or list(CAMERA_NAMES)
    python = sys.executable
    renderer = ROOT / "scripts/render_t0_highres_source_audit.py"
    temp_root = args.output / ".temporary_frames"
    rows = []
    for index, record in enumerate(selected, 1):
        # At the normalized 1920x1080 raster the renderer can process all
        # seven target cameras in one process. This avoids reloading DINO and
        # the structure checkpoint once per camera while keeping one frame as
        # the memory boundary.
        camera_batches = [cameras] if len(cameras) > 1 else [[cameras[0]]]
        for camera_batch in camera_batches:
            camera = camera_batch[0]
            width, height = _native_size(sequence_dir, record, camera)
            # All cameras, including the native 3840x2160 narrow/wide views,
            # are deliberately normalized to one video raster.
            native_width, native_height = args.output_width, args.output_height
            scale = min(1.0, args.compute_max_width / width, args.compute_max_height / height)
            compute_width = max(2, int(round(width * scale)))
            compute_height = max(2, int(round(height * scale)))
            destination_root = args.output / "frames" / args.sequence / camera
            destination_root.mkdir(parents=True, exist_ok=True)
            complete = all((args.output / "frames" / args.sequence / name / pose / f"{record.frame_id:06d}.jpg").exists() for name in camera_batch for pose in POSES)
            if complete:
                rows.append({"frame_id": record.frame_id, "camera": camera, "reused": True})
                continue
            temporary = temp_root / f"{record.frame_id:06d}_{camera}"
            if temporary.exists():
                shutil.rmtree(temporary)
            command = [
                python, str(renderer),
                "--bundle", str(args.bundle),
                "--root", str(args.sequence_root),
                "--sequence", args.sequence,
                "--frame", str(record.frame_id),
                "--target-camera", "all" if len(camera_batch) > 1 else camera,
                "--target-cameras", *camera_batch,
                "--output", str(temporary),
                "--cache-root", str(args.cache_root),
                "--teacher-root", str(args.teacher_root),
                "--dynamic-root", str(args.dynamic_root),
                "--checkpoint", str(args.checkpoint),
                "--distortion", str(args.distortion),
                "--width", str(compute_width), "--height", str(compute_height),
                "--device", args.device,
                "--depth-mode", "linear",
                "--surface-repair-radius", str(args.surface_repair_radius),
            ]
            environment = dict(__import__("os").environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            environment["GCR_NVS_ANYUP_CHECKPOINT"] = str(args.anyup_checkpoint)
            environment["GCR_NVS_ANYUP_ROOT"] = str(args.anyup_root)
            subprocess.run(command, check=True, env=environment)
            for target_camera in camera_batch:
                target_width, target_height = args.output_width, args.output_height
                target_destination_root = args.output / "frames" / args.sequence / target_camera
                for pose in POSES:
                    source = temporary / pose / target_camera / "t0_highres_blue.png"
                    if not source.exists():
                        raise FileNotFoundError(f"missing T0 output: {source}")
                    pose_destination = target_destination_root / pose
                    pose_destination.mkdir(parents=True, exist_ok=True)
                    with Image.open(source) as image:
                        result = image.convert("RGB")
                        if result.size != (target_width, target_height):
                            result = result.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        result.save(pose_destination / f"{record.frame_id:06d}.jpg", quality=95, subsampling=0)
                rows.append({
                "frame_id": record.frame_id,
                "camera": target_camera,
                "resolution": [target_width, target_height],
                "compute_resolution": [compute_width, compute_height],
                "reused": False,
                })
            if not args.keep_temporary:
                shutil.rmtree(temporary, ignore_errors=True)
            print(json.dumps({"completed": len(rows), "frame_id": record.frame_id, "cameras": camera_batch}, ensure_ascii=False), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "sequence": args.sequence,
        "real_frames_requested": len(selected),
        "cameras": cameras,
        "poses": list(POSES),
        "rows": rows,
        "output_resolution": [args.output_width, args.output_height],
        "contract": "T0 reconstruction; every camera is normalized to the requested output raster before video encoding",
        "temporary_cleanup": not args.keep_temporary,
    }
    (args.output / "native_reconstruction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"frames": len(selected), "rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
