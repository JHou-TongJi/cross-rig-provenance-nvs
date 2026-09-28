"""Reproduce the released T0 video from a sequence and external caches.

The default path renders deterministic T0 reconstruction anchors and encodes
them at the requested FPS. ``--with-a-only`` is opt-in and requires the
optional diffusion environment plus the frozen adapter checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("CAM_BACK", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(ROOT / "src")
    if env:
        merged.update(env)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce T0 reconstruction video artifacts")
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distortion", type=Path, required=True)
    parser.add_argument("--anyup-checkpoint", type=Path, required=True)
    parser.add_argument("--anyup-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pose", choices=("mixed_5_10cm", "mixed_10_20cm", "mixed_20_50cm"), default="mixed_10_20cm")
    parser.add_argument("--fps", type=float, choices=(120.0, 240.0), default=120.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--with-a-only", action="store_true", help="also run optional RGB-D-S A-only completion")
    parser.add_argument("--model", type=Path, help="Stable Diffusion inpainting directory; required with --with-a-only")
    parser.add_argument("--adapter-a", type=Path, help="frozen A-only adapter checkpoint; required with --with-a-only")
    parser.add_argument("--completion-python", default=sys.executable)
    args = parser.parse_args()
    if args.with_a_only and (args.model is None or args.adapter_a is None):
        parser.error("--with-a-only requires --model and --adapter-a")

    native = args.output_root / "native_t0"
    native_cmd = [
        sys.executable, str(ROOT / "scripts/run_t0_native_reconstruction_video.py"),
        "--sequence-root", str(args.sequence_root), "--sequence", args.sequence,
        "--cache-root", str(args.cache_root), "--teacher-root", str(args.teacher_root),
        "--dynamic-root", str(args.dynamic_root), "--bundle", str(ROOT),
        "--checkpoint", str(args.checkpoint), "--distortion", str(args.distortion),
        "--anyup-checkpoint", str(args.anyup_checkpoint), "--anyup-root", str(args.anyup_root),
        "--output", str(native), "--device", args.device,
        "--output-width", "1920", "--output-height", "1080",
    ]
    if args.frame_start is not None:
        native_cmd += ["--frame-start", str(args.frame_start)]
    if args.frame_end is not None:
        native_cmd += ["--frame-end", str(args.frame_end)]
    for camera in CAMERAS:
        native_cmd += ["--camera", camera]
    run(native_cmd, env={"GCR_NVS_ANYUP_CHECKPOINT": str(args.anyup_checkpoint), "GCR_NVS_ANYUP_ROOT": str(args.anyup_root)})

    if args.with_a_only:
        a_only_root = args.output_root / "a_only"
        a_only_cmd = [
            args.completion_python, str(ROOT / "scripts/render_t0_a_only_video_frames.py"),
            "--sequence-root", str(args.sequence_root), "--sequence", args.sequence,
            "--cache-root", str(args.cache_root), "--teacher-root", str(args.teacher_root),
            "--dynamic-root", str(args.dynamic_root), "--bundle", str(ROOT),
            "--checkpoint", str(args.checkpoint), "--distortion", str(args.distortion),
            "--anyup-checkpoint", str(args.anyup_checkpoint), "--anyup-root", str(args.anyup_root),
            "--model", str(args.model), "--adapter-a", str(args.adapter_a),
            "--output", str(a_only_root), "--device", args.device,
            "--poses", args.pose,
        ]
        run(a_only_cmd)
        encoded = args.output_root / "comparison"
        run([
            sys.executable, str(ROOT / "scripts/render_t0_video_demo.py"),
            "--sequence", str(args.sequence_root / args.sequence),
            "--anchor-root", str(a_only_root / "anchor_work"),
            "--a-only-root", str(a_only_root / "frames"),
            "--output-root", str(encoded), "--pose", args.pose,
            "--fps", str(int(args.fps)), "--width", "1920", "--height", "1080",
            *sum((["--camera", camera] for camera in CAMERAS), []),
        ])
    else:
        encoded = args.output_root / "t0_video"
        run([
            sys.executable, str(ROOT / "scripts/render_t0_reconstruction_videos.py"),
            "--sequence", str(args.sequence_root / args.sequence),
            "--reconstruction-root", str(native), "--output-root", str(encoded),
            "--pose", args.pose, "--fps", str(args.fps),
            "--width", "1920", "--height", "1080",
            *sum((["--camera", camera] for camera in CAMERAS), []),
        ])

    print(f"Video artifacts written to {encoded / args.pose}")


if __name__ == "__main__":
    main()
