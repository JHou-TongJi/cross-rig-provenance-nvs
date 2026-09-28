"""Public inference entry point for the frozen T0/DA3 reconstruction route.

The caller supplies a source sequence frame and a target camera rig YAML. The
command loads frozen caches/checkpoints, performs rectified calibrated
surface inference, applies the target SE(3)/K/D contract, and writes RGB,
depth, validity, provenance and an audit summary. No training module is
imported by this entry point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end T0 inference from source RGB/LiDAR and target camera parameters")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--target-rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path)
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument("--distortion", type=Path, required=True)
    parser.add_argument("--anyup-checkpoint", type=Path, required=True)
    parser.add_argument("--anyup-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-mode", choices=("linear", "edge"), default="edge")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    render = Path(__file__).with_name("render_t0_highres_source_audit.py")
    command = [
        sys.executable, str(render), "--bundle", str(Path(__file__).resolve().parents[1]),
        "--root", str(args.source_root), "--sequence", args.sequence, "--frame", str(args.frame),
        "--target-rig", str(args.target_rig), "--target-camera", "all", "--output", str(args.output),
        "--cache-root", str(args.cache_root), "--teacher-root", str(args.teacher_root),
        "--dynamic-root", str(args.dynamic_root), "--semantic-cache", str(args.semantic_cache or args.output / "semantic_cache"),
        "--checkpoint", str(args.structure_checkpoint), "--distortion", str(args.distortion),
        "--width", str(args.width), "--height", str(args.height), "--device", args.device,
        "--depth-mode", args.depth_mode,
    ]
    # The target source encoder loads AnyUP from the project default in the
    # historical renderer; expose explicit paths through environment variables
    # for deployment wrappers and record them in the output manifest.
    import os
    env = os.environ.copy()
    env["GCR_NVS_ANYUP_CHECKPOINT"] = str(args.anyup_checkpoint)
    env["GCR_NVS_ANYUP_ROOT"] = str(args.anyup_root)
    result = subprocess.run(command, env=env, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
