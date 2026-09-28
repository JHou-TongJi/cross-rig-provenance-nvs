"""Run the complete submission route and emit final Hybrid RGB results.

The route is intentionally explicit: T0 observed RGB -> DA3/UniWorld
background gate -> RGB-D-S completion on the residual hole only.  Each stage
keeps its own artifacts and timing report so the final image is auditable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def _run(command: list[str], env: dict[str, str] | None = None) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    return time.perf_counter() - started


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="T0 + UniWorld gate + RGB-D-S final Hybrid inference")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--target-rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distortion", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-a", type=Path, required=True)
    parser.add_argument("--anyup-checkpoint", type=Path)
    parser.add_argument("--anyup-root", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--completion-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-steps", type=int, default=20)
    args = parser.parse_args()
    output = args.output
    audit = output / "t0_surface"
    diagnostic = output / "background_layers"
    gate = output / "background_gate"
    final = output / "final_hybrid"
    output.mkdir(parents=True, exist_ok=True)
    common = [
        "--root", str(args.source_root), "--sequence", args.sequence,
        "--frame", str(args.frame), "--target-rig", str(args.target_rig),
        "--cache-root", str(args.cache_root), "--teacher-root", str(args.teacher_root),
        "--dynamic-root", str(args.dynamic_root), "--semantic-cache", str(args.semantic_cache),
        "--checkpoint", str(args.checkpoint), "--distortion", str(args.distortion),
        "--width", str(args.width), "--height", str(args.height), "--device", args.device,
        "--depth-mode", "edge",
    ]
    timings = {}
    timings["t0_surface_s"] = _run([
        sys.executable, str(project / "scripts/render_t0_highres_source_audit.py"),
        *common, "--output", str(audit),
    ])
    timings["background_diagnostic_s"] = _run([
        sys.executable, str(project / "scripts/diagnose_t0_background_layers.py"),
        "--bundle", str(args.source_root), "--source-root", str(args.source_root),
        "--audit-root", str(audit), "--repair-root", str(audit),
        "--target-rig", str(args.target_rig), "--distortion", str(args.distortion), "--output", str(diagnostic),
        "--scale", "0.25", "--depth-layer", "da3",
    ])
    gate_times = {}
    for camera in CAMERAS:
        gate_times[camera] = _run([
            sys.executable, str(project / "scripts/apply_t0_uniview_background_gate.py"),
            "--t0-root", str(audit), "--gate-root", str(diagnostic),
            "--output", str(gate), "--pose", "target_rig", "--camera", camera,
        ])
    timings["background_gate_s"] = sum(gate_times.values())
    completion_env = None
    if args.anyup_checkpoint or args.anyup_root:
        import os
        completion_env = os.environ.copy()
        if args.anyup_checkpoint:
            completion_env["GCR_NVS_ANYUP_CHECKPOINT"] = str(args.anyup_checkpoint)
        if args.anyup_root:
            completion_env["GCR_NVS_ANYUP_ROOT"] = str(args.anyup_root)
    timings["rgbds_completion_s"] = _run([
        str(args.completion_python), str(project / "scripts/apply_t0_rgbds_hybrid_to_2px.py"),
        "--input", str(audit), "--prefilled-root", str(gate),
        "--model", str(args.model), "--adapter-a", str(args.adapter_a),
        "--route", "a", "--output", str(final), "--width", "512", "--height", "288",
        "--inference-steps", str(args.inference_steps), "--poses", "target_rig",
    ], env=completion_env)
    report = {
        "sequence": args.sequence,
        "frame": args.frame,
        "target_rig": str(args.target_rig),
        "output_resolution": [args.width, args.height],
        "final_root": str(final),
        "stages": {
            "t0_surface": str(audit),
            "background_layers": str(diagnostic),
            "background_gate": str(gate),
            "final_hybrid": str(final),
        },
        "timing_s": timings,
        "device": args.device,
        "completion_python": str(args.completion_python),
        "contract": "observed T0 and gate pixels immutable; RGB-D-S generates residual hole only",
    }
    (output / "final_inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
