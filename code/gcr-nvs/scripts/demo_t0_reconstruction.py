"""Minimal, reproducible GCR-NVS demo.

Default: run the frozen T0/DA3/LiDAR route and keep unobserved pixels blue.
Optional flags add surface crack repair, UniWorld/DA3 background recovery,
RGB-D-S residual completion, and OpenCV cleanup.  Every stage is isolated in
its own directory and recorded in ``demo_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CAMERAS = (
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT",
    "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE",
)


def run(command: list[str], *, python: str, env: dict[str, str] | None = None) -> float:
    started = time.perf_counter()
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    if env:
        merged.update(env)
    result = subprocess.run([python, *command], env=merged, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    return time.perf_counter() - started


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Minimal T0 reconstruction demo")
    parser.add_argument("--source-root", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/incoming_sequences"))
    parser.add_argument("--sequence", default="2026-05-26-13-46-53")
    parser.add_argument("--frame", type=int, default=73)
    parser.add_argument("--target-rig", type=Path, default=project / "configs/target_rigs/l4_simulated_3700x1400x2000.yaml")
    parser.add_argument("--output", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/gcr_nvs_demo_frame73"))
    parser.add_argument("--cache-root", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/da3_depth_cache_dense_train_v1"))
    parser.add_argument("--teacher-root", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/geometry_teacher_full_v1"))
    parser.add_argument("--dynamic-root", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/dynamic_masks_full_v1"))
    parser.add_argument("--semantic-cache", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/semantic_cache_dinol_644_full188_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"))
    parser.add_argument("--distortion", type=Path, default=project / "camera_intric.yaml")
    parser.add_argument("--anyup-checkpoint", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/t0_baseline_isolated_20260823/outputs/checkpoints/anyup_multi_backbone.pth"))
    parser.add_argument("--anyup-root", type=Path, default=project / "third_party/anyup")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--surface-repair-radius", type=float, default=0.0, help="0 keeps the blue raster audit; 2 repairs only short target-surface cracks")
    parser.add_argument("--background-gate", action="store_true", help="recover residual holes with DA3/UniWorld triple reprojection")
    parser.add_argument("--rgbds-completion", action="store_true", help="run optional RGB-D-S completion on the residual hole")
    parser.add_argument("--opencv-crack-repair", action="store_true", help="repair 1-2px cracks after RGB-D-S; observed T0 pixels stay locked")
    parser.add_argument("--completion-python", default="/home/heqing/anaconda3/envs/comfyui/bin/python")
    parser.add_argument("--model", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/models/runwayml-stable-diffusion-inpainting"))
    parser.add_argument("--adapter-a", type=Path, default=Path("/media/2T_HD/GCR-NVS_DATA/outputs/checkpoints/t0_rgbds_adapter_a_256x64_v2/adapter_step_001000.pt"))
    args = parser.parse_args()
    if args.rgbds_completion and not args.background_gate:
        args.background_gate = True
    out = args.output
    t0 = out / "01_t0_surface"
    layers = out / "02_background_layers"
    gate = out / "03_background_gate"
    hybrid = out / "04_rgbds_completion"
    repaired = out / "05_opencv_repair"
    distorted = out / "06_distorted_final"
    timings = {}
    common = [
        str(project / "scripts/render_t0_highres_source_audit.py"),
        "--root", str(args.source_root), "--sequence", args.sequence, "--frame", str(args.frame),
        "--target-rig", str(args.target_rig), "--output", str(t0),
        "--cache-root", str(args.cache_root), "--teacher-root", str(args.teacher_root),
        "--dynamic-root", str(args.dynamic_root), "--semantic-cache", str(args.semantic_cache),
        "--checkpoint", str(args.checkpoint), "--distortion", str(args.distortion),
        "--width", str(args.width), "--height", str(args.height), "--device", args.device,
        "--depth-mode", "edge", "--surface-repair-radius", str(args.surface_repair_radius),
    ]
    timings["01_t0_surface_s"] = run(common, python=sys.executable, env={
        "GCR_NVS_ANYUP_CHECKPOINT": str(args.anyup_checkpoint), "GCR_NVS_ANYUP_ROOT": str(args.anyup_root),
    })
    stages = {"01_t0_surface": str(t0)}
    if args.background_gate:
        timings["02_background_layers_s"] = run([
            str(project / "scripts/diagnose_t0_background_layers.py"),
            "--bundle", str(args.source_root), "--source-root", str(args.source_root),
            "--audit-root", str(t0), "--repair-root", str(t0), "--target-rig", str(args.target_rig),
            "--distortion", str(args.distortion), "--output", str(layers), "--scale", "0.25", "--depth-layer", "da3",
        ], python=sys.executable)
        gate_start = time.perf_counter()
        for camera in CAMERAS:
            run([
                str(project / "scripts/apply_t0_uniview_background_gate.py"),
                "--t0-root", str(t0), "--gate-root", str(layers), "--output", str(gate),
                "--pose", "target_rig", "--camera", camera,
            ], python=sys.executable)
        timings["03_background_gate_s"] = time.perf_counter() - gate_start
        stages["02_background_layers"] = str(layers)
        stages["03_background_gate"] = str(gate)
    if args.rgbds_completion:
        timings["04_rgbds_completion_s"] = run([
            str(project / "scripts/apply_t0_rgbds_hybrid_to_2px.py"),
            "--input", str(t0), "--prefilled-root", str(gate), "--model", str(args.model),
            "--adapter-a", str(args.adapter_a), "--route", "a", "--output", str(hybrid),
            "--width", "512", "--height", "288", "--inference-steps", "20", "--poses", "target_rig",
        ], python=args.completion_python)
        stages["04_rgbds_completion"] = str(hybrid)
    if args.opencv_crack_repair:
        if not args.rgbds_completion:
            parser.error("--opencv-crack-repair requires --rgbds-completion")
        timings["05_opencv_repair_s"] = run([
            str(project / "scripts/repair_t0_final_cracks.py"), "--final-root", str(hybrid),
            "--t0-root", str(t0), "--gate-root", str(gate), "--output", str(repaired), "--radius", "2",
        ], python=sys.executable)
        stages["05_opencv_repair"] = str(repaired)
    if args.opencv_crack_repair:
        input_root, image_name = repaired, "rgbds_a_4k_repaired.png"
    elif args.rgbds_completion:
        input_root, image_name = hybrid, "rgbds_a_4k.png"
    elif args.background_gate:
        input_root, image_name = gate, "t0_uniview_background_gate.png"
    else:
        input_root, image_name = t0, "t0_highres_blue.png"
    timings["06_distortion_mapping_s"] = run([
        str(project / "scripts/distort_t0_demo_output.py"),
        "--source-root", str(args.source_root), "--sequence", args.sequence, "--frame", str(args.frame),
        "--target-rig", str(args.target_rig), "--input-root", str(input_root), "--image-name", image_name,
        "--output", str(distorted), "--distortion", str(args.distortion),
    ], python=sys.executable)
    stages["06_distorted_final"] = str(distorted)
    manifest = {
        "sequence": args.sequence, "frame": args.frame, "target_rig": str(args.target_rig),
        "resolution": [args.width, args.height], "default_result": "01_t0_surface/target_rig/<camera>/t0_highres_blue.png",
        "final_result_if_enabled": "06_distorted_final/target_rig/<camera>/final_distorted.png",
        "stages": stages, "timing_s": timings,
        "visual_artifacts": {
            "t0_highres_blue.png": "默认最终审计图：真实 RGB + 蓝色无效/未观测区域",
            "t0_dense_surface_to_rgb_pipeline.png": "去畸变 RGB、DA3、结构校准、目标表面、validity、T0 RGB 六步面板",
            "t0_semantic_surface_2px_repair_pipeline.png": "DINOv2/AnyUP、结构变化、表面修补与 RGB 面板",
            "background_layer_diagnostic.jpg": "DA3 前景/后景、三重回投影和 true disocclusion 诊断",
            "t0_uniview_background_gate.png": "仅有几何后景证据的 hole 填充结果",
            "rgbds_a_4k.png": "仅剩余 residual hole 经 RGB-D-S 补全的结果",
            "rgbds_a_4k_repaired.png": "仅对小裂缝执行 OpenCV 修补后的结果",
            "final_distorted.png": "最终输出：将 rectified 重建图按目标相机 raw K/D 畸变回去",
        },
        "contracts": ["rectified_rgb_newK_D0", "observed_pixels_immutable", "hole_only_completion"],
        "TODO": ["接入 Wan2.1/VACE 14B 时，仅允许在 residual true hole 运行时空生成，锁定 observed/gate latent，实现 4D 重建；当前未启用"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
