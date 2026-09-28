"""Render A-only completed anchors for video, without blue audit pixels."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

from gcr_nvs.datasets.manifest import CAMERA_NAMES, build_sequence_manifest
from gcr_nvs.datasets.t0_rgbds_diffusion import build_rgbds_conditions, resize_hole_mask
from gcr_nvs.dynamics.masks import DynamicMaskProvider
from gcr_nvs.models.dino_anyup_descriptor import FrozenDinoAnyUpDescriptor
from gcr_nvs.inference.rgbds_completion import load_adapter, prompt_embeddings, sample

DEFAULT_POSES = ("mixed_10_20cm",)
CAMERAS = ("CAM_BACK", "CAM_FRONT_NARROW", "CAM_FRONT_RIGHT")

def run(cmd, env=None):
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence-root", type=Path, required=True); p.add_argument("--sequence", required=True)
    p.add_argument("--cache-root", type=Path, required=True); p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--dynamic-root", type=Path, required=True); p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--distortion", type=Path, required=True)
    p.add_argument("--anyup-checkpoint", type=Path, required=True); p.add_argument("--anyup-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--adapter-a", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--compute-width", type=int, default=960)
    p.add_argument("--compute-height", type=int, default=540); p.add_argument("--output-width", type=int, default=1920)
    p.add_argument("--output-height", type=int, default=1080); p.add_argument("--inference-steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--poses", nargs="+", choices=("mixed_5_10cm", "mixed_10_20cm"), default=list(DEFAULT_POSES))
    args = p.parse_args(); started = time.perf_counter(); device = torch.device(args.device)
    records = build_sequence_manifest(args.sequence_root / args.sequence, compute_quality=False)
    work = args.output / "anchor_work"; work.mkdir(parents=True, exist_ok=True)
    renderer = Path(__file__).with_name("render_t0_highres_source_audit.py")
    diagnostic = Path(__file__).with_name("diagnose_t0_background_layers.py")
    gate_script = Path(__file__).with_name("apply_t0_uniview_background_gate.py")
    env = os.environ.copy(); env["PYTHONPATH"] = str(args.bundle / "src")
    env["GCR_NVS_ANYUP_CHECKPOINT"] = str(args.anyup_checkpoint); env["GCR_NVS_ANYUP_ROOT"] = str(args.anyup_root)
    render_started = time.perf_counter()
    for index, record in enumerate(records):
        frame = record.frame_id; audit = work / f"{frame:06d}"; layers = work / f"{frame:06d}_layers"; gate = work / f"{frame:06d}_gate"
        if not (audit / "summary.json").exists():
            run([sys.executable, str(renderer), "--bundle", str(args.bundle), "--root", str(args.sequence_root),
                 "--sequence", args.sequence, "--frame", str(frame), "--target-camera", "all", "--target-cameras", *CAMERAS,
                 "--output", str(audit), "--cache-root", str(args.cache_root), "--teacher-root", str(args.teacher_root),
                 "--dynamic-root", str(args.dynamic_root), "--checkpoint", str(args.checkpoint), "--distortion", str(args.distortion),
                 "--width", str(args.compute_width), "--height", str(args.compute_height), "--device", args.device,
                 "--depth-mode", "linear", "--poses", *args.poses], env=env)
        if not (layers / "report.json").exists():
            run([sys.executable, str(diagnostic), "--bundle", str(args.bundle), "--source-root", str(args.sequence_root),
                 "--audit-root", str(audit), "--repair-root", str(audit), "--output", str(layers), "--scale", "1.0", "--depth-layer", "da3", "--target-cameras", *CAMERAS, "--pose", args.poses[0]], env=env)
        for cam in CAMERAS:
            for pose in args.poses:
                if not (gate / pose / cam / "report.json").exists():
                    run([sys.executable, str(gate_script), "--t0-root", str(audit), "--gate-root", str(layers), "--output", str(gate), "--pose", pose, "--camera", cam], env=env)
        print(json.dumps({"stage":"geometry", "frame":frame, "index":index+1, "total":len(records)}), flush=True)
    geometry_s = time.perf_counter() - render_started
    pipe = StableDiffusionInpaintPipeline.from_pretrained(str(args.model), torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)
    for module in (pipe.vae, pipe.text_encoder, pipe.unet): module.requires_grad_(False).eval()
    prompt = prompt_embeddings(pipe, device).to(torch.float16); adapter = load_adapter(args.adapter_a, device)
    descriptor = FrozenDinoAnyUpDescriptor(model_name="dinov2_vitl14", output_channels=8, dino_input_size=(364,644), anyup_checkpoint=args.anyup_checkpoint, anyup_root=args.anyup_root, q_chunk_size=4096).to(device).eval()
    detector = DynamicMaskProvider(device="cuda", score_threshold=0.5); final_root = args.output / "frames" / args.sequence
    complete_started = time.perf_counter(); rows = []
    for record in records:
        frame = record.frame_id; audit = work / f"{frame:06d}"; gate = work / f"{frame:06d}_gate"
        for pose_index, pose in enumerate(args.poses):
            for camera_index, camera in enumerate(CAMERAS):
                source = audit / pose / camera; gate_cam = gate / pose / camera
                blue = np.asarray(Image.open(source / "t0_highres_blue.png").convert("RGB")); valid = np.load(source / "validity_after_2px_surface_repair.npy").astype(bool)
                depth = np.load(source / "target_surface_depth_2px_repaired.npy").astype(np.float32); base = np.asarray(Image.open(gate_cam / "t0_uniview_background_gate.png").convert("RGB")); gate_mask = np.asarray(Image.open(gate_cam / "uniview_gate_fill_mask.png").convert("L")) > 127
                residual = (~valid) & (~gate_mask); low_size = (512, 288); rgb_low = cv2.resize(base, low_size, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0; hole_low = resize_hole_mask(residual, low_size); rgb_low[hole_low] = 0.0
                depth_low = cv2.resize(depth, low_size, interpolation=cv2.INTER_NEAREST); dynamic, _, _ = detector(np.uint8(rgb_low*255)); image = torch.from_numpy(rgb_low.transpose(2,0,1).copy()).to(device)[None]
                with torch.inference_mode(): semantic = descriptor(image, (low_size[1], low_size[0]))[0].cpu().numpy()
                conditions = build_rgbds_conditions(rgb_low, depth_low, dynamic, semantic, hole_low); batch = {"target_rgb":torch.from_numpy(rgb_low.transpose(2,0,1).copy())[None],"masked_rgb":torch.from_numpy(conditions["masked_rgb"].transpose(2,0,1).copy())[None],"hole_mask":torch.from_numpy(conditions["hole_mask"][None,None].copy()),"control":torch.from_numpy(conditions["control"][None].copy()),"forbid_foreground_extension":torch.from_numpy(conditions["forbid_foreground_extension"][None,None].copy())}
                _, _, _, output = sample(pipe, adapter, batch, prompt, "a", args.inference_steps, 20260826 + pose_index*10000 + record.frame_id*10 + camera_index)
                generated = cv2.resize(output[0].float().cpu().permute(1,2,0).numpy().clip(0,1), (blue.shape[1], blue.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                final = base.copy(); final[residual] = np.uint8(generated[residual]*255); destination = final_root / camera / pose; destination.mkdir(parents=True, exist_ok=True); Image.fromarray(final).resize((args.output_width,args.output_height), Image.Resampling.LANCZOS).save(destination / f"{frame:06d}.jpg", quality=95, subsampling=0)
                rows.append({"frame_id":frame,"pose":pose,"camera":camera,"resolution":[args.output_width,args.output_height],"blue_pixels":int(((final[:,:,2]>180)&(final[:,:,0]<80)&(final[:,:,1]<140)).sum())})
        print(json.dumps({"stage":"a_only", "frame":frame}), flush=True)
    report = {"sequence":args.sequence,"poses":list(args.poses),"cameras":list(CAMERAS),"real_anchors":len(records),"geometry_seconds":geometry_s,"a_only_seconds":time.perf_counter()-complete_started,"total_seconds":time.perf_counter()-started,"resolution":[args.output_width,args.output_height],"contract":"A-only completion; no blue invalid pixels in video anchors","rows":rows}
    (args.output / "a_only_video_frames_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
