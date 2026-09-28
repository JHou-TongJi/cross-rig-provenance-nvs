"""High-resolution source-RGB T0+ render with blue invalid pixels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration, load_calibrations, rectify_image, rectified_calibration
from gcr_nvs.models.source_encoder import SourceImageEncoder
from gcr_nvs.models.t0_structure_constraint import T0StructureConstraintNet
from gcr_nvs.inference.t0_inputs import Ref, load_case as _load_case, to_device as _to_device, semantic_features as _semantic_features
from gcr_nvs.geometry.rig import load_target_rig, resolve_target_calibration
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb
from gcr_nvs.rendering.target_surface_completion import (
    complete_nearest_target_surface,
    inverse_sample_source_rgb,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))


POSES = {"mixed_5_10cm": (0.06, -0.08, 0.05), "mixed_10_20cm": (-0.14, 0.12, 0.18), "mixed_20_50cm": (0.32, -0.24, 0.42)}
BLUE_INVALID = np.asarray([25, 90, 235], np.float32) / 255.0


def _perturb(calibration, translation):
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, np.float64)
    return CameraCalibration(calibration.name, calibration.intrinsic.copy(), calibration.distortion.copy(), np.linalg.inv(camera_to_world), calibration.width, calibration.height)


def _raw_rectified(root: Path, sequence: str, record, camera: str, raw_calibrations, size):
    raw_path = root / sequence / record.cameras[camera]
    raw = np.asarray(Image.open(raw_path).convert("RGB"))
    rectified, calibration = rectify_image(raw, raw_calibrations[camera], output_size=size, alpha=0.0)
    return rectified.astype(np.float32) / 255.0, calibration, str(raw_path)


def _cache_calibration(cache_root: Path, sequence: str, frame: int, camera: str):
    root = cache_root / sequence / f"{frame:06d}" / camera
    report = json.loads((root / "report.json").read_text())
    return CameraCalibration(camera, np.asarray(report["intrinsic_newK"], np.float64), np.zeros_like(np.asarray(report["distortion_after_rectification"], np.float64)), np.asarray(report["external_world_to_camera"], np.float64), int(report["output_size"][0]), int(report["output_size"][1]))


def _load_highres_source(root, bundle, sequence, frame, camera, record, raw_cals, cache_root, teacher_root, dynamic_root, semantic_cache, model, encoder, device, highres, structure_size, depth_mode="linear"):
    started = time.perf_counter()
    rgb, native_cal, raw_path = _raw_rectified(root, sequence, record, camera, raw_cals, highres)
    rgb_rectification_s = time.perf_counter() - started
    stage = time.perf_counter()
    low_cal = _cache_calibration(cache_root, sequence, frame, camera)
    cache = cache_root / sequence / f"{frame:06d}" / camera
    confidence = cv2.resize(np.load(cache / "da3_confidence.npy").astype(np.float32), highres, interpolation=cv2.INTER_LINEAR)
    ref = Ref(sequence, frame, camera, "", str(teacher_root / sequence / f"{frame:06d}.npz"))
    case = _load_case(
        ref, root=root, cache_root=cache_root, teacher_root=teacher_root,
        dynamic_root=dynamic_root, distortion=bundle / "camera_intric.yaml",
        width=int(structure_size[0]), height=int(structure_size[1]),
        keep_ratio=0.58, seed=20260824 + CAMERA_NAMES.index(camera),
    )
    scalar, edges, rays, _, _ = _to_device(case, device)
    input_prepare_s = time.perf_counter() - stage
    stage = time.perf_counter()
    semantic = _semantic_features(case, encoder, device, semantic_cache)
    semantic_s = time.perf_counter() - stage
    stage = time.perf_counter()
    with torch.inference_mode():
        output = model(*scalar, edges, rays, semantic)
    structure_inference_s = time.perf_counter() - stage
    depth_range = output.depth_range_m[0, 0].cpu().numpy()
    camera_rays = np.einsum("ij,jhw->ihw", low_cal.external[:3, :3].astype(np.float32), case["rays"])
    low_z = depth_range * np.maximum(camera_rays[2], 1e-4)
    linear_z = cv2.resize(low_z.astype(np.float32), highres, interpolation=cv2.INTER_LINEAR)
    if depth_mode == "edge":
        nearest_z = cv2.resize(low_z.astype(np.float32), highres, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0)
        threshold = float(np.percentile(magnitude, 88.0))
        edge = cv2.dilate((magnitude > threshold).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        high_z = np.where(edge, nearest_z, linear_z).astype(np.float32)
    else:
        high_z = linear_z.astype(np.float32)
    return rgb, high_z, confidence, native_cal, raw_path, case, output, semantic[0].detach().cpu().numpy(), {
        "rgb_rectification_s": rgb_rectification_s,
        "input_prepare_s": input_prepare_s,
        "dino_anyup_s": semantic_s,
        "structure_network_s": structure_inference_s,
        "source_total_s": time.perf_counter() - started,
    }


def _blue_invalid(rgb, valid):
    result = np.asarray(rgb, np.float32).copy()
    result[~valid] = BLUE_INVALID
    return result


def _save(array, path: Path, blue_invalid_mask=None):
    image = np.asarray(array, np.float32).copy()
    if blue_invalid_mask is not None:
        image = _blue_invalid(image, blue_invalid_mask)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.1)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        log_depth = np.log(np.clip(depth[valid], 0.5, 160.0))
        lo, hi = np.log(0.5), np.log(160.0)
        normalized[valid] = np.rint((log_depth - lo) / (hi - lo) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)[..., ::-1]
    colored[~valid] = np.rint(BLUE_INVALID * 255).astype(np.uint8)
    return colored


def _semantic_visualization(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    channels, height, width = features.shape
    flat = features.reshape(channels, -1).T
    sample = flat[::max(1, len(flat) // 12000)]
    mean = sample.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(sample - mean, full_matrices=False)
    projected = (flat - mean) @ vectors[:3].T
    lo = np.percentile(projected, 2.0, axis=0)
    hi = np.percentile(projected, 98.0, axis=0)
    projected = np.clip((projected - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return np.rint(projected.reshape(height, width, 3) * 255).astype(np.uint8)


def _depth_delta_visualization(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    valid = np.isfinite(before) & np.isfinite(after) & (before > 0.1) & (after > 0.1)
    delta = np.zeros(before.shape, dtype=np.float32)
    delta[valid] = np.abs(np.log(after[valid]) - np.log(before[valid]))
    normalized = np.uint8(np.clip(delta / 0.12, 0.0, 1.0) * 255)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)[..., ::-1]
    colored[~valid] = 0
    return colored


def _pipeline_panel(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    font = ImageFont.truetype(str(font_path), 25) if font_path.exists() else ImageFont.load_default()
    thumb = (960, 540)
    title_height = 42
    canvas = Image.new("RGB", (thumb[0] * 3, (thumb[1] + title_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(entries):
        value = np.asarray(array)
        if value.dtype != np.uint8:
            value = np.rint(np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(value).resize(thumb, Image.Resampling.LANCZOS)
        x = index % 3 * thumb[0]
        y = index // 3 * (thumb[1] + title_height)
        draw.text((x + 12, y + 7), label, fill=(15, 20, 30), font=font)
        canvas.paste(image, (x, y + title_height))
    canvas.save(path, quality=95, subsampling=0)


def _hole_distance_report(validity: np.ndarray) -> dict[str, float]:
    hole = (~np.asarray(validity, dtype=bool)).astype(np.uint8)
    count = max(int(hole.sum()), 1)
    distance = cv2.distanceTransform(hole, cv2.DIST_L2, 5)
    return {
        "hole_fraction": float(hole.mean()),
        "hole_within_1px": float(((distance <= 1.0) & (hole > 0)).sum() / count),
        "hole_within_2px": float(((distance <= 2.0) & (hole > 0)).sum() / count),
        "hole_within_4px": float(((distance <= 4.0) & (hole > 0)).sum() / count),
        "hole_within_8px": float(((distance <= 8.0) & (hole > 0)).sum() / count),
        "maximum_hole_distance_px": float(distance.max(initial=0.0)),
    }


def _contact(output: Path, camera_names, pose):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    font = ImageFont.truetype(font_path, 20) if Path(font_path).exists() else ImageFont.load_default()
    tw, th = 960, 540
    canvas = Image.new("RGB", (tw * 3, (th + 32) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for i, camera in enumerate(camera_names):
        folder = output / pose / camera
        image = Image.open(folder / "t0_highres_blue.png").convert("RGB")
        image.thumbnail((tw, th))
        x = (i % 3) * tw; y = (i // 3) * (th + 32)
        canvas.paste(image, (x, y + 32))
        draw.text((x + 8, y + 6), camera, fill="black", font=font)
    canvas.save(output / f"{pose}_highres_seven_view.png", compress_level=2)


def main():
    default_bundle = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--sequence", default="2026-05-22-10-25-21")
    parser.add_argument("--frame", type=int, default=73)
    parser.add_argument("--target-camera", choices=(*CAMERA_NAMES, "all"), default="all")
    parser.add_argument("--target-cameras", nargs="+", choices=CAMERA_NAMES,
                        help="Render an explicit subset in one process (used by video batching).")
    parser.add_argument("--poses", nargs="+", choices=tuple(POSES),
                        help="Render only selected built-in pose groups.")
    parser.add_argument("--output", type=Path, default=default_bundle / "runs/t0_highres_source_audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-mode", choices=("linear", "edge"), default="linear")
    parser.add_argument("--root", type=Path, default=default_bundle / "datasets")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--dynamic-root", type=Path)
    parser.add_argument("--semantic-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--distortion", type=Path)
    parser.add_argument("--target-rig", type=Path, help="Optional YAML target camera rig with K/D/T and output sizes")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument(
        "--surface-repair-radius", type=float, default=0.0,
        help="maximum target-surface crack radius in pixels; 0 disables RGB repair",
    )
    args = parser.parse_args()
    bundle = args.bundle
    highres = (args.width, args.height)
    device = torch.device(args.device)
    runtime_started = time.perf_counter()
    cache_root = args.cache_root or bundle / "data/da3_depth_cache_dense_train_v1"
    teacher_root = args.teacher_root or bundle / "data/geometry_teacher_full_v1"
    dynamic_root = args.dynamic_root or bundle / "data/dynamic_masks_full_v1"
    semantic_cache = args.semantic_cache or bundle / "data/t0_semantic_cache_v1"
    checkpoint = args.checkpoint or bundle / "checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"
    distortion = args.distortion or bundle / "camera_intric.yaml"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    structure_size = (
        int(state.get("config", {}).get("width", 320)),
        int(state.get("config", {}).get("height", 180)),
    )
    model = T0StructureConstraintNet(int(state["config"].get("channels", 48)), float(state["config"].get("maximum_log_residual", 0.22))).to(device).eval()
    model.load_state_dict(state["model"])
    dino_model = str(state.get("config", {}).get("dino_model", "dinov2_vitb14"))
    dino_width = int(state.get("config", {}).get("dino_input_width", 448))
    dino_height = int(state.get("config", {}).get("dino_input_height", 252))
    anyup_checkpoint = Path(os.environ.get("GCR_NVS_ANYUP_CHECKPOINT", str(bundle / "outputs/checkpoints/anyup_multi_backbone.pth")))
    anyup_root = Path(os.environ.get("GCR_NVS_ANYUP_ROOT", str(bundle / "third_party/anyup")))
    encoder = SourceImageEncoder(feature_dim=32, use_dino=True, dino_model_name=dino_model, dino_pretrained=True, dino_input_size=(dino_height, dino_width), semantic_upsampling="anyup", anyup_checkpoint=anyup_checkpoint, anyup_root=anyup_root, anyup_q_chunk_size=4096).to(device).eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    record = next(row for row in build_sequence_manifest(args.root / args.sequence, compute_quality=False) if row.frame_id == args.frame)
    raw_cals = load_calibrations(args.root / args.sequence / record.camera_config, distortion)
    target_rig = load_target_rig(args.target_rig) if args.target_rig else None
    target_specs = {spec.name: spec for spec in target_rig.cameras} if target_rig else {}
    if target_rig:
        target_cameras = list(target_specs)
        target_source = {name: (spec.inherit_source_camera or name) for name, spec in target_specs.items()}
        unknown = sorted(set(target_source.values()) - set(CAMERA_NAMES))
        if unknown:
            raise ValueError(f"target rig sources are not captured cameras: {unknown}")
        topologies = {name: appearance_source_cameras(source, include_target=True) for name, source in target_source.items()}
    else:
        target_cameras = list(CAMERA_NAMES) if args.target_camera == "all" else [args.target_camera]
    if args.target_cameras:
        target_cameras = list(args.target_cameras)
        target_source = {camera: camera for camera in target_cameras}
        topologies = {camera: appearance_source_cameras(camera, include_target=True) for camera in target_cameras}
    required_sources = sorted({source for topology in topologies.values() for source in topology})
    sources = {}
    source_da3_depths = {}
    source_timings = {}
    for camera in required_sources:
        sources[camera] = _load_highres_source(
            args.root, bundle, args.sequence, args.frame, camera, record, raw_cals,
            cache_root, teacher_root, dynamic_root, semantic_cache, model, encoder,
            device, highres, structure_size, args.depth_mode,
        )
        source_timings[camera] = sources[camera][8]
        source_da3_depths[camera] = cv2.resize(
            np.load(cache_root / args.sequence / f"{args.frame:06d}" / camera / "aligned_dense_depth.npy").astype(np.float32),
            highres, interpolation=cv2.INTER_LINEAR,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    source_sidecars = args.output / "source_sidecars"
    source_sidecars.mkdir(exist_ok=True)
    for camera, source in sources.items():
        folder = source_sidecars / camera
        folder.mkdir(exist_ok=True)
        np.save(folder / "da3_depth_before_structure.npy", source_da3_depths[camera].astype(np.float32))
        np.save(folder / "depth.npy", source[1].astype(np.float32))
        np.save(folder / "confidence.npy", source[2].astype(np.float32))
        np.save(folder / "dino_anyup_semantic_320x180.npy", source[7].astype(np.float16))
        Image.fromarray(_depth_visualization(source_da3_depths[camera])).save(folder / "da3_depth_before_structure.png")
        Image.fromarray(_depth_visualization(source[1])).save(folder / "depth_after_structure.png")
        Image.fromarray(_semantic_visualization(source[7])).save(folder / "dino_anyup_semantic_pca.png")
        Image.fromarray(_depth_delta_visualization(source_da3_depths[camera], source[1])).save(
            folder / "structure_depth_change.png"
        )
    reports = {}
    pose_definitions = {"target_rig": (0.0, 0.0, 0.0)} if target_rig else {name: POSES[name] for name in (args.poses or POSES)}
    for pose_name, translation in pose_definitions.items():
        pose_root = args.output / pose_name
        pose_root.mkdir(exist_ok=True)
        for camera in target_cameras:
            source_camera = target_source[camera]
            topology = topologies[camera]
            if target_rig:
                spec = target_specs[camera]
                raw_target = resolve_target_calibration(spec, raw_cals)
                target_size = (int(spec.width), int(spec.height))
                target = rectified_calibration(raw_target, output_size=target_size, alpha=0.0)
            else:
                target_size = highres
                target = _perturb(sources[source_camera][3], translation)
            items = [sources[name] for name in topology]
            stage_started = time.perf_counter()
            t0, t0_valid, t0_prov, t0_depth = reproject_dense_rgb(
                [x[0] for x in items], [x[1] for x in items], [x[3] for x in items], target,
                source_confidences=[x[2] for x in items], splat_radius=0,
                exclusive_source_priority=True, return_depth=True,
            )
            if args.surface_repair_radius > 0.0:
                surface_candidate = complete_nearest_target_surface(
                    t0_depth, t0_valid, maximum_distance_px=args.surface_repair_radius,
                )
                inverse = inverse_sample_source_rgb(
                    surface_candidate.depth, target,
                    [x[0] for x in items], [x[1] for x in items], [x[3] for x in items],
                    surface_candidate.candidate_mask,
                    source_confidences=[x[2] for x in items],
                    maximum_log_depth_error=0.02,
                )
            else:
                surface_candidate = complete_nearest_target_surface(
                    t0_depth, t0_valid, maximum_distance_px=0.0,
                )
                from gcr_nvs.rendering.target_surface_completion import InverseSampling
                inverse = InverseSampling(
                    np.zeros_like(t0), np.zeros_like(t0_valid),
                    np.full_like(t0_prov, -1, dtype=np.int16),
                    np.full_like(t0_depth, np.inf, dtype=np.float32),
                )
            repaired_rgb = t0.copy()
            repaired_rgb[inverse.validity] = inverse.rgb[inverse.validity]
            repaired_valid = t0_valid | inverse.validity
            repaired_depth = t0_depth.copy()
            repaired_depth[inverse.validity] = surface_candidate.depth[inverse.validity]
            repaired_provenance = t0_prov.copy()
            repaired_provenance[inverse.validity] = inverse.provenance[inverse.validity]
            projection_s = time.perf_counter() - stage_started
            folder = pose_root / camera; folder.mkdir(exist_ok=True)
            _save(sources[source_camera][0], folder / "source_native_rectified.png")
            _save(t0, folder / "t0_highres_blue.png", t0_valid)
            _save(t0_valid[..., None].repeat(3, axis=2).astype(np.float32), folder / "validity.png")
            np.save(folder / "validity.npy", t0_valid)
            np.save(folder / "depth.npy", t0_depth.astype(np.float32))
            np.save(folder / "target_surface_depth_before_rgb.npy", t0_depth.astype(np.float32))
            np.save(folder / "target_surface_depth_2px_candidate.npy", surface_candidate.depth.astype(np.float32))
            np.save(folder / "target_surface_depth_2px_repaired.npy", repaired_depth.astype(np.float32))
            np.save(folder / "surface_2px_candidate_mask.npy", surface_candidate.candidate_mask)
            np.save(folder / "surface_2px_repaired_mask.npy", inverse.validity)
            np.save(folder / "validity_after_2px_surface_repair.npy", repaired_valid)
            np.save(folder / "provenance_after_2px_surface_repair.npy", repaired_provenance.astype(np.int16))
            np.save(folder / "provenance.npy", t0_prov.astype(np.int16))
            target_depth_vis = _depth_visualization(t0_depth)
            Image.fromarray(target_depth_vis).save(folder / "target_surface_depth_before_rgb.png")
            Image.fromarray(_depth_visualization(repaired_depth)).save(folder / "target_surface_depth_after_2px_repair.png")
            _save(inverse.validity[..., None].repeat(3, axis=2).astype(np.float32), folder / "surface_2px_repaired_mask.png")
            _save(repaired_rgb, folder / "t0_surface_2px_repaired_blue.png", repaired_valid)
            _pipeline_panel(folder / "t0_dense_surface_to_rgb_pipeline.png", [
                ("① 去畸变源 RGB", sources[source_camera][0]),
                ("② 原始 DA3 稠密深度", _depth_visualization(source_da3_depths[camera])),
                ("③ 结构网络校准后源表面", _depth_visualization(sources[camera][1])),
                ("④ SE(3)+z-buffer 后目标表面（RGB 前）", target_depth_vis),
                ("⑤ 目标表面有效性 / 空洞", t0_valid[..., None].repeat(3, axis=2).astype(np.float32)),
                ("⑥ 从源图取色后的 T0 RGB", _blue_invalid(t0, t0_valid)),
            ])
            _pipeline_panel(folder / "t0_semantic_surface_2px_repair_pipeline.png", [
                ("① DINOv2-B + AnyUP 隐语义 PCA", cv2.resize(
                    _semantic_visualization(sources[camera][7]), highres, interpolation=cv2.INTER_NEAREST,
                )),
                ("② 语义参与后的深度改变量", _depth_delta_visualization(
                    source_da3_depths[camera], sources[camera][1],
                )),
                ("③ SE(3) 后目标表面（修补前）", target_depth_vis),
                ("④ 2px 表面修补且通过源深度验证", inverse.validity[..., None].repeat(3, axis=2).astype(np.float32)),
                ("⑤ 2px 修补后的目标表面", _depth_visualization(repaired_depth)),
                ("⑥ 只剩大洞的真实源 RGB 结果", _blue_invalid(repaired_rgb, repaired_valid)),
            ])
            hole_report = _hole_distance_report(t0_valid)
            original_holes = max(int((~t0_valid).sum()), 1)
            report = {
                "pose": translation, "resolution": target_size,
                "coverage": float(t0_valid.mean()),
                "source_rgb": "raw 3840x2160 rectified", "splat_radius": 0,
                "invalid_color_rgb": [25, 90, 235], "lidar_direct_rgb_override": False,
                "target_surface_before_rgb": "target_surface_depth_before_rgb.npy",
                "surface_2px_candidate_fraction_of_hole": float(
                    surface_candidate.candidate_mask.sum() / original_holes
                ),
                "surface_repair_radius_px": float(args.surface_repair_radius),
                "surface_2px_recovered_fraction_of_hole": float(inverse.validity.sum() / original_holes),
                "coverage_after_2px_surface_repair": float(repaired_valid.mean()),
                "surface_repair_contract": "nearest target depth candidate; inverse source RGB; source depth log error <= 0.04",
                "timing_s": {
                    **source_timings[source_camera],
                    "target_se3_zbuffer_rgb_sampling_s": projection_s,
                    "camera_total_s": source_timings[source_camera]["source_total_s"] + projection_s,
                },
                **hole_report,
            }
            (folder / "report.json").write_text(json.dumps(report, indent=2))
            reports.setdefault(pose_name, {})[camera] = {
                "coverage": float(t0_valid.mean()), "resolution": target_size,
                "source_topology": list(topology),
                "surface_2px_recovered_fraction_of_hole": float(inverse.validity.sum() / original_holes),
                "coverage_after_2px_surface_repair": float(repaired_valid.mean()),
                "timing_s": report["timing_s"],
                **hole_report,
            }
    for pose in pose_definitions:
        _contact(args.output, target_cameras, pose)
    resolutions = {name: [int(target_specs[name].width), int(target_specs[name].height)] for name in target_cameras} if target_rig else {name: [int(highres[0]), int(highres[1])] for name in target_cameras}
    runtime = {
        "total_s": time.perf_counter() - runtime_started,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" and torch.cuda.is_available() else 0.0,
    }
    (args.output / "summary.json").write_text(json.dumps({"sequence": args.sequence, "frame": args.frame, "target_cameras": target_cameras, "source_topologies": topologies, "reports": reports, "resolution": resolutions, "runtime": runtime, "invalid_color": "blue", "depth_mode": args.depth_mode, "target_rig": str(args.target_rig) if args.target_rig else None, "source_depth_sidecars": "source_sidecars/<camera>/{da3_depth_before_structure,depth,confidence}.npy", "source_semantic_sidecar": "source_sidecars/<camera>/dino_anyup_semantic_320x180.npy", "target_surface_before_rgb": "<pose>/<camera>/target_surface_depth_before_rgb.{npy,png}", "pipeline_panel": "<pose>/<camera>/t0_dense_surface_to_rgb_pipeline.png", "surface_repair_panel": "<pose>/<camera>/t0_semantic_surface_2px_repair_pipeline.png", "surface_repair": "2px target crack candidate + inverse source sampling + source-depth consistency", "semantic_decoder": "DINOv2-B + AnyUP hidden features condition structure depth; PCA is visualization only"}, indent=2))
    print(json.dumps({"output": str(args.output), "resolution": resolutions, "poses": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
