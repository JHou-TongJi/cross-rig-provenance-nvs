"""Export every intermediate artifact of one T0+structure inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gcr_nvs.datasets.manifest import CAMERA_NAMES, appearance_source_cameras, build_sequence_manifest
from gcr_nvs.geometry.calibration import CameraCalibration
from gcr_nvs.models.source_encoder import SourceImageEncoder
from gcr_nvs.models.t0_structure_constraint import T0StructureConstraintNet
from gcr_nvs.rendering.dense_depth_reprojection import reproject_dense_rgb
from gcr_nvs.inference.t0_inputs import Ref, load_case, semantic_features, to_device


POSES = {
    "zero": (0.0, 0.0, 0.0),
    "mixed_10_20cm": (-0.14, 0.12, 0.18),
}


def _rectified_cache(cache_root: Path, sequence: str, frame: int, camera: str):
    root = cache_root / sequence / f"{frame:06d}" / camera
    report = json.loads((root / "report.json").read_text())
    rgb = np.asarray(Image.open(root / "rgb_rectified.jpg").convert("RGB"), np.float32) / 255.0
    depth = np.load(root / "aligned_dense_depth.npy").astype(np.float32)
    confidence = np.load(root / "da3_confidence.npy").astype(np.float32)
    calibration = CameraCalibration(
        camera, np.asarray(report["intrinsic_newK"], np.float64),
        np.zeros_like(np.asarray(report["distortion_after_rectification"], np.float64)),
        np.asarray(report["external_world_to_camera"], np.float64),
        int(report["output_size"][0]), int(report["output_size"][1]),
    )
    return rgb, depth, confidence, calibration


def _perturb(calibration: CameraCalibration, translation):
    camera_to_world = np.linalg.inv(calibration.external)
    camera_to_world[:3, 3] += np.asarray(translation, np.float64)
    return CameraCalibration(
        calibration.name, calibration.intrinsic.copy(), calibration.distortion.copy(),
        np.linalg.inv(camera_to_world), calibration.width, calibration.height,
    )


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _norm(value: np.ndarray, valid: np.ndarray | None = None, cmap=cv2.COLORMAP_TURBO):
    value = np.asarray(value, np.float32)
    unique = np.unique(value[np.isfinite(value)])
    if len(unique) <= 2 and np.all(np.isin(unique, [0.0, 1.0])):
        return np.repeat((value > 0.5).astype(np.float32)[..., None], 3, axis=2)
    finite = np.isfinite(value)
    if valid is not None:
        finite &= valid
    if not finite.any():
        scaled = np.zeros_like(value, np.uint8)
    else:
        lo, hi = np.percentile(value[finite], [2, 98])
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        scaled = np.clip((value - lo) / (hi - lo), 0, 1)
        scaled[~finite] = 0
        scaled = np.rint(scaled * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cmap), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _rgb(value: np.ndarray):
    value = np.asarray(value)
    if value.ndim == 2:
        return np.repeat(value[..., None], 3, axis=2).astype(np.float32)
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = np.moveaxis(value, 0, -1)
    return np.clip(value[..., :3], 0, 1).astype(np.float32)


def _pca_rgb(features: np.ndarray):
    x = np.asarray(features, np.float32).reshape(features.shape[0], -1).T
    x -= x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:3].T
    lo, hi = np.percentile(y, [2, 98], axis=0)
    y = np.clip((y - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return y.reshape(features.shape[1], features.shape[2], 3).astype(np.float32)


def _save(path: Path, array: np.ndarray):
    array = np.asarray(array)
    np.save(path.with_suffix(".npy"), array)
    if array.ndim == 2:
        image = _norm(array)
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        image = _rgb(array)
    else:
        image = _norm(array[0] if array.ndim == 3 else array)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path.with_suffix(".png"))


def _trace_panel(items: list[tuple[str, np.ndarray]], output: Path, tile=(320, 180)):
    cols = 4
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile[0], rows * (tile[1] + 26)), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(16)
    for i, (label, image) in enumerate(items):
        image = cv2.resize(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), tile)
        x = (i % cols) * tile[0]
        y = (i // cols) * (tile[1] + 26)
        canvas.paste(Image.fromarray(image), (x, y + 26))
        draw.text((x + 5, y + 4), label, fill=(20, 20, 20), font=label_font)
    canvas.save(output, quality=95)


def main():
    bundle = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="2026-05-22-10-25-21")
    parser.add_argument("--frame", type=int, default=73)
    parser.add_argument("--camera", choices=CAMERA_NAMES, default="CAM_FRONT_NARROW")
    parser.add_argument("--pose", choices=tuple(POSES), default="mixed_10_20cm")
    parser.add_argument("--output", type=Path, default=bundle / "runs/t0_inference_trace_v2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache_root = bundle / "data/da3_depth_cache_dense_train_v1"
    teacher_root = bundle / "data/geometry_teacher_full_v1"
    dynamic_root = bundle / "data/dynamic_masks_full_v1"
    semantic_cache = bundle / "data/t0_semantic_cache_v1"
    checkpoint = bundle / "checkpoints/t0_structure_rectified_v2_dinob14_320x180_c48_best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = T0StructureConstraintNet(int(config.get("channels", 48)), float(config.get("maximum_log_residual", 0.22))).to(device).eval()
    model.load_state_dict(state["model"])
    encoder = SourceImageEncoder(
        feature_dim=32, use_dino=True, dino_model_name="dinov2_vitb14", dino_pretrained=True,
        dino_input_size=(252, 448), semantic_upsampling="anyup",
        anyup_checkpoint=bundle / "outputs/checkpoints/anyup_multi_backbone.pth",
        anyup_root=bundle / "third_party/anyup", anyup_q_chunk_size=4096,
    ).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    source_rgb, source_depth, source_conf, calibration = _rectified_cache(cache_root, args.sequence, args.frame, args.camera)
    teacher_path = teacher_root / args.sequence / f"{args.frame:06d}.npz"
    ref = Ref(args.sequence, args.frame, args.camera, "", str(teacher_path))
    namespace = argparse.Namespace(
        root=bundle / "datasets", cache_root=cache_root, teacher_root=teacher_root,
        dynamic_root=dynamic_root, distortion=bundle / "camera_intric.yaml", width=320, height=180,
    )
    case = load_case(ref, root=bundle / "datasets", cache_root=cache_root,
                     teacher_root=teacher_root, dynamic_root=dynamic_root,
                     distortion=bundle / "camera_intric.yaml", width=320, height=180,
                     keep_ratio=0.58, seed=20260824)
    scalar, edges, rays, _, _ = to_device(case, device)
    semantic = semantic_features(case, encoder, device, semantic_cache)
    with torch.inference_mode():
        output = model(*scalar, edges, rays, semantic)
    depth_range = output.depth_range_m[0, 0].cpu().numpy()
    residual = output.residual_log_range[0, 0].cpu().numpy()
    gate = output.correction_gate[0, 0].cpu().numpy()
    geom_conf = output.geometry_confidence[0, 0].cpu().numpy()
    boundary = output.boundary_probability[0, 0].cpu().numpy()
    semantic_np = semantic[0].cpu().numpy()
    semantic_pca = _pca_rgb(semantic_np)
    camera_rays = np.einsum("ij,jhw->ihw", calibration.external[:3, :3].astype(np.float32), case["rays"])
    corrected_z = cv2.resize(depth_range * np.maximum(camera_rays[2], 1e-4), (source_rgb.shape[1], source_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    adapted = (source_rgb, corrected_z, source_conf, calibration)
    topology = appearance_source_cameras(args.camera, include_target=True)
    source_items = [_rectified_cache(cache_root, args.sequence, args.frame, name) for name in topology]
    adapted_items = [adapted if name == args.camera else _rectified_cache(cache_root, args.sequence, args.frame, name) for name in topology]
    target = _perturb(calibration, POSES[args.pose])
    t0_rgb, t0_valid, t0_prov = reproject_dense_rgb([x[0] for x in source_items], [x[1] for x in source_items], [x[3] for x in source_items], target, source_confidences=[x[2] for x in source_items], splat_radius=1, exclusive_source_priority=True)
    new_rgb, new_valid, new_prov = reproject_dense_rgb([x[0] for x in adapted_items], [x[1] for x in adapted_items], [x[3] for x in adapted_items], target, source_confidences=[x[2] for x in adapted_items], splat_radius=1, exclusive_source_priority=True)
    raw_path = None
    records = build_sequence_manifest(bundle / "datasets" / args.sequence, compute_quality=False)
    record = next(item for item in records if item.frame_id == args.frame)
    raw_path = bundle / "datasets" / args.sequence / record.cameras[args.camera]
    raw_rgb = np.asarray(Image.open(raw_path).convert("RGB"), np.float32) / 255.0
    rectified_rgb = source_rgb
    artifacts = {
        "01_raw_rgb": raw_rgb,
        "02_rectified_rgb": rectified_rgb,
        "03_da3_depth_z": cv2.resize(np.load(cache_root / args.sequence / f"{args.frame:06d}" / args.camera / "aligned_dense_depth.npy"), (320, 180), interpolation=cv2.INTER_LINEAR),
        "04_da3_confidence": cv2.resize(np.load(cache_root / args.sequence / f"{args.frame:06d}" / args.camera / "da3_confidence.npy"), (320, 180), interpolation=cv2.INTER_LINEAR),
        "05_current_lidar_range": case["current_range"],
        "06_current_lidar_validity": case["current_valid"],
        "07_temporal_lidar_range": case["temporal_range"],
        "08_temporal_lidar_validity": case["temporal_valid"],
        "09_lidar_support_distance_px": case["seed_distance"],
        "10_rgb_edges": np.moveaxis(case["rgb_edges"], 0, -1),
        "11_dinov2_anyup_pca": semantic_pca,
        "12_corrected_range_m": depth_range,
        "13_residual_log_range": residual,
        "14_correction_gate": gate,
        "15_geometry_confidence": geom_conf,
        "16_boundary_probability": boundary,
        "17_corrected_depth_z": corrected_z,
        "18_t0_original_rgb": t0_rgb * t0_valid[..., None],
        "19_t0_structure_rgb": new_rgb * new_valid[..., None],
        "20_t0_original_validity": t0_valid,
        "21_t0_structure_validity": new_valid,
        "22_t0_structure_provenance": new_prov.astype(np.float32),
    }
    names_cn = {name: name.replace("_", " ") for name in artifacts}
    items = []
    for name, array in artifacts.items():
        path = args.output / name
        _save(path, array)
        items.append((names_cn[name], _rgb(array)))
    _trace_panel(items, args.output / "00_full_inference_trace.jpg")
    report = {
        "route": "T0_structure_constraint_v2_inference_trace",
        "sequence": args.sequence, "frame": args.frame, "camera": args.camera,
        "pose": {"name": args.pose, "translation_xyz_m": POSES[args.pose]},
        "source_topology": list(topology), "resolution": [int(source_rgb.shape[1]), int(source_rgb.shape[0])],
        "training_checkpoint": str(checkpoint), "checkpoint_step": state.get("step"),
        "artifacts": sorted(artifacts), "raw_rgb_path": str(raw_path),
        "coverage": {"t0": float(t0_valid.mean()), "t0_structure": float(new_valid.mean())},
        "depth_stats": {"da3_z_median": float(np.median(artifacts["03_da3_depth_z"])), "corrected_z_median": float(np.median(corrected_z)), "residual_abs_mean": float(np.mean(np.abs(residual))), "gate_mean": float(gate.mean()), "geometry_confidence_mean": float(geom_conf.mean())},
        "rgb_policy": "source RGB z-buffer sampling; LiDAR never supplies RGB; completion disabled",
    }
    (args.output / "trace_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
